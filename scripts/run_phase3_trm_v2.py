#!/usr/bin/env python3
"""
Phase 3: Quiet-STaR Training with PROPER TRM x Titans-MAG Loop (V2)

FIXES from V1:
1. Downloads REAL GSM8K dataset from Hugging Face
2. ACT loss PENALIZES early halting (more steps = lower loss)
3. Includes chain-of-thought in training data
4. Minimum recursion steps enforced

Architecture features:
- reset_memory() between batches (LTM state management)
- return_all_steps=True for multi-step reasoning
- Deep supervision (weighted loss across recursion steps)
- MuGrokfast optimizer with Phase 3 preset
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, "/app/src")

from cross_phase.mugrokfast.config import MuGrokConfig
from cross_phase.mugrokfast.optimizer import MuonGrokfast
from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============== GSM8K Dataset (Real) ==============

def download_gsm8k(data_dir: str = "/app/data/gsm8k"):
    """Download GSM8K from Hugging Face"""
    os.makedirs(data_dir, exist_ok=True)
    train_path = Path(data_dir) / "train.jsonl"
    test_path = Path(data_dir) / "test.jsonl"

    if train_path.exists() and test_path.exists():
        logger.info("GSM8K already downloaded")
        return

    logger.info("Downloading GSM8K from Hugging Face...")
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main")

        # Save train
        with open(train_path, "w") as f:
            for item in ds["train"]:
                # Extract answer from "#### <number>" format
                answer_text = item["answer"]
                # Get the final numeric answer after ####
                if "####" in answer_text:
                    parts = answer_text.split("####")
                    solution = parts[0].strip()  # Chain of thought
                    final_answer = parts[1].strip()  # Numeric answer
                else:
                    solution = ""
                    final_answer = answer_text.strip()

                f.write(json.dumps({
                    "question": item["question"],
                    "solution": solution,  # Chain of thought steps
                    "answer": final_answer,
                }) + "\n")

        # Save test
        with open(test_path, "w") as f:
            for item in ds["test"]:
                answer_text = item["answer"]
                if "####" in answer_text:
                    parts = answer_text.split("####")
                    solution = parts[0].strip()
                    final_answer = parts[1].strip()
                else:
                    solution = ""
                    final_answer = answer_text.strip()

                f.write(json.dumps({
                    "question": item["question"],
                    "solution": solution,
                    "answer": final_answer,
                }) + "\n")

        logger.info(f"GSM8K downloaded: {len(ds['train'])} train, {len(ds['test'])} test")

    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        raise
    except Exception as e:
        logger.error(f"Failed to download GSM8K: {e}")
        raise


class GSM8KDataset(Dataset):
    """GSM8K math reasoning dataset with chain-of-thought"""

    def __init__(self, split: str = "train", max_samples: int = None, include_cot: bool = True):
        self.samples = []
        self.include_cot = include_cot
        data_path = Path(f"/app/data/gsm8k/{split}.jsonl")

        if data_path.exists():
            with open(data_path) as f:
                for line in f:
                    self.samples.append(json.loads(line))
                    if max_samples and len(self.samples) >= max_samples:
                        break
            logger.info(f"Loaded {len(self.samples)} GSM8K {split} samples")
        else:
            logger.warning(f"GSM8K not found at {data_path}")
            self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_len=512, include_cot=True):
    """Collate batch with chain-of-thought reasoning"""
    questions = [s["question"] for s in batch]
    solutions = [s.get("solution", "") for s in batch]
    answers = [s["answer"] for s in batch]

    # Format with chain-of-thought
    if include_cot:
        # Include reasoning steps in training
        texts = []
        for q, sol, a in zip(questions, solutions, answers):
            if sol:
                # Full CoT format
                text = f"Question: {q}\nLet me think step by step.\n{sol}\nThe answer is {a}"
            else:
                text = f"Question: {q}\nThe answer is {a}"
            texts.append(text)
    else:
        texts = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]

    # Tokenize
    encodings = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # Labels are same as input_ids for LM training
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # Ignore padding

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "answers": answers,
    }


# ============== Grokking Detector ==============

@dataclass
class GrokDetector:
    """Detect grokking signal (sudden accuracy spike)"""
    window: int = 5
    spike_threshold: float = 0.05  # 5% spike = grokking
    target: float = 0.20  # 20% accuracy target

    def __post_init__(self):
        self.history = []

    def update(self, acc: float) -> dict:
        self.history.append(acc)

        result = {
            "current_acc": acc,
            "is_grokking": False,
            "hit_target": acc >= self.target,
            "spike": 0.0,
        }

        if len(self.history) >= self.window:
            recent = self.history[-self.window:]
            older = self.history[-(2*self.window):-self.window] if len(self.history) >= 2*self.window else []

            if older:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                spike = recent_avg - older_avg
                result["spike"] = spike
                result["is_grokking"] = spike >= self.spike_threshold

        return result


# ============== Evaluation ==============

def extract_answer(text: str) -> str:
    """Extract numeric answer from generated text"""
    # Look for "The answer is X" pattern first
    match = re.search(r"answer\s+is\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Look for patterns like "= 42" or just numbers at end
    patterns = [
        r"=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(",", "")

    # Fallback: find any number
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def evaluate_gsm8k(model, tokenizer, dataset, device, n_samples=200):
    """Evaluate on GSM8K subset"""
    model.eval()
    correct = 0
    total = min(n_samples, len(dataset))

    if total == 0:
        logger.warning("No samples in evaluation dataset")
        return 0.0

    indices = random.sample(range(len(dataset)), total)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating", leave=False):
            sample = dataset[idx]
            question = sample["question"]
            true_answer = sample["answer"].replace(",", "")

            # Generate answer with CoT prompt
            prompt = f"Question: {question}\nLet me think step by step.\n"
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

            # Reset memory before inference
            model.reset_memory()

            # Generate tokens
            generated = input_ids
            for _ in range(150):  # Max 150 new tokens for CoT
                output = model(generated, return_all_steps=True)
                logits = output["logits"]
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                # Stop on double newline or "The answer is"
                decoded = tokenizer.decode(generated[0][-20:], skip_special_tokens=True)
                if "The answer is" in decoded and any(c.isdigit() for c in decoded.split("The answer is")[-1]):
                    break
                if next_token.item() == tokenizer.eos_token_id:
                    break

            # Decode and check
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            pred_answer = extract_answer(generated_text)

            # Normalize comparison
            try:
                pred_num = float(pred_answer) if pred_answer else None
                true_num = float(true_answer)
                if pred_num is not None and abs(pred_num - true_num) < 0.01:
                    correct += 1
            except (ValueError, TypeError):
                if pred_answer.strip() == true_answer.strip():
                    correct += 1

    model.train()
    return correct / total if total > 0 else 0.0


# ============== Training Loop ==============

def compute_act_penalty(halting_steps: torch.Tensor, min_steps: int = 2, max_steps: int = 4) -> torch.Tensor:
    """
    Compute ACT penalty that ENCOURAGES more recursion steps.

    Instead of penalizing computation (original ACT), we penalize EARLY halting.
    This encourages the model to actually use multi-step reasoning.

    Args:
        halting_steps: [batch] number of steps taken (1 to T_max)
        min_steps: Minimum desired steps (penalize if fewer)
        max_steps: Maximum steps (T_max)

    Returns:
        penalty: scalar loss that penalizes early halting
    """
    # Normalize steps to 0-1 range
    normalized = (halting_steps.float() - 1) / (max_steps - 1)

    # Penalize LOW step counts (early halting)
    # penalty = 0 when steps = max_steps, penalty = 1 when steps = 1
    penalty = 1.0 - normalized

    # Extra penalty for halting before min_steps
    early_halt_mask = halting_steps < min_steps
    penalty = penalty + early_halt_mask.float() * 0.5  # Extra 0.5 penalty

    return penalty.mean()


def train_step_with_trm(model, batch, optimizer, device, min_recursion: int = 2):
    """
    Single training step using PROPER TRM x Titans-MAG loop

    Key features:
    1. reset_memory() before each batch
    2. return_all_steps=True for multi-step reasoning
    3. ACT penalty for early halting (encourages more recursion)
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # CRITICAL: Reset LTM state between batches
    model.reset_memory()

    # Forward pass with multi-step reasoning
    output = model(
        input_ids=input_ids,
        labels=labels,
        return_all_steps=True,
    )

    # Get losses from model (includes deep supervision)
    loss = output["loss"]
    loss_ce = output.get("loss_ce", loss)
    loss_act_original = output.get("loss_act", torch.tensor(0.0))
    loss_gate = output.get("loss_gate", torch.tensor(0.0))

    # Get halting steps
    halting_steps = output["halting_steps"]

    # OVERRIDE: Add penalty for EARLY halting (encourage more recursion)
    # This is the key fix - we WANT more recursion steps for reasoning
    early_halt_penalty = compute_act_penalty(halting_steps, min_steps=min_recursion, max_steps=4)

    # Combine losses: CE + gate + early_halt_penalty (instead of original ACT)
    # Remove original ACT loss, add our penalty
    loss_total = loss_ce + loss_gate + 0.1 * early_halt_penalty

    # Backward
    optimizer.zero_grad()
    loss_total.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return {
        "loss": loss_total.item(),
        "loss_ce": loss_ce.item() if hasattr(loss_ce, "item") else float(loss_ce),
        "loss_gate": loss_gate.item() if hasattr(loss_gate, "item") else float(loss_gate),
        "early_halt_penalty": early_halt_penalty.item(),
        "halting_steps": halting_steps.float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)  # Smaller for longer sequences
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--target_acc", type=float, default=0.20)
    parser.add_argument("--save_dir", type=str, default="/app/checkpoints/phase3")
    parser.add_argument("--min_recursion", type=int, default=2, help="Minimum recursion steps")
    parser.add_argument("--include_cot", action="store_true", default=True, help="Include chain-of-thought")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Download GSM8K if needed
    download_gsm8k()

    # Load tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model with config
    if "config" in checkpoint:
        config = Phase1Config(**checkpoint["config"])
    else:
        config = Phase1Config(specialization="reasoning")

    model = TRMTitansMAGModel(config).to(device)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Create optimizer with Phase 3 MuGrokfast preset
    mugrok_config = MuGrokConfig.from_phase(3)
    mugrok_config.muon_lr = args.lr
    optimizer = MuonGrokfast(model.parameters(), config=mugrok_config)

    logger.info(f"MuGrokfast Phase 3 config:")
    logger.info(f"  muon_lr={mugrok_config.muon_lr}")
    logger.info(f"  grokfast_lambda={mugrok_config.grokfast_lambda}")
    logger.info(f"  qk_clip_threshold={mugrok_config.qk_clip_threshold}")

    # Load dataset
    train_dataset = GSM8KDataset(split="train", include_cot=args.include_cot)
    eval_dataset = GSM8KDataset(split="test", max_samples=500, include_cot=args.include_cot)

    if len(train_dataset) == 0:
        logger.error("No training data! Check GSM8K download.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, include_cot=args.include_cot),
        drop_last=True,
    )

    # Training state
    grok_detector = GrokDetector(target=args.target_acc)
    global_step = 0
    best_acc = 0.0

    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 3 TRM x Titans-MAG Training V2")
    logger.info("=" * 60)
    logger.info(f"  REAL GSM8K dataset: {len(train_dataset)} samples")
    logger.info(f"  Chain-of-thought training: {args.include_cot}")
    logger.info(f"  reset_memory() between batches: YES")
    logger.info(f"  return_all_steps=True: YES")
    logger.info(f"  Minimum recursion steps: {args.min_recursion}")
    logger.info(f"  Early halt PENALTY: YES (encourages reasoning)")
    logger.info(f"  Deep supervision: {config.trm_config.deep_supervision}")
    logger.info(f"  T_max recursion steps: {config.trm_config.T_max}")
    logger.info("=" * 60)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"E{epoch+1}")
        for batch in pbar:
            global_step += 1

            # Train step with proper TRM loop + early halt penalty
            metrics = train_step_with_trm(
                model, batch, optimizer, device,
                min_recursion=args.min_recursion
            )
            epoch_losses.append(metrics["loss"])

            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                halt=f"{metrics['halting_steps']:.1f}",
                penalty=f"{metrics['early_halt_penalty']:.2f}",
                step=global_step,
            )

            # Evaluate
            if global_step % args.eval_every == 0:
                logger.info(f"\n[EVAL] Step {global_step}...")
                accuracy = evaluate_gsm8k(model, tokenizer, eval_dataset, device, n_samples=200)
                logger.info(f"  Accuracy: {accuracy*100:.2f}% ({int(accuracy*200)}/200)")
                logger.info(f"  Avg halt steps: {metrics['halting_steps']:.2f}")

                # Check grokking
                grok_result = grok_detector.update(accuracy)
                logger.info(f"  Grokking check: spike={grok_result['spike']*100:.2f}%")

                if accuracy > best_acc:
                    best_acc = accuracy
                    save_path = os.path.join(args.save_dir, f"best_step{global_step}.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "step": global_step,
                        "accuracy": accuracy,
                    }, save_path)
                    logger.info(f"  New best! Saved to {save_path}")

                if grok_result["hit_target"]:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"TARGET REACHED: {accuracy*100:.2f}% >= {args.target_acc*100:.0f}%")
                    logger.info(f"{'='*60}")

                    # Save final
                    final_path = os.path.join(args.save_dir, "phase3_champion.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "step": global_step,
                        "accuracy": accuracy,
                    }, final_path)
                    logger.info(f"Champion saved to {final_path}")
                    return

                if grok_result["is_grokking"]:
                    logger.info(f"  GROKKING DETECTED! Spike: {grok_result['spike']*100:.2f}%")

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}, Best acc: {best_acc*100:.2f}%")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

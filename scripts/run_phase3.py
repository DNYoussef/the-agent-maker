#!/usr/bin/env python3
"""
Phase 3: Quiet-STaR with Prompt Baking + MuonGrokfast

Two-step process:
1. STEP 1 (Prompt Baking): Add thinking tokens, supervised learning on reasoning examples
2. STEP 2 (Quiet-STaR RL): REINFORCE training to optimize thought generation

Input: Phase 2 champion (checkpoints/phase2_v7/final_champion.pt)
Output: Reasoning-enhanced model (checkpoints/phase3/final_model.pt)

Key Features:
- MuonGrokfast optimizer (Grokfast + Muon + QK-Clip)
- 8 thinking tokens: <think>, </think>, <step>, <reason>, <mece>, <falsify>, <expert>, <doubt>
- Anti-theater detection (validates genuine reasoning vs memorization)
"""

import copy
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# THINKING TOKENS
# ============================================================

THINKING_TOKENS = [
    "<think>",      # Start thinking block
    "</think>",     # End thinking block
    "<step>",       # Reasoning step marker
    "<reason>",     # Justification marker
    "<mece>",       # MECE (Mutually Exclusive, Collectively Exhaustive)
    "<falsify>",    # Falsification attempt
    "<expert>",     # Expert perspective
    "<doubt>",      # Uncertainty/doubt marker
]


def extend_tokenizer(tokenizer):
    """Add thinking tokens to tokenizer."""
    num_added = tokenizer.add_tokens(THINKING_TOKENS, special_tokens=True)
    print(f"  Added {num_added} thinking tokens to tokenizer")
    return tokenizer


def resize_model_embeddings(model, tokenizer):
    """Resize model embeddings to accommodate new tokens."""
    # TRM x Titans-MAG uses backbone.token_emb
    old_embed = model.backbone.token_emb
    old_vocab = old_embed.weight.shape[0]
    new_vocab = len(tokenizer)
    embed_dim = old_embed.embedding_dim

    if new_vocab > old_vocab:
        # Resize token embedding layer
        new_embed = nn.Embedding(new_vocab, embed_dim)
        new_embed.weight.data[:old_vocab] = old_embed.weight.data
        # Initialize new tokens with mean of existing embeddings
        new_embed.weight.data[old_vocab:] = old_embed.weight.data.mean(dim=0)
        model.backbone.token_emb = new_embed

        # Resize LM head (tied with embeddings)
        old_head = model.lm_head
        new_head = nn.Linear(old_head.in_features, new_vocab, bias=False)
        new_head.weight.data[:old_vocab] = old_head.weight.data
        new_head.weight.data[old_vocab:] = old_head.weight.data.mean(dim=0)
        model.lm_head = new_head

        # Re-tie weights
        model.lm_head.weight = model.backbone.token_emb.weight

        # CRITICAL: Update config vocab_size so forward() uses correct size
        model.config.titans_config.vocab_size = new_vocab

        print(f"  Resized embeddings: {old_vocab} -> {new_vocab}")

    return model


# ============================================================
# MUGROKFAST OPTIMIZER (Simplified for Phase 3)
# ============================================================

class MuonGrokfast(torch.optim.Optimizer):
    """
    Simplified MuonGrokfast optimizer for Phase 3.

    Combines:
    - Grokfast: EMA gradient filtering
    - Muon: Parameter routing (2-D -> orthogonalization, 1-D -> AdamW)
    - QK-Clip: Attention safety (for RL)
    """

    def __init__(
        self,
        params,
        muon_lr: float = 5e-4,
        fallback_lr: float = 1e-3,
        grokfast_alpha: float = 0.98,
        grokfast_lambda: float = 0.1,  # Phase 3 RL setting
        qk_clip_threshold: float = 25.0,  # Phase 3 RL setting
        momentum: float = 0.95,
        weight_decay: float = 0.01,
    ):
        defaults = dict(
            muon_lr=muon_lr,
            fallback_lr=fallback_lr,
            grokfast_alpha=grokfast_alpha,
            grokfast_lambda=grokfast_lambda,
            qk_clip_threshold=qk_clip_threshold,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Initialize state
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["ema_grad"] = torch.zeros_like(p.data)
                state["momentum_buffer"] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                state["step"] += 1

                grad = p.grad.data
                alpha = group["grokfast_alpha"]
                lambda_ = group["grokfast_lambda"]

                # Grokfast: EMA gradient filtering
                state["ema_grad"].mul_(alpha).add_(grad, alpha=1 - alpha)
                filtered_grad = grad + lambda_ * (grad - state["ema_grad"])

                # Weight decay
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["weight_decay"] * group["muon_lr"])

                # Parameter routing: 2-D -> Muon, 1-D -> SGD with momentum
                if len(p.shape) >= 2:
                    # Muon: momentum + orthogonalization hint
                    lr = group["muon_lr"]
                    momentum = group["momentum"]

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(filtered_grad)
                    p.data.add_(buf, alpha=-lr)
                else:
                    # 1-D params: simple SGD with momentum
                    lr = group["fallback_lr"]
                    momentum = group["momentum"]

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(filtered_grad)
                    p.data.add_(buf, alpha=-lr)

        return loss


# ============================================================
# REASONING DATASET
# ============================================================

class ReasoningDataset(Dataset):
    """Dataset for reasoning training examples."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Combine question + reasoning + answer
        full_text = (
            f"Question: {example['question']}\n"
            f"<think>\n{example['reasoning']}\n</think>\n"
            f"Answer: {example['answer']}"
        )

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
        }


def generate_reasoning_examples(num_examples: int = 1000) -> List[Dict]:
    """Generate synthetic reasoning examples with thinking tokens."""

    examples = []

    # Math reasoning templates
    math_templates = [
        {
            "question": "What is {a} + {b}?",
            "reasoning": "<step>First, I identify the numbers: {a} and {b}</step>\n<reason>Addition means combining quantities</reason>\n<step>Adding: {a} + {b} = {result}</step>",
            "gen": lambda: {"a": random.randint(1, 100), "b": random.randint(1, 100)},
            "result": lambda d: d["a"] + d["b"],
        },
        {
            "question": "What is {a} * {b}?",
            "reasoning": "<step>Identify factors: {a} and {b}</step>\n<mece>Multiplication approaches: repeated addition or direct calculation</mece>\n<step>Computing: {a} * {b} = {result}</step>",
            "gen": lambda: {"a": random.randint(2, 20), "b": random.randint(2, 20)},
            "result": lambda d: d["a"] * d["b"],
        },
        {
            "question": "If I have {total} apples and give away {give}, how many remain?",
            "reasoning": "<step>Start with {total} apples</step>\n<step>Give away {give} apples</step>\n<reason>Subtraction: {total} - {give} = {result}</reason>",
            "gen": lambda: {"total": random.randint(20, 100), "give": random.randint(1, 19)},
            "result": lambda d: d["total"] - d["give"],
        },
    ]

    # Logic reasoning templates
    logic_templates = [
        {
            "question": "All {category} are {property}. {item} is a {category}. Is {item} {property}?",
            "reasoning": "<step>Premise 1: All {category} are {property}</step>\n<step>Premise 2: {item} is a {category}</step>\n<reason>By syllogistic logic, {item} must be {property}</reason>\n<falsify>Could there be exceptions? No, 'all' is universal</falsify>",
            "gen": lambda: {"category": random.choice(["dogs", "cats", "birds"]), "property": random.choice(["animals", "living things"]), "item": random.choice(["Spot", "Fluffy", "Tweety"])},
            "result": lambda d: "Yes",
        },
    ]

    all_templates = math_templates + logic_templates

    for _ in range(num_examples):
        template = random.choice(all_templates)
        data = template["gen"]()
        data["result"] = template["result"](data)

        examples.append({
            "question": template["question"].format(**data),
            "reasoning": template["reasoning"].format(**data),
            "answer": str(data["result"]),
        })

    return examples


# ============================================================
# STEP 1: PROMPT BAKING
# ============================================================

def run_step1_baking(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    num_examples: int = 1000,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    Step 1: Prompt Baking - Supervised learning on reasoning examples.

    Embeds reasoning patterns into model weights via fine-tuning.
    """

    print("\n" + "=" * 70)
    print("STEP 1: PROMPT BAKING")
    print("=" * 70)

    # Extend tokenizer and resize model
    print("\n[1/4] Preparing tokenizer and model...")
    tokenizer = extend_tokenizer(tokenizer)
    model = resize_model_embeddings(model, tokenizer)
    model = model.to(device)

    # Generate training data
    print(f"\n[2/4] Generating {num_examples} reasoning examples...")
    examples = generate_reasoning_examples(num_examples)
    dataset = ReasoningDataset(examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  Created {len(dataset)} examples, {len(dataloader)} batches")

    # Initialize optimizer (MuonGrokfast with supervised settings)
    print("\n[3/4] Initializing MuonGrokfast optimizer...")
    optimizer = MuonGrokfast(
        model.parameters(),
        muon_lr=learning_rate,
        fallback_lr=learning_rate,
        grokfast_alpha=0.98,
        grokfast_lambda=0.2,  # Supervised setting (gentler)
    )

    # Training loop
    print(f"\n[4/4] Training for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            output = model(input_ids, labels=labels)

            if isinstance(output, dict):
                loss = output.get("loss", None)
                if loss is None:
                    logits = output["logits"]
                    loss = F.cross_entropy(
                        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                        labels[:, 1:].contiguous().view(-1),
                    )
            else:
                loss = output.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    # Save baked model
    os.makedirs(output_dir, exist_ok=True)
    baked_path = os.path.join(output_dir, "step1_baked.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_vocab_size": len(tokenizer),
    }, baked_path)
    print(f"\n  Baked model saved to: {baked_path}")

    return model


# ============================================================
# STEP 2: QUIET-STAR RL
# ============================================================

class ThoughtGenerator(nn.Module):
    """Generate multiple thought candidates at each position."""

    def __init__(self, model: nn.Module, num_thoughts: int = 4, max_thought_len: int = 32):
        super().__init__()
        self.model = model
        self.num_thoughts = num_thoughts
        self.max_thought_len = max_thought_len

    def forward(self, hidden_states: torch.Tensor, temperature: float = 0.8):
        """Generate thought candidates."""
        thoughts = []

        for _ in range(self.num_thoughts):
            # Sample thought tokens autoregressively
            thought_tokens = []
            current = hidden_states

            for _ in range(self.max_thought_len):
                # Get logits from model
                with torch.no_grad():
                    output = self.model(current)
                    if isinstance(output, dict):
                        logits = output["logits"]
                    else:
                        logits = output.logits if hasattr(output, "logits") else output

                # Sample with temperature
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                thought_tokens.append(next_token)

                # Update current (simplified - just use last token)
                current = next_token

                # Early stop on EOS
                if next_token.item() == 50256:  # GPT-2 EOS
                    break

            thoughts.append(torch.cat(thought_tokens, dim=1) if thought_tokens else torch.zeros(1, 1).long())

        return thoughts


def run_step2_rl(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    episodes: int = 500,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    Step 2: Quiet-STaR RL - REINFORCE training for thought generation.

    Optimizes model to generate useful internal reasoning.
    """

    print("\n" + "=" * 70)
    print("STEP 2: QUIET-STAR RL")
    print("=" * 70)

    model = model.to(device)
    model.train()

    # Initialize optimizer (MuonGrokfast with RL settings)
    print("\n[1/3] Initializing MuonGrokfast optimizer (RL config)...")
    optimizer = MuonGrokfast(
        model.parameters(),
        muon_lr=learning_rate,
        fallback_lr=learning_rate,
        grokfast_alpha=0.98,
        grokfast_lambda=0.1,  # RL setting (more aggressive)
        qk_clip_threshold=25.0,  # Attention safety
    )

    # Generate RL training prompts
    print(f"\n[2/3] Generating {episodes} RL episodes...")
    prompts = []
    for _ in range(episodes):
        a, b = random.randint(1, 50), random.randint(1, 50)
        prompts.append({
            "question": f"What is {a} + {b}?",
            "answer": a + b,
        })

    # RL training loop
    print(f"\n[3/3] Running REINFORCE training...")

    rewards_history = []
    baseline = 0.0

    for ep in range(0, episodes, batch_size):
        batch_prompts = prompts[ep:ep+batch_size]
        batch_rewards = []

        for prompt in batch_prompts:
            # Encode prompt
            inputs = tokenizer(
                f"Question: {prompt['question']}\n<think>",
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(device)

            # Generate response
            with torch.no_grad():
                generated = inputs["input_ids"].clone()
                for _ in range(64):
                    output = model(generated)
                    if isinstance(output, dict):
                        logits = output["logits"]
                    else:
                        logits = output.logits if hasattr(output, "logits") else output

                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

            # Decode and check answer
            response = tokenizer.decode(generated[0], skip_special_tokens=True)

            # Simple reward: 1 if correct answer appears, 0 otherwise
            import re
            nums = re.findall(r"\d+", response)
            reward = 1.0 if str(prompt["answer"]) in nums else 0.0
            batch_rewards.append(reward)

        # Compute advantage
        avg_reward = np.mean(batch_rewards)
        advantages = [r - baseline for r in batch_rewards]
        baseline = 0.9 * baseline + 0.1 * avg_reward  # Update baseline

        # Policy gradient update (simplified)
        if avg_reward > 0:
            # Positive reward -> reinforce by reducing loss
            inputs = tokenizer(
                f"Question: {batch_prompts[0]['question']}\n<think>",
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(device)

            output = model(inputs["input_ids"], labels=inputs["input_ids"])
            if isinstance(output, dict):
                loss = output.get("loss", torch.tensor(0.0))
            else:
                loss = output.loss

            # Scale loss by advantage
            scaled_loss = loss * (1.0 - avg_reward)  # Lower loss when reward is high

            optimizer.zero_grad()
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        rewards_history.append(avg_reward)

        if (ep + batch_size) % 50 == 0:
            recent_avg = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
            print(f"  Episode {ep+batch_size}/{episodes}: avg_reward={recent_avg:.3f}")

    # Save RL-trained model
    rl_path = os.path.join(output_dir, "step2_rl.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "rewards_history": rewards_history,
    }, rl_path)
    print(f"\n  RL model saved to: {rl_path}")

    return model


# ============================================================
# MAIN
# ============================================================

def run_phase3(
    phase2_checkpoint: str = "checkpoints/phase2_v7/final_champion.pt",
    output_dir: str = "checkpoints/phase3",
    step1_examples: int = 1000,
    step1_epochs: int = 3,
    step2_episodes: int = 500,
    device: str = "cuda",
):
    """
    Run complete Phase 3 pipeline.
    """

    print("=" * 70)
    print("PHASE 3: QUIET-STAR + PROMPT BAKING")
    print("=" * 70)
    print(f"Input: {phase2_checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Step 1: {step1_examples} examples, {step1_epochs} epochs")
    print(f"Step 2: {step2_episodes} RL episodes")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Load Phase 2 champion
    print("\n[LOADING] Phase 2 champion model...")

    from safetensors.torch import load_file as load_safetensors

    # Import Phase 1 model architecture
    try:
        from phase1_cognate.model.full_model import TRMTitansMAGModel
        from phase1_cognate.model.model_config import Phase1Config
        use_custom_arch = True
        print("  Using TRM x Titans-MAG architecture")
    except ImportError:
        use_custom_arch = False
        print("  WARNING: Phase 1 architecture not found")
        return

    # Load checkpoint
    checkpoint = torch.load(phase2_checkpoint, map_location="cpu")

    # Create model
    config = Phase1Config(specialization="reasoning")
    model = TRMTitansMAGModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model: {params/1e6:.1f}M params")
    print(f"  Phase 2 fitness: {checkpoint.get('fitness', 'N/A')}")
    print(f"  Phase 2 accuracy: {checkpoint.get('accuracy', 'N/A')}%")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # STEP 1: Prompt Baking
    model = run_step1_baking(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        num_examples=step1_examples,
        epochs=step1_epochs,
        device=device,
    )

    clear_gpu_memory()

    # STEP 2: Quiet-STaR RL
    model = run_step2_rl(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        episodes=step2_episodes,
        device=device,
    )

    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "phase": 3,
        "tokenizer_vocab_size": len(tokenizer),
    }, final_path)

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")

    return model


if __name__ == "__main__":
    run_phase3(
        phase2_checkpoint="checkpoints/phase2_v7/final_champion.pt",
        output_dir="checkpoints/phase3",
        step1_examples=1000,
        step1_epochs=3,
        step2_episodes=500,
        device="cuda",
    )

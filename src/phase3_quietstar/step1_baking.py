"""
Phase 3 Step 1: Prompt Baking Implementation

Supervised learning to embed 7 reasoning strategies into model weights.

Key Innovation: Provides "jumpstart effect" for Step 2 (RL training)
by giving model reasoning foundation before RL begins.

Training Flow:
1. Load Phase 2 champion model
2. Add 8 thinking tokens to vocabulary
3. Load 20K reasoning examples (from data_generator.py)
4. Train with 7 strategies using existing Prompt Baking system
5. Validate ≥85% convergence threshold
6. Save baked model for Step 2

Duration: ~5 hours (5 epochs × 1 hour/epoch)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json

from .config import QuietSTaRConfig, PromptBakingConfig
from .vocabulary import prepare_model_for_phase3, compute_thinking_token_usage
from .wandb_logger import WandBLogger

# ISS-004: Secure checkpoint utilities
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from cross_phase.utils.checkpoint_utils import save_checkpoint as secure_save
from ..cross_phase.mugrokfast import MuonGrokfast, MuGrokConfig
from ..cross_phase.prompt_baking import PromptBaker


class ReasoningDataset(Dataset):
    """
    Dataset for reasoning training examples.

    Each example contains:
    - question: Original question
    - reasoning: Reasoning with thinking tokens
    - answer: Final answer
    - strategy: Reasoning strategy used
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized example."""
        example = self.examples[idx]

        # Combine question + reasoning + answer
        full_text = (
            f"Question: {example['question']}\n"
            f"{example['reasoning']}\n"
            f"Answer: {example['answer']}"
        )

        # Tokenize
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
            "strategy": example["strategy"],
        }


class PromptBakingTrainer:
    """
    Step 1 trainer using Prompt Baking system.

    Integrates existing cross_phase.prompt_baking with Phase 3.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: QuietSTaRConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Prepare model with thinking tokens
        self.model, self.tokenizer, self.vocab = prepare_model_for_phase3(
            model, tokenizer, use_extended_tokens=False
        )

        self.model = self.model.to(device)

        # Initialize components
        self._init_optimizer()
        self._init_prompt_baker()
        self._init_wandb()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.strategy_accuracies: Dict[str, List[float]] = {
            strategy: [] for strategy in self._get_strategies()
        }

    def _init_optimizer(self):
        """Initialize MuGrokfast optimizer for baking."""
        optimizer_config = MuGrokConfig(
            muon_lr=self.config.baking.muon_lr,
            grokfast_lambda=self.config.baking.grokfast_lambda,
            qk_clip_threshold=self.config.baking.qk_clip_threshold,
            kl_coefficient=self.config.baking.kl_coefficient,
            weight_decay=self.config.baking.weight_decay,
        )

        self.optimizer = MuonGrokfast(
            self.model.parameters(), config=optimizer_config
        )

    def _init_prompt_baker(self):
        """Initialize Prompt Baking system."""
        from ..cross_phase.prompt_baking import PromptBaker

        self.prompt_baker = PromptBaker(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        self.wandb_logger = WandBLogger(
            project="agent-forge-v2",
            name="phase3-step1-baking",
            config=self.config.to_dict(),
            tags=["phase3", "step1", "prompt-baking", "supervised"],
        )

    def _get_strategies(self) -> List[str]:
        """Get list of 7 reasoning strategies."""
        return [
            "chain_of_thought",
            "mece_decomposition",
            "falsification_testing",
            "expert_perspective",
            "orthogonal_wisdom",
            "self_doubt",
            "bayesian_rationalist",
        ]

    def load_dataset(self, data_path: Path) -> ReasoningDataset:
        """Load reasoning dataset from JSON."""
        with open(data_path) as f:
            examples = json.load(f)

        print(f"Loaded {len(examples)} reasoning examples")

        return ReasoningDataset(
            examples, self.tokenizer, max_length=512
        )

    def train_epoch(
        self, dataloader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        strategy_correct: Dict[str, int] = {s: 0 for s in self._get_strategies()}
        strategy_total: Dict[str, int] = {s: 0 for s in self._get_strategies()}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            strategies = batch["strategy"]

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()

            # Compute accuracy per strategy
            with torch.no_grad():
                predictions = outputs.logits.argmax(dim=-1)
                for i, strategy in enumerate(strategies):
                    correct = (predictions[i] == labels[i]).sum().item()
                    total = labels[i].numel()
                    strategy_correct[strategy] += correct
                    strategy_total[strategy] += total

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Log to W&B
            if self.global_step % 10 == 0:
                self._log_step(loss.item(), epoch, batch_idx)

            self.global_step += 1

        # Compute epoch metrics
        avg_loss = total_loss / len(dataloader)
        strategy_accs = {
            s: strategy_correct[s] / max(strategy_total[s], 1)
            for s in self._get_strategies()
        }
        overall_acc = sum(strategy_correct.values()) / sum(strategy_total.values())

        # Store accuracies
        for strategy, acc in strategy_accs.items():
            self.strategy_accuracies[strategy].append(acc)

        return {
            "loss": avg_loss,
            "overall_accuracy": overall_acc,
            **{f"accuracy_{s}": acc for s, acc in strategy_accs.items()},
        }

    def _log_step(self, loss: float, epoch: int, batch_idx: int):
        """Log training step to W&B."""
        metrics = {
            "baking/step": self.global_step,
            "baking/epoch": epoch,
            "baking/loss": loss,
            "baking/learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        self.wandb_logger.log(metrics)

    def _log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics to W&B."""
        wandb_metrics = {
            f"baking/{k}": v for k, v in metrics.items()
        }
        wandb_metrics["baking/epoch"] = epoch

        # Add strategy-specific accuracies
        for strategy in self._get_strategies():
            acc_key = f"accuracy_{strategy}"
            if acc_key in metrics:
                wandb_metrics[f"baking/{acc_key}"] = metrics[acc_key]

        # Add thinking token usage (if available)
        # This would be computed on validation set
        wandb_metrics["baking/convergence_progress"] = (
            metrics["overall_accuracy"] / self.config.baking.convergence_threshold
        )

        self.wandb_logger.log(wandb_metrics)

    def validate(
        self, dataloader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Validate model and compute thinking token usage."""
        self.model.eval()
        total_correct = 0
        total_tokens = 0
        strategy_correct: Dict[str, int] = {s: 0 for s in self._get_strategies()}
        strategy_total: Dict[str, int] = {s: 0 for s in self._get_strategies()}

        all_outputs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                strategies = batch["strategy"]

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # Compute accuracy
                predictions = outputs.logits.argmax(dim=-1)
                correct = (predictions == labels).sum().item()
                total = labels.numel()

                total_correct += correct
                total_tokens += total

                # Per-strategy accuracy
                for i, strategy in enumerate(strategies):
                    s_correct = (predictions[i] == labels[i]).sum().item()
                    s_total = labels[i].numel()
                    strategy_correct[strategy] += s_correct
                    strategy_total[strategy] += s_total

                # Decode for token usage analysis
                for i in range(input_ids.size(0)):
                    output_text = self.tokenizer.decode(
                        predictions[i], skip_special_tokens=False
                    )
                    all_outputs.append(output_text)

        # Overall accuracy
        overall_acc = total_correct / total_tokens

        # Strategy accuracies
        strategy_accs = {
            s: strategy_correct[s] / max(strategy_total[s], 1)
            for s in self._get_strategies()
        }

        # Thinking token usage
        token_usage = compute_thinking_token_usage(
            all_outputs, self.vocab
        )

        return overall_acc, strategy_accs, token_usage

    def check_convergence(
        self, overall_acc: float, strategy_accs: Dict[str, float]
    ) -> bool:
        """Check if model has converged (≥85% threshold)."""
        threshold = self.config.baking.convergence_threshold

        # Overall accuracy check
        if overall_acc < threshold:
            return False

        # Check each strategy meets threshold
        for strategy, acc in strategy_accs.items():
            if acc < threshold * 0.9:  # Allow 10% slack per strategy
                return False

        return True

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Complete training loop for Step 1 (Prompt Baking).

        Returns:
            Final validation metrics
        """
        print("\n" + "=" * 60)
        print("PHASE 3 - STEP 1: PROMPT BAKING")
        print("=" * 60)

        num_epochs = self.config.baking.num_epochs
        best_acc = 0.0
        best_model_state = None

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            train_metrics = self.train_epoch(train_dataloader, epoch)

            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Accuracy: {train_metrics['overall_accuracy']:.4f}")

            # Validate
            val_acc, strategy_accs, token_usage = self.validate(
                val_dataloader
            )

            print(f"Val Accuracy: {val_acc:.4f}")
            print("\nStrategy Accuracies:")
            for strategy, acc in strategy_accs.items():
                print(f"  {strategy}: {acc:.4f}")

            print("\nThinking Token Usage:")
            for token_type, usage in token_usage.items():
                print(f"  {token_type}: {usage:.2%}")

            # Log epoch metrics
            epoch_metrics = {
                **train_metrics,
                "val_accuracy": val_acc,
                **{f"val_{k}": v for k, v in strategy_accs.items()},
                **{f"token_usage_{k}": v for k, v in token_usage.items()},
            }
            self._log_epoch(epoch, epoch_metrics)

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = self.model.state_dict()

            # Check convergence
            if self.check_convergence(val_acc, strategy_accs):
                print(f"\n✅ Convergence achieved at epoch {epoch + 1}!")
                print(f"Overall accuracy: {val_acc:.4f} (≥{self.config.baking.convergence_threshold})")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Final validation
        print("\n" + "=" * 60)
        print("FINAL VALIDATION")
        print("=" * 60)
        final_acc, final_strategy_accs, final_token_usage = self.validate(
            val_dataloader
        )

        print(f"Final Accuracy: {final_acc:.4f}")
        print("\nFinal Strategy Accuracies:")
        for strategy, acc in final_strategy_accs.items():
            print(f"  {strategy}: {acc:.4f}")

        print("\nFinal Thinking Token Usage:")
        for token_type, usage in final_token_usage.items():
            print(f"  {token_type}: {usage:.2%}")

        # Check if convergence met
        converged = self.check_convergence(
            final_acc, final_strategy_accs
        )

        if not converged:
            print(
                f"\n⚠️  Warning: Convergence threshold not met "
                f"(target: {self.config.baking.convergence_threshold:.2%})"
            )

        return {
            "final_accuracy": final_acc,
            "strategy_accuracies": final_strategy_accs,
            "token_usage": final_token_usage,
            "converged": converged,
        }

    def save_baked_model(self, output_path: Path):
        """Save baked model for Step 2 using secure SafeTensors format (ISS-004)."""
        # Remove extension if present
        base_path = output_path.with_suffix('') if output_path.suffix else output_path

        # Metadata (JSON-serializable)
        baking_metadata = {
            "tokenizer_vocab": self.tokenizer.get_vocab(),
            "thinking_tokens": self.vocab.thinking_tokens,
            "strategy_accuracies": self.strategy_accuracies,
        }

        secure_save(
            model=self.model,
            output_path=base_path,
            config=self.config.to_dict(),
            metadata=baking_metadata,
        )

        print(f"\n[OK] Baked model saved to: {base_path}.safetensors")

        # Save to W&B as artifact
        self.wandb_logger.save_artifact(
            output_path,
            name="phase3-baked-model",
            type="model",
            metadata={
                "final_accuracy": self.strategy_accuracies,
            },
        )


def run_step1_baking(
    model: nn.Module,
    tokenizer,
    data_path: Path,
    output_path: Path,
    config: Optional[QuietSTaRConfig] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run Step 1 (Prompt Baking) training.

    Args:
        model: Base model (from Phase 2)
        tokenizer: Tokenizer
        data_path: Path to reasoning dataset JSON
        output_path: Where to save baked model
        config: Phase 3 configuration
        device: Device to use

    Returns:
        Final validation metrics
    """
    if config is None:
        config = QuietSTaRConfig()

    # Initialize trainer
    trainer = PromptBakingTrainer(model, tokenizer, config, device)

    # Load dataset
    dataset = trainer.load_dataset(data_path)

    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.baking.batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.baking.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Train
    final_metrics = trainer.train(train_dataloader, val_dataloader)

    # Save baked model
    trainer.save_baked_model(output_path)

    return final_metrics

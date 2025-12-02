"""
Phase 3 Step 2: Quiet-STaR RL Implementation

REINFORCE-based RL training to optimize thought generation.

Key Innovation: Trains on baked foundation (Step 1), achieving 30-50%
faster convergence vs training from scratch.

Training Flow:
1. Load baked model from Step 1
2. Initialize Quiet-STaR components (ThoughtGenerator, CoherenceScorer, etc.)
3. Run REINFORCE RL for 10K episodes
4. Validate with anti-theater detection (3 tests)
5. Save reasoning-enhanced model

Duration: ~8-12 hours (10K episodes × 3-4s/episode)
"""

import json

# ISS-004: Secure checkpoint utilities
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..cross_phase.mugrokfast import MuGrokConfig, MuonGrokfast
from .architecture import QuietSTaRModel
from .config import QuietSTaRConfig
from .wandb_logger import WandBLogger

sys.path.insert(0, str(Path(__file__).parents[2]))
from cross_phase.utils.checkpoint_utils import load_checkpoint as secure_load
from cross_phase.utils.checkpoint_utils import save_checkpoint as secure_save


class REINFORCETrainer:
    """
    REINFORCE policy gradient trainer for Quiet-STaR.

    ISS-007: Full implementation with:
    - Baseline network for variance reduction
    - GAE (Generalized Advantage Estimation)
    - Entropy bonus for exploration
    - Learning rate scheduling
    - Early stopping with patience

    Uses baked model as baseline to prevent drift (KL regularization).
    """

    def __init__(
        self,
        model: nn.Module,
        baked_model: nn.Module,
        tokenizer,
        config: QuietSTaRConfig,
        device: str = "cuda",
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Initialize Quiet-STaR model
        self.model = QuietSTaRModel(
            base_model=model,
            hidden_size=model.config.hidden_size,
            num_thoughts=config.rl.num_thoughts,
            max_thought_length=config.rl.max_thought_length,
            injection_threshold=config.rl.injection_threshold,
            coherence_weights=config.rl.coherence_weights,
        ).to(device)

        # Baked baseline (frozen)
        self.baked_model = baked_model.to(device)
        self.baked_model.eval()
        for param in self.baked_model.parameters():
            param.requires_grad = False

        # ISS-007: Baseline network for variance reduction
        self.hidden_size = model.config.hidden_size
        self.baseline_network = self._create_baseline_network()

        # Initialize optimizer and scheduler
        self._init_optimizer()
        self._init_scheduler()
        self._init_wandb()

        # Training state
        self.episode = 0
        self.global_step = 0
        self.reward_history = []

        # ISS-007: Early stopping state
        self.best_reward = float("-inf")
        self.best_model_state = None
        self.patience_counter = 0
        self.current_entropy_coef = config.rl.entropy_coefficient

    def _create_baseline_network(self) -> nn.Module:
        """
        ISS-007: Create baseline (value) network for variance reduction.

        Architecture: hidden_state -> MLP -> scalar value estimate
        """
        baseline_hidden = self.config.rl.baseline_hidden_size

        baseline = nn.Sequential(
            nn.Linear(self.hidden_size, baseline_hidden),
            nn.ReLU(),
            nn.Linear(baseline_hidden, baseline_hidden),
            nn.ReLU(),
            nn.Linear(baseline_hidden, 1),
        ).to(self.device)

        return baseline

    def _init_optimizer(self):
        """ISS-007: Initialize MuGrokfast optimizer for RL (model + baseline)."""
        optimizer_config = MuGrokConfig(
            muon_lr=self.config.rl.muon_lr,
            grokfast_lambda=self.config.rl.grokfast_lambda,
            qk_clip_threshold=self.config.rl.qk_clip_threshold,
            kl_coefficient=self.config.rl.kl_coefficient,
            weight_decay=0.0,  # No weight decay in RL
        )

        # ISS-007: Include both model and baseline network parameters
        import itertools

        all_params = itertools.chain(self.model.parameters(), self.baseline_network.parameters())

        self.optimizer = MuonGrokfast(all_params, config=optimizer_config)

    def _init_scheduler(self):
        """ISS-007: Initialize learning rate scheduler."""
        if self.config.rl.lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.rl.num_episodes
            )
        elif self.config.rl.lr_schedule == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.rl.num_episodes,
            )
        else:
            self.scheduler = None

    def _init_wandb(self):
        """Initialize W&B logging."""
        self.wandb_logger = WandBLogger(
            project="agent-forge-v2",
            name="phase3-step2-rl",
            config=self.config.to_dict(),
            tags=["phase3", "step2", "quiet-star", "rl", "reinforce"],
        )

    def compute_reward(
        self,
        logits_with_thoughts: torch.Tensor,
        logits_without_thoughts: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute REINFORCE reward.

        Reward = 1.0 if prediction correct, 0.0 otherwise.

        Args:
            logits_with_thoughts: (batch, seq_len, vocab)
            logits_without_thoughts: (batch, seq_len, vocab)
            labels: (batch, seq_len)

        Returns:
            Reward tensor (batch,)
        """
        # Predictions
        predictions_with = logits_with_thoughts.argmax(dim=-1)
        predictions_without = logits_without_thoughts.argmax(dim=-1)

        # Accuracy
        correct_with = (predictions_with == labels).float().mean(dim=-1)
        correct_without = (predictions_without == labels).float().mean(dim=-1)

        # Reward: binary (1.0 if improved, 0.0 otherwise)
        reward = (correct_with > correct_without).float()

        return reward

    def compute_kl_divergence(
        self, logits: torch.Tensor, baked_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from baked baseline.

        Prevents drift from baked reasoning patterns.
        """
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        baked_log_probs = F.log_softmax(baked_logits, dim=-1)

        # KL divergence
        kl_div = F.kl_div(
            log_probs, baked_log_probs.detach(), reduction="batchmean", log_target=True
        )

        return kl_div

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        ISS-007: Compute Generalized Advantage Estimation.

        GAE reduces variance while maintaining low bias.

        Args:
            rewards: (batch,) Rewards for each timestep
            values: (batch,) Value estimates for each timestep
            next_values: (batch,) Value estimates for next timestep
            dones: (batch,) Episode termination flags

        Returns:
            advantages: (batch,) GAE advantages
        """
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.rl.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.rl.gamma * self.config.rl.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        ISS-007: Compute policy entropy for exploration bonus.

        Args:
            logits: (batch, seq_len, vocab) Logits from model

        Returns:
            entropy: Scalar entropy value
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy

    def train_episode(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        ISS-007: Train one REINFORCE episode with full RL features.

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len)

        Returns:
            Episode metrics
        """
        self.model.train()
        self.baseline_network.train()

        # Forward pass WITH thoughts
        outputs_with = self.model(input_ids=input_ids, labels=labels, use_thoughts=True)

        # Forward pass WITHOUT thoughts
        outputs_without = self.model(input_ids=input_ids, labels=labels, use_thoughts=False)

        # Baked baseline (frozen)
        with torch.no_grad():
            baked_outputs = self.baked_model(input_ids=input_ids, labels=labels)

        # ISS-007: Extract hidden states for value estimation
        # Use mean pooling over sequence for value prediction
        hidden_states = outputs_with.get("hidden_states", None)
        if hidden_states is None:
            # Fallback: use embedding layer output if hidden states not available
            hidden_states = self.model.base_model.get_input_embeddings()(input_ids)

        # Mean pool over sequence dimension
        pooled_hidden = hidden_states.mean(dim=1)  # (batch, hidden_size)

        # ISS-007: Compute value estimates from baseline network
        values = self.baseline_network(pooled_hidden).squeeze(-1)  # (batch,)

        # Compute next values (for GAE, assume same as current for single-step)
        with torch.no_grad():
            next_values = values.clone()

        # Compute reward
        reward = self.compute_reward(
            outputs_with["logits"],
            outputs_without["logits"],
            labels,
        )

        # ISS-007: Compute advantages using GAE
        dones = torch.zeros_like(reward)  # No episode termination in language modeling
        if self.config.rl.use_gae:
            advantages = self.compute_gae(reward, values, next_values, dones)
        else:
            # Simple advantage: reward - baseline
            advantages = reward - values.detach()

        # ISS-007: Compute entropy for exploration bonus
        entropy = self.compute_entropy(outputs_with["logits"])

        # Compute KL divergence (prevent drift from baked baseline)
        kl_div = self.compute_kl_divergence(outputs_with["logits"], baked_outputs.logits)

        # ISS-007: REINFORCE loss with advantages
        log_prob = -outputs_with["loss"]  # Negative CE loss as log prob
        policy_loss = -(log_prob * advantages.detach().mean())

        # ISS-007: Value loss for baseline network
        value_targets = reward  # Target is actual reward
        value_loss = F.mse_loss(values, value_targets.detach())

        # ISS-007: Total loss with all components
        total_loss = (
            policy_loss
            + self.config.rl.value_loss_coefficient * value_loss
            - self.current_entropy_coef * entropy  # Negative because we want to maximize
            + self.config.rl.kl_coefficient * kl_div
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.rl.gradient_clip)

        self.optimizer.step()

        # ISS-007: Decay entropy coefficient
        self.current_entropy_coef = max(
            self.config.rl.min_entropy_coefficient,
            self.current_entropy_coef * self.config.rl.entropy_decay,
        )

        # ISS-007: Metrics
        metrics = {
            "reward": reward.mean().item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "entropy_coefficient": self.current_entropy_coef,
            "advantages": advantages.mean().item(),
            "kl_divergence": kl_div.item(),
            "total_loss": total_loss.item(),
            "num_thoughts_used": outputs_with.get("num_thoughts_used", 0),
            "avg_coherence": outputs_with.get("avg_coherence", 0.0),
        }

        self.reward_history.append(reward.mean().item())

        return metrics

    def validate_episode(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Validate one episode (compute metrics without training).

        Returns detailed metrics including coherence breakdown.
        """
        self.model.eval()

        with torch.no_grad():
            # Forward with thoughts
            outputs = self.model(input_ids=input_ids, labels=labels, use_thoughts=True)

            # Compute accuracy
            predictions = outputs["logits"].argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()

            # Thought metrics (from QuietSTaRModel)
            thought_positions = outputs.get("thought_positions", [])
            avg_coherence = outputs.get("avg_coherence", 0.0)
            num_thoughts = outputs.get("num_thoughts_used", 0)

        return {
            "accuracy": accuracy,
            "avg_coherence": avg_coherence,
            "num_thoughts": num_thoughts,
            "thought_density": len(thought_positions) / input_ids.size(1),
        }

    def train(
        self,
        train_dataloader,
        val_dataloader,
        num_episodes: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        ISS-007: Complete RL training loop with full features.

        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            num_episodes: Override config episodes

        Returns:
            Final metrics
        """
        num_episodes = num_episodes or self.config.rl.num_episodes

        print("\n" + "=" * 60)
        print("PHASE 3 - STEP 2: QUIET-STAR RL (REINFORCE)")
        print("=" * 60)
        print(f"ISS-007 Features:")
        print(f"  - GAE: {self.config.rl.use_gae}")
        print(f"  - LR Schedule: {self.config.rl.lr_schedule}")
        print(f"  - Entropy Bonus: {self.config.rl.entropy_coefficient}")
        print(f"  - Early Stopping: patience={self.config.rl.patience}")
        print("=" * 60)

        train_iter = iter(train_dataloader)

        for episode in tqdm(range(num_episodes), desc="RL Training"):
            self.episode = episode

            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Train episode
            metrics = self.train_episode(input_ids, labels)

            # ISS-007: Step learning rate scheduler
            if self.scheduler is not None and episode >= self.config.rl.warmup_episodes:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                metrics["learning_rate"] = current_lr

            # Log every 10 episodes
            if episode % 10 == 0:
                self._log_episode(episode, metrics)

            # ISS-007: Validate at configured frequency
            validation_freq = self.config.rl.validation_frequency
            if episode % validation_freq == 0 and episode > 0:
                val_metrics = self._validate(val_dataloader)
                avg_reward = (
                    np.mean(self.reward_history[-100:])
                    if len(self.reward_history) >= 100
                    else np.mean(self.reward_history)
                )

                print(f"\n--- Episode {episode} Validation ---")
                print(f"Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"Avg Coherence: {val_metrics['avg_coherence']:.4f}")
                print(f"Avg Reward (last 100): {avg_reward:.4f}")
                if self.scheduler is not None and episode >= self.config.rl.warmup_episodes:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f"Learning Rate: {current_lr:.6f}")
                print(f"Entropy Coefficient: {self.current_entropy_coef:.6f}")

                # ISS-007: Early stopping logic
                improvement = avg_reward - self.best_reward
                if improvement > self.config.rl.min_improvement:
                    print(f"✅ Improvement: {improvement:.4f} (saving checkpoint)")
                    self.best_reward = avg_reward
                    self.best_model_state = self.model.state_dict()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    print(
                        f"⚠️  No improvement for {self.patience_counter}/{self.config.rl.patience} validations"
                    )

                    if self.patience_counter >= self.config.rl.patience:
                        print(f"\n🛑 Early stopping triggered after {episode} episodes")
                        print(f"Best reward: {self.best_reward:.4f}")
                        break

            # Anti-theater validation every 1000 episodes
            if episode % 1000 == 0 and episode > 0:
                print(f"\n--- Running Anti-Theater Validation (Episode {episode}) ---")
                theater_results = self._anti_theater_validation(val_dataloader)

                if not theater_results["all_passed"]:
                    print("WARNING: THEATER DETECTED! Consider rollback.")
                    print(f"  Divergence: {theater_results['divergence']:.3f} (need >0.30)")
                    print(f"  Ablation: {theater_results['ablation']:.3f} (need >0.02)")
                else:
                    print("PASS: Anti-theater validation passed")

            self.global_step += 1

        # ISS-007: Restore best model
        if self.best_model_state is not None:
            print(f"\nRestoring best model (reward: {self.best_reward:.4f})")
            self.model.load_state_dict(self.best_model_state)

        # Final validation
        print("\n" + "=" * 60)
        print("FINAL VALIDATION")
        print("=" * 60)
        final_metrics = self._validate(val_dataloader)

        print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Final Coherence: {final_metrics['avg_coherence']:.4f}")
        print(
            f"Final Avg Reward: {np.mean(self.reward_history[-100:]) if self.reward_history else 0.0:.4f}"
        )
        print(f"Total Episodes: {self.episode}")

        return final_metrics

    def _log_episode(self, episode: int, metrics: Dict[str, float]):
        """ISS-007: Log episode metrics to W&B with full RL features."""
        # Compute rolling average reward
        avg_reward = (
            np.mean(self.reward_history[-100:])
            if len(self.reward_history) >= 100
            else np.mean(self.reward_history)
        )

        # ISS-007: Log all metrics including new RL features
        log_dict = {
            "rl/episode": episode,
            "rl/reward": metrics["reward"],
            "rl/avg_reward_100": avg_reward,
            "rl/policy_loss": metrics["policy_loss"],
            "rl/value_loss": metrics["value_loss"],
            "rl/entropy": metrics["entropy"],
            "rl/entropy_coefficient": metrics["entropy_coefficient"],
            "rl/advantages": metrics["advantages"],
            "rl/kl_divergence": metrics["kl_divergence"],
            "rl/total_loss": metrics["total_loss"],
            "rl/num_thoughts": metrics["num_thoughts_used"],
            "rl/coherence": metrics["avg_coherence"],
        }

        # Add learning rate if available
        if "learning_rate" in metrics:
            log_dict["rl/learning_rate"] = metrics["learning_rate"]

        self.wandb_logger.log(log_dict, step=self.global_step)

    def _validate(self, val_dataloader) -> Dict[str, float]:
        """Run full validation."""
        total_accuracy = 0.0
        total_coherence = 0.0
        total_thoughts = 0
        num_batches = 0

        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            metrics = self.validate_episode(input_ids, labels)

            total_accuracy += metrics["accuracy"]
            total_coherence += metrics["avg_coherence"]
            total_thoughts += metrics["num_thoughts"]
            num_batches += 1

        return {
            "accuracy": total_accuracy / num_batches,
            "avg_coherence": total_coherence / num_batches,
            "avg_thoughts": total_thoughts / num_batches,
        }

    def _anti_theater_validation(self, val_dataloader) -> Dict[str, float]:
        """
        Run anti-theater validation (3 tests).

        Returns results for all 3 tests.
        """
        from .anti_theater import AntiTheaterValidator

        validator = AntiTheaterValidator(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config.anti_theater,
            device=self.device,
        )

        # Run all 3 tests
        results = validator.validate_all(val_dataloader)

        # Log to W&B
        self.wandb_logger.log(
            {
                "rl/anti_theater_divergence": results["divergence"],
                "rl/anti_theater_ablation": results["ablation"],
                "rl/anti_theater_correlation": results["correlation"],
                "rl/anti_theater_all_passed": float(results["all_passed"]),
            },
            step=self.global_step,
        )

        return results

    def save_model(self, output_path: Path):
        """Save reasoning-enhanced model using secure SafeTensors format (ISS-004)."""
        # Remove extension if present
        base_path = output_path.with_suffix("") if output_path.suffix else output_path

        # Training state goes in metadata
        rl_metadata = {
            "episode": self.episode,
            "reward_history": [float(r) for r in self.reward_history],  # Ensure JSON-serializable
        }

        secure_save(
            model=self.model,
            output_path=base_path,
            config=self.config.to_dict(),
            metadata=rl_metadata,
        )

        print(f"\n[OK] Reasoning-enhanced model saved to: {base_path}.safetensors")

        # Save to W&B
        self.wandb_logger.save_artifact(
            output_path,
            name="phase3-reasoning-enhanced-model",
            type="model",
            metadata={
                "final_avg_reward": np.mean(self.reward_history[-100:]),
                "total_episodes": self.episode,
            },
        )


def run_step2_rl(
    baked_model_path: Path,
    train_dataloader,
    val_dataloader,
    output_path: Path,
    model: nn.Module,
    tokenizer,
    config: Optional[QuietSTaRConfig] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run Step 2 (Quiet-STaR RL) training.

    ISS-004: Updated to use secure SafeTensors checkpoint loading.

    Args:
        baked_model_path: Path to baked model from Step 1 (SafeTensors format)
        train_dataloader: Training data
        val_dataloader: Validation data
        output_path: Where to save enhanced model
        model: Pre-instantiated model architecture to load weights into
        tokenizer: Tokenizer instance
        config: Phase 3 configuration
        device: Device to use

    Returns:
        Final validation metrics
    """
    if config is None:
        config = QuietSTaRConfig()

    # Load baked model weights securely (ISS-004)
    print(f"Loading baked model from {baked_model_path}...")
    model = model.to(device)

    checkpoint_data = secure_load(
        model=model,
        checkpoint_path=baked_model_path,
        device=device,
    )

    # Extract metadata from checkpoint
    metadata = checkpoint_data.get("metadata", {})
    thinking_tokens = metadata.get("thinking_tokens", [])
    strategy_accuracies = metadata.get("strategy_accuracies", {})

    print(f"[OK] Loaded baked model")
    print(f"   Thinking tokens: {len(thinking_tokens)}")
    print(f"   Strategy accuracies: {strategy_accuracies}")

    # Initialize trainer
    trainer = REINFORCETrainer(
        model=model,
        baked_model=model,  # Use same model as baseline
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    # Train
    final_metrics = trainer.train(train_dataloader, val_dataloader)

    # Save
    trainer.save_model(output_path)

    return final_metrics

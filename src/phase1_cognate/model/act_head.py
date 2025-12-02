"""
Adaptive Computation Time (ACT) Head

Learns when to halt recursion based on confidence and correctness.
Uses EMA calibration to prevent pathological halting behavior.

Reference: Graves, A. (2016). Adaptive Computation Time for RNNs
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import ACTConfig


class ACTHead(nn.Module):
    """
    Adaptive Computation Time Head

    Predicts halt probability for each token. Uses EMA-based
    calibration to learn when to stop iterating.
    """

    def __init__(self, d_model: int, config: ACTConfig):
        super().__init__()
        self.config = config
        self.d_model = d_model

        # Halting predictor (single logit per token)
        self.w_halt = nn.Linear(d_model, 1)

        # EMA step accuracies (tracked during training)
        self.register_buffer("ema_step_acc", torch.zeros(10))  # Support up to T_max=10
        self.register_buffer("step_count", torch.zeros(10))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute halt probabilities

        Args:
            z: [batch, seq_len, d_model] latent state

        Returns:
            q: [batch, seq_len, 1] halt probability
        """
        halt_logit = self.w_halt(z)
        q = torch.sigmoid(halt_logit)
        return q

    def should_halt(self, q: torch.Tensor, threshold: Optional[float] = None) -> bool:
        """
        Determine if should halt based on halt probabilities

        Args:
            q: [batch, seq_len, 1] halt probabilities
            threshold: Override config threshold (optional)

        Returns:
            bool: True if should halt
        """
        if threshold is None:
            threshold = self.config.halt_threshold

        avg_halt_prob = q.mean().item()
        return avg_halt_prob > threshold

    def compute_act_loss(
        self, q: torch.Tensor, step: int, is_correct: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ACT loss with EMA calibration + diversity regularization

        Args:
            q: [batch, seq_len, 1] halt probabilities
            step: Current recursion step (0-indexed)
            is_correct: [batch] binary correctness (optional)

        Returns:
            loss: Scalar ACT loss
        """
        batch_size = q.shape[0]

        # If correctness provided, update EMA
        if is_correct is not None:
            acc = is_correct.float().mean()
            self.update_ema_step_acc(step, acc)

        # Target halt probability based on EMA
        # Halt if current step is better than EMA
        target_halt = (
            (is_correct.float().mean() > self.ema_step_acc[step])
            if is_correct is not None
            else torch.ones(batch_size, device=q.device) * 0.5
        )

        # BCE loss
        target_halt = target_halt.view(batch_size, 1, 1).expand_as(q)
        loss_bce = F.binary_cross_entropy(q, target_halt)

        # Entropy regularization (prevent saturation)
        eps = 1e-8
        entropy = -(q * torch.log(q + eps) + (1 - q) * torch.log(1 - q + eps))
        loss_entropy = -self.config.entropy_reg * entropy.mean()

        # Diversity regularization (encourage variance across tokens)
        # FIX for ACT variance=0 issue
        q_mean = q.mean(dim=1, keepdim=True)  # [batch, 1, 1]
        q_variance = ((q - q_mean) ** 2).mean()  # Scalar

        # Penalize LOW variance (want tokens to have different halt probs)
        target_variance = 0.1  # Target variance (tunable)
        diversity_loss = torch.clamp(target_variance - q_variance, min=0.0)

        return loss_bce + loss_entropy + 0.01 * diversity_loss

    def update_ema_step_acc(self, step: int, accuracy: float) -> None:
        """
        Update EMA of step accuracy

        Args:
            step: Recursion step (0-indexed)
            accuracy: Accuracy at this step (0-1)
        """
        if step >= len(self.ema_step_acc):
            return  # Out of bounds

        # EMA update
        alpha = self.config.ema_decay
        self.ema_step_acc[step] = alpha * self.ema_step_acc[step] + (1 - alpha) * accuracy
        self.step_count[step] += 1

    def get_ema_stats(self) -> Dict[str, float]:
        """Get EMA statistics for W&B logging"""
        stats = {}
        for i in range(10):
            if self.step_count[i] > 0:
                stats[f"act/ema_acc_step{i}"] = self.ema_step_acc[i].item()
        return stats

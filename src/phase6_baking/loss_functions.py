"""
Phase 6: Loss Functions

Implements KL divergence loss and other loss functions for Phase 6 baking.

KL divergence measures how one probability distribution P differs from another Q:
    KL(P || Q) = sum P(x) * log(P(x) / Q(x))

For self-discovery, we want the model's output distribution to match a target
distribution during B-cycle persona refinement.

Research: "Prompt Baking" (arXiv:2409.13697v1)
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    reduction: Literal["none", "batchmean", "sum", "mean"] = "batchmean",
    temperature: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute KL divergence loss between model predictions and target distribution.

    KL(target || pred) = sum target * (log(target) - log(pred))

    Theory:
        KL divergence is asymmetric and always non-negative (Gibbs' inequality).
        KL = 0 if and only if P = Q almost everywhere.
        We use KL(target || pred) which penalizes the model for assigning
        low probability to tokens that have high probability in target.

    Args:
        logits: [batch, seq_len, vocab_size] Raw model outputs before softmax
        target_probs: [batch, seq_len, vocab_size] Target probability distribution
                      (already normalized, sums to 1 along vocab dim)
        reduction: How to reduce the loss:
            - 'none': No reduction, return full tensor
            - 'batchmean': Sum over all elements, divide by batch size (default)
            - 'sum': Sum over all elements
            - 'mean': Mean over all elements
        temperature: Softmax temperature (>1 = softer, <1 = sharper)
                     Higher temperature produces more uniform distributions.
        epsilon: Small constant for numerical stability (prevents log(0))

    Returns:
        KL divergence loss value (non-negative scalar or tensor based on reduction)

    Note:
        PyTorch's kl_div expects log(pred) as first argument, target as second.
        The 'batchmean' reduction divides by batch size, not total elements.

    Example:
        >>> batch, seq, vocab = 2, 10, 1000
        >>> logits = torch.randn(batch, seq, vocab)
        >>> target = torch.softmax(torch.randn(batch, seq, vocab), dim=-1)
        >>> loss = kl_divergence_loss(logits, target)
        >>> assert loss.item() >= 0, "KL divergence must be non-negative"
    """
    # Temperature scaling and log-softmax for numerical stability
    # log_softmax is more stable than softmax followed by log
    log_probs = F.log_softmax(logits / temperature, dim=-1)

    # Clamp target_probs to avoid log(0) in target entropy term
    target_probs = target_probs.clamp(min=epsilon)

    # KL divergence: target * (log(target) - log(pred))
    # F.kl_div expects (log_pred, target) and computes target * (log(target) - log_pred)
    loss = F.kl_div(
        log_probs,
        target_probs,
        reduction=reduction,
        log_target=False,  # target_probs is NOT in log space
    )

    return loss


def reverse_kl_divergence_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    reduction: Literal["none", "batchmean", "sum", "mean"] = "batchmean",
    temperature: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute reverse KL divergence: KL(pred || target).

    This is the "mode-seeking" variant that encourages the model to
    concentrate on modes of the target distribution rather than covering all of it.

    Use this when you want the model to be confident on specific behaviors
    rather than trying to match the full target distribution.

    Args:
        logits: [batch, seq_len, vocab_size] Raw model outputs
        target_probs: [batch, seq_len, vocab_size] Target probabilities
        reduction: Reduction method
        temperature: Softmax temperature
        epsilon: Numerical stability constant

    Returns:
        Reverse KL divergence loss (non-negative)
    """
    # Convert logits to probabilities
    pred_probs = F.softmax(logits / temperature, dim=-1)
    pred_probs = pred_probs.clamp(min=epsilon)

    # Clamp targets
    target_probs = target_probs.clamp(min=epsilon)

    # Log of both distributions
    log_pred = torch.log(pred_probs)
    log_target = torch.log(target_probs)

    # KL(pred || target) = sum pred * (log(pred) - log(target))
    kl = pred_probs * (log_pred - log_target)

    # Reduce
    if reduction == "none":
        return kl
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "mean":
        return kl.mean()
    elif reduction == "batchmean":
        return kl.sum() / logits.size(0)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def jensen_shannon_divergence(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    temperature: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence (symmetric, bounded).

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    JS divergence is symmetric and bounded in [0, log(2)].
    Useful when you want a balanced measure between distributions.

    Args:
        logits: [batch, seq_len, vocab_size] Raw model outputs
        target_probs: [batch, seq_len, vocab_size] Target probabilities
        temperature: Softmax temperature
        epsilon: Numerical stability constant

    Returns:
        JS divergence (scalar, bounded in [0, log(2)])
    """
    # Get prediction probabilities
    pred_probs = F.softmax(logits / temperature, dim=-1)

    # Clamp both
    pred_probs = pred_probs.clamp(min=epsilon)
    target_probs = target_probs.clamp(min=epsilon)

    # Mixture distribution M = 0.5 * (P + Q)
    m = 0.5 * (pred_probs + target_probs)
    m = m.clamp(min=epsilon)

    # JS = 0.5 * KL(pred || m) + 0.5 * KL(target || m)
    log_m = torch.log(m)

    kl_pred_m = (pred_probs * (torch.log(pred_probs) - log_m)).sum(dim=-1).mean()
    kl_target_m = (target_probs * (torch.log(target_probs) - log_m)).sum(dim=-1).mean()

    js = 0.5 * (kl_pred_m + kl_target_m)

    return js


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Knowledge distillation loss combining soft and hard targets.

    L = alpha * KL(teacher_soft || student_soft) + (1-alpha) * CE(labels, student)

    This is useful for self-discovery B-cycle where we want the model to
    learn from its own improved outputs (teacher = previous iteration).

    Args:
        student_logits: [batch, seq_len, vocab_size] Student model outputs
        teacher_logits: [batch, seq_len, vocab_size] Teacher model outputs
        labels: [batch, seq_len] Optional hard labels for CE loss
        temperature: Temperature for softening distributions
        alpha: Weight for soft target loss (0 = only hard, 1 = only soft)
        epsilon: Numerical stability constant

    Returns:
        Combined distillation loss
    """
    # Soft targets from teacher
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # KL loss with soft targets
    soft_loss = kl_divergence_loss(
        student_logits, teacher_probs, temperature=temperature, epsilon=epsilon
    )

    # Scale by T^2 (Hinton et al. recommendation)
    soft_loss = soft_loss * (temperature**2)

    if labels is not None and alpha < 1.0:
        # Hard target CE loss
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        return alpha * soft_loss + (1 - alpha) * hard_loss
    else:
        return soft_loss


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence Loss as a PyTorch Module.

    Wraps the functional kl_divergence_loss for use in model pipelines.
    """

    def __init__(
        self, reduction: str = "batchmean", temperature: float = 1.0, epsilon: float = 1e-8
    ):
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output logits
            target_probs: Target probability distribution

        Returns:
            KL divergence loss
        """
        return kl_divergence_loss(
            logits,
            target_probs,
            reduction=self.reduction,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )


__all__ = [
    "kl_divergence_loss",
    "reverse_kl_divergence_loss",
    "jensen_shannon_divergence",
    "distillation_loss",
    "KLDivergenceLoss",
]

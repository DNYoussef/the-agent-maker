"""
Spectral Gap Regularization Utilities

Diversity regularizers and loss functions for maintaining representation diversity.

Phase Applications:
    - Phase 3 (Quiet-STaR): Thought diversity regularization
    - Phase 7 (Experts): Expert diversity regularization
    - All phases: Anti-collapse regularization
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

# Layer 1 import only
from ..spectral_gap import (
    SpectralGapRegularizer,
    compute_expert_diversity,
    compute_thought_diversity,
    thought_diversity_loss,
)


@dataclass
class DiversityRegularizerConfig:
    """Configuration for diversity regularizers."""

    # Target diversity (spectral gap)
    target_diversity: float = 0.5

    # Loss weight
    weight: float = 0.1

    # Minimum gap before penalty kicks in
    min_gap: float = 0.01

    # Penalty type: "linear", "quadratic", "exponential"
    penalty_type: str = "quadratic"

    # Warmup (don't penalize early in training)
    warmup_steps: int = 0


class EmbeddingDiversityRegularizer:
    """Callable anti-collapse regularizer for embedding tensors."""

    def __init__(self, target_diversity: float = 0.5, weight: float = 0.1):
        self.target_diversity = target_diversity
        self.weight = weight

    def __call__(self, embeddings):
        return (
            thought_diversity_loss(
                embeddings,
                target_diversity=self.target_diversity,
            )
            * self.weight
        )


class ExpertDiversityRegularizer:
    """Callable anti-collapse regularizer for expert weight lists."""

    def __init__(self, target_diversity: float = 0.3, weight: float = 0.1):
        self.target_diversity = target_diversity
        self.weight = weight

    def __call__(self, expert_weights):
        return expert_diversity_loss(
            expert_weights,
            target_diversity=self.target_diversity,
            weight=self.weight,
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_thought_regularizer(
    target_diversity: float = 0.5,
    weight: float = 0.1,
    config: Optional[DiversityRegularizerConfig] = None,
) -> SpectralGapRegularizer:
    """
    Create a regularizer for thought diversity (Phase 3).

    Encourages diverse thinking by penalizing low spectral gap
    in the thought embedding space.

    Args:
        target_diversity: Target spectral gap value
        weight: Loss weight
        config: Optional full configuration

    Returns:
        SpectralGapRegularizer instance

    Example:
        >>> regularizer = create_thought_regularizer(target_diversity=0.5, weight=0.1)
        >>> for step, batch in enumerate(dataloader):
        ...     thoughts = model.generate_thoughts(batch)
        ...     task_loss = compute_task_loss(thoughts)
        ...     diversity_loss = regularizer(thoughts)
        ...     total_loss = task_loss + diversity_loss
    """
    if config is not None:
        target_diversity = config.target_diversity
        weight = config.weight

    return EmbeddingDiversityRegularizer(
        target_diversity=target_diversity,
        weight=weight,
    )


def create_expert_regularizer(
    target_diversity: float = 0.3,
    weight: float = 0.1,
    config: Optional[DiversityRegularizerConfig] = None,
) -> SpectralGapRegularizer:
    """
    Create a regularizer for expert diversity (Phase 7).

    Encourages diverse experts by penalizing low spectral gap
    in the expert weight space.

    Args:
        target_diversity: Target spectral gap value
        weight: Loss weight
        config: Optional full configuration

    Returns:
        SpectralGapRegularizer instance

    Example:
        >>> regularizer = create_expert_regularizer(target_diversity=0.3, weight=0.1)
        >>> expert_weights = [e.weight for e in experts]
        >>> diversity_loss = regularizer(torch.stack(expert_weights))
    """
    if config is not None:
        target_diversity = config.target_diversity
        weight = config.weight

    return ExpertDiversityRegularizer(
        target_diversity=target_diversity,
        weight=weight,
    )


def create_generic_regularizer(
    target_diversity: float = 0.5,
    weight: float = 0.1,
    config: Optional[DiversityRegularizerConfig] = None,
) -> SpectralGapRegularizer:
    """
    Create a generic diversity regularizer.

    Args:
        target_diversity: Target spectral gap value
        weight: Loss weight
        config: Optional full configuration

    Returns:
        SpectralGapRegularizer instance
    """
    if config is not None:
        target_diversity = config.target_diversity
        weight = config.weight

    return SpectralGapRegularizer(
        target_gap=target_diversity,
        weight=weight,
    )


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


def diversity_loss(
    embeddings,
    target_diversity: float = 0.5,
    weight: float = 0.1,
    config: Optional[DiversityRegularizerConfig] = None,
) -> float:
    """
    Compute diversity loss based on spectral gap.

    Loss is higher when spectral gap is below target.

    Args:
        embeddings: Embedding tensor [N, D]
        target_diversity: Target spectral gap
        weight: Loss weight
        config: Optional configuration

    Returns:
        Diversity loss value

    Example:
        >>> loss = diversity_loss(thought_embeddings, target_diversity=0.5, weight=0.1)
        >>> total_loss = task_loss + loss
    """
    config = config or DiversityRegularizerConfig(
        target_diversity=target_diversity,
        weight=weight,
    )

    return (
        thought_diversity_loss(
            embeddings,
            target_diversity=config.target_diversity,
        )
        * config.weight
    )


def anti_collapse_loss(
    embeddings,
    min_gap: float = 0.01,
    weight: float = 1.0,
    config: Optional[DiversityRegularizerConfig] = None,
) -> float:
    """
    Compute anti-collapse loss that activates only when gap is critically low.

    Uses exponential penalty to strongly discourage collapse.

    Args:
        embeddings: Embedding tensor [N, D]
        min_gap: Minimum acceptable gap before strong penalty
        weight: Loss weight
        config: Optional configuration

    Returns:
        Anti-collapse loss value

    Example:
        >>> # Add to loss when representations might collapse
        >>> collapse_loss = anti_collapse_loss(embeddings, min_gap=0.01)
        >>> total_loss = task_loss + collapse_loss
    """
    config = config or DiversityRegularizerConfig(min_gap=min_gap, weight=weight)

    # Compute current gap
    result = compute_thought_diversity(embeddings)
    current_gap = result.get("gap", 0.0)

    # Only penalize if below minimum
    if current_gap >= config.min_gap:
        return 0.0

    # Exponential penalty as gap approaches zero
    # Loss = weight * exp(-gap / min_gap)
    penalty = math.exp(-current_gap / config.min_gap) - 1

    return config.weight * max(0, penalty)


def thought_diversity_loss_with_warmup(
    embeddings,
    step: int,
    target_diversity: float = 0.5,
    weight: float = 0.1,
    warmup_steps: int = 100,
) -> float:
    """
    Thought diversity loss with warmup period.

    No penalty during warmup to allow initial learning.

    Args:
        embeddings: Embedding tensor
        step: Current training step
        target_diversity: Target diversity
        weight: Loss weight
        warmup_steps: Steps before penalty activates

    Returns:
        Diversity loss value (0 during warmup)
    """
    if step < warmup_steps:
        return 0.0

    # Gradually increase weight after warmup
    warmup_factor = min(1.0, (step - warmup_steps) / warmup_steps)
    effective_weight = weight * warmup_factor

    return diversity_loss(embeddings, target_diversity, effective_weight)


def expert_diversity_loss(
    expert_weights: list,
    target_diversity: float = 0.3,
    weight: float = 0.1,
) -> float:
    """
    Compute diversity loss for expert weights.

    Args:
        expert_weights: List of expert weight tensors
        target_diversity: Target diversity
        weight: Loss weight

    Returns:
        Expert diversity loss value

    Example:
        >>> expert_weights = [expert.weight for expert in experts]
        >>> loss = expert_diversity_loss(expert_weights)
    """
    if not expert_weights:
        return 0.0

    result = compute_expert_diversity(expert_weights)
    current_gap = result.get("gap", 0.0)

    # Penalty for being below target
    if current_gap >= target_diversity:
        return 0.0

    deficit = target_diversity - current_gap
    return weight * deficit * deficit  # Quadratic penalty


# =============================================================================
# COMBINED REGULARIZER
# =============================================================================


class CombinedDiversityRegularizer:
    """
    Combined regularizer with multiple diversity objectives.

    Example:
        >>> reg = CombinedDiversityRegularizer()
        >>> reg.add_term("thoughts", target=0.5, weight=0.1)
        >>> reg.add_term("experts", target=0.3, weight=0.05)
        >>> loss = reg.compute_loss(thoughts=thought_emb, experts=expert_weights)
    """

    def __init__(self):
        self.terms: dict = {}

    def add_term(
        self,
        name: str,
        target: float = 0.5,
        weight: float = 0.1,
        loss_fn: Optional[Callable] = None,
    ) -> "CombinedDiversityRegularizer":
        """
        Add a diversity term.

        Args:
            name: Term name (used as kwarg key in compute_loss)
            target: Target diversity
            weight: Loss weight
            loss_fn: Optional custom loss function

        Returns:
            Self for chaining
        """
        self.terms[name] = {
            "target": target,
            "weight": weight,
            "loss_fn": loss_fn or diversity_loss,
        }
        return self

    def compute_loss(self, **kwargs) -> float:
        """
        Compute combined loss from all terms.

        Args:
            **kwargs: Named embedding tensors matching term names

        Returns:
            Combined diversity loss
        """
        total_loss = 0.0

        for name, config in self.terms.items():
            if name in kwargs and kwargs[name] is not None:
                embeddings = kwargs[name]
                loss_fn = config["loss_fn"]
                term_loss = loss_fn(
                    embeddings,
                    target_diversity=config["target"],
                    weight=config["weight"],
                )
                total_loss += term_loss

        return total_loss

    def get_term_losses(self, **kwargs) -> dict:
        """
        Get individual term losses for logging.

        Args:
            **kwargs: Named embedding tensors

        Returns:
            Dictionary of term losses
        """
        losses = {}

        for name, config in self.terms.items():
            if name in kwargs and kwargs[name] is not None:
                embeddings = kwargs[name]
                loss_fn = config["loss_fn"]
                losses[name] = loss_fn(
                    embeddings,
                    target_diversity=config["target"],
                    weight=config["weight"],
                )
            else:
                losses[name] = 0.0

        losses["total"] = sum(losses.values())
        return losses

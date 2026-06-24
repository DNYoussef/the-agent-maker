"""
Phase 3: Quiet-STaR Facade

Two-step process: Prompt Baking FIRST, then Quiet-STaR RL training.
This facade provides:
- Thought diversity monitoring via spectral gap
- k(L) adaptive thought count
- Information-weighted token selection
- Policy gradient in log-space

Usage:
    from src.cross_phase.meta_calculus.phase_facades import phase3

    # Get adaptive thought count based on input complexity
    num_thoughts = phase3.get_thought_count(input_entropy=2.5)

    # Create thought diversity regularizer
    reg = phase3.create_thought_regularizer()
    loss = reg(thought_embeddings)

    # Check where to insert thoughts
    positions = phase3.get_thought_positions(token_entropies)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..gap_utils.regularization import (
    DiversityRegularizerConfig,
    create_thought_regularizer,
    thought_diversity_loss_with_warmup,
)

# Layer 2 imports (utilities)
from ..k_utils.adaptive import (
    AdaptiveConfig,
    get_thought_count,
    get_thought_count_batch,
    should_insert_thought,
)
from ..meta_grokfast import GrokfastConfig, MetaGrokfast
from ..moo_utils.hyperparams import ThoughtHyperparamProblem, optimize_thought_hyperparams

# Layer 1 imports (core)
from ..spectral_gap import SpectralGapMonitor, thought_diversity_loss
from ..transform_utils.log_space import (
    LogSpaceConfig,
    log_space_kl_divergence,
    log_space_policy_gradient,
)

# Phase 3 specific defaults
PHASE3_CONFIG = GrokfastConfig(
    alpha=0.99,
    lamb=0.1,  # Very gentle for RL stability
    filter_type="bigeometric",
    warmup_steps=100,
    qk_clip=True,  # Attention safety for RL
    qk_clip_value=10.0,
)


def create_optimizer(
    model: Any,
    lr: float = 1e-5,
    config: Optional[GrokfastConfig] = None,
    **adamw_kwargs,
) -> MetaGrokfast:
    """
    Create Phase 3 optimizer for Quiet-STaR training.

    Uses very gentle gradient filtering for RL stability.

    Args:
        model: PyTorch model
        lr: Learning rate (typically lower for RL)
        config: Optional custom config
        **adamw_kwargs: Additional AdamW kwargs

    Returns:
        MetaGrokfast configured for Phase 3
    """
    if config is None:
        return MetaGrokfast.for_phase("phase3_quietstar", model, lr, **adamw_kwargs)

    return MetaGrokfast(model, config=config, lr=lr, **adamw_kwargs)


def get_adaptive_thought_count(
    input_entropy: float,
    base_count: int = 4,
) -> int:
    """
    Get k(L)-adaptive number of thoughts for given input.

    High entropy inputs get more thoughts for exploration.
    Low entropy inputs get fewer thoughts (already clear).

    Args:
        input_entropy: Entropy of input tokens
        base_count: Base number of thoughts

    Returns:
        Recommended number of thoughts
    """
    config = AdaptiveConfig(base_thought_count=base_count)
    return get_thought_count(input_entropy, config=config)


def get_batch_thought_counts(
    entropies: List[float],
    base_count: int = 4,
) -> List[int]:
    """
    Get thought counts for a batch of inputs.

    Args:
        entropies: List of input entropies
        base_count: Base number of thoughts

    Returns:
        List of thought counts
    """
    config = AdaptiveConfig(base_thought_count=base_count)
    return get_thought_count_batch(entropies, config=config)


def get_thought_positions(
    token_entropies: List[float],
    threshold_percentile: float = 0.7,
) -> List[int]:
    """
    Get positions where thoughts should be inserted.

    Inserts at high-entropy tokens (uncertain/information-rich).

    Args:
        token_entropies: Entropy of each token
        threshold_percentile: Insert at tokens above this percentile

    Returns:
        List of token indices for thought insertion
    """
    if not token_entropies:
        return []

    threshold = np.percentile(token_entropies, threshold_percentile * 100)
    return [i for i, e in enumerate(token_entropies) if e >= threshold]


def create_diversity_regularizer(
    target_gap: float = 0.1,
    weight: float = 0.01,
    warmup_steps: int = 100,
) -> Callable:
    """
    Create thought diversity regularizer.

    Penalizes thought collapse (when thoughts become too similar).

    Args:
        target_gap: Target spectral gap
        weight: Regularization weight
        warmup_steps: Warmup steps before full regularization

    Returns:
        Regularizer function
    """
    config = DiversityRegularizerConfig(
        target_diversity=target_gap,
        weight=weight,
        warmup_steps=warmup_steps,
    )
    return create_thought_regularizer(config=config)


def compute_diversity_loss(
    thought_embeddings: Any,
    step: int = 0,
    warmup_steps: int = 100,
) -> float:
    """
    Compute diversity loss for thought embeddings.

    Args:
        thought_embeddings: Tensor of thought embeddings (n_thoughts, dim)
        step: Current training step
        warmup_steps: Warmup steps

    Returns:
        Diversity loss value
    """
    return thought_diversity_loss_with_warmup(
        thought_embeddings,
        step=step,
        warmup_steps=warmup_steps,
    )


def log_policy_gradient(
    log_probs: Any,
    advantages: Any,
) -> Any:
    """
    Compute policy gradient in log-space.

    More stable than standard policy gradient for sparse rewards.

    Args:
        log_probs: Log probabilities of actions
        advantages: Advantage values

    Returns:
        Policy gradient (to maximize)
    """
    return log_space_policy_gradient(log_probs, advantages)


def compute_kl_divergence(
    log_probs_new: Any,
    log_probs_old: Any,
) -> float:
    """
    Compute KL divergence in log-space.

    Args:
        log_probs_new: New policy log probabilities
        log_probs_old: Old policy log probabilities

    Returns:
        KL divergence
    """
    return log_space_kl_divergence(log_probs_new, log_probs_old)


def optimize_thought_params(
    evaluator: Callable,
    n_generations: int = 30,
) -> Any:
    """
    Optimize thought generation hyperparameters via MOO.

    Objectives: coherence, diversity, compute_cost, quality_improvement

    Args:
        evaluator: Function (config) -> objective_values
        n_generations: Evolution generations

    Returns:
        Pareto-optimal thought configurations
    """
    from ..moo_utils.hyperparams import HyperparamSearchConfig

    config = HyperparamSearchConfig(n_generations=n_generations)
    return optimize_thought_hyperparams(evaluator, config=config)


__all__ = [
    "create_optimizer",
    "get_adaptive_thought_count",
    "get_batch_thought_counts",
    "get_thought_positions",
    "create_diversity_regularizer",
    "compute_diversity_loss",
    "log_policy_gradient",
    "compute_kl_divergence",
    "optimize_thought_params",
    "PHASE3_CONFIG",
]

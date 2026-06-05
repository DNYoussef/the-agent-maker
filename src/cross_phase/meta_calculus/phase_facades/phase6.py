"""
Phase 6: Tool & Persona Baking Facade

A/B cycle optimization with half-baking mechanism.
A-cycle: Tool use baking
B-cycle: Persona baking
This facade provides:
- k(L) adaptive baking strength
- Half-baking ratio computation
- A/B cycle interleaving optimization
- Baking quality monitoring

Usage:
    from src.cross_phase.meta_calculus.phase_facades import phase6

    # Get baking strength for layer
    strength = phase6.get_baking_strength(layer_idx=3, total_layers=8)

    # Get half-baking ratio
    ratio = phase6.get_half_baking_ratio(variance=0.5)

    # Optimize A/B interleaving
    result = phase6.optimize_interleaving(evaluator)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

# Layer 1 imports (core)
from ..meta_grokfast import MetaGrokfast, GrokfastConfig

# Layer 2 imports (utilities)
from ..k_utils.adaptive import (
    get_baking_strength,
    get_baking_strengths_for_model,
    get_half_baking_ratio,
    AdaptiveConfig,
)
from ..gap_utils.monitoring import PhaseGapMonitor
from ..moo_utils.hyperparams import (
    optimize_ab_interleaving,
    ABCycleInterleavingProblem,
)

# Phase 6 specific defaults
PHASE6_CONFIG = GrokfastConfig(
    alpha=0.98,
    lamb=0.5,
    filter_type="bigeometric",
    warmup_steps=50,
)


def create_optimizer(
    model: Any,
    lr: float = 5e-5,
    config: Optional[GrokfastConfig] = None,
    **adamw_kwargs,
) -> MetaGrokfast:
    """
    Create Phase 6 optimizer for prompt baking.

    Args:
        model: PyTorch model
        lr: Learning rate (typically lower for baking)
        config: Optional custom config
        **adamw_kwargs: Additional AdamW kwargs

    Returns:
        MetaGrokfast configured for Phase 6
    """
    if config is None:
        return MetaGrokfast.for_phase("phase6_baking", model, lr, **adamw_kwargs)

    return MetaGrokfast(model, config=config, lr=lr, **adamw_kwargs)


def get_layer_baking_strength(
    layer_idx: int,
    total_layers: int,
    parameter_variance: Optional[float] = None,
) -> float:
    """
    Get k(L)-adaptive baking strength for a layer.

    High variance layers: lower strength (more uncertain)
    Low variance layers: higher strength (more stable)

    Args:
        layer_idx: Layer index
        total_layers: Total layers
        parameter_variance: Optional variance of layer parameters

    Returns:
        Baking strength [0.0-1.0]
    """
    if parameter_variance is not None:
        return get_baking_strength(parameter_variance)
    else:
        # Use layer position as proxy
        L = (layer_idx + 1) / total_layers
        from ..k_formula import compute_k
        k = compute_k(L)
        return 0.5 + 0.5 * k  # Higher k -> higher strength


def get_all_baking_strengths(
    model: Any,
) -> Dict[str, float]:
    """
    Get baking strengths for all layers in model.

    Args:
        model: PyTorch model

    Returns:
        Dict mapping layer name to baking strength
    """
    return get_baking_strengths_for_model(model)


def compute_half_baking_ratio(
    variance: float,
    base_ratio: float = 0.5,
) -> float:
    """
    Compute half-baking ratio based on parameter variance.

    Higher variance: more half-baking (cautious)
    Lower variance: less half-baking (confident)

    Args:
        variance: Parameter variance
        base_ratio: Base half-baking ratio

    Returns:
        Half-baking ratio [0.0-1.0]
    """
    return get_half_baking_ratio(variance, base_ratio=base_ratio)


def create_gap_monitor() -> PhaseGapMonitor:
    """
    Create spectral gap monitor for baking quality.

    Returns:
        PhaseGapMonitor configured for Phase 6
    """
    return PhaseGapMonitor("phase6_baking")


def get_a_cycle_config(
    tool_complexity: float = 0.5,
) -> Dict[str, Any]:
    """
    Get configuration for A-cycle (tool baking).

    Args:
        tool_complexity: Complexity of tool (0-1)

    Returns:
        A-cycle configuration
    """
    from ..k_formula import compute_k
    k = compute_k(tool_complexity)

    return {
        "baking_steps": int(100 / (k + 0.1)),  # More steps for complex tools
        "baking_strength": 0.3 + 0.4 * k,  # Stronger for simple tools
        "kl_weight": 0.1 * (1 + k),  # More regularization for simple
    }


def get_b_cycle_config(
    persona_depth: float = 0.5,
) -> Dict[str, Any]:
    """
    Get configuration for B-cycle (persona baking).

    Args:
        persona_depth: Depth/complexity of persona (0-1)

    Returns:
        B-cycle configuration
    """
    from ..k_formula import compute_k
    k = compute_k(persona_depth)

    return {
        "baking_steps": int(150 / (k + 0.1)),  # More steps for deep personas
        "baking_strength": 0.4 + 0.3 * k,  # Moderate strength
        "consistency_weight": 0.2 * (1 - k),  # More consistency for deep personas
    }


def optimize_interleaving(
    evaluator: Callable,
    n_generations: int = 30,
) -> Any:
    """
    Optimize A/B cycle interleaving via MOO.

    Objectives: tool_performance, persona_consistency, interference

    Args:
        evaluator: Function (interleaving_config) -> objectives
        n_generations: Evolution generations

    Returns:
        Pareto-optimal interleaving strategies
    """
    from ..moo_utils.hyperparams import HyperparamSearchConfig

    config = HyperparamSearchConfig(n_generations=n_generations)
    return optimize_ab_interleaving(evaluator, config=config)


def compute_baking_loss(
    logits_baked: Any,
    logits_original: Any,
    target_distribution: Any,
    baking_strength: float = 0.5,
) -> Any:
    """
    Compute baking loss combining KL divergence and target matching.

    Args:
        logits_baked: Logits from baked model
        logits_original: Logits from original model
        target_distribution: Target distribution to bake
        baking_strength: How strongly to bake

    Returns:
        Combined baking loss
    """
    import torch.nn.functional as F

    # KL from original (stay close)
    kl_original = F.kl_div(
        F.log_softmax(logits_baked, dim=-1),
        F.softmax(logits_original, dim=-1),
        reduction='batchmean',
    )

    # KL to target (move toward)
    kl_target = F.kl_div(
        F.log_softmax(logits_baked, dim=-1),
        target_distribution,
        reduction='batchmean',
    )

    # Weighted combination
    return (1 - baking_strength) * kl_original + baking_strength * kl_target


__all__ = [
    "create_optimizer",
    "get_layer_baking_strength",
    "get_all_baking_strengths",
    "compute_half_baking_ratio",
    "create_gap_monitor",
    "get_a_cycle_config",
    "get_b_cycle_config",
    "optimize_interleaving",
    "compute_baking_loss",
    "PHASE6_CONFIG",
]

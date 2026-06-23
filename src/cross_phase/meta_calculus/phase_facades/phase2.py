"""
Phase 2: EvoMerge Facade

Evolutionary merge over 50 generations using 6 merge techniques.
This facade provides:
- MOO for multi-objective merge optimization
- k(L) layer merge ratios
- Bigeometric merge (novel log-space technique)
- Merge quality gates

Usage:
    from src.cross_phase.meta_calculus.phase_facades import phase2

    # Get per-layer merge ratios
    ratios = phase2.get_merge_ratios(num_layers=8)

    # Run multi-objective merge optimization
    result = phase2.run_moo(evaluator)

    # Bigeometric merge (novel technique)
    merged = phase2.bigeometric_merge(model_a, model_b, ratio=0.5)

    # Check merge quality
    quality = phase2.check_merge_quality(before_models, after_model)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..gap_utils.gates import MergeQualityResult, QualityGateConfig
from ..gap_utils.gates import check_merge_quality as _check_merge_quality

# Layer 2 imports (utilities)
from ..k_utils.layer_ratios import LayerRatioConfig, get_all_merge_ratios, get_layer_merge_ratio

# Layer 1 imports (core)
from ..moo_bridge import EvoMergeProblem, MOOConfig, MOORunner
from ..moo_utils.selection import SelectionResult, select_balanced, select_knee_point
from ..transform_utils.weights import WeightMergeConfig
from ..transform_utils.weights import bigeometric_merge as _bigeometric_merge
from ..transform_utils.weights import bigeometric_merge_models


def get_merge_ratios(
    num_layers: int,
    strategy: str = "balanced",
) -> List[float]:
    """
    Get k(L)-adaptive merge ratios for all layers.

    Early layers get conservative ratios (preserve features).
    Later layers get aggressive ratios (allow innovation).

    Args:
        num_layers: Total number of layers
        strategy: "balanced", "conservative", or "aggressive"

    Returns:
        List of merge ratios [0.0-1.0] for each layer
    """
    config = LayerRatioConfig()

    if strategy == "conservative":
        config.sensitivity = 0.7
    elif strategy == "aggressive":
        config.sensitivity = 1.3

    return get_all_merge_ratios(num_layers, config=config)


def run_moo(
    evaluator: Callable,
    n_generations: int = 50,
    pop_size: int = 30,
    n_layers: int = 8,
    n_techniques: int = 6,
) -> Any:
    """
    Run multi-objective optimization for merge selection.

    Objectives: perplexity, spectral_gap, weight_norm, invariance, diversity

    Args:
        evaluator: Function (merge_config) -> objective_values
        n_generations: Evolution generations
        pop_size: Population size

    Returns:
        pymoo Result with Pareto-optimal merge configurations
    """
    problem = EvoMergeProblem(
        n_layers=n_layers,
        n_techniques=n_techniques,
        model_evaluator=evaluator,
    )
    runner = MOORunner(MOOConfig(n_generations=n_generations, population_size=pop_size))
    return runner.optimize(problem)


def select_best_merge(
    result: Any,
    method: str = "knee",
) -> SelectionResult:
    """
    Select best merge from Pareto front.

    Args:
        result: MOO result with Pareto front
        method: "knee" (best tradeoff) or "balanced" (equal objectives)

    Returns:
        SelectionResult with best merge configuration
    """
    if method == "knee":
        return select_knee_point(result)
    else:
        return select_balanced(result)


def bigeometric_merge(
    model_a: Any,
    model_b: Any,
    ratio: float = 0.5,
    per_layer_ratios: Optional[List[float]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Merge two models using bigeometric (log-space) interpolation.

    This is a NOVEL merge technique that:
    - Works in log-space (natural for log-normal weight distributions)
    - Preserves multiplicative relationships between weights
    - Handles sign differences gracefully

    Args:
        model_a: First PyTorch model
        model_b: Second PyTorch model
        ratio: Global merge ratio (0.0 = model_a, 1.0 = model_b)
        per_layer_ratios: Optional per-layer ratios (overrides ratio)

    Returns:
        Tuple of (merged_state_dict, merge_stats)
    """
    layer_alphas = None
    if per_layer_ratios is not None:
        layer_alphas = {}
        state_names = list(model_a.state_dict().keys())
        for idx, name in enumerate(state_names[: len(per_layer_ratios)]):
            layer_alphas[name] = per_layer_ratios[idx]

    result = bigeometric_merge_models(
        model_a,
        model_b,
        layer_alphas=layer_alphas,
        default_alpha=ratio,
    )
    return result["merged_state_dict"], result["stats"]


def check_merge_quality(
    models_before: List[Any],
    model_after: Any,
    min_gap_retention: float = 0.9,
) -> MergeQualityResult:
    """
    Check if merge maintains quality (spectral gap, weight norms).

    Args:
        models_before: List of models before merge
        model_after: Merged model
        min_gap_retention: Minimum fraction of original gap to retain

    Returns:
        MergeQualityResult with accept/reject decision
    """
    config = QualityGateConfig(
        min_retention=min_gap_retention,
    )
    return _check_merge_quality(models_before, model_after, config)


def get_merge_techniques() -> List[str]:
    """
    Get list of available merge techniques.

    Returns:
        List of technique names
    """
    return [
        "linear",  # Linear interpolation
        "slerp",  # Spherical linear interpolation
        "ties",  # Task Interference Elimination via Sign
        "dare",  # Drop And REscale
        "franken",  # FrankenMerge (layer-wise from different models)
        "dfs",  # Depth-first search combination
        "bigeometric",  # Log-space merge (NOVEL)
    ]


__all__ = [
    "get_merge_ratios",
    "run_moo",
    "select_best_merge",
    "bigeometric_merge",
    "check_merge_quality",
    "get_merge_techniques",
]

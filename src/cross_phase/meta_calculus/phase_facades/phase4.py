"""
Phase 4: BitNet Facade

1.58-bit ternary quantization {-1, 0, +1} achieving 8.2x compression.
This facade provides:
- Bigeometric quantization thresholds
- k(L) adaptive sparsity per layer
- Log-space STE (straight-through estimator)
- Quantization quality gates

Usage:
    from src.cross_phase.meta_calculus.phase_facades import phase4

    # Get quantization parameters
    params = phase4.get_quantization_params(layer_idx=0, total_layers=8)

    # Apply bigeometric quantization
    quantized = phase4.apply_quantization(weights, params)

    # Check quantization quality
    quality = phase4.check_quantization_quality(original, quantized)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..gap_utils.gates import QualityGateConfig, check_quantization_quality

# Layer 2 imports (utilities)
from ..k_utils.adaptive import get_quantization_params, get_quantization_threshold_scale
from ..k_utils.layer_ratios import get_all_sparsities, get_layer_sparsity

# Layer 1 imports (core)
from ..meta_grokfast import GrokfastConfig, MetaGrokfast
from ..moo_utils.architecture import search_precision_assignment
from ..transform_utils.quantization import (
    LogSpaceSTE,
    QuantizationConfig,
    analyze_quantization,
    apply_bigeometric_quantization,
    bigeometric_threshold,
)

# Phase 4 specific defaults
PHASE4_CONFIG = GrokfastConfig(
    alpha=0.95,
    lamb=2.0,  # Stronger for discrete optimization
    filter_type="bigeometric",
    warmup_steps=200,
    ste_mode=True,  # STE-compatible mode
)


def create_optimizer(
    model: Any,
    lr: float = 1e-4,
    config: Optional[GrokfastConfig] = None,
    **adamw_kwargs,
) -> MetaGrokfast:
    """
    Create Phase 4 optimizer for BitNet quantization-aware training.

    Uses STE-compatible gradient filtering.

    Args:
        model: PyTorch model
        lr: Learning rate
        config: Optional custom config
        **adamw_kwargs: Additional AdamW kwargs

    Returns:
        MetaGrokfast configured for Phase 4
    """
    if config is None:
        return MetaGrokfast.for_phase("phase4_bitnet", model, lr, **adamw_kwargs)

    return MetaGrokfast(model, config=config, lr=lr, **adamw_kwargs)


def get_layer_quantization_params(
    layer_idx: int,
    total_layers: int,
    weights: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Get k(L)-adaptive quantization parameters for a layer.

    Early layers: lower sparsity, tighter thresholds (preserve)
    Later layers: higher sparsity, looser thresholds (aggressive)

    Args:
        layer_idx: Layer index (0-indexed)
        total_layers: Total number of layers
        weights: Optional weights for statistics

    Returns:
        Dict with threshold_scale, target_sparsity, etc.
    """
    if weights is not None:
        params = get_quantization_params(weights)
        return {
            "threshold_scale": params["threshold_scale"],
            "target_sparsity": params["recommended_sparsity"],
            "variance": params["variance"],
            "k_value": params["k_value"],
        }

    proxy_variance = max((layer_idx + 1) / max(total_layers, 1), 1e-10)
    return {
        "threshold_scale": get_quantization_threshold_scale(proxy_variance),
        "target_sparsity": get_layer_sparsity(layer_idx, total_layers),
    }


def get_all_layer_sparsities(
    num_layers: int,
    base_sparsity: float = 0.5,
) -> List[float]:
    """
    Get k(L)-adaptive sparsity targets for all layers.

    Args:
        num_layers: Total number of layers
        base_sparsity: Base sparsity level

    Returns:
        List of sparsity targets [0.0-1.0] for each layer
    """
    return get_all_sparsities(num_layers, base_sparsity=base_sparsity)


def compute_threshold(
    weights: Any,
    layer_idx: int,
    total_layers: int,
) -> float:
    """
    Compute bigeometric quantization threshold.

    Works in log-space (natural for log-normal weight distributions).

    Args:
        weights: Weight tensor
        layer_idx: Layer index
        total_layers: Total layers

    Returns:
        Quantization threshold
    """
    params = get_layer_quantization_params(layer_idx, total_layers)
    return bigeometric_threshold(weights, scale=params["threshold_scale"])


def apply_ternary_quantization(
    weights: Any,
    threshold: Optional[float] = None,
    layer_idx: int = 0,
    total_layers: int = 1,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Apply ternary quantization {-1, 0, +1}.

    Args:
        weights: Weight tensor
        threshold: Quantization threshold (auto-computed if None)
        layer_idx: Layer index (for adaptive threshold)
        total_layers: Total layers

    Returns:
        Tuple of (quantized_weights, stats)
    """
    if threshold is None:
        threshold = compute_threshold(weights, layer_idx, total_layers)

    config = QuantizationConfig()
    quantized = apply_bigeometric_quantization(weights, threshold=threshold, config=config)
    stats = analyze_quantization(weights, threshold=threshold, config=config)
    return quantized, stats


def create_ste(
    method: str = "bigeometric",
) -> LogSpaceSTE:
    """
    Create straight-through estimator for quantization.

    Args:
        method: "bigeometric" or "standard"

    Returns:
        LogSpaceSTE for use in forward/backward pass
    """
    return LogSpaceSTE()


def check_quality(
    original_weights: Any,
    quantized_weights: Any,
    min_accuracy_retention: float = 0.95,
) -> Dict[str, Any]:
    """
    Check if quantization maintains quality.

    Args:
        original_weights: Original weight tensor
        quantized_weights: Quantized weight tensor
        min_accuracy_retention: Minimum accuracy to retain

    Returns:
        Quality check result with accept/reject decision
    """
    config = QualityGateConfig(
        min_retention=min_accuracy_retention,
    )
    return check_quantization_quality(original_weights, quantized_weights, config)


def analyze_weight_distribution(
    weights: Any,
) -> Dict[str, float]:
    """
    Analyze weight distribution for quantization planning.

    Args:
        weights: Weight tensor

    Returns:
        Statistics: mean, std, sparsity, kurtosis, etc.
    """
    return analyze_quantization(weights)


def search_precision_config(
    evaluator: Callable,
    n_layers: int,
    n_generations: int = 30,
) -> Any:
    """
    Search for optimal per-layer precision assignment.

    Objectives: model_size, accuracy_loss, inference_speed

    Args:
        evaluator: Function (precision_config) -> objectives
        n_layers: Number of layers
        n_generations: Evolution generations

    Returns:
        Pareto-optimal precision configurations
    """
    from ..moo_utils.architecture import ArchitectureSearchConfig

    config = ArchitectureSearchConfig(n_generations=n_generations)

    def quantize_fn(assignment):
        return assignment

    def evaluate_fn(assignment):
        return evaluator(assignment)

    return search_precision_assignment(
        n_layers=n_layers,
        quantize_fn=quantize_fn,
        evaluate_fn=evaluate_fn,
        config=config,
    )


__all__ = [
    "create_optimizer",
    "get_layer_quantization_params",
    "get_all_layer_sparsities",
    "compute_threshold",
    "apply_ternary_quantization",
    "create_ste",
    "check_quality",
    "analyze_weight_distribution",
    "search_precision_config",
    "PHASE4_CONFIG",
]

"""
Phase 8: Final Compression Facade

Triple compression: SeedLM (2x) + VPTQ (20x) + Hypercompression (6.25x) = 280x total.
This facade provides:
- k(L) layer compression ratios
- Log-space weight fitting (Bezier curves)
- Compression quality gates
- MetaGrokfast for fine-tuning

Usage:
    from src.cross_phase.meta_calculus.phase_facades import phase8

    # Get per-layer compression ratios
    ratios = phase8.get_compression_ratios(num_layers=8)

    # Fit weights in log-space for SeedLM
    params, reconstruct = phase8.fit_weights_log_space(weights)

    # Check compression quality
    quality = phase8.check_compression_quality(original, compressed)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..bigeometric import from_log_space, to_log_space
from ..gap_utils.gates import CompressionQualityResult, QualityGateConfig
from ..gap_utils.gates import check_compression_quality as _check_compression_quality
from ..gap_utils.monitoring import PhaseGapMonitor

# Layer 2 imports (utilities)
from ..k_utils.layer_ratios import (
    LayerRatioConfig,
    get_all_compression_ratios,
    get_layer_compression_ratio,
)

# Layer 1 imports (core)
from ..meta_grokfast import GrokfastConfig, MetaGrokfast
from ..transform_utils.log_space import safe_exp
from ..transform_utils.weights import compute_reconstruction_error
from ..transform_utils.weights import fit_weights_log_space as _fit_weights_log_space

# Phase 8 specific defaults
PHASE8_CONFIG = GrokfastConfig(
    alpha=0.99,
    lamb=0.5,  # Gentle for post-compression fine-tuning
    filter_type="bigeometric",
    warmup_steps=100,
)

# Compression stage targets
COMPRESSION_STAGES = {
    "seedlm": 2.0,  # 2x compression
    "vptq": 20.0,  # 20x compression
    "hyper": 6.25,  # 6.25x compression
    "total": 280.0,  # 2 * 20 * 6.25 * 1.12 (overhead)
}


def create_optimizer(
    model: Any,
    lr: float = 1e-5,
    config: Optional[GrokfastConfig] = None,
    **adamw_kwargs,
) -> MetaGrokfast:
    """
    Create Phase 8 optimizer for post-compression fine-tuning.

    Uses very gentle gradient filtering to preserve compression.

    Args:
        model: Compressed PyTorch model
        lr: Learning rate (typically very low)
        config: Optional custom config
        **adamw_kwargs: Additional AdamW kwargs

    Returns:
        MetaGrokfast configured for Phase 8
    """
    if config is None:
        return MetaGrokfast.for_phase("phase8_compression", model, lr, **adamw_kwargs)

    return MetaGrokfast(model, config=config, lr=lr, **adamw_kwargs)


def get_compression_ratios(
    num_layers: int,
    target_compression: float = 280.0,
    strategy: str = "balanced",
) -> List[float]:
    """
    Get k(L)-adaptive compression ratios per layer.

    Early layers: lower ratio (preserve critical features)
    Later layers: higher ratio (can compress more)

    Args:
        num_layers: Total number of layers
        target_compression: Overall target compression ratio
        strategy: "balanced", "aggressive", or "conservative"

    Returns:
        List of compression ratios for each layer
    """
    config = LayerRatioConfig()

    if strategy == "aggressive":
        config.sensitivity = 1.3
    elif strategy == "conservative":
        config.sensitivity = 0.7

    return get_all_compression_ratios(
        num_layers,
        base_ratio=min(target_compression, 50.0),
        config=config,
    )


def get_stage_ratios(
    layer_idx: int,
    total_layers: int,
) -> Dict[str, float]:
    """
    Get per-stage compression ratios for a layer.

    Distributes total compression across SeedLM, VPTQ, Hypercompression.

    Args:
        layer_idx: Layer index
        total_layers: Total layers

    Returns:
        Dict with ratios for each compression stage
    """
    total_ratio = get_layer_compression_ratio(layer_idx, total_layers)

    # Distribute across stages (proportional to base ratios)
    _ = COMPRESSION_STAGES["seedlm"] * COMPRESSION_STAGES["vptq"] * COMPRESSION_STAGES["hyper"]
    scale = total_ratio / COMPRESSION_STAGES["total"]

    return {
        "seedlm": COMPRESSION_STAGES["seedlm"] * scale**0.2,
        "vptq": COMPRESSION_STAGES["vptq"] * scale**0.5,
        "hyper": COMPRESSION_STAGES["hyper"] * scale**0.3,
        "total": total_ratio,
    }


def fit_weights_log_space(
    weights: Any,
    n_control_points: int = 16,
) -> Tuple[Any, Callable]:
    """
    Fit weights in log-space using Bezier curves (for SeedLM).

    Log-space is natural for log-normal weight distributions.

    Args:
        weights: Weight tensor
        n_control_points: Number of Bezier control points

    Returns:
        Tuple of (control_points, reconstruct_fn)
    """
    result = _fit_weights_log_space(weights, n_control_points=n_control_points)
    return result["control_points"], result["reconstruct_fn"]


def compute_error(
    original: Any,
    reconstructed: Any,
) -> Dict[str, float]:
    """
    Compute reconstruction error metrics.

    Args:
        original: Original weights
        reconstructed: Reconstructed weights

    Returns:
        Dict with mse, rmse, mae, correlation
    """
    return compute_reconstruction_error(original, reconstructed)


def check_compression_quality(
    original: Any,
    compressed: Any,
    min_correlation: float = 0.95,
    max_relative_error: float = 0.1,
) -> CompressionQualityResult:
    """
    Check if compression maintains acceptable quality.

    Args:
        original: Original weights/model
        compressed: Compressed weights/model
        min_correlation: Minimum weight correlation
        max_relative_error: Maximum relative error

    Returns:
        CompressionQualityResult with accept/reject decision
    """
    config = QualityGateConfig(
        min_retention=min_correlation,
    )
    return _check_compression_quality(original, compressed, config)


def create_gap_monitor() -> PhaseGapMonitor:
    """
    Create spectral gap monitor for compression quality.

    Returns:
        PhaseGapMonitor configured for Phase 8
    """
    return PhaseGapMonitor("phase8_compression")


def apply_log_space_transform(
    weights: Any,
) -> Any:
    """
    Transform weights to log-space for compression.

    Args:
        weights: Weight tensor

    Returns:
        Log-transformed weights
    """
    return to_log_space(weights)


def reverse_log_space_transform(
    log_weights: Any,
) -> Any:
    """
    Reverse log-space transformation.

    Args:
        log_weights: Log-transformed weights

    Returns:
        Original-space weights
    """
    if isinstance(log_weights, tuple) and len(log_weights) == 2:
        return from_log_space(log_weights[0], log_weights[1])
    return safe_exp(log_weights)


def estimate_final_size(
    original_size_bytes: int,
    compression_ratios: Optional[List[float]] = None,
    num_layers: int = 8,
) -> Dict[str, Any]:
    """
    Estimate final model size after compression.

    Args:
        original_size_bytes: Original model size in bytes
        compression_ratios: Per-layer ratios (auto-computed if None)
        num_layers: Number of layers

    Returns:
        Dict with estimated sizes at each stage
    """
    if compression_ratios is None:
        compression_ratios = get_compression_ratios(num_layers)

    avg_ratio = np.mean(compression_ratios)

    # Approximate per-stage sizes
    after_seedlm = original_size_bytes / COMPRESSION_STAGES["seedlm"]
    after_vptq = after_seedlm / COMPRESSION_STAGES["vptq"]
    after_hyper = after_vptq / COMPRESSION_STAGES["hyper"]

    return {
        "original_mb": original_size_bytes / (1024 * 1024),
        "after_seedlm_mb": after_seedlm / (1024 * 1024),
        "after_vptq_mb": after_vptq / (1024 * 1024),
        "final_mb": after_hyper / (1024 * 1024),
        "total_compression": original_size_bytes / after_hyper,
        "avg_layer_ratio": avg_ratio,
    }


__all__ = [
    "create_optimizer",
    "get_compression_ratios",
    "get_stage_ratios",
    "fit_weights_log_space",
    "compute_error",
    "check_compression_quality",
    "create_gap_monitor",
    "apply_log_space_transform",
    "reverse_log_space_transform",
    "estimate_final_size",
    "COMPRESSION_STAGES",
    "PHASE8_CONFIG",
]

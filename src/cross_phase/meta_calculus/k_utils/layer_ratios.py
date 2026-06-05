"""
k-Formula Layer Ratio Utilities

Per-layer ratios for merging, sparsity, and compression based on
k(L) = -0.0137*log10(L) + 0.1593 where L = layer_index / total_layers.

Key Insight:
    - Early layers (low index, high k): Conservative treatment (preserve)
    - Later layers (high index, low k): Aggressive treatment (modify/compress)

Phase Applications:
    - Phase 2 (EvoMerge): Layer-wise merge ratios
    - Phase 4 (BitNet): Layer-wise sparsity for ternary quantization
    - Phase 8 (Compression): Layer-wise compression ratios
"""

from dataclasses import dataclass
from typing import List, Optional
import math

# Layer 1 import only
from ..k_formula import k_from_layer_index, normalize_k_value


@dataclass
class LayerRatioConfig:
    """Configuration for layer ratio computations."""

    # Base ratio (center of range)
    base_ratio: float = 0.5

    # Ratio range
    min_ratio: float = 0.1
    max_ratio: float = 0.9

    # k-formula interpretation
    # True: high k = high ratio, False: high k = low ratio
    k_proportional: bool = True

    # Sensitivity to k changes
    sensitivity: float = 1.0


# =============================================================================
# MERGE RATIOS (Phase 2 EvoMerge)
# =============================================================================

def get_layer_merge_ratio(
    layer_idx: int,
    total_layers: int,
    base_ratio: float = 0.5,
    config: Optional[LayerRatioConfig] = None,
) -> float:
    """
    Get merge ratio for a specific layer using k(L) formula.

    Early layers (high k) -> conservative merge (ratio closer to base)
    Later layers (low k) -> aggressive merge (ratio deviates from base)

    Args:
        layer_idx: Layer index (0-indexed)
        total_layers: Total number of layers
        base_ratio: Base merge ratio (default 0.5 = equal blend)
        config: Optional configuration

    Returns:
        Merge ratio for the layer (0 to 1)

    Example:
        >>> for i in range(8):
        ...     ratio = get_layer_merge_ratio(i, 8)
        ...     print(f"Layer {i}: merge ratio = {ratio:.3f}")
        Layer 0: merge ratio = 0.520  # Conservative (high k)
        Layer 7: merge ratio = 0.480  # Slightly more aggressive (low k)
    """
    config = config or LayerRatioConfig(base_ratio=base_ratio)

    if total_layers <= 0:
        return config.base_ratio

    # Get k value for this layer
    k = k_from_layer_index(layer_idx, total_layers)

    # Normalize k to [0, 1]
    k_normalized = normalize_k_value(k)

    # Apply sensitivity
    k_effect = k_normalized * config.sensitivity

    if config.k_proportional:
        # High k (early layers) -> ratio closer to max (conservative)
        ratio = config.base_ratio + (config.max_ratio - config.base_ratio) * k_effect
    else:
        # High k (early layers) -> ratio closer to min
        ratio = config.base_ratio - (config.base_ratio - config.min_ratio) * k_effect

    return max(config.min_ratio, min(config.max_ratio, ratio))


def get_all_merge_ratios(
    total_layers: int,
    base_ratio: float = 0.5,
    config: Optional[LayerRatioConfig] = None,
) -> List[float]:
    """
    Get merge ratios for all layers.

    Args:
        total_layers: Number of layers
        base_ratio: Base merge ratio
        config: Optional configuration

    Returns:
        List of merge ratios per layer

    Example:
        >>> ratios = get_all_merge_ratios(8)
        >>> print([f"{r:.3f}" for r in ratios])
    """
    return [
        get_layer_merge_ratio(i, total_layers, base_ratio, config)
        for i in range(total_layers)
    ]


# =============================================================================
# SPARSITY RATIOS (Phase 4 BitNet)
# =============================================================================

def get_layer_sparsity(
    layer_idx: int,
    total_layers: int,
    base_sparsity: float = 0.5,
    config: Optional[LayerRatioConfig] = None,
) -> float:
    """
    Get sparsity ratio for a specific layer (BitNet ternary quantization).

    Sparsity = proportion of weights mapped to 0 in {-1, 0, +1}.
    Early layers (high k) -> lower sparsity (preserve information)
    Later layers (low k) -> higher sparsity (more zeros OK)

    Formula: sparsity = base * (1 - k_normalized)

    Args:
        layer_idx: Layer index (0-indexed)
        total_layers: Total number of layers
        base_sparsity: Maximum sparsity for deepest layers
        config: Optional configuration

    Returns:
        Sparsity ratio for the layer (0 to 1)

    Example:
        >>> for i in range(8):
        ...     sparsity = get_layer_sparsity(i, 8)
        ...     print(f"Layer {i}: sparsity = {sparsity:.3f}")
    """
    config = config or LayerRatioConfig(
        base_ratio=base_sparsity,
        min_ratio=0.0,
        max_ratio=0.8,
        k_proportional=False,  # High k = LOW sparsity
    )

    if total_layers <= 0:
        return config.base_ratio

    # Get k value
    k = k_from_layer_index(layer_idx, total_layers)

    # Normalize k to [0, 1]
    k_normalized = normalize_k_value(k)

    # Inverse relationship: high k -> low sparsity
    sparsity = config.base_ratio * (1 - k_normalized * config.sensitivity)

    return max(config.min_ratio, min(config.max_ratio, sparsity))


def get_all_sparsities(
    total_layers: int,
    base_sparsity: float = 0.5,
    config: Optional[LayerRatioConfig] = None,
) -> List[float]:
    """
    Get sparsity ratios for all layers.

    Args:
        total_layers: Number of layers
        base_sparsity: Maximum sparsity
        config: Optional configuration

    Returns:
        List of sparsity ratios per layer
    """
    return [
        get_layer_sparsity(i, total_layers, base_sparsity, config)
        for i in range(total_layers)
    ]


# =============================================================================
# COMPRESSION RATIOS (Phase 8)
# =============================================================================

def get_layer_compression_ratio(
    layer_idx: int,
    total_layers: int,
    base_ratio: float = 10.0,
    min_ratio: float = 2.0,
    max_ratio: float = 50.0,
    config: Optional[LayerRatioConfig] = None,
) -> float:
    """
    Get compression ratio for a specific layer.

    Early layers (high k) -> lower compression (preserve capacity)
    Later layers (low k) -> higher compression (more redundant)

    Args:
        layer_idx: Layer index (0-indexed)
        total_layers: Total number of layers
        base_ratio: Base compression ratio
        min_ratio: Minimum compression
        max_ratio: Maximum compression
        config: Optional configuration

    Returns:
        Compression ratio for the layer

    Example:
        >>> for i in range(8):
        ...     ratio = get_layer_compression_ratio(i, 8)
        ...     print(f"Layer {i}: compression = {ratio:.1f}x")
    """
    config = config or LayerRatioConfig(
        base_ratio=base_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        k_proportional=False,  # High k = LOW compression
    )

    if total_layers <= 0:
        return config.base_ratio

    # Get k value
    k = k_from_layer_index(layer_idx, total_layers)

    # Normalize k to [0, 1]
    k_normalized = normalize_k_value(k)

    # Inverse: high k (early) -> lower compression
    # Range: min_ratio at k=k_max, max_ratio at k=k_min
    ratio_range = config.max_ratio - config.min_ratio
    compression = config.min_ratio + ratio_range * (1 - k_normalized)

    return max(config.min_ratio, min(config.max_ratio, compression))


def get_all_compression_ratios(
    total_layers: int,
    base_ratio: float = 10.0,
    min_ratio: float = 2.0,
    max_ratio: float = 50.0,
    config: Optional[LayerRatioConfig] = None,
) -> List[float]:
    """
    Get compression ratios for all layers.

    Args:
        total_layers: Number of layers
        base_ratio: Base compression ratio
        min_ratio: Minimum compression
        max_ratio: Maximum compression
        config: Optional configuration

    Returns:
        List of compression ratios per layer
    """
    return [
        get_layer_compression_ratio(i, total_layers, base_ratio, min_ratio, max_ratio, config)
        for i in range(total_layers)
    ]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_layer_priority_order(total_layers: int) -> List[int]:
    """
    Get layer indices ordered by priority (high k first).

    Useful for iterating through layers in order of importance.

    Args:
        total_layers: Number of layers

    Returns:
        List of layer indices, sorted by k value (high to low)
    """
    layers_with_k = [
        (i, k_from_layer_index(i, total_layers))
        for i in range(total_layers)
    ]
    # Sort by k descending (early layers = high priority)
    layers_with_k.sort(key=lambda x: x[1], reverse=True)
    return [layer_idx for layer_idx, _ in layers_with_k]


def print_layer_ratio_table(
    total_layers: int,
    include_merge: bool = True,
    include_sparsity: bool = True,
    include_compression: bool = True,
) -> None:
    """
    Print a table of all layer ratios.

    Args:
        total_layers: Number of layers
        include_merge: Include merge ratios
        include_sparsity: Include sparsity ratios
        include_compression: Include compression ratios
    """
    print(f"\nLayer Ratio Table ({total_layers} layers)")
    print("-" * 60)

    header = f"{'Layer':>6} | {'k value':>8}"
    if include_merge:
        header += f" | {'Merge':>8}"
    if include_sparsity:
        header += f" | {'Sparsity':>8}"
    if include_compression:
        header += f" | {'Compress':>8}"

    print(header)
    print("-" * 60)

    for i in range(total_layers):
        k = k_from_layer_index(i, total_layers)
        row = f"{i:>6} | {k:>8.4f}"

        if include_merge:
            merge = get_layer_merge_ratio(i, total_layers)
            row += f" | {merge:>8.3f}"

        if include_sparsity:
            sparsity = get_layer_sparsity(i, total_layers)
            row += f" | {sparsity:>8.3f}"

        if include_compression:
            compression = get_layer_compression_ratio(i, total_layers)
            row += f" | {compression:>7.1f}x"

        print(row)

    print("-" * 60)


def compute_aggregate_compression(
    layer_ratios: List[float],
    layer_sizes: Optional[List[int]] = None,
) -> float:
    """
    Compute aggregate compression ratio from per-layer ratios.

    Args:
        layer_ratios: Compression ratio per layer
        layer_sizes: Optional parameter counts per layer (for weighted average)

    Returns:
        Aggregate compression ratio
    """
    if not layer_ratios:
        return 1.0

    if layer_sizes is None:
        # Simple average
        return sum(layer_ratios) / len(layer_ratios)

    # Weighted average by layer size
    total_original = sum(layer_sizes)
    total_compressed = sum(size / ratio for size, ratio in zip(layer_sizes, layer_ratios))

    return total_original / total_compressed if total_compressed > 0 else 1.0

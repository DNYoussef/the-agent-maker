"""
k-Formula Adaptive Parameter Utilities

Adaptive parameters based on input/parameter properties using
k(L) = -0.0137*log10(L) + 0.1593.

Key Insight:
    - Parameters adapt to context (entropy, variance, etc.)
    - k formula provides physics-motivated scaling

Phase Applications:
    - Phase 3 (Quiet-STaR): Adaptive thought count based on input entropy
    - Phase 4 (BitNet): Quantization threshold scaling
    - Phase 6 (Baking): Adaptive baking strength based on parameter variance
"""

import math
from dataclasses import dataclass
from typing import Optional, Union

# Layer 1 import only
from ..k_formula import compute_k, k_from_entropy, k_from_parameter_variance, normalize_k_value


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive parameters."""

    # General parameters
    sensitivity: float = 1.0  # How strongly k affects the output

    # Thought count (Phase 3)
    base_thought_count: int = 4
    min_thought_count: int = 1
    max_thought_count: int = 8

    # Baking strength (Phase 6)
    base_baking_strength: float = 0.5
    min_baking_strength: float = 0.1
    max_baking_strength: float = 0.9

    # Quantization (Phase 4)
    base_threshold_scale: float = 1.0
    min_threshold_scale: float = 0.5
    max_threshold_scale: float = 2.0


# =============================================================================
# THOUGHT COUNT (Phase 3 Quiet-STaR)
# =============================================================================


def get_thought_count(
    input_entropy: float,
    config: Optional[AdaptiveConfig] = None,
) -> int:
    """
    Get adaptive thought count based on input entropy.

    High entropy (uncertainty) -> more thoughts needed
    Low entropy (certainty) -> fewer thoughts sufficient

    Formula: count = base * (1 + scale / k)

    Args:
        input_entropy: Entropy of input distribution (higher = more uncertain)
        config: Optional configuration

    Returns:
        Number of thoughts to generate

    Example:
        >>> # Uncertain input (high entropy) - needs more thinking
        >>> count_uncertain = get_thought_count(5.0)
        >>> print(f"High entropy: {count_uncertain} thoughts")
        High entropy: 7 thoughts

        >>> # Certain input (low entropy) - quick answer
        >>> count_certain = get_thought_count(0.5)
        >>> print(f"Low entropy: {count_certain} thoughts")
        Low entropy: 3 thoughts
    """
    config = config or AdaptiveConfig()

    # Get k from entropy (high entropy -> low k)
    k = k_from_entropy(input_entropy)

    # Inverse relationship: low k (high entropy) -> more thoughts.
    # K_MIN is zero, so do not use 1 / K_MIN normalization here.
    k_normalized = normalize_k_value(k)
    uncertainty = max(0.0, min(1.0, (1.0 - k_normalized) * config.sensitivity))

    # Map to thought count range.
    count_range = config.max_thought_count - config.min_thought_count
    count = config.min_thought_count + int(round(count_range * uncertainty))

    return max(config.min_thought_count, min(config.max_thought_count, count))


def get_thought_count_batch(
    entropies: list,
    config: Optional[AdaptiveConfig] = None,
) -> list:
    """
    Get thought counts for a batch of inputs.

    Args:
        entropies: List of entropy values
        config: Optional configuration

    Returns:
        List of thought counts
    """
    return [get_thought_count(e, config) for e in entropies]


def should_insert_thought(
    token_info_weight: float,
    threshold: float = 0.5,
) -> bool:
    """
    Decide whether to insert a thought at a specific token position.

    High information weight -> insert thought (important token)
    Low information weight -> skip (unimportant token)

    Args:
        token_info_weight: Information weight of the token (0 to 1)
        threshold: Minimum weight to trigger thought insertion

    Returns:
        True if thought should be inserted

    Example:
        >>> # High-info token (complex concept)
        >>> should_insert_thought(0.8)
        True

        >>> # Low-info token (punctuation, filler)
        >>> should_insert_thought(0.2)
        False
    """
    return token_info_weight >= threshold


# =============================================================================
# BAKING STRENGTH (Phase 6)
# =============================================================================


def get_baking_strength(
    param_variance: float,
    config: Optional[AdaptiveConfig] = None,
) -> float:
    """
    Get adaptive baking strength based on parameter variance.

    High variance parameters -> lower strength (preserve diversity)
    Low variance parameters -> higher strength (safe to modify)

    Formula: strength = base + scale * (k - k_mid)

    Args:
        param_variance: Variance of the parameter tensor
        config: Optional configuration

    Returns:
        Baking strength (0 to 1)

    Example:
        >>> # High variance parameter (important, diverse)
        >>> strength_high_var = get_baking_strength(1.0)
        >>> print(f"High variance: strength = {strength_high_var:.3f}")
        High variance: strength = 0.350

        >>> # Low variance parameter (stable, less important)
        >>> strength_low_var = get_baking_strength(0.01)
        >>> print(f"Low variance: strength = {strength_low_var:.3f}")
        Low variance: strength = 0.650
    """
    config = config or AdaptiveConfig()

    # Get k from parameter variance (high variance -> low k)
    k = k_from_parameter_variance(param_variance)

    # Normalize k to [0, 1]
    k_normalized = normalize_k_value(k)

    # Direct relationship: high k (low variance) -> higher strength
    strength_range = config.max_baking_strength - config.min_baking_strength
    strength = config.min_baking_strength + strength_range * k_normalized * config.sensitivity

    return max(config.min_baking_strength, min(config.max_baking_strength, strength))


def get_baking_strengths_for_model(
    model,
    config: Optional[AdaptiveConfig] = None,
) -> dict:
    """
    Get baking strengths for all parameters in a model.

    Args:
        model: PyTorch model
        config: Optional configuration

    Returns:
        Dictionary mapping parameter names to baking strengths

    Example:
        >>> strengths = get_baking_strengths_for_model(model)
        >>> for name, strength in strengths.items():
        ...     print(f"{name}: {strength:.3f}")
    """
    strengths = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            variance = param.var().item()
            strengths[name] = get_baking_strength(variance, config)

    return strengths


# =============================================================================
# HALF-BAKING RATIO (Phase 6)
# =============================================================================


def get_half_baking_ratio(
    param_variance: float,
    base_ratio: float = 0.5,
    config: Optional[AdaptiveConfig] = None,
) -> float:
    """
    Get half-baking ratio for gradual integration.

    Half-baking applies changes at reduced strength for smoother integration.
    Formula: ratio = base * (1 + k_normalized)

    High variance -> lower ratio (more conservative)
    Low variance -> higher ratio (more aggressive)

    Args:
        param_variance: Variance of the parameter tensor
        base_ratio: Base half-baking ratio (default 0.5 = 50%)
        config: Optional configuration

    Returns:
        Half-baking ratio (0 to 1)

    Example:
        >>> ratio = get_half_baking_ratio(0.1)
        >>> # Apply: new_param = old_param + ratio * (baked_param - old_param)
    """
    config = config or AdaptiveConfig()

    # Get k from variance
    k = k_from_parameter_variance(param_variance)

    # Normalize k to [0, 1]
    k_normalized = normalize_k_value(k)

    # High k (low variance) -> higher ratio (more baking OK)
    ratio = base_ratio * (1 + k_normalized * config.sensitivity * 0.5)

    return max(0.1, min(0.9, ratio))


# =============================================================================
# QUANTIZATION THRESHOLD (Phase 4 BitNet)
# =============================================================================


def get_quantization_threshold_scale(
    layer_variance: float,
    config: Optional[AdaptiveConfig] = None,
) -> float:
    """
    Get threshold scaling factor for ternary quantization.

    High variance layers -> higher threshold (more zeros)
    Low variance layers -> lower threshold (preserve values)

    Args:
        layer_variance: Variance of layer weights
        config: Optional configuration

    Returns:
        Threshold scale factor

    Example:
        >>> scale = get_quantization_threshold_scale(0.1)
        >>> threshold = base_threshold * scale
        >>> # Values within threshold mapped to 0
    """
    config = config or AdaptiveConfig()

    # Get k from variance
    k = k_from_parameter_variance(layer_variance)

    # Normalize k to [0, 1]
    k_normalized = normalize_k_value(k)

    # Inverse: high k (low variance) -> lower threshold scale
    scale_range = config.max_threshold_scale - config.min_threshold_scale
    scale = config.max_threshold_scale - scale_range * k_normalized * config.sensitivity

    return max(config.min_threshold_scale, min(config.max_threshold_scale, scale))


def get_quantization_params(
    weights,
    config: Optional[AdaptiveConfig] = None,
) -> dict:
    """
    Get all quantization parameters for a weight tensor.

    Args:
        weights: Weight tensor (torch.Tensor or numpy array)
        config: Optional configuration

    Returns:
        Dictionary with threshold_scale, recommended_sparsity, etc.
    """
    config = config or AdaptiveConfig()

    # Get variance
    try:
        variance = weights.var().item()
    except AttributeError:
        import numpy as np

        variance = np.var(weights)

    k = k_from_parameter_variance(variance)
    threshold_scale = get_quantization_threshold_scale(variance, config)

    # Recommended sparsity based on k
    k_normalized = normalize_k_value(k)
    recommended_sparsity = 0.3 + 0.4 * (1 - k_normalized)  # 30-70% range

    return {
        "variance": variance,
        "k_value": k,
        "threshold_scale": threshold_scale,
        "recommended_sparsity": recommended_sparsity,
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_adaptive_param(
    input_value: float,
    base: float,
    min_val: float,
    max_val: float,
    inverse: bool = False,
    sensitivity: float = 1.0,
) -> float:
    """
    Generic adaptive parameter computation using k formula.

    Args:
        input_value: Input value for k computation
        base: Base parameter value
        min_val: Minimum output value
        max_val: Maximum output value
        inverse: If True, high k -> low output
        sensitivity: Scaling factor for k effect

    Returns:
        Adapted parameter value
    """
    k = compute_k(max(input_value, 1e-10))
    k_normalized = normalize_k_value(k)

    if inverse:
        k_normalized = 1 - k_normalized

    k_normalized *= sensitivity
    k_normalized = max(0, min(1, k_normalized))

    return min_val + (max_val - min_val) * k_normalized


def print_adaptive_params_table(
    value_range: tuple = (0.01, 10.0),
    num_samples: int = 10,
    config: Optional[AdaptiveConfig] = None,
) -> None:
    """
    Print a table of adaptive parameters across a value range.

    Args:
        value_range: (min, max) input values
        num_samples: Number of samples
        config: Optional configuration
    """
    config = config or AdaptiveConfig()

    print(f"\nAdaptive Parameters Table")
    print("-" * 75)
    print(
        f"{'Input':>8} | {'k value':>8} | {'Thoughts':>8} | "
        f"{'Baking':>8} | {'Half-Bake':>9} | {'Quant Scale':>11}"
    )
    print("-" * 75)

    min_v, max_v = value_range
    for i in range(num_samples):
        # Use log scale for better distribution
        v = min_v * ((max_v / min_v) ** (i / (num_samples - 1)))

        k = compute_k(v)
        thoughts = get_thought_count(v, config)
        baking = get_baking_strength(v, config)
        half_bake = get_half_baking_ratio(v, config=config)
        quant = get_quantization_threshold_scale(v, config)

        print(
            f"{v:>8.3f} | {k:>8.4f} | {thoughts:>8} | "
            f"{baking:>8.3f} | {half_bake:>9.3f} | {quant:>11.3f}"
        )

    print("-" * 75)

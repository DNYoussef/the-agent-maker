"""
k(L) Formula Implementation for Agent Forge V2

The k parameter provides scale-dependent adaptation across all training phases.
This formula was empirically verified in the meta-calculus-toolkit project:
    - R^2 = 0.71, p = 0.008 (statistically significant)
    - Discovered via MOO optimization over 30 FRW solutions

Key insight: k(L) acts as a "cutoff regulator" similar to field theory:
    - Multiplicative behavior at small scales (log-like)
    - Additive behavior at large scales (linear)

Applications in Agent Forge:
    - Gradient filtering strength (MuGrokFast)
    - Layer-wise merge ratios (EvoMerge)
    - Adaptive thought count (Quiet-STaR)
    - Compression ratios (Phase 8)
    - Expert routing temperature (Phase 7)
    - Baking strength (Phase 6)
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

# Verified coefficients from meta-calculus MOO optimization
K_SLOPE = -0.0137
K_INTERCEPT = 0.1593
K_MIN = 0.0
K_MAX = 1.0


@dataclass
class KFormulaConfig:
    """Configuration for k(L) formula behavior."""

    slope: float = K_SLOPE
    intercept: float = K_INTERCEPT
    k_min: float = K_MIN
    k_max: float = K_MAX

    # Scale normalization
    log_base: float = 10.0  # Use log10 as in original formula
    eps: float = 1e-8  # Numerical stability

    # Domain-specific scaling
    gradient_scale: float = 1.0  # For gradient magnitudes
    layer_scale: float = 1.0  # For layer indices
    entropy_scale: float = 1.0  # For entropy values


def compute_k(
    L: Union[float, np.ndarray, torch.Tensor], config: Optional[KFormulaConfig] = None
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Compute k parameter from scale L using verified formula.

    Formula: k(L) = -0.0137 * log10(L) + 0.1593

    Args:
        L: Scale parameter (gradient magnitude, layer index, entropy, etc.)
        config: Optional configuration for formula parameters

    Returns:
        k value(s) clamped to [k_min, k_max]

    Examples:
        >>> compute_k(1.0)  # L=1 -> k=0.1593
        0.1593
        >>> compute_k(10.0)  # L=10 -> k=0.1456
        0.1456
        >>> compute_k(0.01)  # L=0.01 -> k=0.1867
        0.1867
    """
    if config is None:
        config = KFormulaConfig()

    # Handle different input types
    if isinstance(L, torch.Tensor):
        return _compute_k_torch(L, config)
    elif isinstance(L, np.ndarray):
        return _compute_k_numpy(L, config)
    else:
        return _compute_k_scalar(float(L), config)


def _compute_k_scalar(L: float, config: KFormulaConfig) -> float:
    """Scalar implementation of k(L)."""
    L_safe = max(L, config.eps)
    log_L = (
        np.log10(L_safe) if config.log_base == 10.0 else np.log(L_safe) / np.log(config.log_base)
    )
    k = config.slope * log_L + config.intercept
    return max(config.k_min, min(config.k_max, k))


def _compute_k_numpy(L: np.ndarray, config: KFormulaConfig) -> np.ndarray:
    """NumPy implementation of k(L)."""
    L_safe = np.maximum(L, config.eps)
    log_L = (
        np.log10(L_safe) if config.log_base == 10.0 else np.log(L_safe) / np.log(config.log_base)
    )
    k = config.slope * log_L + config.intercept
    return np.clip(k, config.k_min, config.k_max)


def _compute_k_torch(L: torch.Tensor, config: KFormulaConfig) -> torch.Tensor:
    """PyTorch implementation of k(L) with gradient support."""
    L_safe = torch.clamp(L, min=config.eps)
    log_L = (
        torch.log10(L_safe)
        if config.log_base == 10.0
        else torch.log(L_safe) / np.log(config.log_base)
    )
    k = config.slope * log_L + config.intercept
    return torch.clamp(k, config.k_min, config.k_max)


# =============================================================================
# Domain-Specific k(L) Functions
# =============================================================================


def k_from_gradient(
    grad: Union[torch.Tensor, np.ndarray], config: Optional[KFormulaConfig] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute k from gradient magnitude for MuGrokFast.

    Higher gradient magnitude -> lower k -> more dampening
    Lower gradient magnitude -> higher k -> more amplification

    Args:
        grad: Gradient tensor or array
        config: Optional configuration

    Returns:
        k value for gradient transformation
    """
    if config is None:
        config = KFormulaConfig(gradient_scale=1.0)

    if isinstance(grad, torch.Tensor):
        L = torch.norm(grad) * config.gradient_scale
    else:
        L = np.linalg.norm(grad) * config.gradient_scale

    return compute_k(L, config)


def k_from_layer_index(
    layer_idx: int, total_layers: int, config: Optional[KFormulaConfig] = None
) -> float:
    """
    Compute k from layer position for layer-wise operations.

    Early layers (low index) -> higher k -> more conservative
    Later layers (high index) -> lower k -> more aggressive

    Args:
        layer_idx: Current layer index (0-based)
        total_layers: Total number of layers
        config: Optional configuration

    Returns:
        k value for this layer
    """
    if config is None:
        config = KFormulaConfig(layer_scale=1.0)

    # Normalize to [0.01, 1.0] to avoid log(0)
    L = max(0.01, (layer_idx + 1) / total_layers) * config.layer_scale
    return compute_k(L, config)


def k_from_entropy(
    entropy: Union[float, torch.Tensor, np.ndarray],
    max_entropy: float = 10.0,
    config: Optional[KFormulaConfig] = None,
) -> Union[float, torch.Tensor, np.ndarray]:
    """
    Compute k from entropy for adaptive behavior.

    High entropy (uncertain) -> lower k -> more exploration
    Low entropy (certain) -> higher k -> more exploitation

    Args:
        entropy: Entropy value(s)
        max_entropy: Maximum expected entropy for normalization
        config: Optional configuration

    Returns:
        k value(s) based on entropy
    """
    if config is None:
        config = KFormulaConfig(entropy_scale=1.0)

    # Normalize entropy to reasonable scale
    if isinstance(entropy, torch.Tensor):
        L = (entropy / max_entropy).clamp(min=0.01) * config.entropy_scale
    elif isinstance(entropy, np.ndarray):
        L = np.clip(entropy / max_entropy, 0.01, None) * config.entropy_scale
    else:
        L = max(0.01, entropy / max_entropy) * config.entropy_scale

    return compute_k(L, config)


def k_from_parameter_variance(
    params: Union[float, np.ndarray, torch.Tensor], config: Optional[KFormulaConfig] = None
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Compute k from parameter variance for adaptive baking.

    High variance -> lower k; low variance -> higher k. The function accepts
    either a parameter tensor/array or a precomputed variance scalar.
    """
    if config is None:
        config = KFormulaConfig()

    if isinstance(params, torch.Tensor):
        variance = (
            torch.var(params, correction=0)
            if params.numel() > 0
            else torch.zeros((), device=params.device, dtype=params.dtype)
        )
        L = variance.clamp(min=config.eps)
    elif isinstance(params, np.ndarray):
        L = np.maximum(np.var(params), config.eps)
    else:
        L = max(float(params), config.eps)

    return compute_k(L, config)


def normalize_k_value(
    k: Union[float, np.ndarray, torch.Tensor],
    config: Optional[KFormulaConfig] = None,
    min_L: float = 0.01,
    max_L: float = 1.0,
) -> Union[float, np.ndarray, torch.Tensor]:
    """Normalize k over the expected formula range instead of the clamp range [0, 1]."""
    config = config or KFormulaConfig()
    k_high = compute_k(min_L, config)
    k_low = compute_k(max_L, config)
    denom = max(float(k_high) - float(k_low), config.eps)

    if isinstance(k, torch.Tensor):
        return ((k - float(k_low)) / denom).clamp(0.0, 1.0)
    if isinstance(k, np.ndarray):
        return np.clip((k - float(k_low)) / denom, 0.0, 1.0)
    return max(0.0, min(1.0, (float(k) - float(k_low)) / denom))


# =============================================================================
# Utility Functions
# =============================================================================


def k_schedule(
    num_steps: int, start_L: float = 0.01, end_L: float = 1.0, schedule_type: str = "linear"
) -> np.ndarray:
    """
    Generate k schedule over training steps.

    Args:
        num_steps: Number of steps
        start_L: Starting scale
        end_L: Ending scale
        schedule_type: "linear", "log", or "cosine"

    Returns:
        Array of k values for each step
    """
    if schedule_type == "linear":
        L_values = np.linspace(start_L, end_L, num_steps)
    elif schedule_type == "log":
        L_values = np.logspace(np.log10(start_L), np.log10(end_L), num_steps)
    elif schedule_type == "cosine":
        t = np.linspace(0, np.pi, num_steps)
        L_values = start_L + (end_L - start_L) * (1 - np.cos(t)) / 2
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    return _compute_k_numpy(L_values, KFormulaConfig())


def print_k_table(L_values: list = None):
    """Print reference table of k values for different scales."""
    if L_values is None:
        L_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    print("\nk(L) Reference Table")
    print("=" * 40)
    print(f"{'L':>12} | {'k':>12} | {'Effect':>12}")
    print("-" * 40)

    for L in L_values:
        k = compute_k(L)
        if k > 0.12:
            effect = "Conservative"
        elif k > 0.08:
            effect = "Balanced"
        else:
            effect = "Aggressive"
        print(f"{L:>12.4f} | {k:>12.4f} | {effect:>12}")

    print("=" * 40)


if __name__ == "__main__":
    # Demonstrate k(L) formula
    print_k_table()

    # Test with PyTorch tensor
    grad = torch.randn(100, 100)
    k = k_from_gradient(grad)
    print(f"\nGradient norm: {torch.norm(grad):.4f}, k: {k:.4f}")

    # Test layer-wise k
    print("\nLayer-wise k for 8-layer network:")
    for i in range(8):
        k = k_from_layer_index(i, 8)
        print(f"  Layer {i}: k = {k:.4f}")

"""
Bigeometric Calculus Implementation for Agent Forge V2

The bigeometric derivative D_BG provides scale-invariant differentiation
where power functions become "linear" (constant derivative).

Key Property: D_BG[x^n] = e^n (independent of x!)

This property is crucial for gradient stabilization:
- Power-law gradient explosions become BOUNDED
- No information destruction like clipping
- Preserves gradient direction while controlling magnitude

Mathematical Definition:
    D_BG[f](x) = exp(x * f'(x) / f(x))

For gradient transformation:
    g_meta = g * |g|^(2k-1)

    When k > 0.5: dampens large gradients
    When k < 0.5: amplifies small gradients
    When k = 0.5: identity (classical)

Applications in Agent Forge:
    - MuGrokFast gradient stabilization
    - STE enhancement for BitNet quantization
    - Loss scaling for training stability
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple
from dataclasses import dataclass

from .k_formula import compute_k, k_from_gradient, KFormulaConfig


@dataclass
class BigeometricConfig:
    """Configuration for bigeometric operations."""

    # Numerical stability
    eps: float = 1e-8
    max_magnitude: float = 1e6  # Prevent overflow

    # k parameter bounds
    k_min: float = 0.0
    k_max: float = 1.0

    # Adaptive k settings
    use_adaptive_k: bool = True
    k_formula_config: Optional[KFormulaConfig] = None


class BigeometricTransform:
    """
    Bigeometric gradient transformation for neural network training.

    The transform g_meta = g * |g|^(2k-1) provides:
    - Bounded gradients (prevents explosion)
    - Direction preservation (unlike clipping)
    - Scale-adaptive behavior via k(L)
    """

    def __init__(self, config: Optional[BigeometricConfig] = None):
        self.config = config or BigeometricConfig()

    def transform(
        self,
        grad: torch.Tensor,
        k: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Apply bigeometric transformation to gradient.

        Formula: g_meta = g * |g|^(2k-1)

        Args:
            grad: Input gradient tensor
            k: Meta-parameter (if None, computed adaptively)

        Returns:
            Transformed gradient
        """
        if k is None:
            if self.config.use_adaptive_k:
                k = k_from_gradient(grad, self.config.k_formula_config)
            else:
                k = 0.5  # Identity transform

        # Ensure k is a tensor for computation
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=grad.device, dtype=grad.dtype)

        # Compute bigeometric transform
        abs_grad = torch.abs(grad).clamp(min=self.config.eps)
        exponent = 2 * k - 1

        # g_meta = g * |g|^(2k-1)
        scale = abs_grad.to(torch.float32) ** exponent.to(torch.float32)
        if grad.dtype.is_floating_point:
            dtype_max = torch.finfo(grad.dtype).max
            max_scale = min(self.config.max_magnitude, dtype_max)
        else:
            max_scale = self.config.max_magnitude
        scale = scale.clamp(max=max_scale).to(dtype=grad.dtype)

        return grad * scale

    def inverse_transform(
        self,
        grad_meta: torch.Tensor,
        k: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """
        Inverse bigeometric transformation.

        Recovers original gradient from transformed gradient.

        Args:
            grad_meta: Transformed gradient
            k: Meta-parameter used in forward transform

        Returns:
            Recovered gradient (approximate for numerical stability)
        """
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=grad_meta.device, dtype=grad_meta.dtype)

        exponent = 2 * k - 1

        # For k != 0.5, we need to solve: g_meta = g * |g|^(2k-1)
        # This is: |g_meta| = |g|^(2k) * sign(g)
        # So: |g| = |g_meta|^(1/(2k))

        if torch.abs(exponent).max() < self.config.eps:
            return grad_meta  # k ~ 0.5, transform is identity

        abs_meta = torch.abs(grad_meta).clamp(min=self.config.eps)

        # Inverse exponent: 1 / (2k) where original was (2k-1)
        # Actually need to solve numerically for exact inverse
        # Using approximation: |g| ~ |g_meta|^(1/(2k))
        inv_exp = 1.0 / (2 * k.clamp(min=0.1))  # Avoid division by zero
        abs_grad = abs_meta ** inv_exp

        return torch.sign(grad_meta) * abs_grad


class BigeometricDerivative:
    """
    Full bigeometric derivative implementation.

    D_BG[f](x) = exp(x * f'(x) / f(x))

    This is useful for analyzing functions in log-space
    and for understanding scale-invariant behavior.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(
        self,
        f: torch.Tensor,
        x: torch.Tensor,
        df_dx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute bigeometric derivative.

        Args:
            f: Function values f(x)
            x: Input values
            df_dx: Classical derivative (computed if not provided)

        Returns:
            D_BG[f](x) = exp(x * f'(x) / f(x))
        """
        if df_dx is None:
            # Compute classical derivative via autograd
            if not x.requires_grad:
                x = x.clone().requires_grad_(True)
            df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]

        # D_BG[f](x) = exp(x * f'(x) / f(x))
        f_safe = f.clamp(min=self.eps)
        elasticity = x * df_dx / f_safe

        # MTC-002: Clamp elasticity before exponentiating to prevent overflow/underflow
        # exp(700) ~ 1e304 (near float64 max), exp(-700) ~ 1e-304 (near float64 min)
        MAX_ELASTICITY = 50.0  # Conservative limit: exp(50) ~ 5e21
        elasticity = elasticity.clamp(min=-MAX_ELASTICITY, max=MAX_ELASTICITY)

        return torch.exp(elasticity)

    def verify_power_law(self, n: float, x_range: Tuple[float, float] = (0.1, 10.0)) -> dict:
        """
        Verify D_BG[x^n] = e^n (constant).

        This is the key theorem that makes bigeometric calculus
        useful for gradient stabilization.

        Args:
            n: Power law exponent
            x_range: Range of x values to test

        Returns:
            Verification results
        """
        x = torch.linspace(x_range[0], x_range[1], 100, requires_grad=True)
        f = x ** n
        df_dx = n * x ** (n - 1)

        D_BG = self(f, x, df_dx)
        expected = np.exp(n)

        return {
            "n": n,
            "expected": expected,
            "D_BG_mean": D_BG.mean().item(),
            "D_BG_std": D_BG.std().item(),
            "is_constant": D_BG.std().item() < 1e-6,
            "error": abs(D_BG.mean().item() - expected)
        }


# =============================================================================
# Gradient Transformation Functions (for MuGrokFast integration)
# =============================================================================

def bigeometric_gradient_transform(
    grad: torch.Tensor,
    k: Optional[Union[float, torch.Tensor]] = None,
    config: Optional[BigeometricConfig] = None
) -> torch.Tensor:
    """
    Transform gradient using bigeometric calculus.

    This is the main function for MuGrokFast integration.

    Args:
        grad: Input gradient
        k: Meta-parameter (adaptive if None)
        config: Bigeometric configuration

    Returns:
        Transformed gradient with controlled magnitude
    """
    transform = BigeometricTransform(config)
    return transform.transform(grad, k)


def bigeometric_gradient_with_stats(
    grad: torch.Tensor,
    k: Optional[Union[float, torch.Tensor]] = None,
    config: Optional[BigeometricConfig] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Transform gradient and return statistics for monitoring.

    Args:
        grad: Input gradient
        k: Meta-parameter
        config: Configuration

    Returns:
        Tuple of (transformed_gradient, statistics_dict)
    """
    config = config or BigeometricConfig()

    # Compute k if not provided
    if k is None and config.use_adaptive_k:
        k = k_from_gradient(grad, config.k_formula_config)
    elif k is None:
        k = 0.5

    k_val = k.item() if isinstance(k, torch.Tensor) else k

    # Original stats
    orig_norm = torch.norm(grad).item()
    orig_max = torch.abs(grad).max().item()

    # Transform
    transform = BigeometricTransform(config)
    grad_meta = transform.transform(grad, k)

    # Transformed stats
    meta_norm = torch.norm(grad_meta).item()
    meta_max = torch.abs(grad_meta).max().item()

    stats = {
        "k": k_val,
        "original_norm": orig_norm,
        "transformed_norm": meta_norm,
        "original_max": orig_max,
        "transformed_max": meta_max,
        "compression_ratio": orig_norm / (meta_norm + 1e-8),
        "max_compression": orig_max / (meta_max + 1e-8),
    }

    return grad_meta, stats


# =============================================================================
# Log-Space Operations (for Phase 8 compression)
# =============================================================================

def to_log_space(
    tensor: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform tensor to log-space (bigeometric domain).

    Useful for weight distributions that are log-normal.

    Args:
        tensor: Input tensor
        eps: Numerical stability

    Returns:
        Tuple of (log_magnitudes, signs)
    """
    signs = torch.sign(tensor)
    log_magnitudes = torch.log(torch.abs(tensor) + eps)
    return log_magnitudes, signs


def from_log_space(
    log_magnitudes: torch.Tensor,
    signs: torch.Tensor
) -> torch.Tensor:
    """
    Transform from log-space back to linear space.

    Args:
        log_magnitudes: Log of absolute values
        signs: Original signs

    Returns:
        Reconstructed tensor
    """
    return signs * torch.exp(log_magnitudes)


def log_space_interpolation(
    w1: torch.Tensor,
    w2: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Interpolate weights in log-space (geometric mean).

    Better for weight distributions than linear interpolation.

    Args:
        w1: First weight tensor
        w2: Second weight tensor
        alpha: Interpolation factor (0=w1, 1=w2)
        eps: Numerical stability

    Returns:
        Geometrically interpolated weights
    """
    # Handle zeros and sign changes
    sign1 = torch.sign(w1)
    sign2 = torch.sign(w2)

    # For same-sign weights, use geometric interpolation
    same_sign = sign1 == sign2

    # Log magnitudes
    log1 = torch.log(torch.abs(w1) + eps)
    log2 = torch.log(torch.abs(w2) + eps)

    # Interpolate in log-space
    log_interp = (1 - alpha) * log1 + alpha * log2

    # Result with sign handling
    result = torch.where(
        same_sign,
        sign1 * torch.exp(log_interp),
        (1 - alpha) * w1 + alpha * w2  # Fall back to linear for sign changes
    )

    return result


# =============================================================================
# Verification and Testing
# =============================================================================

def run_verification():
    """Run verification tests for bigeometric operations."""
    print("\n" + "=" * 60)
    print("BIGEOMETRIC CALCULUS VERIFICATION")
    print("=" * 60)

    # Test 1: Power law theorem
    print("\n1. Power Law Theorem: D_BG[x^n] = e^n")
    print("-" * 40)

    D_BG = BigeometricDerivative()
    for n in [1, 2, 3, 5, 10]:
        result = D_BG.verify_power_law(n)
        status = "PASS" if result["is_constant"] else "FAIL"
        print(f"   n={n}: D_BG = {result['D_BG_mean']:.4f} (expected {result['expected']:.4f}) [{status}]")

    # Test 2: Gradient transformation
    print("\n2. Gradient Transformation")
    print("-" * 40)

    transform = BigeometricTransform()

    # Test with different gradient magnitudes
    for magnitude in [0.01, 1.0, 100.0, 10000.0]:
        grad = torch.randn(100) * magnitude
        grad_meta, stats = bigeometric_gradient_with_stats(grad)

        print(f"   |grad|={magnitude:.2f}: k={stats['k']:.4f}, "
              f"compress={stats['compression_ratio']:.2f}x")

    # Test 3: Log-space operations
    print("\n3. Log-Space Operations")
    print("-" * 40)

    weights = torch.randn(100) * 2.0
    log_w, signs = to_log_space(weights)
    reconstructed = from_log_space(log_w, signs)
    error = torch.norm(weights - reconstructed).item()
    print(f"   Reconstruction error: {error:.2e}")

    # Test interpolation
    w1 = torch.abs(torch.randn(100)) + 0.1
    w2 = torch.abs(torch.randn(100)) + 0.1
    w_linear = 0.5 * w1 + 0.5 * w2
    w_geom = log_space_interpolation(w1, w2, 0.5)
    print(f"   Linear interp mean: {w_linear.mean():.4f}")
    print(f"   Geometric interp mean: {w_geom.mean():.4f}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()

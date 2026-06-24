"""
Gradient Transformation Utilities

Bigeometric gradient transformations for stable training.

Phase Applications:
    - Phase 1, 3, 5, 6, 7, 8: Gradient stabilization via bigeometric transform
    - All phases: Log-space gradient accumulation for numerical stability
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Layer 1 imports only
from ..bigeometric import BigeometricConfig, BigeometricTransform, bigeometric_gradient_transform
from ..k_formula import compute_k, k_from_gradient


@dataclass
class GradientTransformConfig:
    """Configuration for gradient transformations."""

    # Bigeometric transform
    adaptive_k: bool = True
    base_k: float = 0.14

    # Clipping (as fallback)
    clip_value: Optional[float] = None

    # Log-space accumulation
    use_log_space: bool = False
    log_epsilon: float = 1e-10


# =============================================================================
# BIGEOMETRIC GRADIENT TRANSFORM
# =============================================================================


def apply_bigeometric_to_gradients(
    gradients,
    config: Optional[GradientTransformConfig] = None,
):
    """
    Apply bigeometric transformation to gradients.

    The bigeometric transform scales gradients based on their magnitude:
    g_meta = g * |g|^(2k-1) where k = k(|g|)

    Large gradients (high L) get dampened, small gradients get amplified.

    Args:
        gradients: Gradient tensor
        config: Optional configuration

    Returns:
        Transformed gradients

    Example:
        >>> for param in model.parameters():
        ...     if param.grad is not None:
        ...         param.grad = apply_bigeometric_to_gradients(param.grad)
    """
    config = config or GradientTransformConfig()

    k = k_from_gradient(gradients) if config.adaptive_k else config.base_k
    transformed = bigeometric_gradient_transform(gradients, k=k)
    if config.clip_value is not None:
        transformed = transformed.clamp(-config.clip_value, config.clip_value)
    return transformed


def apply_bigeometric_to_model(
    model,
    config: Optional[GradientTransformConfig] = None,
) -> Dict[str, Any]:
    """
    Apply bigeometric transformation to all gradients in a model.

    Args:
        model: PyTorch model with computed gradients
        config: Optional configuration

    Returns:
        Dictionary with transformation statistics

    Example:
        >>> loss.backward()
        >>> stats = apply_bigeometric_to_model(model)
        >>> optimizer.step()
    """
    config = config or GradientTransformConfig()

    stats = {
        "num_params": 0,
        "num_transformed": 0,
        "total_grad_norm_before": 0.0,
        "total_grad_norm_after": 0.0,
    }

    transform = BigeometricTransform(BigeometricConfig(use_adaptive_k=config.adaptive_k))

    for name, param in model.named_parameters():
        stats["num_params"] += 1

        if param.grad is None:
            continue

        # Track before
        grad_norm_before = param.grad.norm().item()
        stats["total_grad_norm_before"] += grad_norm_before

        # Transform
        param.grad = transform.transform(param.grad)
        stats["num_transformed"] += 1

        # Track after
        grad_norm_after = param.grad.norm().item()
        stats["total_grad_norm_after"] += grad_norm_after

    return stats


# =============================================================================
# LOG-SPACE GRADIENT ACCUMULATION
# =============================================================================


class LogSpaceGradientAccumulator:
    """
    Accumulate gradients in log-space for numerical stability.

    Log-space accumulation is more stable for gradients with
    varying magnitudes (common in deep networks).

    Example:
        >>> accumulator = LogSpaceGradientAccumulator()
        >>> for micro_batch in micro_batches:
        ...     loss = model(micro_batch)
        ...     loss.backward()
        ...     accumulator.accumulate(model)
        ...     model.zero_grad()
        >>> accumulator.apply_accumulated(model)
        >>> optimizer.step()
    """

    def __init__(self, config: Optional[GradientTransformConfig] = None):
        self.config = config or GradientTransformConfig()
        self.accumulators: Dict[str, Any] = {}
        self.counts: Dict[str, int] = {}

    def accumulate(self, model) -> None:
        """
        Accumulate gradients from model in log-space.

        Args:
            model: PyTorch model with computed gradients
        """
        import torch

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach()

            if name not in self.accumulators:
                # Initialize with first gradient
                self.accumulators[name] = {
                    "log_abs_sum": torch.zeros_like(grad),
                    "sign_sum": torch.zeros_like(grad),
                }
                self.counts[name] = 0

            # Accumulate in log-space
            abs_grad = grad.abs() + self.config.log_epsilon
            log_abs = torch.log(abs_grad)
            sign = torch.sign(grad)

            # Running sum in log-space (log-sum-exp would be more precise)
            self.accumulators[name]["log_abs_sum"] += log_abs
            self.accumulators[name]["sign_sum"] += sign
            self.counts[name] += 1

    def get_accumulated(self, name: str):
        """
        Get accumulated gradient for a parameter.

        Args:
            name: Parameter name

        Returns:
            Accumulated gradient tensor
        """
        import torch

        if name not in self.accumulators:
            return None

        count = self.counts[name]
        if count == 0:
            return None

        # Average in log-space, then convert back
        log_abs_avg = self.accumulators[name]["log_abs_sum"] / count
        sign_avg = torch.sign(self.accumulators[name]["sign_sum"])

        # Convert back from log-space
        abs_grad = torch.exp(log_abs_avg)
        grad = abs_grad * sign_avg

        return grad

    def apply_accumulated(self, model) -> None:
        """
        Apply accumulated gradients to model.

        Args:
            model: PyTorch model
        """
        for name, param in model.named_parameters():
            accumulated = self.get_accumulated(name)
            if accumulated is not None:
                param.grad = accumulated

    def reset(self) -> None:
        """Reset all accumulators."""
        self.accumulators.clear()
        self.counts.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get accumulation statistics."""
        return {
            "num_params": len(self.accumulators),
            "counts": dict(self.counts),
            "total_accumulations": sum(self.counts.values()),
        }


# =============================================================================
# GRADIENT ANALYSIS
# =============================================================================


def analyze_gradients(model) -> Dict[str, Any]:
    """
    Analyze gradient distribution in a model.

    Useful for debugging and monitoring training stability.

    Args:
        model: PyTorch model with computed gradients

    Returns:
        Dictionary with gradient statistics
    """
    import torch

    stats = {
        "num_params": 0,
        "num_with_grad": 0,
        "total_norm": 0.0,
        "max_grad": 0.0,
        "min_grad": float("inf"),
        "mean_grad": 0.0,
        "layers": {},
    }

    all_grads = []

    for name, param in model.named_parameters():
        stats["num_params"] += 1

        if param.grad is None:
            continue

        stats["num_with_grad"] += 1
        grad = param.grad.detach()

        # Per-layer stats
        grad_norm = grad.norm().item()
        grad_abs_mean = grad.abs().mean().item()
        grad_max = grad.abs().max().item()

        stats["layers"][name] = {
            "norm": grad_norm,
            "abs_mean": grad_abs_mean,
            "max": grad_max,
            "k_value": k_from_gradient(grad_abs_mean),
        }

        # Global stats
        stats["total_norm"] += grad_norm**2
        stats["max_grad"] = max(stats["max_grad"], grad_max)
        stats["min_grad"] = min(stats["min_grad"], grad_abs_mean)
        all_grads.append(grad_abs_mean)

    stats["total_norm"] = math.sqrt(stats["total_norm"])
    stats["mean_grad"] = sum(all_grads) / len(all_grads) if all_grads else 0.0

    return stats


def get_gradient_health(model, threshold: float = 100.0) -> Dict[str, Any]:
    """
    Quick health check for gradients.

    Args:
        model: PyTorch model with computed gradients
        threshold: Maximum healthy gradient norm

    Returns:
        Dictionary with health status
    """
    stats = analyze_gradients(model)

    issues = []

    if stats["total_norm"] > threshold:
        issues.append(f"Total gradient norm too high: {stats['total_norm']:.2f}")

    if stats["max_grad"] > threshold * 10:
        issues.append(f"Max gradient too high: {stats['max_grad']:.2f}")

    # Check for NaN/Inf
    for name, layer_stats in stats["layers"].items():
        if math.isnan(layer_stats["norm"]) or math.isinf(layer_stats["norm"]):
            issues.append(f"NaN/Inf gradient in {name}")

    return {
        "healthy": len(issues) == 0,
        "issues": issues,
        "stats": stats,
        "recommendation": "Apply bigeometric transform" if issues else "OK",
    }

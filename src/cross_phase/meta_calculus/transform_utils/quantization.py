"""
Bigeometric Quantization Utilities

Log-space quantization for BitNet ternary compression.

Phase Applications:
    - Phase 4 (BitNet): Bigeometric thresholds for ternary quantization
    - Phase 8 (Compression): Log-space quantization for hypercompression
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math

# Layer 1 imports only
from ..k_formula import compute_k, k_from_parameter_variance
from ..bigeometric import to_log_space, from_log_space


@dataclass
class QuantizationConfig:
    """Configuration for bigeometric quantization."""

    # Numerical stability
    epsilon: float = 1e-10

    # Threshold scaling
    threshold_scale: float = 1.0

    # Ternary values
    positive_value: float = 1.0
    negative_value: float = -1.0
    zero_value: float = 0.0

    # STE options
    ste_scale: float = 1.0


# =============================================================================
# BIGEOMETRIC THRESHOLD (Phase 4)
# =============================================================================

def bigeometric_threshold(
    weights,
    scale: float = 1.0,
    config: Optional[QuantizationConfig] = None,
):
    """
    Compute quantization threshold in log-space.

    Threshold is set based on the log-space mean of weight magnitudes,
    which is more appropriate for log-normal weight distributions.

    Formula: threshold = exp(k * mean(log|w|)) * scale

    Args:
        weights: Weight tensor
        scale: Additional scaling factor
        config: Optional configuration

    Returns:
        Threshold value for ternary quantization

    Example:
        >>> threshold = bigeometric_threshold(layer.weight)
        >>> # Values in [-threshold, threshold] -> 0
        >>> # Values > threshold -> +1
        >>> # Values < -threshold -> -1
    """
    config = config or QuantizationConfig()

    try:
        import torch
        if isinstance(weights, torch.Tensor):
            # Compute in log-space
            abs_weights = weights.abs() + config.epsilon
            log_weights = torch.log(abs_weights)
            mean_log = log_weights.mean()

            # Get k based on variance. k gently raises the geometric threshold
            # without replacing the actual log-space magnitude scale.
            k = k_from_parameter_variance(weights)
            threshold = torch.exp(mean_log) * (1.0 + k) * scale * config.threshold_scale

            return threshold
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(weights, np.ndarray):
            abs_weights = np.abs(weights) + config.epsilon
            log_weights = np.log(abs_weights)
            mean_log = np.mean(log_weights)

            k = k_from_parameter_variance(weights)

            threshold = np.exp(mean_log) * (1.0 + k) * scale * config.threshold_scale

            return threshold
    except ImportError:
        pass

    raise TypeError("weights must be torch.Tensor or numpy.ndarray")


def compute_adaptive_threshold(
    weights,
    target_sparsity: float = 0.5,
    config: Optional[QuantizationConfig] = None,
) -> float:
    """
    Compute threshold to achieve target sparsity.

    Args:
        weights: Weight tensor
        target_sparsity: Desired proportion of zeros
        config: Optional configuration

    Returns:
        Threshold value
    """
    config = config or QuantizationConfig()

    try:
        import torch
        if isinstance(weights, torch.Tensor):
            # Sort absolute values
            abs_sorted = weights.abs().flatten().sort()[0]
            n = len(abs_sorted)

            # Find threshold at target percentile
            idx = int(n * target_sparsity)
            threshold = abs_sorted[min(idx, n - 1)]

            return threshold.item()
    except ImportError:
        pass

    raise TypeError("weights must be torch.Tensor")


# =============================================================================
# TERNARY QUANTIZATION
# =============================================================================

def apply_bigeometric_quantization(
    weights,
    threshold: Optional[float] = None,
    config: Optional[QuantizationConfig] = None,
):
    """
    Apply ternary quantization using bigeometric threshold.

    Maps weights to {-1, 0, +1} based on log-space threshold.

    Args:
        weights: Weight tensor
        threshold: Optional pre-computed threshold
        config: Optional configuration

    Returns:
        Quantized weights

    Example:
        >>> quantized = apply_bigeometric_quantization(layer.weight)
        >>> # quantized contains only {-1, 0, +1}
    """
    config = config or QuantizationConfig()

    if threshold is None:
        threshold = bigeometric_threshold(weights, config=config)

    try:
        import torch
        if isinstance(weights, torch.Tensor):
            # Ternary mapping
            quantized = torch.zeros_like(weights)
            quantized[weights > threshold] = config.positive_value
            quantized[weights < -threshold] = config.negative_value
            # Values in [-threshold, threshold] remain 0

            return quantized
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(weights, np.ndarray):
            quantized = np.zeros_like(weights)
            quantized[weights > threshold] = config.positive_value
            quantized[weights < -threshold] = config.negative_value

            return quantized
    except ImportError:
        pass

    raise TypeError("weights must be torch.Tensor or numpy.ndarray")


def get_ternary_mapping(
    weights,
    threshold: Optional[float] = None,
    config: Optional[QuantizationConfig] = None,
) -> Dict[str, Any]:
    """
    Get ternary quantization mapping with statistics.

    Args:
        weights: Weight tensor
        threshold: Optional pre-computed threshold
        config: Optional configuration

    Returns:
        Dictionary with mapping and statistics
    """
    config = config or QuantizationConfig()

    if threshold is None:
        threshold = bigeometric_threshold(weights, config=config)

    try:
        import torch
        if isinstance(weights, torch.Tensor):
            n_total = weights.numel()

            positive_mask = weights > threshold
            negative_mask = weights < -threshold
            zero_mask = ~(positive_mask | negative_mask)

            n_positive = positive_mask.sum().item()
            n_negative = negative_mask.sum().item()
            n_zero = zero_mask.sum().item()

            return {
                "threshold": threshold.item() if isinstance(threshold, torch.Tensor) else threshold,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_zero": n_zero,
                "n_total": n_total,
                "sparsity": n_zero / n_total,
                "positive_ratio": n_positive / n_total,
                "negative_ratio": n_negative / n_total,
                "effective_bits": math.log2(3),  # log2(3) = 1.58 bits
            }
    except ImportError:
        pass

    raise TypeError("weights must be torch.Tensor")


# =============================================================================
# LOG-SPACE STE (Straight-Through Estimator)
# =============================================================================

class LogSpaceSTE:
    """
    Straight-Through Estimator in log-space.

    Provides gradients scaled by quantization error in log-space,
    which is more appropriate for log-normal weight distributions.

    Example:
        >>> ste = LogSpaceSTE()
        >>> # In forward pass:
        >>> quantized = ste.forward(weights, threshold)
        >>> # In backward pass (automatic via autograd):
        >>> # Gradients are scaled by log-space error
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def forward(self, weights, threshold: Optional[float] = None):
        """
        Forward pass: quantize weights.

        Args:
            weights: Weight tensor
            threshold: Quantization threshold

        Returns:
            Quantized weights (with gradient connection)
        """
        try:
            import torch

            if threshold is None:
                threshold = bigeometric_threshold(weights, config=self.config)

            return LogSpaceSTEFunction.apply(
                weights,
                threshold,
                self.config.ste_scale,
                self.config.epsilon,
            )
        except ImportError:
            raise ImportError("PyTorch required for LogSpaceSTE")


try:
    import torch

    class LogSpaceSTEFunction(torch.autograd.Function):
        """PyTorch autograd function for log-space STE."""

        @staticmethod
        def forward(ctx, weights, threshold, ste_scale, epsilon):
            # Quantize
            quantized = torch.zeros_like(weights)
            quantized[weights > threshold] = 1.0
            quantized[weights < -threshold] = -1.0

            # Save for backward
            ctx.save_for_backward(weights, quantized)
            ctx.threshold = threshold
            ctx.ste_scale = ste_scale
            ctx.epsilon = epsilon

            return quantized

        @staticmethod
        def backward(ctx, grad_output):
            weights, quantized = ctx.saved_tensors

            # Compute quantization error in log-space
            abs_weights = weights.abs() + ctx.epsilon
            abs_quantized = quantized.abs() + ctx.epsilon

            log_error = torch.abs(torch.log(abs_weights) - torch.log(abs_quantized))

            # Scale gradient by log-space error
            # Small error -> full gradient, large error -> dampened gradient
            error_scale = 1.0 / (1.0 + log_error)
            scaled_grad = grad_output * error_scale * ctx.ste_scale

            return scaled_grad, None, None, None

except ImportError:
    # PyTorch not available
    LogSpaceSTEFunction = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_quantization(
    weights,
    threshold: Optional[float] = None,
    config: Optional[QuantizationConfig] = None,
) -> Dict[str, Any]:
    """
    Analyze quantization quality.

    Args:
        weights: Original weights
        threshold: Quantization threshold
        config: Optional configuration

    Returns:
        Dictionary with quality metrics
    """
    config = config or QuantizationConfig()

    if threshold is None:
        threshold = bigeometric_threshold(weights, config=config)

    quantized = apply_bigeometric_quantization(weights, threshold, config)
    mapping = get_ternary_mapping(weights, threshold, config)

    try:
        import torch
        if isinstance(weights, torch.Tensor):
            # Reconstruction error
            error = weights - quantized
            mse = (error ** 2).mean().item()
            mae = error.abs().mean().item()

            # Correlation
            flat_w = weights.flatten()
            flat_q = quantized.flatten()
            corr = torch.corrcoef(torch.stack([flat_w, flat_q]))[0, 1].item()

            return {
                **mapping,
                "mse": mse,
                "rmse": math.sqrt(mse),
                "mae": mae,
                "correlation": corr,
                "compression_ratio": 32 / mapping["effective_bits"],  # vs float32
            }
    except ImportError:
        pass

    return mapping

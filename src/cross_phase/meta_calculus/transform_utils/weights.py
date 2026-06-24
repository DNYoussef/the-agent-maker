"""
Weight Transformation Utilities

Bigeometric weight merging and log-space fitting.

Phase Applications:
    - Phase 2 (EvoMerge): Bigeometric merge as novel merge technique
    - Phase 8 (Compression): Log-space weight fitting for Bezier curves
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Layer 1 imports only
from ..bigeometric import from_log_space, log_space_interpolation, to_log_space


@dataclass
class WeightMergeConfig:
    """Configuration for weight merging."""

    # Default merge ratio
    default_alpha: float = 0.5

    # Epsilon for numerical stability
    epsilon: float = 1e-10

    # Handle negative weights
    handle_negative: str = "sign_preserve"  # "sign_preserve", "abs", "clamp"


# =============================================================================
# BIGEOMETRIC MERGE (Phase 2 - NOVEL TECHNIQUE)
# =============================================================================


def bigeometric_merge(
    weights1,
    weights2,
    alpha: float = 0.5,
    config: Optional[WeightMergeConfig] = None,
):
    """
    Merge weights using bigeometric interpolation.

    Unlike linear interpolation, bigeometric merge operates in log-space:
    log(w_merged) = (1-alpha) * log(w1) + alpha * log(w2)

    This is equivalent to geometric mean for alpha=0.5, and provides
    more stable merging for weights with varying magnitudes.

    Args:
        weights1: First weight tensor
        weights2: Second weight tensor
        alpha: Interpolation factor (0=weights1, 1=weights2)
        config: Optional configuration

    Returns:
        Merged weights

    Example:
        >>> merged = bigeometric_merge(model1.fc.weight, model2.fc.weight, alpha=0.6)
        >>> # merged = (w1^0.4 * w2^0.6) with sign preservation
    """
    config = config or WeightMergeConfig()

    return log_space_interpolation(weights1, weights2, alpha)


def bigeometric_merge_tensors(
    tensor1,
    tensor2,
    alpha: float = 0.5,
    config: Optional[WeightMergeConfig] = None,
):
    """
    Merge two tensors using bigeometric interpolation.

    Handles sign preservation and numerical stability.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        alpha: Interpolation factor
        config: Optional configuration

    Returns:
        Merged tensor
    """
    import torch

    config = config or WeightMergeConfig()

    # Handle signs
    sign1 = torch.sign(tensor1)
    sign2 = torch.sign(tensor2)

    # Work with absolute values in log-space
    abs1 = tensor1.abs() + config.epsilon
    abs2 = tensor2.abs() + config.epsilon

    log1 = torch.log(abs1)
    log2 = torch.log(abs2)

    # Interpolate in log-space
    log_merged = (1 - alpha) * log1 + alpha * log2
    abs_merged = torch.exp(log_merged)

    # Merge signs (use weighted vote)
    if config.handle_negative == "sign_preserve":
        same_sign = sign1 == sign2
        voted_sign = torch.sign((1 - alpha) * sign1 + alpha * sign2 + config.epsilon)
        sign_merged = torch.where(same_sign, sign1, voted_sign)
    elif config.handle_negative == "abs":
        sign_merged = torch.ones_like(abs_merged)
    else:  # clamp
        sign_merged = sign1  # Keep first model's sign

    return abs_merged * sign_merged


def bigeometric_merge_models(
    model1,
    model2,
    layer_alphas: Optional[Dict[str, float]] = None,
    default_alpha: float = 0.5,
    config: Optional[WeightMergeConfig] = None,
) -> Dict[str, Any]:
    """
    Merge two models using bigeometric interpolation.

    Args:
        model1: First PyTorch model
        model2: Second PyTorch model
        layer_alphas: Optional per-layer alpha values
        default_alpha: Default alpha if not specified per-layer
        config: Optional configuration

    Returns:
        Dictionary with merged state dict and statistics

    Example:
        >>> result = bigeometric_merge_models(model1, model2, default_alpha=0.6)
        >>> merged_model.load_state_dict(result["merged_state_dict"])
    """
    config = config or WeightMergeConfig()
    layer_alphas = layer_alphas or {}

    state1 = model1.state_dict()
    state2 = model2.state_dict()

    merged_state = {}
    stats = {"layers_merged": 0, "layers_skipped": 0}

    for name in state1.keys():
        if name not in state2:
            # Only in model1
            merged_state[name] = state1[name]
            stats["layers_skipped"] += 1
            continue

        tensor1 = state1[name]
        tensor2 = state2[name]

        # Skip non-float tensors (e.g., batch norm running stats integers)
        if not tensor1.is_floating_point():
            merged_state[name] = tensor1
            stats["layers_skipped"] += 1
            continue

        # Get alpha for this layer
        alpha = layer_alphas.get(name, default_alpha)

        # Merge
        merged_state[name] = bigeometric_merge_tensors(tensor1, tensor2, alpha, config)
        stats["layers_merged"] += 1

    return {
        "merged_state_dict": merged_state,
        "stats": stats,
        "layer_alphas_used": {
            name: layer_alphas.get(name, default_alpha) for name in state1.keys()
        },
    }


# =============================================================================
# LOG-SPACE WEIGHT FITTING (Phase 8)
# =============================================================================


def fit_weights_log_space(
    weights,
    n_control_points: int = 8,
    config: Optional[WeightMergeConfig] = None,
) -> Dict[str, Any]:
    """
    Fit Bezier curve to weights in log-space.

    Log-space fitting is more appropriate for neural network weights
    which typically follow log-normal distributions.

    Args:
        weights: Weight tensor to fit
        n_control_points: Number of Bezier control points
        config: Optional configuration

    Returns:
        Dictionary with control points and reconstruction info

    Example:
        >>> result = fit_weights_log_space(layer.weight, n_control_points=8)
        >>> control_points = result["control_points"]
        >>> reconstructed = result["reconstruct_fn"](control_points)
    """
    import torch

    config = config or WeightMergeConfig()

    # Flatten weights
    flat = weights.flatten()
    n_weights = flat.numel()

    # Work in log-space
    signs = torch.sign(flat)
    log_abs = torch.log(flat.abs() + config.epsilon)

    # Sort for fitting
    sorted_log, sort_indices = torch.sort(log_abs)

    # Select control points (evenly spaced in sorted order)
    indices = torch.linspace(0, n_weights - 1, n_control_points).long()
    control_points = sorted_log[indices]

    # Store info for reconstruction
    def reconstruct_fn(ctrl_pts):
        """Reconstruct weights from control points."""
        # Linear interpolation between control points
        x = torch.linspace(0, 1, n_weights, device=weights.device)
        ctrl_x = torch.linspace(0, 1, len(ctrl_pts), device=weights.device)

        # Interpolate
        reconstructed_log = torch.zeros(n_weights, device=weights.device)
        for i in range(len(ctrl_pts) - 1):
            mask = (x >= ctrl_x[i]) & (x < ctrl_x[i + 1])
            t = (x[mask] - ctrl_x[i]) / (ctrl_x[i + 1] - ctrl_x[i] + 1e-10)
            reconstructed_log[mask] = (1 - t) * ctrl_pts[i] + t * ctrl_pts[i + 1]

        # Handle last point
        reconstructed_log[x >= ctrl_x[-1]] = ctrl_pts[-1]

        # Convert back from log-space
        reconstructed_abs = torch.exp(reconstructed_log)

        # Unsort and reshape
        _, unsort_indices = torch.sort(sort_indices)
        reconstructed_flat = reconstructed_abs[unsort_indices] * signs

        return reconstructed_flat.view(weights.shape)

    # Compute compression ratio
    original_size = n_weights * 4  # float32
    compressed_size = n_control_points * 4 + n_weights // 8  # ctrl + sign bits approx
    compression_ratio = original_size / compressed_size

    return {
        "control_points": control_points,
        "n_control_points": n_control_points,
        "original_shape": weights.shape,
        "reconstruct_fn": reconstruct_fn,
        "compression_ratio": compression_ratio,
        "log_space_stats": {
            "mean": log_abs.mean().item(),
            "std": log_abs.std().item(),
            "min": log_abs.min().item(),
            "max": log_abs.max().item(),
        },
    }


def compute_reconstruction_error(
    original,
    reconstructed,
) -> Dict[str, float]:
    """
    Compute reconstruction error metrics.

    Args:
        original: Original weights
        reconstructed: Reconstructed weights

    Returns:
        Dictionary with error metrics
    """
    import torch

    diff = original - reconstructed

    mse = (diff**2).mean().item()
    mae = diff.abs().mean().item()
    max_error = diff.abs().max().item()

    # Relative error
    rel_error = (diff.abs() / (original.abs() + 1e-10)).mean().item()

    # Correlation
    flat_orig = original.flatten()
    flat_recon = reconstructed.flatten()
    correlation = torch.corrcoef(torch.stack([flat_orig, flat_recon]))[0, 1].item()

    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "max_error": max_error,
        "relative_error": rel_error,
        "correlation": correlation,
    }

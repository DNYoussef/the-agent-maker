"""
Phase 4 Utility Functions
Compression metrics, validation, and helper functions
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def calculate_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes

    Args:
        model: PyTorch model

    Returns:
        Size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024**2)
    return size_mb


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
    }


def calculate_sparsity_ratio(model: nn.Module) -> float:
    """
    Calculate fraction of zero weights

    Args:
        model: PyTorch model

    Returns:
        Sparsity ratio (0.0 to 1.0)
    """
    total_elements = 0
    zero_elements = 0

    for param in model.parameters():
        total_elements += param.numel()
        zero_elements += (param.abs() < 1e-8).sum().item()

    return zero_elements / total_elements if total_elements > 0 else 0.0


def test_gradient_flow(model: nn.Module, device: str = "cuda") -> Tuple[bool, Optional[str]]:
    """
    Test if model supports gradient backpropagation

    Args:
        model: PyTorch model
        device: Device for testing

    Returns:
        Tuple of (success, error_message)
    """
    model.train()
    model.to(device)

    try:
        # Create dummy input
        dummy_input = torch.randn(1, 512, dtype=torch.float32).to(device)
        dummy_labels = torch.randint(0, 100, (1,), dtype=torch.long).to(device)

        # Forward pass
        if hasattr(model, "forward"):
            output = model(dummy_input)
        else:
            raise ValueError("Model has no forward method")

        # Calculate loss
        if output.dim() > 1:
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), dummy_labels)
        else:
            loss = output.mean()

        # Backward pass
        loss.backward()

        # Check if gradients were computed
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)

        if not has_gradients:
            return False, "No gradients computed"

        return True, None

    except Exception as e:
        return False, str(e)

    finally:
        model.zero_grad()


def calculate_compression_ratio(original_size_mb: float, compressed_size_mb: float) -> float:
    """
    Calculate compression ratio

    Args:
        original_size_mb: Original model size
        compressed_size_mb: Compressed model size

    Returns:
        Compression ratio (e.g., 8.2 for 8.2x compression)
    """
    if compressed_size_mb == 0:
        return 0.0
    return original_size_mb / compressed_size_mb


def save_compression_metadata(output_dir: Path, metadata: Dict):
    """
    Save compression metadata to JSON

    Args:
        output_dir: Output directory
        metadata: Metadata dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "compression_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_compression_metadata(output_dir: Path) -> Dict:
    """
    Load compression metadata from JSON

    Args:
        output_dir: Output directory

    Returns:
        Metadata dictionary
    """
    output_dir = Path(output_dir)
    metadata_path = output_dir / "compression_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata


def should_preserve_layer(layer_name: str, preserve_patterns: list) -> bool:
    """
    Check if layer should be preserved in higher precision

    Args:
        layer_name: Name of the layer
        preserve_patterns: List of patterns to preserve

    Returns:
        True if layer should be preserved
    """
    layer_name_lower = layer_name.lower()

    for pattern in preserve_patterns:
        if pattern.lower() in layer_name_lower:
            return True

    return False


def validate_compression_quality(
    pre_perplexity: float, post_perplexity: float, max_accuracy_drop: float
) -> Tuple[bool, float]:
    """
    Validate compression quality

    Args:
        pre_perplexity: Pre-compression perplexity
        post_perplexity: Post-compression perplexity
        max_accuracy_drop: Maximum acceptable drop

    Returns:
        Tuple of (is_valid, degradation_ratio)
    """
    if pre_perplexity == 0:
        return False, float("inf")

    degradation = (post_perplexity - pre_perplexity) / pre_perplexity

    is_valid = degradation <= max_accuracy_drop

    return is_valid, degradation


def estimate_inference_speedup(original_size_mb: float, compressed_size_mb: float) -> float:
    """
    Estimate inference speedup from compression

    Args:
        original_size_mb: Original model size
        compressed_size_mb: Compressed model size

    Returns:
        Estimated speedup (e.g., 2.5 for 2.5x faster)
    """
    compression_ratio = calculate_compression_ratio(original_size_mb, compressed_size_mb)

    # Empirical formula from BitNet paper
    # Speedup is sublinear with compression
    speedup = compression_ratio**0.7

    # Clamp to reasonable range (1.5x - 4.0x)
    speedup = max(1.5, min(4.0, speedup))

    return speedup

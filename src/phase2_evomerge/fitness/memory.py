"""
Memory usage measurement for fitness evaluation.

This module provides functions for measuring peak VRAM usage during model inference,
which is one component of the composite fitness score (10% weight).
"""

import torch
import torch.nn as nn
from typing import Optional


def measure_memory_usage(
    model: nn.Module,
    benchmark_batch: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Measure peak VRAM usage during inference (MB).

    This function clears the CUDA cache, resets peak memory statistics,
    runs a forward pass, and measures the peak memory allocated.

    Args:
        model: Model to measure memory usage for
        benchmark_batch: Representative input batch (batch_size, seq_len)
        device: Device to use (must be 'cuda' for memory measurement)

    Returns:
        Peak memory usage in MB (float, lower is better)

    Raises:
        RuntimeError: If device is not 'cuda'
        RuntimeError: If CUDA is not available

    Example:
        >>> model = SimpleModel().cuda()
        >>> batch = torch.randint(0, 1000, (32, 512)).cuda()
        >>> memory_mb = measure_memory_usage(model, batch)
        >>> print(f"Peak memory: {memory_mb:.2f} MB")
    """
    if device != 'cuda':
        raise RuntimeError(
            f"Memory measurement requires CUDA, got device='{device}'"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")

    # Clear CUDA cache to get accurate baseline
    torch.cuda.empty_cache()

    # Reset peak memory statistics
    torch.cuda.reset_peak_memory_stats(device=device)

    # Move model to device if needed
    model = model.to(device)
    model.eval()

    # Run forward pass (no gradients needed)
    with torch.no_grad():
        _ = model(benchmark_batch)

    # Get peak memory allocated (in bytes)
    peak_memory_bytes = torch.cuda.max_memory_allocated(device=device)

    # Convert bytes to MB
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)

    return peak_memory_mb

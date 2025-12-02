"""
Inference speed benchmark for fitness evaluation.

This module provides functions for measuring model inference speed (tokens/second),
which is one component of the composite fitness score (20% weight).
"""

import time
import torch
import torch.nn as nn
from typing import Optional


def benchmark_speed(
    model: nn.Module,
    benchmark_batch: torch.Tensor,
    device: str = 'cuda',
    num_warmup: int = 10,
    num_iterations: int = 100
) -> float:
    """
    Benchmark inference speed (tokens/second).

    This function runs a warmup period to avoid cold start effects,
    then measures average inference time over multiple iterations
    with proper CUDA synchronization for accurate timing.

    Args:
        model: Model to benchmark
        benchmark_batch: Representative batch (batch_size, seq_len)
        device: Device to use ('cuda' or 'cpu')
        num_warmup: Number of warmup iterations (default: 10)
        num_iterations: Number of measurement iterations (default: 100)

    Returns:
        Tokens per second (float, higher is better)

    Example:
        >>> model = SimpleModel().cuda()
        >>> batch = torch.randint(0, 1000, (32, 512)).cuda()
        >>> tokens_per_sec = benchmark_speed(model, batch)
        >>> print(f"Speed: {tokens_per_sec:.0f} tokens/sec")
    """
    # Move model to device
    model = model.to(device)
    model.eval()

    # Move batch to device if needed
    if benchmark_batch.device.type != device:
        benchmark_batch = benchmark_batch.to(device)

    # Get batch dimensions
    batch_size = benchmark_batch.size(0)
    seq_len = benchmark_batch.size(1)
    total_tokens = batch_size * seq_len

    # Warmup: avoid cold start effects
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(benchmark_batch)

    # Synchronize before timing (if CUDA)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Start timing
    start_time = time.time()

    # Run inference iterations
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(benchmark_batch)

    # Synchronize after timing (if CUDA)
    if device == 'cuda':
        torch.cuda.synchronize()

    # End timing
    end_time = time.time()

    # Calculate tokens per second
    total_time = end_time - start_time
    total_tokens_processed = total_tokens * num_iterations
    tokens_per_second = total_tokens_processed / total_time

    return tokens_per_second

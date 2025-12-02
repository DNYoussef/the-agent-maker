"""
Accuracy measurement for fitness evaluation.

This module provides functions for calculating model accuracy on test datasets,
which is one component of the composite fitness score (30% weight).

Updated to support real task evaluation using benchmarks (GSM8K, MGSM)
in addition to standard next-token prediction.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from phase2_evomerge.fitness.benchmarks import BenchmarkConfig, evaluate_benchmark

    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False


def calculate_accuracy(
    model: nn.Module,
    test_dataset: Optional[DataLoader] = None,
    task_type: str = "next_token",
    device: str = "cuda",
    max_batches: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    benchmark_name: Optional[str] = None,
    benchmark_config: Optional["BenchmarkConfig"] = None,
) -> float:
    """
    Calculate accuracy on test dataset or benchmark.

    Supports different task types:
    - 'next_token': Next token prediction (language modeling)
    - 'classification': Multi-class classification
    - 'qa': Question answering
    - 'benchmark': Real task evaluation (GSM8K, MGSM)

    Args:
        model: Model to evaluate
        test_dataset: DataLoader with test data (batches of input_ids, labels).
            Not required if using benchmark evaluation.
        task_type: Type of task ('next_token', 'classification', 'qa', 'benchmark')
        device: Device to use ('cuda' or 'cpu')
        max_batches: Limit evaluation to N batches (None = all batches)
        tokenizer: Tokenizer for benchmark evaluation (required if task_type='benchmark')
        benchmark_name: Name of benchmark to use (gsm8k, mgsm)
        benchmark_config: Configuration for benchmark evaluation

    Returns:
        Accuracy (float, 0.0-1.0, higher is better)

    Example (next-token):
        >>> model = SimpleModel().cuda()
        >>> test_loader = DataLoader(test_dataset, batch_size=32)
        >>> acc = calculate_accuracy(model, test_loader)
        >>> print(f"Accuracy: {acc:.2%}")

    Example (benchmark):
        >>> from transformers import AutoTokenizer
        >>> model = MyModel().cuda()
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> acc = calculate_accuracy(
        ...     model,
        ...     task_type="benchmark",
        ...     tokenizer=tokenizer,
        ...     benchmark_name="gsm8k"
        ... )
        >>> print(f"GSM8K Accuracy: {acc:.2%}")
    """
    # Benchmark evaluation mode
    if task_type == "benchmark":
        if not BENCHMARKS_AVAILABLE:
            raise ImportError(
                "Benchmark evaluation requires benchmarks.py. "
                "Install required dependencies: pip install datasets"
            )
        if tokenizer is None:
            raise ValueError("tokenizer is required for benchmark evaluation")
        if benchmark_name is None:
            benchmark_name = "gsm8k"

        return evaluate_benchmark(model, tokenizer, benchmark_name, benchmark_config)

    # Standard evaluation modes (next_token, classification, qa)
    if test_dataset is None:
        raise ValueError("test_dataset is required for non-benchmark evaluation")
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    num_batches_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataset):
            # Early stopping if max_batches specified
            if max_batches and batch_idx >= max_batches:
                break

            # Unpack batch
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
            elif isinstance(batch, (list, tuple)):
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            # Forward pass
            logits = model(input_ids)

            # Calculate predictions
            predictions = torch.argmax(logits, dim=-1)

            # Count correct predictions
            correct += (predictions == labels).sum().item()
            total += labels.numel()
            num_batches_processed += 1

    # Calculate accuracy
    if total == 0:
        return 0.0

    accuracy = correct / total
    return accuracy

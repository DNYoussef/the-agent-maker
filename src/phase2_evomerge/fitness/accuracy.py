"""
Accuracy measurement for fitness evaluation.

This module provides functions for calculating model accuracy on test datasets,
which is one component of the composite fitness score (30% weight).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_accuracy(
    model: nn.Module,
    test_dataset: DataLoader,
    task_type: str = "next_token",
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> float:
    """
    Calculate accuracy on test dataset.

    Supports different task types:
    - 'next_token': Next token prediction (language modeling)
    - 'classification': Multi-class classification
    - 'qa': Question answering

    Args:
        model: Model to evaluate
        test_dataset: DataLoader with test data (batches of input_ids, labels)
        task_type: Type of task ('next_token', 'classification', 'qa')
        device: Device to use ('cuda' or 'cpu')
        max_batches: Limit evaluation to N batches (None = all batches)

    Returns:
        Accuracy (float, 0.0-1.0, higher is better)

    Example:
        >>> model = SimpleModel().cuda()
        >>> test_loader = DataLoader(test_dataset, batch_size=32)
        >>> acc = calculate_accuracy(model, test_loader)
        >>> print(f"Accuracy: {acc:.2%}")
    """
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

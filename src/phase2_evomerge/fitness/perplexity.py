"""
Perplexity calculation for fitness evaluation.

This module provides functions for calculating model perplexity on validation datasets,
which is one component of the composite fitness score (40% weight - highest priority).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import math


def calculate_perplexity(
    model: nn.Module,
    validation_dataset: DataLoader,
    device: str = 'cuda',
    mixed_precision: bool = True,
    max_batches: Optional[int] = None
) -> float:
    """
    Calculate perplexity on validation dataset.

    Perplexity = exp(average cross-entropy loss)
    Lower perplexity indicates better language modeling performance.

    Args:
        model: Model to evaluate
        validation_dataset: DataLoader with validation data
        device: Device to use ('cuda' or 'cpu')
        mixed_precision: Use torch.amp for faster evaluation
        max_batches: Limit evaluation to N batches (None = all batches)

    Returns:
        Perplexity value (float, lower is better)

    Raises:
        ValueError: If perplexity is NaN or Inf

    Example:
        >>> model = SimpleModel().cuda()
        >>> val_loader = DataLoader(val_dataset, batch_size=32)
        >>> ppl = calculate_perplexity(model, val_loader)
        >>> print(f"Perplexity: {ppl:.2f}")
    """
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    num_batches = 0

    # Use automatic mixed precision if enabled
    autocast_ctx = (
        torch.amp.autocast(device_type=device)
        if mixed_precision and device == 'cuda'
        else torch.no_grad()
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_dataset):
            # Early stopping if max_batches specified
            if max_batches and batch_idx >= max_batches:
                break

            # Unpack batch
            input_ids, labels = _unpack_batch(batch, device)

            # Forward pass with optional mixed precision
            with autocast_ctx:
                logits = model(input_ids)

                # Calculate cross-entropy loss
                loss = _compute_cross_entropy_loss(
                    logits, labels
                )

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

    # Calculate average loss
    if num_batches == 0:
        raise ValueError("No batches processed")

    avg_loss = total_loss / num_batches

    # Calculate perplexity: exp(average loss)
    perplexity = math.exp(avg_loss)

    # Validate result
    if math.isnan(perplexity) or math.isinf(perplexity):
        raise ValueError(
            f"Invalid perplexity: {perplexity} "
            f"(avg_loss={avg_loss}, num_batches={num_batches})"
        )

    return perplexity


def _unpack_batch(batch, device: str):
    """
    Unpack batch into (input_ids, labels).

    Handles different batch formats:
    - Dict: {'input_ids': ..., 'labels': ...}
    - Tuple/List: (input_ids, labels)

    Args:
        batch: Batch from DataLoader
        device: Target device

    Returns:
        Tuple of (input_ids, labels) on device
    """
    if isinstance(batch, dict):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
    elif isinstance(batch, (list, tuple)):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
    else:
        raise ValueError(f"Unsupported batch format: {type(batch)}")

    return input_ids, labels


def _compute_cross_entropy_loss(logits, labels):
    """
    Compute cross-entropy loss between logits and labels.

    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)

    Returns:
        Cross-entropy loss (scalar tensor)
    """
    # Flatten logits and labels for cross-entropy
    # logits: (batch_size * seq_len, vocab_size)
    # labels: (batch_size * seq_len,)
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute loss
    loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')

    return loss

"""
Diversity tracking for evolutionary optimization.

This module provides functions for measuring population diversity
via pairwise L2 distance between model parameters.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np


def compute_diversity(
    population: List[nn.Module],
    expected_distance: Optional[float] = None
) -> float:
    """
    Compute population diversity via average pairwise L2 distance.

    Calculates L2 distance between all pairs of models and returns
    normalized average distance as diversity score.

    Args:
        population: List of models
        expected_distance: Normalization factor (None = use avg distance)

    Returns:
        Diversity score (0.0 = identical, 1.0 = maximally diverse)

    Example:
        >>> diversity = compute_diversity(population)
        >>> if diversity < 0.2:
        ...     print("Warning: Low diversity!")
        >>> elif diversity > 0.3:
        ...     print("Healthy diversity")
    """
    if len(population) < 2:
        return 0.0

    # Compute pairwise distances
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Flatten parameters
            params_i = _flatten_model_params(population[i])
            params_j = _flatten_model_params(population[j])

            # L2 distance
            dist = torch.norm(params_i - params_j).item()
            distances.append(dist)

    # Average distance
    avg_distance = np.mean(distances)

    # Normalize
    if expected_distance is None:
        # Use average distance as baseline (first generation)
        expected_distance = avg_distance if avg_distance > 0 else 1.0

    if expected_distance == 0:
        return 0.0

    normalized_diversity = avg_distance / expected_distance

    return normalized_diversity


def _flatten_model_params(model: nn.Module) -> torch.Tensor:
    """
    Flatten all model parameters into single vector.

    Args:
        model: PyTorch model

    Returns:
        1D tensor of all parameters concatenated
    """
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().flatten())

    if not params:
        return torch.tensor([])

    return torch.cat(params)

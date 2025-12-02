"""
Population initialization for evolutionary optimization.

This module provides functions for creating the initial population
of 8 models using all binary merge combinations.
"""

import torch.nn as nn
from typing import List

from src.phase2_evomerge.merge import MergeTechniques


def initialize_population(
    base_models: List[nn.Module]
) -> List[nn.Module]:
    """
    Create initial population of 8 models using all binary combinations.

    Takes 3 Phase 1 models and creates 8 merged models by applying
    all 8 binary merge combinations (000 to 111).

    Args:
        base_models: List of exactly 3 Phase 1 models

    Returns:
        List of 8 merged models (one per binary combo)

    Raises:
        ValueError: If not exactly 3 base models provided

    Example:
        >>> model1, model2, model3 = load_phase1_models()
        >>> population = initialize_population([model1, model2, model3])
        >>> len(population)
        8
    """
    if len(base_models) != 3:
        raise ValueError(
            f"Expected exactly 3 base models, got {len(base_models)}"
        )

    # Initialize merge techniques
    merger = MergeTechniques()

    # Create all 8 binary combinations
    population = []
    for combo_id in range(8):  # 000 to 111
        merged_model = merger.apply_combo(base_models, combo_id)
        population.append(merged_model)

    return population

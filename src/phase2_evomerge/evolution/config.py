"""
Evolution configuration for Phase 2 (EvoMerge).

This module provides configuration dataclass for evolutionary optimization parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvolutionConfig:
    """
    Configuration for evolutionary optimization.

    Attributes:
        generations: Maximum number of generations to run
        population_size: Number of models per generation (fixed at 8)
        elite_count: Number of top models to preserve (fixed at 2)
        mutation_sigma: Standard deviation of Gaussian noise
        mutation_rate: Fraction of weights to mutate (0.01 = 1%)
        min_diversity: Healthy diversity threshold
        diversity_reseed_threshold: Re-seed below this threshold
        early_stopping: Enable early stopping on convergence
        convergence_threshold: Fitness improvement threshold (0.001 = 0.1%)
        convergence_patience: Generations to wait for improvement
        device: Device for computations ('cuda' or 'cpu')

    Example:
        >>> config = EvolutionConfig(generations=50)
        >>> config.validate()  # Check config is valid
    """

    # Evolution parameters
    generations: int = 50
    population_size: int = 8
    elite_count: int = 2

    # Mutation
    mutation_sigma: float = 0.01  # 1% noise std
    mutation_rate: float = 0.01  # 1% of weights

    # Diversity
    min_diversity: float = 0.3
    diversity_reseed_threshold: float = 0.2

    # Convergence
    early_stopping: bool = True
    convergence_threshold: float = 0.001  # 0.1% improvement
    convergence_patience: int = 5  # Generations

    # Device
    device: str = "cuda"

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            AssertionError: If any validation check fails
        """
        assert self.generations > 0, "Generations must be positive"
        assert self.population_size == 8, "Population size must be 8 (binary combos)"
        assert self.elite_count == 2, "Elite count must be 2 (for 6 children)"
        assert 0 < self.mutation_rate < 1, "Mutation rate must be in (0, 1)"
        assert self.mutation_sigma > 0, "Mutation sigma must be positive"
        assert 0 <= self.min_diversity <= 1, "Diversity must be in [0, 1]"
        assert self.convergence_patience > 0, "Convergence patience must be positive"

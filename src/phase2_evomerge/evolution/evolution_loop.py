"""
Main evolutionary optimization loop for Phase 2 (EvoMerge).

This module implements the complete evolution loop with:
- Elite preservation (top 2 → 6 children via mutation)
- Loser merging (bottom 6 → 2 children via combo merging)
- Diversity tracking and re-seeding
- Early stopping on convergence
"""

import random
import torch.nn as nn
from typing import List, Dict, Any
import numpy as np

from .config import EvolutionConfig
from .population import initialize_population
from .mutation import mutate_model
from .diversity import compute_diversity
from src.phase2_evomerge.fitness import FitnessEvaluator
from src.phase2_evomerge.merge import MergeTechniques


class EvolutionLoop:
    """
    Main evolutionary optimization loop.

    Implements 50-generation evolution with elite-loser strategy:
    - Elite: Top 2 models → mutate 3× each → 6 children
    - Loser: Bottom 6 models → merge in 2 groups → 2 children

    Example:
        >>> config = EvolutionConfig(generations=50)
        >>> evolution = EvolutionLoop(config, fitness_evaluator)
        >>> result = evolution.evolve([model1, model2, model3])
        >>> print(f"Best fitness: {result['fitness']:.4f}")
    """

    def __init__(
        self,
        config: EvolutionConfig,
        fitness_evaluator: FitnessEvaluator
    ):
        """
        Initialize evolution loop.

        Args:
            config: Evolution configuration
            fitness_evaluator: Fitness evaluator from Day 6
        """
        config.validate()
        self.config = config
        self.evaluator = fitness_evaluator
        self.merger = MergeTechniques()

    def evolve(
        self,
        base_models: List[nn.Module]
    ) -> Dict[str, Any]:
        """
        Run complete evolution loop.

        Args:
            base_models: List of 3 Phase 1 models

        Returns:
            Dictionary with results:
            {
                'champion': best_model,
                'fitness': 0.185,
                'initial_fitness': 0.150,
                'improvement': 0.035,
                'improvement_pct': 0.235,  # 23.5%
                'generations': 38,
                'convergence_reason': 'threshold_met',
                'final_diversity': 0.35
            }
        """
        # Initialize population (8 models from all binary combos)
        population = initialize_population(base_models)

        # Initial fitness evaluation
        initial_fitness_scores = self.evaluator.evaluate_batch(population)
        initial_best_fitness = max(initial_fitness_scores)

        # Track champion
        champion = None
        champion_fitness = -float('inf')
        fitness_history = []

        # Evolution loop
        for generation in range(1, self.config.generations + 1):
            # Evaluate fitness
            fitness_scores = self.evaluator.evaluate_batch(population)

            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]

            # Update champion
            if fitness_scores[0] > champion_fitness:
                champion = population[0]
                champion_fitness = fitness_scores[0]

            fitness_history.append(champion_fitness)

            # Check convergence
            if (
                self.config.early_stopping
                and generation > self.config.convergence_patience
            ):
                if self._check_convergence(fitness_history):
                    convergence_reason = 'threshold_met'
                    break
            else:
                convergence_reason = 'max_generations'

            # Elite preservation (top 2 → 6 children)
            elite_children = self._elite_preservation(population)

            # Loser merging (bottom 6 → 2 children)
            loser_children = self._loser_merging(population, base_models)

            # New population
            population = elite_children + loser_children

            # Diversity management
            diversity = compute_diversity(population)
            population = self._reseed_if_needed(
                population, diversity, base_models
            )

        # Final metrics
        final_diversity = compute_diversity(population)
        improvement = champion_fitness - initial_best_fitness
        improvement_pct = improvement / initial_best_fitness if initial_best_fitness > 0 else 0

        return {
            'champion': champion,
            'fitness': champion_fitness,
            'initial_fitness': initial_best_fitness,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'generations': generation,
            'convergence_reason': convergence_reason,
            'final_diversity': final_diversity,
            'fitness_history': fitness_history
        }

    def _elite_preservation(
        self,
        population: List[nn.Module]
    ) -> List[nn.Module]:
        """
        Create 6 children from top 2 elites via mutation.

        Args:
            population: Sorted population (best first)

        Returns:
            List of 6 mutated children
        """
        elite1, elite2 = population[0], population[1]
        elite_children = []

        for elite in [elite1, elite2]:
            for _ in range(3):  # 3 mutations per elite
                child = mutate_model(
                    elite,
                    sigma=self.config.mutation_sigma,
                    rate=self.config.mutation_rate,
                    device=self.config.device
                )
                elite_children.append(child)

        return elite_children  # 6 children total

    def _loser_merging(
        self,
        population: List[nn.Module],
        base_models: List[nn.Module]
    ) -> List[nn.Module]:
        """
        Create 2 children from bottom 6 via combo merging.

        Args:
            population: Sorted population (best first)
            base_models: Original 3 Phase 1 models

        Returns:
            List of 2 merged children
        """
        # Bottom 6 models
        losers = population[-6:]

        # Split into 2 groups of 3
        group1, group2 = losers[0:3], losers[3:6]

        # Random merge combos
        combo1 = random.randint(0, 7)
        combo2 = random.randint(0, 7)

        # Merge each group
        child1 = self.merger.apply_combo(group1, combo1)
        child2 = self.merger.apply_combo(group2, combo2)

        return [child1, child2]

    def _check_convergence(
        self,
        fitness_history: List[float]
    ) -> bool:
        """
        Check if evolution has converged.

        Convergence: Improvement < threshold for patience generations.

        Args:
            fitness_history: List of best fitness per generation

        Returns:
            True if converged, False otherwise
        """
        if len(fitness_history) < self.config.convergence_patience:
            return False

        # Recent fitness values
        recent = fitness_history[-self.config.convergence_patience:]

        # Check improvement
        improvement = max(recent) - min(recent)

        return improvement < self.config.convergence_threshold

    def _reseed_if_needed(
        self,
        population: List[nn.Module],
        diversity: float,
        base_models: List[nn.Module]
    ) -> List[nn.Module]:
        """
        Re-seed bottom 2 models if diversity too low.

        Args:
            population: Current population
            diversity: Current diversity score
            base_models: Original 3 Phase 1 models

        Returns:
            Population with re-seeded models if needed
        """
        if diversity < self.config.diversity_reseed_threshold:
            # Replace bottom 2 with new random combos
            combo1 = random.randint(0, 7)
            combo2 = random.randint(0, 7)

            population[-2] = self.merger.apply_combo(base_models, combo1)
            population[-1] = self.merger.apply_combo(base_models, combo2)

        return population

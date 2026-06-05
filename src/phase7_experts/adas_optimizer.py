"""Compatibility exports for the modular ADAS implementation."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch.nn as nn

from .adas.config import ADASConfig, ADASResult, Individual
from .adas.evaluation import evaluate_individual, evaluate_population
from .adas.nsga2 import (
    assign_ranks,
    calculate_crowding_distance,
    survivor_selection,
    tournament_selection,
)
from .adas.operators import create_offspring, crossover, mutate
from .adas.optimizer import ADASOptimizer as _ModularADASOptimizer


class ADASOptimizer(_ModularADASOptimizer):
    """Backward-compatible facade over ``phase7_experts.adas.optimizer``."""

    def _evaluate_population(
        self,
        model: nn.Module,
        experts: List[Any],
        tokenizer: Any,
        evaluator: Optional[Callable[..., Any]] = None,
    ) -> None:
        evaluate_population(self.population, model, experts, tokenizer, evaluator)

    def _evaluate_individual(
        self,
        individual: Individual,
        model: nn.Module,
        experts: List[Any],
        tokenizer: Any,
        evaluator: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, float]:
        return evaluate_individual(individual, model, experts, tokenizer, evaluator)

    def _assign_ranks(self) -> None:
        assign_ranks(self.population, self.config)

    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        from .adas.nsga2 import _dominates

        return _dominates(ind1, ind2, self.config)

    def _calculate_crowding_distance(self) -> None:
        calculate_crowding_distance(self.population, self.config)

    def _tournament_selection(self) -> List[Individual]:
        return tournament_selection(self.population, self.config)

    def _create_offspring(self, parents: List[Individual], num_experts: int) -> List[Individual]:
        return create_offspring(parents, num_experts, self.config)

    def _crossover(
        self, parent1: Individual, parent2: Individual, num_experts: int
    ) -> Tuple[Individual, Individual]:
        return crossover(parent1, parent2, num_experts)

    def _mutate(self, individual: Individual, num_experts: int) -> None:
        mutate(individual, num_experts)

    def _survivor_selection(self, offspring: List[Individual]) -> None:
        self.population = survivor_selection(self.population, offspring, self.config)


__all__ = ["ADASOptimizer", "ADASConfig", "ADASResult", "Individual"]

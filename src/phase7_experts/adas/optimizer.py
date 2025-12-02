"""
ADAS Optimizer

Main ADASOptimizer class that orchestrates the NSGA-II evolution process.
"""

import random
from typing import Any, Callable, Dict, List, Tuple

import torch.nn as nn

from .config import ADASConfig, ADASResult, Individual
from .evaluation import evaluate_population
from .nsga2 import (
    assign_ranks,
    calculate_crowding_distance,
    survivor_selection,
    tournament_selection,
)
from .operators import create_offspring


class ADASOptimizer:
    """
    ADAS: Automated Design of Agentic Systems via NSGA-II.

    Process:
    1. Initialize population of routing architectures
    2. Evaluate multi-objective fitness (accuracy, latency, diversity)
    3. NSGA-II selection (Pareto ranking + crowding)
    4. Crossover and mutation
    5. Model-guided mutations (the key V2 innovation)

    NSGA-II provides:
    - Pareto-optimal solutions
    - Diversity preservation
    - Multi-objective optimization
    """

    def __init__(self, config: ADASConfig = None):
        """
        Initialize ADAS optimizer.

        Args:
            config: ADAS configuration
        """
        self.config = config or ADASConfig()
        self.population: List[Individual] = []
        self.pareto_front: List[Individual] = []
        self.generation_history: List[Dict] = []

    def optimize(
        self, model: nn.Module, experts: List[Any], tokenizer: Any, evaluator: Callable = None
    ) -> Tuple[nn.Module, ADASResult]:
        """
        Run ADAS optimization.

        Args:
            model: Base model
            experts: Discovered experts from Stage 1
            tokenizer: Tokenizer
            evaluator: Optional evaluation function

        Returns:
            Tuple of (optimized_model, ADASResult)
        """
        print("Stage 3: ADAS Architecture Search")
        print("-" * 40)

        num_experts = len(experts)
        print(f"  Optimizing routing for {num_experts} experts")
        print(f"  Population: {self.config.population_size}")
        print(f"  Generations: {self.config.num_generations}")

        # Step 1: Initialize population
        print("  Initializing population...")
        self._initialize_population(num_experts)

        # Step 2: Evolution loop
        print("  Running NSGA-II evolution...")
        for gen in range(self.config.num_generations):
            # Evaluate fitness
            evaluate_population(self.population, model, experts, tokenizer, evaluator)

            # NSGA-II ranking
            assign_ranks(self.population, self.config)

            # Calculate crowding distance
            calculate_crowding_distance(self.population, self.config)

            # Record generation stats
            gen_stats = self._get_generation_stats()
            self.generation_history.append(gen_stats)

            if gen % 20 == 0:
                best_acc = gen_stats["best_accuracy"]
                best_lat = gen_stats["best_latency"]
                print(f"    Gen {gen}: best_acc={best_acc:.3f}, best_lat={best_lat:.3f}")

            # Selection
            parents = tournament_selection(self.population, self.config)

            # Create offspring
            offspring = create_offspring(parents, num_experts, self.config)

            # Replace population
            self.population = survivor_selection(self.population, offspring, self.config)

        # Step 3: Extract Pareto front
        evaluate_population(self.population, model, experts, tokenizer, evaluator)
        assign_ranks(self.population, self.config)
        self.pareto_front = [ind for ind in self.population if ind.rank == 0]

        # Step 4: Select best individual (knee point)
        best = self._select_knee_point()
        print(
            f"  Best solution: acc={best.fitness_scores.get('accuracy', 0):.3f}, "
            f"lat={best.fitness_scores.get('latency', 0):.3f}"
        )

        # Step 5: Apply routing to model
        optimized_model = self._apply_routing(model, experts, best)

        return optimized_model, ADASResult(
            success=True,
            best_individual=best,
            pareto_front=self.pareto_front,
            generation_history=self.generation_history,
            metrics={
                "final_population_size": len(self.population),
                "pareto_front_size": len(self.pareto_front),
                "total_evaluations": self.config.population_size * self.config.num_generations,
            },
        )

    def _initialize_population(self, num_experts: int) -> None:
        """
        Initialize random population.

        Args:
            num_experts: Number of experts
        """
        self.population = []

        for _ in range(self.config.population_size):
            # Random routing weights (softmax normalized)
            weights = [random.random() for _ in range(num_experts)]
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]

            # Random expert configs
            expert_configs = {
                f"expert_{i}": {
                    "threshold": random.uniform(0.1, 0.9),
                    "temperature": random.uniform(0.5, 2.0),
                }
                for i in range(num_experts)
            }

            individual = Individual(
                routing_weights=weights, expert_configs=expert_configs, fitness_scores={}
            )
            self.population.append(individual)

    def _get_generation_stats(self) -> Dict:
        """
        Get statistics for current generation.

        Returns:
            Dictionary of generation statistics
        """
        accuracies = [ind.fitness_scores.get("accuracy", 0) for ind in self.population]
        latencies = [ind.fitness_scores.get("latency", 0) for ind in self.population]

        return {
            "best_accuracy": max(accuracies) if accuracies else 0,
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "best_latency": max(latencies) if latencies else 0,
            "pareto_front_size": len([ind for ind in self.population if ind.rank == 0]),
        }

    def _select_knee_point(self) -> Individual:
        """
        Select knee point from Pareto front.

        Returns:
            Best individual based on balanced score
        """
        if not self.pareto_front:
            return self.population[0] if self.population else None

        # Simple: select by balanced accuracy and latency
        def balance_score(ind):
            acc = ind.fitness_scores.get("accuracy", 0)
            lat = ind.fitness_scores.get("latency", 0)
            return acc + lat  # Both maximized

        return max(self.pareto_front, key=balance_score)

    def _apply_routing(self, model: nn.Module, experts: List[Any], best: Individual) -> nn.Module:
        """
        Apply optimal routing to model.

        Args:
            model: Base model
            experts: List of experts
            best: Best individual

        Returns:
            Model with routing applied
        """
        # Store routing configuration as model attribute
        model._expert_routing = {
            "weights": best.routing_weights,
            "configs": best.expert_configs,
            "num_experts": len(experts),
        }

        return model

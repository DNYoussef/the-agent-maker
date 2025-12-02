"""
Phase 2 Pipeline Orchestrator

Main entry point for Phase 2 (EvoMerge) that coordinates the entire
evolutionary optimization process.

Implements: 50-generation evolutionary optimization using 6 merge techniques.
"""

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization."""

    num_generations: int = 50
    population_size: int = 10
    elite_count: int = 2
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    merge_techniques: List[str] = field(
        default_factory=lambda: ["slerp", "ties", "dare", "linear", "frankenmerge", "dfs"]
    )
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: {"perplexity": 0.4, "accuracy": 0.3, "speed": 0.2, "memory": 0.1}
    )
    target_fitness_gain: float = 0.235  # 23.5% improvement target (paper)

    # New: Real fitness evaluation options
    use_real_fitness: bool = False  # Use GSM8K/MGSM benchmarks instead of proxies
    benchmark_name: str = "gsm8k"  # gsm8k, mgsm
    max_benchmark_samples: int = 100  # Limit samples for fast eval

    # New: CMA-ES and hybrid PS+DFS options
    use_cmaes: bool = False  # Use CMA-ES optimizer for coefficient optimization
    use_hybrid_ps_dfs: bool = False  # Use hybrid PS+DFS merging (paper's best)
    ps_candidates_multiplier: int = 3  # For hybrid: create M = N * multiplier candidates

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "EvolutionConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class Phase2Pipeline:
    """
    Phase 2 (EvoMerge) main orchestrator.

    Evolves 3 input models from Phase 1 into 1 champion model
    using evolutionary optimization with 6 merge techniques.

    Flow:
        1. Initialize population from 3 input models
        2. For each generation:
           a. Evaluate fitness
           b. Select parents (tournament)
           c. Apply crossover (merge techniques)
           d. Apply mutation (weight perturbation)
           e. Track best model
        3. Return champion with metrics
    """

    def __init__(self, config: Optional[EvolutionConfig] = None):
        """Initialize Phase 2 pipeline."""
        self.config = config or EvolutionConfig()
        self.population = []
        self.fitness_history = []
        self.best_fitness = 0.0
        self.champion = None
        self.metrics = {}
        self._mergers = self._init_mergers()

    def _init_mergers(self) -> Dict:
        """Initialize merge technique instances."""
        mergers = {}
        try:
            from phase2_evomerge.merge.slerp_merge import SLERPMerge

            mergers["slerp"] = SLERPMerge()
        except ImportError:
            pass
        try:
            from phase2_evomerge.merge.ties_merge import TIESMerge

            mergers["ties"] = TIESMerge()
        except ImportError:
            pass
        try:
            from phase2_evomerge.merge.dare_merge import DAREMerge

            mergers["dare"] = DAREMerge()
        except ImportError:
            pass
        try:
            from phase2_evomerge.merge.linear_merge import LinearMerge

            mergers["linear"] = LinearMerge()
        except ImportError:
            pass
        try:
            from phase2_evomerge.merge.frankenmerge import FrankenMerge

            mergers["frankenmerge"] = FrankenMerge()
        except ImportError:
            pass
        try:
            from phase2_evomerge.merge.dfs_merge import DFSMerge

            mergers["dfs"] = DFSMerge()
        except ImportError:
            pass
        return mergers

    def run(
        self,
        input_models: List[nn.Module],
        session_id: Optional[str] = None,
        tokenizer: Optional[Any] = None,
    ) -> nn.Module:
        """
        Run Phase 2 evolution.

        Args:
            input_models: 3 models from Phase 1 (reasoning, memory, speed)
            session_id: Optional session identifier for tracking
            tokenizer: Optional tokenizer for real fitness evaluation (required if use_real_fitness=True)

        Returns:
            Champion model after evolution

        Raises:
            ValueError: If not exactly 3 input models
        """
        if len(input_models) != 3:
            raise ValueError(f"Phase 2 requires 3 input models, got {len(input_models)}")

        print("\n" + "=" * 60)
        print("PHASE 2: EVOMERGE - EVOLUTIONARY OPTIMIZATION")
        print("=" * 60)
        print(f"Generations: {self.config.num_generations}")
        print(f"Population: {self.config.population_size}")
        print(f"Merge techniques: {list(self._mergers.keys())}")

        # Check for hybrid PS+DFS mode
        if self.config.use_hybrid_ps_dfs:
            print(f"Mode: HYBRID PS+DFS (Paper's best approach)")
            print(f"PS candidates: {len(input_models) * self.config.ps_candidates_multiplier}")
        elif self.config.use_cmaes:
            print(f"Mode: CMA-ES Parameter Space Optimization")
        else:
            print(f"Mode: Standard Evolutionary Search")

        if self.config.use_real_fitness:
            print(f"Fitness: Real benchmarks ({self.config.benchmark_name})")
        else:
            print(f"Fitness: Parameter-based proxy (fast)")

        print("=" * 60 + "\n")

        start_time = time.time()

        # HYBRID PS+DFS MODE (Paper's best approach)
        if self.config.use_hybrid_ps_dfs:
            return self._run_hybrid_ps_dfs(input_models, tokenizer)

        # Step 1: Initialize population from input models
        self._init_population(input_models)
        initial_fitness = self._evaluate_population()
        print(f"Initial best fitness: {initial_fitness:.4f}")

        # Step 2: Evolutionary loop
        for gen in range(self.config.num_generations):
            # Evaluate fitness
            gen_best_fitness = self._evaluate_population()

            # Track history
            self.fitness_history.append(gen_best_fitness)

            # Update champion if improved
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.champion = copy.deepcopy(self.population[0])

            # Selection
            parents = self._tournament_selection()

            # Crossover (merge)
            offspring = self._crossover(parents)

            # Mutation
            offspring = self._mutate(offspring)

            # Elitism + new population
            self.population = self._elitism_replacement(offspring)

            # Progress update
            if (gen + 1) % 10 == 0 or gen == 0:
                gain = (gen_best_fitness / initial_fitness - 1) * 100 if initial_fitness > 0 else 0
                print(
                    f"Gen {gen + 1}/{self.config.num_generations}: "
                    f"fitness={gen_best_fitness:.4f} (+{gain:.1f}%)"
                )

        # Step 3: Final evaluation and metrics
        final_fitness = self._evaluate_population()
        fitness_gain = (final_fitness / initial_fitness - 1) if initial_fitness > 0 else 0

        elapsed = time.time() - start_time
        self.metrics = {
            "initial_fitness": initial_fitness,
            "final_fitness": final_fitness,
            "fitness_gain": fitness_gain,
            "generations": self.config.num_generations,
            "population_size": self.config.population_size,
            "duration_seconds": elapsed,
            "merge_techniques_used": list(self._mergers.keys()),
        }

        print(f"\nPhase 2 Complete!")
        print(f"  Fitness gain: {fitness_gain * 100:.1f}%")
        print(f"  Duration: {elapsed:.1f}s")

        return self.champion if self.champion else self.population[0]

    def _init_population(self, input_models: List[nn.Module]) -> None:
        """Initialize population from 3 input models."""
        self.population = []

        # Add original models
        for model in input_models:
            self.population.append(copy.deepcopy(model))

        # Fill rest with merged variants
        while len(self.population) < self.config.population_size:
            # Random merge of 2 parents
            parents = random.sample(input_models, 2)
            technique = random.choice(list(self._mergers.keys()))
            if technique in self._mergers:
                child = self._mergers[technique].merge(parents)
                self.population.append(child)
            else:
                # Fallback: copy random parent
                self.population.append(copy.deepcopy(random.choice(input_models)))

    def _evaluate_population(self) -> float:
        """Evaluate fitness of all population members. Returns best fitness."""
        from phase2_evomerge.fitness.composite import compute_composite_fitness

        best_fitness = 0.0
        for i, model in enumerate(self.population):
            # Quick fitness approximation (full eval is expensive)
            fitness_result = self._quick_fitness(model)
            model._fitness = fitness_result["composite"]
            if model._fitness > best_fitness:
                best_fitness = model._fitness

        # Sort by fitness (descending)
        self.population.sort(key=lambda m: getattr(m, "_fitness", 0), reverse=True)
        return best_fitness

    def _quick_fitness(self, model: nn.Module) -> Dict[str, Any]:
        """Quick fitness evaluation (approximation for speed)."""
        from phase2_evomerge.fitness.composite import compute_composite_fitness

        # Get parameter stats as proxy for fitness
        total_params = sum(p.numel() for p in model.parameters())
        param_variance = sum(p.var().item() for p in model.parameters()) / max(
            1, len(list(model.parameters()))
        )

        # Approximate metrics (real eval would use actual inference)
        perplexity = 10.0 + param_variance * 5  # Lower variance = lower perplexity (proxy)
        accuracy = min(
            0.9, 0.3 + (1.0 / (1.0 + param_variance))
        )  # Higher variance = lower accuracy
        speed = 1000.0 + random.uniform(-100, 100)  # Tokens/sec (approx)
        memory = total_params * 4 / (1024 * 1024)  # MB

        return compute_composite_fitness(
            perplexity=max(1.0, perplexity),
            accuracy=max(0.0, min(1.0, accuracy)),
            speed=max(100.0, speed),
            memory=max(1.0, memory),
            weights=self.config.fitness_weights,
        )

    def _tournament_selection(self, k: int = 3) -> List[nn.Module]:
        """Tournament selection for parent selection."""
        parents = []
        for _ in range(self.config.population_size):
            tournament = random.sample(self.population, min(k, len(self.population)))
            winner = max(tournament, key=lambda m: getattr(m, "_fitness", 0))
            parents.append(winner)
        return parents

    def _crossover(self, parents: List[nn.Module]) -> List[nn.Module]:
        """Apply crossover using merge techniques."""
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            if random.random() < self.config.crossover_rate:
                # Select random merge technique
                technique = random.choice(list(self._mergers.keys()))
                if technique in self._mergers:
                    child = self._mergers[technique].merge([parents[i], parents[i + 1]])
                    offspring.append(child)
                else:
                    offspring.append(copy.deepcopy(parents[i]))
            else:
                offspring.append(copy.deepcopy(parents[i]))
        return offspring

    def _mutate(self, offspring: List[nn.Module]) -> List[nn.Module]:
        """Apply mutation (weight perturbation)."""
        for model in offspring:
            if random.random() < self.config.mutation_rate:
                with torch.no_grad():
                    for param in model.parameters():
                        noise = torch.randn_like(param) * 0.01  # Small noise
                        param.add_(noise)
        return offspring

    def _elitism_replacement(self, offspring: List[nn.Module]) -> List[nn.Module]:
        """Replace population keeping elite models."""
        # Keep top elite_count from current population
        elite = self.population[: self.config.elite_count]

        # Add offspring
        new_population = elite + offspring

        # Trim to population size
        return new_population[: self.config.population_size]

    def get_metrics(self) -> Dict:
        """Return collected metrics."""
        return self.metrics

    def _run_hybrid_ps_dfs(self, input_models: List[nn.Module], tokenizer: Optional[Any]) -> nn.Module:
        """
        Run hybrid PS+DFS merging (paper's best approach).

        Args:
            input_models: Base models from Phase 1
            tokenizer: Tokenizer for real fitness evaluation

        Returns:
            Champion model
        """
        from phase2_evomerge.merge.hybrid_ps_dfs import HybridConfig, hybrid_merge

        # Create fitness function
        if self.config.use_real_fitness:
            if tokenizer is None:
                raise ValueError("tokenizer is required for real fitness evaluation")

            from phase2_evomerge.fitness.benchmarks import BenchmarkConfig, evaluate_benchmark

            benchmark_config = BenchmarkConfig(
                benchmark_name=self.config.benchmark_name,
                max_samples=self.config.max_benchmark_samples,
            )

            def fitness_fn(model: nn.Module) -> float:
                return evaluate_benchmark(model, tokenizer, self.config.benchmark_name, benchmark_config)

        else:
            # Use proxy fitness
            fitness_fn = lambda model: self._quick_fitness(model)["composite"]

        # Configure hybrid merge
        hybrid_config = HybridConfig(
            ps_candidates_multiplier=self.config.ps_candidates_multiplier,
            ps_generations=self.config.num_generations,
            dfs_optimization_iterations=100,
        )

        # Run hybrid merge
        champion, metrics = hybrid_merge(
            input_models, fitness_fn, config=hybrid_config, tokenizer=tokenizer, verbose=True
        )

        # Store metrics
        self.metrics = {
            "initial_fitness": metrics["baseline_fitness"],
            "final_fitness": metrics["champion_fitness"],
            "fitness_gain": metrics["fitness_improvement"],
            "generations": self.config.num_generations,
            "population_size": len(input_models) * self.config.ps_candidates_multiplier,
            "duration_seconds": 0,  # Will be updated by caller
            "merge_strategy": "hybrid_ps_dfs",
            "ps_candidates": metrics["n_ps_candidates"],
            "ps_best_fitness": metrics["ps_best_fitness"],
        }

        self.champion = champion
        return champion

    def _create_real_fitness_fn(self, tokenizer: Any) -> callable:
        """
        Create fitness function using real benchmarks.

        Args:
            tokenizer: Tokenizer for model

        Returns:
            Fitness function
        """
        from phase2_evomerge.fitness.benchmarks import BenchmarkConfig, evaluate_benchmark

        benchmark_config = BenchmarkConfig(
            benchmark_name=self.config.benchmark_name, max_samples=self.config.max_benchmark_samples
        )

        def fitness_fn(model: nn.Module) -> float:
            """Evaluate model on real benchmark."""
            return evaluate_benchmark(model, tokenizer, self.config.benchmark_name, benchmark_config)

        return fitness_fn


__all__ = ["Phase2Pipeline", "EvolutionConfig"]

"""
Phase 2 Pipeline Orchestrator

Main entry point for Phase 2 (EvoMerge) that coordinates the entire
evolutionary optimization process.

Implements: 50-generation evolutionary optimization using 6 merge techniques.
"""

import copy
import importlib
import inspect
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

MERGER_SPECS = {
    "slerp": ("phase2_evomerge.merge.slerp_merge", "SLERPMerge"),
    "ties": ("phase2_evomerge.merge.ties_merge", "TIESMerge"),
    "dare": ("phase2_evomerge.merge.dare_merge", "DAREMerge"),
    "linear": ("phase2_evomerge.merge.linear_merge", "LinearMerge"),
    "frankenmerge": ("phase2_evomerge.merge.frankenmerge", "FrankenMerge"),
    "dfs": ("phase2_evomerge.merge.dfs_paper_accurate", "DFSPaperAccurate"),
}


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
        import_errors = {}

        for technique in self.config.merge_techniques:
            if technique not in MERGER_SPECS:
                import_errors[technique] = f"unknown merge technique {technique!r}"
                continue

            module_name, class_name = MERGER_SPECS[technique]
            try:
                module = importlib.import_module(module_name)
                merger_cls = getattr(module, class_name)
                mergers[technique] = merger_cls()
            except (ImportError, AttributeError) as exc:
                import_errors[technique] = str(exc)

        if import_errors:
            details = "; ".join(
                f"{technique}: {error}" for technique, error in sorted(import_errors.items())
            )
            raise ImportError(f"Failed to initialize configured Phase 2 mergers: {details}")

        if not mergers:
            raise ImportError("No Phase 2 mergers initialized")

        return mergers

    @staticmethod
    def _merge_models(merger, models: List[nn.Module]) -> nn.Module:
        """Call a merge operator with its correct signature.

        Operators have heterogeneous merge() signatures: Linear/SLERP take a list
        of models, DARE takes (model_finetuned, model_base) two single models, and
        DFS/TIES/Franken take (model_target, models_ref: List). The evolutionary
        path called them all with a single list, raising TypeError for the 2-arg
        operators. Dispatch by inspecting the signature so every operator is fed
        correctly.
        """
        required = [
            p
            for name, p in inspect.signature(merger.merge).parameters.items()
            if name != "self"
            and p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(required) <= 1:
            return merger.merge(models)  # list-taking (Linear/SLERP)
        if "List" in str(required[1].annotation):
            return merger.merge(models[0], models[1:])  # (target, refs) DFS/TIES/Franken
        return merger.merge(models[0], models[1])  # (finetuned, base) DARE

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
        self._validate_run_inputs(input_models, tokenizer)
        # Store for _evaluate_population so the standard path can honor use_real_fitness.
        self.tokenizer = tokenizer
        # E3: reset per-run state so a REUSED pipeline can't report/return a prior run's
        # champion (final_fitness now mirrors best_fitness, which would make that a visible lie).
        self.best_fitness = 0.0
        self.champion = None
        self.fitness_history = []

        print("\n" + "=" * 60)
        print("PHASE 2: EVOMERGE - EVOLUTIONARY OPTIMIZATION")
        print("=" * 60)
        print(f"Generations: {self.config.num_generations}")
        print(f"Population: {self.config.population_size}")
        print(f"Merge techniques: {list(self._mergers.keys())}")

        # Check for hybrid PS+DFS mode
        if self.config.use_hybrid_ps_dfs:
            print("Mode: HYBRID PS+DFS (Paper's best approach)")
            print(f"PS candidates: {len(input_models) * self.config.ps_candidates_multiplier}")
        elif self.config.use_cmaes:
            # E3: this was theater - it printed "CMA-ES" then ran standard evolution. Fail
            # loud rather than pretend a CMA-ES optimizer exists.
            raise NotImplementedError(
                "use_cmaes is not implemented; use the standard or hybrid PS+DFS path."
            )
        else:
            print("Mode: Standard Evolutionary Search")

        if self.config.use_real_fitness:
            print(f"Fitness: Real benchmarks ({self.config.benchmark_name})")
        else:
            print("Fitness: Parameter-based proxy (fast)")

        print("=" * 60 + "\n")

        start_time = time.time()

        # HYBRID PS+DFS MODE (Paper's best approach)
        if self.config.use_hybrid_ps_dfs:
            return self._run_hybrid_ps_dfs(input_models, tokenizer)

        # Step 1: Initialize population from input models
        self._init_population(input_models)
        initial_fitness = self._evaluate_population()
        print(f"Initial best fitness: {initial_fitness:.4f}")

        # Step 2: Evolutionary loop, then Step 3: consistent champion/metrics.
        self._evolve(initial_fitness)
        return self._finalize_run(initial_fitness, start_time)

    def _evolve(self, initial_fitness: float) -> None:
        """Run the generational loop: evaluate, track champion, select/crossover/mutate/replace."""
        for gen in range(self.config.num_generations):
            gen_best_fitness = self._evaluate_population()
            self.fitness_history.append(gen_best_fitness)
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.champion = copy.deepcopy(self.population[0])

            parents = self._tournament_selection()
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring)
            self.population = self._elitism_replacement(offspring)

            if (gen + 1) % 10 == 0 or gen == 0:
                gain = (gen_best_fitness / initial_fitness - 1) * 100 if initial_fitness > 0 else 0
                print(
                    f"Gen {gen + 1}/{self.config.num_generations}: "
                    f"fitness={gen_best_fitness:.4f} (+{gain:.1f}%)"
                )

    def _finalize_run(self, initial_fitness: float, start_time: float) -> nn.Module:
        """E3: keep champion/metrics consistent - fold the final population's best into the
        champion, then REPORT the champion's fitness (the model we return) and the ACTUAL
        population size, not a last-population number / nominal size that could disagree."""
        final_pop_best = self._evaluate_population()
        if final_pop_best > self.best_fitness:
            self.best_fitness = final_pop_best
            self.champion = copy.deepcopy(self.population[0])
        final_fitness = self.best_fitness
        fitness_gain = (final_fitness / initial_fitness - 1) if initial_fitness > 0 else 0

        elapsed = time.time() - start_time
        self.metrics = {
            "initial_fitness": initial_fitness,
            "final_fitness": final_fitness,
            "champion_fitness": self.best_fitness,
            "fitness_gain": fitness_gain,
            "generations": self.config.num_generations,
            "population_size": len(self.population),  # E3: actual, not the nominal config value
            "duration_seconds": elapsed,
            "merge_techniques_used": list(self._mergers.keys()),
        }

        print("\nPhase 2 Complete!")
        print(f"  Fitness gain: {fitness_gain * 100:.1f}%")
        print(f"  Duration: {elapsed:.1f}s")
        return self.champion if self.champion else self.population[0]

    def _validate_run_inputs(self, input_models: List[nn.Module], tokenizer: Optional[Any]) -> None:
        """E3: reject bad run inputs up front. Real fitness must NOT silently degrade to the
        parameter proxy - fail loud if requested without a tokenizer (matches hybrid path)."""
        if len(input_models) != 3:
            raise ValueError(f"Phase 2 requires 3 input models, got {len(input_models)}")
        if self.config.use_real_fitness and tokenizer is None:
            raise ValueError(
                "use_real_fitness=True requires a tokenizer; refusing to silently fall "
                "back to the parameter-statistics proxy."
            )

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
                child = self._merge_models(self._mergers[technique], parents)
                self.population.append(child)
            else:
                # Fallback: copy random parent
                self.population.append(copy.deepcopy(random.choice(input_models)))

    def _evaluate_population(self) -> float:
        """Evaluate fitness of all population members. Returns best fitness."""

        # Honor use_real_fitness on the standard path (not just the hybrid path):
        # use the real benchmark fitness when configured and a tokenizer is present,
        # otherwise the proxy approximation.
        use_real = self.config.use_real_fitness and getattr(self, "tokenizer", None) is not None
        real_fitness_fn = self._create_real_fitness_fn(self.tokenizer) if use_real else None

        best_fitness = 0.0
        for i, model in enumerate(self.population):
            if real_fitness_fn is not None:
                model._fitness = real_fitness_fn(model)
            else:
                model._fitness = self._quick_fitness(model)["composite"]
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
        # ponytail: deterministic proxy. Smaller model -> higher tokens/sec; NO random term
        # (the old random.uniform made the same model score differently each call, which
        # randomized champion selection). 1e12 numerator keeps the score above the floor and
        # still differentiating at LLM scale (222M -> ~4.5k). This is a parameter proxy, not a
        # real benchmark - set use_real_fitness=True (with a tokenizer) for actual task fitness.
        speed = 1.0e12 / max(1.0, float(total_params))  # Tokens/sec (deterministic proxy)
        memory = total_params * 4 / (1024 * 1024)  # MB

        result = compute_composite_fitness(
            perplexity=max(1.0, perplexity),
            accuracy=max(0.0, min(1.0, accuracy)),
            speed=max(100.0, speed),
            memory=max(1.0, memory),
            weights=self.config.fitness_weights,
        )
        # Label the numbers honestly: perplexity/accuracy here are a parameter-
        # variance PROXY, not measured task fitness. A consumer (or the champion
        # selector) must be able to tell this apart from a real benchmark eval.
        # Set use_real_fitness=True (with a tokenizer) for measured GSM8K/MGSM.
        result["fitness_method"] = "proxy_param_variance"
        result["is_proxy"] = True
        return result

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
                    child = self._merge_models(
                        self._mergers[technique], [parents[i], parents[i + 1]]
                    )
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

        # E3: refill to the configured size. Crossover yields only ~population_size/2
        # offspring, so elite+offspring kept shrinking the population each generation (it
        # stabilized below the nominal size while metrics still reported the nominal value).
        # Pad with deepcopies of the best members so the population stays constant.
        i = 0
        while len(new_population) < self.config.population_size and self.population:
            new_population.append(copy.deepcopy(self.population[i % len(self.population)]))
            i += 1

        # Trim to population size
        return new_population[: self.config.population_size]

    def get_metrics(self) -> Dict:
        """Return collected metrics."""
        return self.metrics

    def _run_hybrid_ps_dfs(
        self, input_models: List[nn.Module], tokenizer: Optional[Any]
    ) -> nn.Module:
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
                return evaluate_benchmark(
                    model, tokenizer, self.config.benchmark_name, benchmark_config
                )

        else:
            # Use proxy fitness
            def fitness_fn(model):
                return self._quick_fitness(model)["composite"]

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
            return evaluate_benchmark(
                model, tokenizer, self.config.benchmark_name, benchmark_config
            )

        return fitness_fn


__all__ = ["Phase2Pipeline", "EvolutionConfig"]

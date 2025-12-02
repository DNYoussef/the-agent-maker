"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimizer

Implements the CMA-ES algorithm for optimizing merge coefficients in
Parameter Space (PS) merging, as described in the Sakana AI EvoMerge paper.

CMA-ES is a stochastic derivative-free optimization algorithm that adapts
the covariance matrix of a multivariate normal distribution to search for
optimal parameter configurations.

Paper: Evolutionary Optimization of Model Merging Recipes (arXiv:2403.13187v1)

Key Features:
- Adapts search distribution based on successful mutations
- Maintains population diversity
- Efficient for high-dimensional optimization
- No gradient computation required

Algorithm:
    1. Initialize mean vector and covariance matrix
    2. Sample population from N(mean, sigma^2 * C)
    3. Evaluate fitness of each candidate
    4. Select top mu candidates
    5. Update mean, covariance, and step size
    6. Repeat until convergence
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch

logger = logging.getLogger(__name__)


@dataclass
class CMAESConfig:
    """Configuration for CMA-ES optimizer."""

    population_size: int = 50  # Lambda: number of offspring per generation
    n_parents: int = 25  # Mu: number of parents for recombination
    sigma: float = 0.3  # Initial step size (exploration vs exploitation)
    max_generations: int = 1000  # Maximum number of generations
    target_fitness: Optional[float] = None  # Stop if fitness reaches this value
    tolerance: float = 1e-6  # Convergence tolerance
    patience: int = 50  # Early stopping patience (generations without improvement)
    bounds: Tuple[float, float] = (0.0, 1.0)  # Parameter bounds for merge coefficients
    seed: Optional[int] = None  # Random seed for reproducibility

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "CMAESConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class CMAESOptimizer:
    """
    CMA-ES optimizer for Parameter Space (PS) merging coefficient optimization.

    Uses Optuna's built-in CMA-ES sampler for efficient implementation.
    Optimizes merge coefficients alpha = [alpha_1, ..., alpha_M] where M is
    the number of models being merged.

    Example:
        >>> config = CMAESConfig(population_size=50, sigma=0.3)
        >>> optimizer = CMAESOptimizer(config)
        >>> def fitness_fn(coeffs):
        ...     # Merge models with coefficients and evaluate
        ...     merged_model = merge_models(models, coeffs)
        ...     return evaluate_model(merged_model)
        >>> best_coeffs, best_fitness = optimizer.optimize(
        ...     fitness_fn,
        ...     n_dimensions=3,  # 3 models
        ...     n_trials=1000
        ... )
    """

    def __init__(self, config: Optional[CMAESConfig] = None):
        """
        Initialize CMA-ES optimizer.

        Args:
            config: CMA-ES configuration. If None, uses defaults.
        """
        self.config = config or CMAESConfig()
        self.best_params: Optional[np.ndarray] = None
        self.best_fitness: float = float("-inf")
        self.fitness_history: List[float] = []
        self.generation_count: int = 0
        self.stagnation_count: int = 0

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        n_dimensions: int,
        n_trials: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize merge coefficients using CMA-ES.

        Args:
            objective_fn: Function that takes coefficients (numpy array of shape
                [n_dimensions]) and returns fitness score (float, higher is better)
            n_dimensions: Number of dimensions (number of models being merged)
            n_trials: Number of trials (if None, uses max_generations * population_size)
            verbose: Whether to print progress

        Returns:
            Tuple of (best_coefficients, best_fitness)

        Example:
            >>> def fitness(coeffs):
            ...     return -np.sum((coeffs - 0.5)**2)  # Maximize closeness to 0.5
            >>> optimizer = CMAESOptimizer()
            >>> best_coeffs, best_fitness = optimizer.optimize(fitness, n_dimensions=3)
        """
        if n_trials is None:
            n_trials = self.config.max_generations * self.config.population_size

        if verbose:
            logger.info(
                f"Starting CMA-ES optimization: "
                f"dimensions={n_dimensions}, trials={n_trials}, "
                f"population={self.config.population_size}"
            )

        # Create Optuna study with CMA-ES sampler
        sampler = optuna.samplers.CMAESSampler(
            n_startup_trials=10,  # Random warmup trials
            seed=self.config.seed,
            # CMA-ES specific parameters
            with_margin=False,
            # Population size
            # Optuna uses default formula: 4 + 3*log(n_dimensions)
        )

        study = optuna.create_study(
            direction="maximize",  # Maximize fitness
            sampler=sampler,
        )

        # Create objective function for Optuna
        def optuna_objective(trial: optuna.Trial) -> float:
            # Suggest coefficients in [0, 1] range
            coeffs = np.array(
                [
                    trial.suggest_float(f"coeff_{i}", self.config.bounds[0], self.config.bounds[1])
                    for i in range(n_dimensions)
                ]
            )

            # Normalize coefficients to sum to 1 (for proper merging)
            coeffs = coeffs / (np.sum(coeffs) + 1e-8)

            # Evaluate fitness
            try:
                fitness = objective_fn(coeffs)
            except Exception as e:
                logger.warning(f"Fitness evaluation failed: {e}")
                return float("-inf")

            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_params = coeffs.copy()
                self.stagnation_count = 0
                if verbose:
                    logger.info(
                        f"New best fitness: {fitness:.4f} " f"at coeffs: {coeffs.round(3)}"
                    )
            else:
                self.stagnation_count += 1

            # Track history
            self.fitness_history.append(fitness)
            self.generation_count += 1

            # Early stopping check
            if self.config.target_fitness and fitness >= self.config.target_fitness:
                logger.info(f"Target fitness {self.config.target_fitness} reached!")
                study.stop()

            if self.stagnation_count >= self.config.patience:
                logger.info(f"Early stopping: No improvement for {self.config.patience} trials")
                study.stop()

            return fitness

        # Run optimization
        study.optimize(
            optuna_objective,
            n_trials=n_trials,
            show_progress_bar=verbose,
            catch=(Exception,),
        )

        # Extract best parameters
        if self.best_params is None:
            # Fallback: use best trial from study
            best_trial = study.best_trial
            self.best_params = np.array(
                [best_trial.params[f"coeff_{i}"] for i in range(n_dimensions)]
            )
            self.best_params = self.best_params / (np.sum(self.best_params) + 1e-8)
            self.best_fitness = best_trial.value

        if verbose:
            logger.info(f"\nOptimization complete!")
            logger.info(f"  Best fitness: {self.best_fitness:.4f}")
            logger.info(f"  Best coefficients: {self.best_params.round(3)}")
            logger.info(f"  Total generations: {self.generation_count}")

        return self.best_params, self.best_fitness

    def get_history(self) -> List[float]:
        """Return fitness history."""
        return self.fitness_history

    def reset(self):
        """Reset optimizer state."""
        self.best_params = None
        self.best_fitness = float("-inf")
        self.fitness_history = []
        self.generation_count = 0
        self.stagnation_count = 0


def ps_merge_with_cmaes(
    models: List[torch.nn.Module],
    fitness_fn: Callable[[torch.nn.Module], float],
    config: Optional[CMAESConfig] = None,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, np.ndarray, float]:
    """
    Parameter Space (PS) merging using CMA-ES optimization.

    Optimizes merge coefficients to maximize fitness of the merged model.

    Args:
        models: List of M models to merge
        fitness_fn: Function that evaluates a model and returns fitness score
        config: CMA-ES configuration
        verbose: Whether to print progress

    Returns:
        Tuple of (merged_model, best_coefficients, best_fitness)

    Example:
        >>> models = [model1, model2, model3]
        >>> def fitness(model):
        ...     return evaluate_on_benchmark(model)
        >>> merged, coeffs, fitness = ps_merge_with_cmaes(models, fitness)
    """
    config = config or CMAESConfig()
    optimizer = CMAESOptimizer(config)

    def objective(coeffs: np.ndarray) -> float:
        """Merge models with given coefficients and evaluate."""
        # Merge models using weighted averaging
        merged = _weighted_merge(models, coeffs)
        # Evaluate fitness
        return fitness_fn(merged)

    # Optimize
    best_coeffs, best_fitness = optimizer.optimize(
        objective, n_dimensions=len(models), verbose=verbose
    )

    # Create final merged model with best coefficients
    final_merged = _weighted_merge(models, best_coeffs)

    return final_merged, best_coeffs, best_fitness


def _weighted_merge(models: List[torch.nn.Module], weights: np.ndarray) -> torch.nn.Module:
    """
    Merge models using weighted averaging.

    Args:
        models: List of models to merge
        weights: Weights for each model (should sum to 1)

    Returns:
        Merged model
    """
    import copy

    # Create result as copy of first model
    result = copy.deepcopy(models[0])

    with torch.no_grad():
        for param_name, param in result.named_parameters():
            # Weighted sum of parameters
            merged_param = torch.zeros_like(param)
            for i, model in enumerate(models):
                model_param = dict(model.named_parameters())[param_name]
                merged_param += weights[i] * model_param

            param.copy_(merged_param)

    return result


__all__ = ["CMAESOptimizer", "CMAESConfig", "ps_merge_with_cmaes"]

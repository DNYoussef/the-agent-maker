"""
Hybrid PS + DFS Merging Pipeline

Implements the paper's key innovation: combining Parameter Space (PS) merging
with Dataflow Selection (DFS) for best results.

Paper: Evolutionary Optimization of Model Merging Recipes (arXiv:2403.13187v1)

Strategy:
    1. Phase 1 (PS Merging):
       - Apply PS merging to create M enlarged collection of models
       - Uses CMA-ES to optimize merge coefficients
       - Creates diverse candidate models

    2. Phase 2 (DFS Merging):
       - Apply DFS on the enlarged collection
       - Select best layers using indicator array I
       - Fine-tune with scaling matrix W
       - Produces final champion model

Benefits:
    - Combines global optimization (PS) with local routing (DFS)
    - PS phase explores parameter space broadly
    - DFS phase exploits layer-wise structure
    - Better than either technique alone (paper's key finding)

Algorithm:
    PS Phase:
        Input: N base models
        Output: M candidate models (M > N, typically M = 2-3N)
        Method: CMA-ES optimized weighted merging

    DFS Phase:
        Input: M candidate models from PS phase
        Output: 1 champion model
        Method: Layer-wise routing with indicator array + scaling matrix
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from phase2_evomerge.evolution.cma_es import CMAESConfig, CMAESOptimizer
from phase2_evomerge.merge.dfs_paper_accurate import DFSConfig, DFSPaperAccurate

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid PS + DFS merging."""

    # PS phase configuration
    ps_candidates_multiplier: int = 3  # Create M = N * multiplier candidates
    ps_cmaes_config: Optional[CMAESConfig] = None
    ps_generations: int = 50  # CMA-ES generations for PS phase

    # DFS phase configuration
    dfs_config: Optional[DFSConfig] = None
    dfs_optimization_iterations: int = 100  # Iterations for optimizing I and W

    # General
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class HybridPSDFS:
    """
    Hybrid PS + DFS merger combining global and local optimization.

    This implements the paper's best-performing merge strategy.
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        """
        Initialize hybrid merger.

        Args:
            config: Hybrid configuration
        """
        self.config = config or HybridConfig()

        # Initialize PS optimizer
        ps_config = self.config.ps_cmaes_config or CMAESConfig(
            population_size=50, sigma=0.3, max_generations=self.config.ps_generations
        )
        self.ps_optimizer = CMAESOptimizer(ps_config)

        # Initialize DFS merger
        dfs_config = self.config.dfs_config or DFSConfig()
        self.dfs_merger = DFSPaperAccurate(dfs_config)

        # Tracking
        self.ps_candidates: List[nn.Module] = []
        self.ps_fitness_scores: List[float] = []
        self.final_champion: Optional[nn.Module] = None
        self.final_fitness: float = 0.0

    def merge(
        self,
        base_models: List[nn.Module],
        fitness_fn: Callable[[nn.Module], float],
        tokenizer: Optional[Any] = None,
        verbose: bool = True,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Perform hybrid PS + DFS merging.

        Args:
            base_models: N base models to merge (typically 3 from Phase 1)
            fitness_fn: Function that evaluates model fitness
            tokenizer: Optional tokenizer for benchmark evaluation
            verbose: Whether to print progress

        Returns:
            Tuple of (champion_model, metrics_dict)

        Example:
            >>> models = [model1, model2, model3]
            >>> def fitness(model):
            ...     return evaluate_gsm8k(model, tokenizer)
            >>> hybrid = HybridPSDFS()
            >>> champion, metrics = hybrid.merge(models, fitness)
        """
        N = len(base_models)
        M = N * self.config.ps_candidates_multiplier

        if verbose:
            logger.info("=" * 70)
            logger.info("HYBRID PS + DFS MERGING")
            logger.info("=" * 70)
            logger.info(f"Input: {N} base models")
            logger.info(f"PS Phase: Creating {M} candidate models")
            logger.info(f"DFS Phase: Selecting champion from {M} candidates")
            logger.info("=" * 70)

        # Phase 1: PS Merging
        if verbose:
            logger.info("\n[PHASE 1: PARAMETER SPACE MERGING]")

        ps_candidates, ps_scores = self._ps_phase(base_models, fitness_fn, M, verbose)
        self.ps_candidates = ps_candidates
        self.ps_fitness_scores = ps_scores

        if verbose:
            logger.info(f"PS Phase complete: {len(ps_candidates)} candidates created")
            logger.info(f"PS best fitness: {max(ps_scores):.4f}")
            logger.info(f"PS mean fitness: {np.mean(ps_scores):.4f}")

        # Phase 2: DFS Merging
        if verbose:
            logger.info("\n[PHASE 2: DATAFLOW SELECTION MERGING]")

        champion, champion_fitness = self._dfs_phase(ps_candidates, fitness_fn, verbose)
        self.final_champion = champion
        self.final_fitness = champion_fitness

        if verbose:
            logger.info(f"DFS Phase complete")
            logger.info(f"Champion fitness: {champion_fitness:.4f}")

        # Compute metrics
        baseline_fitness = max([fitness_fn(model) for model in base_models])
        improvement = (champion_fitness - baseline_fitness) / baseline_fitness if baseline_fitness > 0 else 0

        metrics = {
            "n_base_models": N,
            "n_ps_candidates": M,
            "baseline_fitness": baseline_fitness,
            "ps_best_fitness": max(ps_scores),
            "ps_mean_fitness": float(np.mean(ps_scores)),
            "champion_fitness": champion_fitness,
            "fitness_improvement": improvement,
            "ps_candidates_fitness": ps_scores,
        }

        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("HYBRID MERGE COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Baseline fitness: {baseline_fitness:.4f}")
            logger.info(f"Champion fitness: {champion_fitness:.4f}")
            logger.info(f"Improvement: {improvement * 100:.2f}%")
            logger.info("=" * 70)

        return champion, metrics

    def _ps_phase(
        self, base_models: List[nn.Module], fitness_fn: Callable, M: int, verbose: bool
    ) -> Tuple[List[nn.Module], List[float]]:
        """
        Phase 1: Parameter Space merging to create M candidate models.

        Args:
            base_models: N base models
            fitness_fn: Fitness evaluation function
            M: Number of candidates to create
            verbose: Print progress

        Returns:
            Tuple of (candidate_models, fitness_scores)
        """
        N = len(base_models)
        candidates = []
        scores = []

        # Create M candidates by optimizing merge coefficients
        for i in range(M):
            if verbose:
                logger.info(f"\nPS Candidate {i + 1}/{M}:")

            # Create objective function for this candidate
            def objective(coeffs: np.ndarray) -> float:
                """Merge with given coefficients and evaluate."""
                merged = self._weighted_merge(base_models, coeffs)
                return fitness_fn(merged)

            # Optimize merge coefficients
            # Use fewer trials per candidate to speed up
            trials_per_candidate = max(10, self.config.ps_generations // M)
            best_coeffs, best_fitness = self.ps_optimizer.optimize(
                objective, n_dimensions=N, n_trials=trials_per_candidate, verbose=False
            )

            # Create candidate with best coefficients
            candidate = self._weighted_merge(base_models, best_coeffs)
            candidates.append(candidate)
            scores.append(best_fitness)

            if verbose:
                logger.info(f"  Coefficients: {best_coeffs.round(3)}")
                logger.info(f"  Fitness: {best_fitness:.4f}")

            # Reset optimizer for next candidate
            self.ps_optimizer.reset()

        return candidates, scores

    def _dfs_phase(
        self, ps_candidates: List[nn.Module], fitness_fn: Callable, verbose: bool
    ) -> Tuple[nn.Module, float]:
        """
        Phase 2: DFS merging on PS candidates to select champion.

        Args:
            ps_candidates: M candidate models from PS phase
            fitness_fn: Fitness evaluation function
            verbose: Print progress

        Returns:
            Tuple of (champion_model, champion_fitness)
        """
        # Optimize indicator array and scaling matrix
        if verbose:
            logger.info("Optimizing DFS indicators and scaling matrix...")

        best_indicators, best_scaling = self.dfs_merger.optimize_indicators_and_scaling(
            ps_candidates, fitness_fn, n_iterations=self.config.dfs_optimization_iterations
        )

        # Merge with optimized parameters
        champion = self.dfs_merger.merge(
            ps_candidates, indicator_array=best_indicators, scaling_matrix=best_scaling
        )

        # Evaluate champion
        champion_fitness = fitness_fn(champion)

        if verbose:
            logger.info(f"Indicators selected: {best_indicators.sum()}/{len(best_indicators)}")
            logger.info(f"Scaling matrix mean: {best_scaling.mean():.3f}")

        return champion, champion_fitness

    def _weighted_merge(self, models: List[nn.Module], weights: np.ndarray) -> nn.Module:
        """
        Merge models using weighted averaging.

        Args:
            models: List of models
            weights: Merge coefficients (normalized to sum to 1)

        Returns:
            Merged model
        """
        result = copy.deepcopy(models[0])

        with torch.no_grad():
            for param_name, param in result.named_parameters():
                merged_param = torch.zeros_like(param)
                for i, model in enumerate(models):
                    model_param = dict(model.named_parameters())[param_name]
                    merged_param += weights[i] * model_param
                param.copy_(merged_param)

        return result

    def get_ps_candidates(self) -> List[nn.Module]:
        """Return PS phase candidate models."""
        return self.ps_candidates

    def get_champion(self) -> Optional[nn.Module]:
        """Return final champion model."""
        return self.final_champion


def hybrid_merge(
    base_models: List[nn.Module],
    fitness_fn: Callable[[nn.Module], float],
    config: Optional[HybridConfig] = None,
    tokenizer: Optional[Any] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function for hybrid PS + DFS merging.

    Args:
        base_models: Base models to merge
        fitness_fn: Fitness evaluation function
        config: Hybrid configuration
        tokenizer: Optional tokenizer for benchmarks
        verbose: Print progress

    Returns:
        Tuple of (champion_model, metrics)

    Example:
        >>> models = [model1, model2, model3]
        >>> def fitness(model):
        ...     return evaluate_model(model)
        >>> champion, metrics = hybrid_merge(models, fitness)
        >>> print(f"Improvement: {metrics['fitness_improvement'] * 100:.1f}%")
    """
    merger = HybridPSDFS(config)
    return merger.merge(base_models, fitness_fn, tokenizer, verbose)


__all__ = ["HybridPSDFS", "HybridConfig", "hybrid_merge"]

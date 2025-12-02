"""
Fitness evaluation system for Phase 2 (EvoMerge) evolutionary optimization.

This module provides comprehensive fitness evaluation combining four metrics:
- Perplexity (40% weight) - Language modeling quality
- Accuracy (30% weight) - Task performance
- Speed (20% weight) - Inference efficiency
- Memory (10% weight) - Resource usage

Example:
    >>> from src.phase2_evomerge.fitness import FitnessEvaluator
    >>> evaluator = FitnessEvaluator(val_dataset, test_dataset)
    >>> fitness = evaluator.evaluate(model)
    >>> print(f"Composite fitness: {fitness['composite']:.4f}")
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional

from .perplexity import calculate_perplexity
from .accuracy import calculate_accuracy
from .speed import benchmark_speed
from .memory import measure_memory_usage
from .composite import compute_composite_fitness, DEFAULT_WEIGHTS, DEFAULT_EXPECTED
from .cache import FitnessCache


__all__ = [
    'FitnessEvaluator',
    'calculate_perplexity',
    'calculate_accuracy',
    'benchmark_speed',
    'measure_memory_usage',
    'compute_composite_fitness',
    'FitnessCache',
    'DEFAULT_WEIGHTS',
    'DEFAULT_EXPECTED'
]


class FitnessEvaluator:
    """
    Main API for comprehensive fitness evaluation.

    Combines four component metrics into a single composite fitness score
    using weighted averaging. Supports caching for performance.

    Example:
        >>> # Basic usage
        >>> evaluator = FitnessEvaluator(
        ...     validation_dataset=val_loader,
        ...     test_dataset=test_loader
        ... )
        >>> fitness = evaluator.evaluate(model)
        >>> print(f"Fitness: {fitness['composite']:.4f}")

        >>> # Batch evaluation (for population)
        >>> fitness_scores = evaluator.evaluate_batch(population)
        >>> print(f"Best fitness: {max(fitness_scores):.4f}")

        >>> # Custom weights (prioritize speed)
        >>> evaluator = FitnessEvaluator(
        ...     validation_dataset=val_loader,
        ...     fitness_weights={'perplexity': 0.3, 'accuracy': 0.2,
        ...                      'speed': 0.4, 'memory': 0.1}
        ... )
    """

    def __init__(
        self,
        validation_dataset: DataLoader,
        test_dataset: Optional[DataLoader] = None,
        fitness_weights: Optional[Dict[str, float]] = None,
        expected_values: Optional[Dict[str, float]] = None,
        cache_enabled: bool = True,
        device: str = 'cuda',
        mixed_precision: bool = True,
        max_batches: Optional[int] = None,
        benchmark_batch_size: int = 32,
        benchmark_seq_len: int = 512
    ):
        """
        Initialize fitness evaluator with datasets and configuration.

        Args:
            validation_dataset: DataLoader for perplexity calculation
            test_dataset: DataLoader for accuracy (None = use validation)
            fitness_weights: Custom weights (None = use defaults)
            expected_values: Normalization baselines (None = use defaults)
            cache_enabled: Enable fitness caching (default: True)
            device: Device to use ('cuda' or 'cpu')
            mixed_precision: Use mixed precision for perplexity
            max_batches: Limit evaluation to N batches per dataset
            benchmark_batch_size: Batch size for speed/memory benchmarks
            benchmark_seq_len: Sequence length for benchmarks
        """
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset or validation_dataset
        self.fitness_weights = fitness_weights or DEFAULT_WEIGHTS
        self.expected_values = expected_values or DEFAULT_EXPECTED
        self.device = device
        self.mixed_precision = mixed_precision
        self.max_batches = max_batches

        # Create benchmark batch for speed/memory tests
        self.benchmark_batch = torch.randint(
            0, 1000,  # Vocab size dummy
            (benchmark_batch_size, benchmark_seq_len),
            device=device
        )

        # Initialize cache
        self.cache = FitnessCache() if cache_enabled else None

    def evaluate(self, model: nn.Module) -> Dict[str, Any]:
        """
        Evaluate single model fitness (with caching).

        Calculates all four component metrics, combines into composite score,
        and caches the result for future lookups.

        Args:
            model: Model to evaluate

        Returns:
            Dictionary with composite score and components:
            {
                'composite': 0.185,
                'components': {
                    'perplexity': 15.2,
                    'perplexity_score': 0.0658,
                    'accuracy': 0.48,
                    'speed': 1250.0,
                    'speed_score': 1.042,
                    'memory': 520.0,
                    'memory_score': 0.962
                }
            }

        Example:
            >>> fitness = evaluator.evaluate(model)
            >>> print(f"Composite: {fitness['composite']:.4f}")
            >>> print(f"Perplexity: {fitness['components']['perplexity']:.2f}")
        """
        # Check cache first
        if self.cache:
            cached_fitness = self.cache.get(model)
            if cached_fitness is not None:
                return cached_fitness

        # Calculate component metrics
        perplexity = calculate_perplexity(
            model,
            self.validation_dataset,
            device=self.device,
            mixed_precision=self.mixed_precision,
            max_batches=self.max_batches
        )

        accuracy = calculate_accuracy(
            model,
            self.test_dataset,
            device=self.device,
            max_batches=self.max_batches
        )

        speed = benchmark_speed(
            model,
            self.benchmark_batch,
            device=self.device
        )

        # Memory measurement requires CUDA
        if self.device == 'cuda':
            memory = measure_memory_usage(
                model,
                self.benchmark_batch,
                device=self.device
            )
        else:
            # Use expected value for CPU (no CUDA memory)
            memory = self.expected_values['memory']

        # Compute composite fitness
        fitness = compute_composite_fitness(
            perplexity=perplexity,
            accuracy=accuracy,
            speed=speed,
            memory=memory,
            weights=self.fitness_weights,
            expected_values=self.expected_values
        )

        # Cache result
        if self.cache:
            self.cache.put(model, fitness)

        return fitness

    def evaluate_batch(self, models: List[nn.Module]) -> List[float]:
        """
        Evaluate batch of models (returns composite scores only).

        This is the primary API for evolutionary optimization loops,
        where only the composite scores are needed for ranking.

        Args:
            models: List of models to evaluate

        Returns:
            List of composite fitness scores (one per model)

        Example:
            >>> population = [model1, model2, model3, ..., model8]
            >>> scores = evaluator.evaluate_batch(population)
            >>> best_idx = scores.index(max(scores))
            >>> print(f"Best model: {best_idx}, fitness: {scores[best_idx]:.4f}")
        """
        composite_scores = []

        for model in models:
            fitness = self.evaluate(model)
            composite_scores.append(fitness['composite'])

        return composite_scores

    def clear_cache(self) -> None:
        """
        Clear fitness cache.

        Useful for memory management or when starting a new evolution run.

        Example:
            >>> evaluator.clear_cache()
        """
        if self.cache:
            self.cache.clear()

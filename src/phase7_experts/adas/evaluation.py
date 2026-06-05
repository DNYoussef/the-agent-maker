"""
Evaluation Logic

Implements fitness evaluation for individuals in the ADAS population.
"""

from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn

from .config import Individual


def evaluate_individual(
    individual: Individual,
    model: nn.Module,
    experts: List[Any],
    tokenizer: Any,
    evaluator: Callable = None,
) -> Dict[str, float]:
    """
    Evaluate a single individual.

    Args:
        individual: Individual to evaluate
        model: Base model
        experts: List of ExpertProfile
        tokenizer: Tokenizer
        evaluator: Optional custom evaluation function

    Returns:
        Dictionary of fitness scores
    """
    if evaluator is None:
        raise RuntimeError(
            "ADAS fitness evaluator is not configured; refusing to use synthetic "
            "entropy/random-noise scores as measured fitness."
        )

    return evaluator(individual, model, experts, tokenizer)


def evaluate_population(
    population: List[Individual],
    model: nn.Module,
    experts: List[Any],
    tokenizer: Any,
    evaluator: Callable = None,
) -> None:
    """
    Evaluate fitness for all individuals in population.

    Args:
        population: List of individuals to evaluate
        model: Base model
        experts: List of ExpertProfile
        tokenizer: Tokenizer
        evaluator: Optional custom evaluation function
    """
    for individual in population:
        if not individual.fitness_scores:
            individual.fitness_scores = evaluate_individual(
                individual, model, experts, tokenizer, evaluator
            )

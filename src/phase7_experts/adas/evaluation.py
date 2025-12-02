"""
Evaluation Logic

Implements fitness evaluation for individuals in the ADAS population.
"""

import math
import random
from typing import Dict, List, Callable, Any
import torch.nn as nn

from .config import Individual


def evaluate_individual(
    individual: Individual,
    model: nn.Module,
    experts: List[Any],
    tokenizer: Any,
    evaluator: Callable = None
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
    if evaluator is not None:
        return evaluator(individual, model, experts, tokenizer)

    # Default multi-objective evaluation
    scores = {}

    # Objective 1: Accuracy (simulated based on routing diversity)
    weights = individual.routing_weights
    entropy = -sum(w * (math.log(w + 1e-10)) for w in weights) if weights else 0
    max_entropy = math.log(len(weights) + 1e-10) if weights else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Higher entropy = more balanced routing = potentially better
    scores['accuracy'] = 0.5 + 0.5 * normalized_entropy + random.uniform(-0.1, 0.1)
    scores['accuracy'] = max(0, min(1, scores['accuracy']))

    # Objective 2: Latency (lower is better, sparse routing is faster)
    max_weight = max(weights) if weights else 0
    sparsity = max_weight  # Higher max = more sparse = faster
    scores['latency'] = sparsity + random.uniform(-0.1, 0.1)
    scores['latency'] = max(0, min(1, scores['latency']))

    # Objective 3: Diversity (expert utilization)
    active_experts = sum(1 for w in weights if w > 0.1)
    scores['diversity'] = active_experts / len(weights) if weights else 0

    return scores


def evaluate_population(
    population: List[Individual],
    model: nn.Module,
    experts: List[Any],
    tokenizer: Any,
    evaluator: Callable = None
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

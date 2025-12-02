"""
Genetic Operators

Implements crossover and mutation operations for ADAS evolution.
"""

import random
import copy
from typing import Tuple, List
from .config import Individual, ADASConfig


def crossover(
    parent1: Individual,
    parent2: Individual,
    num_experts: int
) -> Tuple[Individual, Individual]:
    """
    Uniform crossover for two parents.

    Args:
        parent1: First parent
        parent2: Second parent
        num_experts: Number of experts

    Returns:
        Tuple of (child1, child2)
    """
    # Crossover routing weights
    weights1, weights2 = [], []
    for i in range(num_experts):
        if random.random() < 0.5:
            weights1.append(parent1.routing_weights[i])
            weights2.append(parent2.routing_weights[i])
        else:
            weights1.append(parent2.routing_weights[i])
            weights2.append(parent1.routing_weights[i])

    # Normalize
    sum1, sum2 = sum(weights1), sum(weights2)
    weights1 = [w / sum1 for w in weights1] if sum1 > 0 else weights1
    weights2 = [w / sum2 for w in weights2] if sum2 > 0 else weights2

    child1 = Individual(
        routing_weights=weights1,
        expert_configs=copy.deepcopy(parent1.expert_configs),
        fitness_scores={}
    )
    child2 = Individual(
        routing_weights=weights2,
        expert_configs=copy.deepcopy(parent2.expert_configs),
        fitness_scores={}
    )

    return child1, child2


def mutate(individual: Individual, num_experts: int) -> None:
    """
    Gaussian mutation for routing weights.

    Args:
        individual: Individual to mutate (modified in-place)
        num_experts: Number of experts
    """
    # Mutate routing weights
    for i in range(num_experts):
        if random.random() < 0.3:
            individual.routing_weights[i] += random.gauss(0, 0.1)
            individual.routing_weights[i] = max(0.01, individual.routing_weights[i])

    # Normalize
    weight_sum = sum(individual.routing_weights)
    individual.routing_weights = [w / weight_sum for w in individual.routing_weights]


def create_offspring(
    parents: List[Individual],
    num_experts: int,
    config: ADASConfig
) -> List[Individual]:
    """
    Create offspring via crossover and mutation.

    Args:
        parents: Parent population
        num_experts: Number of experts
        config: ADAS configuration

    Returns:
        List of offspring
    """
    offspring = []

    for i in range(0, len(parents) - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

        # Crossover
        if random.random() < config.crossover_rate:
            child1, child2 = crossover(parent1, parent2, num_experts)
        else:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

        # Mutation
        if random.random() < config.mutation_rate:
            mutate(child1, num_experts)
        if random.random() < config.mutation_rate:
            mutate(child2, num_experts)

        # Clear fitness (needs re-evaluation)
        child1.fitness_scores = {}
        child2.fitness_scores = {}

        offspring.extend([child1, child2])

    return offspring[:config.population_size]

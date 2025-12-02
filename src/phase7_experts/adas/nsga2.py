"""
NSGA-II Algorithm Components

Implements Pareto ranking and crowding distance calculation
for multi-objective optimization.
"""

from typing import List
from .config import Individual, ADASConfig


def assign_ranks(population: List[Individual], config: ADASConfig) -> None:
    """
    Assign Pareto ranks to population (NSGA-II).

    Args:
        population: List of individuals to rank
        config: ADAS configuration with objectives
    """
    # Reset ranks
    for ind in population:
        ind.rank = 0

    remaining = list(population)
    current_rank = 0

    while remaining:
        non_dominated = []

        for ind in remaining:
            dominated = False
            for other in remaining:
                if other is not ind and _dominates(other, ind, config):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(ind)

        for ind in non_dominated:
            ind.rank = current_rank
            remaining.remove(ind)

        current_rank += 1


def _dominates(ind1: Individual, ind2: Individual, config: ADASConfig) -> bool:
    """
    Check if ind1 Pareto-dominates ind2.

    Args:
        ind1: First individual
        ind2: Second individual
        config: ADAS configuration with objectives

    Returns:
        True if ind1 dominates ind2
    """
    at_least_one_better = False

    for obj in config.objectives:
        v1 = ind1.fitness_scores.get(obj, 0)
        v2 = ind2.fitness_scores.get(obj, 0)

        # Assume maximization for all objectives
        if v1 < v2:
            return False
        if v1 > v2:
            at_least_one_better = True

    return at_least_one_better


def calculate_crowding_distance(
    population: List[Individual],
    config: ADASConfig
) -> None:
    """
    Calculate crowding distance for diversity preservation.

    Args:
        population: List of individuals
        config: ADAS configuration with objectives
    """
    # Group by rank
    ranks = {}
    for ind in population:
        if ind.rank not in ranks:
            ranks[ind.rank] = []
        ranks[ind.rank].append(ind)

    for rank_inds in ranks.values():
        n = len(rank_inds)
        if n == 0:
            continue

        # Initialize distances
        for ind in rank_inds:
            ind.crowding_distance = 0.0

        # For each objective
        for obj in config.objectives:
            # Sort by objective
            rank_inds.sort(key=lambda x: x.fitness_scores.get(obj, 0))

            # Boundary points get infinite distance
            rank_inds[0].crowding_distance = float('inf')
            rank_inds[-1].crowding_distance = float('inf')

            # Calculate distance for others
            obj_range = (
                rank_inds[-1].fitness_scores.get(obj, 0) -
                rank_inds[0].fitness_scores.get(obj, 0)
            )

            if obj_range > 0:
                for i in range(1, n - 1):
                    dist = (
                        rank_inds[i + 1].fitness_scores.get(obj, 0) -
                        rank_inds[i - 1].fitness_scores.get(obj, 0)
                    ) / obj_range
                    rank_inds[i].crowding_distance += dist


def tournament_selection(
    population: List[Individual],
    config: ADASConfig,
    num_parents: int = None
) -> List[Individual]:
    """
    Tournament selection based on rank and crowding distance.

    Args:
        population: Current population
        config: ADAS configuration
        num_parents: Number of parents to select (default: population_size)

    Returns:
        Selected parents
    """
    import random

    if num_parents is None:
        num_parents = config.population_size

    parents = []

    for _ in range(num_parents):
        tournament = random.sample(population, config.tournament_size)

        # Select best by rank, then crowding distance
        winner = min(tournament, key=lambda x: (x.rank, -x.crowding_distance))
        parents.append(winner)

    return parents


def survivor_selection(
    population: List[Individual],
    offspring: List[Individual],
    config: ADASConfig
) -> List[Individual]:
    """
    Elitist survivor selection.

    Args:
        population: Current population
        offspring: New offspring
        config: ADAS configuration

    Returns:
        New population
    """
    # Combine parents and offspring
    combined = population + offspring

    # Re-evaluate and rank
    assign_ranks(combined, config)
    calculate_crowding_distance(combined, config)

    # Sort by rank, then crowding distance
    combined.sort(key=lambda x: (x.rank, -x.crowding_distance))

    # Keep best
    return combined[:config.population_size]

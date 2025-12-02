"""
ADAS Configuration and Data Classes

Defines configuration, individual representation, and result structures
for the ADAS optimization system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ADASConfig:
    """Configuration for ADAS optimization."""

    population_size: int = 50
    num_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    tournament_size: int = 3
    objectives: List[str] = field(default_factory=lambda: ["accuracy", "latency", "diversity"])
    elite_ratio: float = 0.1


@dataclass
class Individual:
    """An individual in the NSGA-II population."""

    routing_weights: List[float]  # Expert routing probabilities
    expert_configs: Dict[str, Any]  # Per-expert configuration
    fitness_scores: Dict[str, float]  # Multi-objective fitness
    rank: int = 0  # Pareto rank
    crowding_distance: float = 0.0


@dataclass
class ADASResult:
    """Result from ADAS optimization."""

    success: bool
    best_individual: Individual
    pareto_front: List[Individual]
    generation_history: List[Dict]
    metrics: Dict

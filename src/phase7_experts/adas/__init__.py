"""
ADAS (Automated Design of Agentic Systems) - Modular Package

Re-exports all components for backward compatibility.
"""

from .config import ADASConfig, Individual, ADASResult
from .nsga2 import assign_ranks, calculate_crowding_distance
from .operators import crossover, mutate
from .evaluation import evaluate_individual
from .optimizer import ADASOptimizer

__all__ = [
    'ADASOptimizer',
    'ADASConfig',
    'ADASResult',
    'Individual',
    'assign_ranks',
    'calculate_crowding_distance',
    'crossover',
    'mutate',
    'evaluate_individual'
]

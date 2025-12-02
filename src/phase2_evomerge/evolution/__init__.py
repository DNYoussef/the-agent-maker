"""
Evolution module for Phase 2 (EvoMerge).

Implements evolutionary optimization with:
- Population initialization from binary combos (8 models)
- Elite preservation via Gaussian mutation (top 2 → 6 children)
- Loser merging via combo application (bottom 6 → 2 children)
- Diversity tracking and re-seeding
- Early stopping on convergence

Example:
    >>> from src.phase2_evomerge.evolution import EvolutionLoop, EvolutionConfig
    >>> from src.phase2_evomerge.fitness import FitnessEvaluator
    >>>
    >>> # Setup
    >>> config = EvolutionConfig(generations=50)
    >>> evaluator = FitnessEvaluator(val_dataset, test_dataset)
    >>> evolution = EvolutionLoop(config, evaluator)
    >>>
    >>> # Run evolution
    >>> result = evolution.evolve([model1, model2, model3])
    >>> print(f"Best fitness: {result['fitness']:.4f}")
    >>> print(f"Improvement: {result['improvement_pct']:.1%}")
"""

from .config import EvolutionConfig
from .population import initialize_population
from .mutation import mutate_model
from .diversity import compute_diversity
from .evolution_loop import EvolutionLoop

__all__ = [
    'EvolutionConfig',
    'EvolutionLoop',
    'initialize_population',
    'mutate_model',
    'compute_diversity'
]

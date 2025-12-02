"""
Phase 2 (EvoMerge): Evolutionary Model Optimization

This module implements evolutionary optimization that takes 3 specialized models
from Phase 1 (Cognate) and evolves them over 50 generations using 6 merge
techniques across 8 binary combinations to produce a single, highly-optimized
merged model.

Key Components:
    - merge: 6 merge techniques (Linear, SLERP, DARE, TIES, FrankenMerge, DFS)
    - fitness: Composite fitness evaluation (perplexity, accuracy, speed, memory)
    - population: Population management (elite mutation, loser merging)
    - evolution: 50-generation evolution engine with convergence detection
    - monitoring: W&B logging (370 metrics)

Target Performance:
    - Fitness Improvement: ≥20% (target: 23.5%)
    - Evolution Time: ≤90 minutes on GTX 1660
    - Memory Usage: <6GB VRAM (12 models max)
    - Population Diversity: >0.3 maintained
"""

__version__ = "1.0.0"
__author__ = "Agent Forge V2 Team"

from .phase2_pipeline import Phase2Pipeline

__all__ = ["Phase2Pipeline"]

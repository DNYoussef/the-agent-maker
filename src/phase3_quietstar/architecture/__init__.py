"""
Phase 3 Quiet-STaR Architecture Components

Re-exports all classes for backward compatibility.
"""

from .coherence_scorer import CoherenceScorer
from .dataclasses import CoherenceScores, ThoughtOutput
from .mixing_head import MixingHead
from .quiet_star_model import QuietSTaRModel
from .thought_generator import ThoughtGenerator
from .thought_injector import ThoughtInjector

__all__ = [
    "ThoughtOutput",
    "CoherenceScores",
    "ThoughtGenerator",
    "CoherenceScorer",
    "MixingHead",
    "ThoughtInjector",
    "QuietSTaRModel",
]

"""
Phase 3: Quiet-STaR - Reasoning Enhancement via Thought Generation

This phase implements the Quiet-STaR algorithm with a two-step process:
1. Prompt Baking: Supervised learning to embed thinking tokens and reasoning patterns
2. Quiet-STaR RL: REINFORCE-based RL training on baked foundation

Key innovation: 30-50% faster RL convergence by providing reasoning foundation
before RL training begins (the "jumpstart effect").
"""

from .architecture import (
    ThoughtGenerator,
    CoherenceScorer,
    MixingHead,
    ThoughtInjector,
    QuietSTaRModel,
)
from .config import QuietSTaRConfig

__all__ = [
    "ThoughtGenerator",
    "CoherenceScorer",
    "MixingHead",
    "ThoughtInjector",
    "QuietSTaRModel",
    "QuietSTaRConfig",
]

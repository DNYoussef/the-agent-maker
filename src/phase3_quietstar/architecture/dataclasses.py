"""
Phase 3 Quiet-STaR Data Classes

Core data structures for thought generation and coherence scoring.
"""

import torch
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ThoughtOutput:
    """Output from thought generation."""

    thoughts: torch.Tensor  # (batch, num_thoughts, thought_len)
    thought_ids: List[List[int]]  # Generated token IDs
    log_probs: torch.Tensor  # Log probabilities
    attention_weights: Optional[torch.Tensor] = None


@dataclass
class CoherenceScores:
    """Coherence scoring output."""

    semantic: torch.Tensor  # (batch, num_thoughts)
    syntactic: torch.Tensor  # (batch, num_thoughts)
    predictive: torch.Tensor  # (batch, num_thoughts)
    composite: torch.Tensor  # Weighted average

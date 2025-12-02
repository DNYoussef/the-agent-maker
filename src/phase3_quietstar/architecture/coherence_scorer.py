"""
Phase 3 Quiet-STaR Coherence Scorer

Score thought quality across 3 dimensions:
- Semantic: 40% weight (embedding similarity)
- Syntactic: 30% weight (grammar validity)
- Predictive: 30% weight (helps next-token prediction)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataclasses import CoherenceScores


class CoherenceScorer(nn.Module):
    """
    Score thought quality across 3 dimensions:
    - Semantic: 40% weight (embedding similarity)
    - Syntactic: 30% weight (grammar validity)
    - Predictive: 30% weight (helps next-token prediction)

    Returns composite score = 0.4*semantic + 0.3*syntactic + 0.3*predictive
    """

    def __init__(
        self,
        hidden_size: int,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Default weights
        self.weights = weights or {
            "semantic": 0.4,
            "syntactic": 0.3,
            "predictive": 0.3,
        }

        # Semantic scorer
        self.semantic_projection = nn.Linear(hidden_size, hidden_size)

        # Syntactic scorer
        self.syntactic_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Predictive scorer
        self.predictive_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        base_hidden: torch.Tensor,
        thought_hiddens: torch.Tensor,
        next_token_logits: Optional[torch.Tensor] = None,
    ) -> CoherenceScores:
        """
        Score thoughts against base hidden state.

        Args:
            base_hidden: (batch, hidden_size) - Base representation
            thought_hiddens: (batch, num_thoughts, thought_len, hidden_size)
            next_token_logits: (batch, vocab_size) - For predictive scoring
        """
        batch_size, num_thoughts = thought_hiddens.shape[:2]

        # Average thought representations
        thought_avg = thought_hiddens.mean(dim=2)

        # Semantic coherence (cosine similarity)
        semantic = self._semantic_coherence(base_hidden, thought_avg)

        # Syntactic coherence (grammar validity)
        syntactic = self._syntactic_coherence(thought_avg)

        # Predictive coherence (helps prediction)
        predictive = self._predictive_coherence(thought_avg, next_token_logits)

        # Composite score
        composite = (
            self.weights["semantic"] * semantic
            + self.weights["syntactic"] * syntactic
            + self.weights["predictive"] * predictive
        )

        return CoherenceScores(
            semantic=semantic,
            syntactic=syntactic,
            predictive=predictive,
            composite=composite,
        )

    def _semantic_coherence(self, base: torch.Tensor, thoughts: torch.Tensor) -> torch.Tensor:
        """Semantic similarity via cosine distance."""
        base_proj = self.semantic_projection(base)
        base_norm = F.normalize(base_proj, p=2, dim=-1)

        thought_norm = F.normalize(thoughts, p=2, dim=-1)

        # Cosine similarity
        similarity = torch.bmm(thought_norm, base_norm.unsqueeze(-1)).squeeze(-1)

        # Scale to [0, 1]
        return (similarity + 1.0) / 2.0

    def _syntactic_coherence(self, thoughts: torch.Tensor) -> torch.Tensor:
        """Grammar validity via learned MLP."""
        return self.syntactic_mlp(thoughts).squeeze(-1)

    def _predictive_coherence(
        self, thoughts: torch.Tensor, logits: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """How much thought helps next-token prediction."""
        if logits is None:
            return torch.ones(thoughts.size(0), thoughts.size(1), device=thoughts.device)

        # Predict utility
        utility = self.predictive_head(thoughts).squeeze(-1)
        return torch.sigmoid(utility)

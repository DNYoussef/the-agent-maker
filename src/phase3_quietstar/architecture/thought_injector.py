"""
Phase 3 Quiet-STaR Thought Injector

Identify difficult token positions for thought injection.
Uses 3 difficulty metrics: entropy, attention dispersion, and loss.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThoughtInjector(nn.Module):
    """
    Identify difficult token positions for thought injection.

    Uses 3 difficulty metrics:
    1. Entropy (high uncertainty)
    2. Attention dispersion (spread attention)
    3. Loss (high prediction error)

    Injects thoughts when composite difficulty > threshold.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        min_interval: int = 3,
    ):
        super().__init__()
        self.threshold = threshold
        self.min_interval = min_interval
        self.last_injection = -min_interval

    def forward(
        self,
        logits: torch.Tensor,
        attention_weights: Optional[torch.Tensor],
        loss: Optional[torch.Tensor],
        position: int,
    ) -> bool:
        """Determine if thought injection needed at position."""
        # Check minimum interval
        if position - self.last_injection < self.min_interval:
            return False

        # Compute difficulty metrics
        entropy = self._compute_entropy(logits)
        dispersion = self._compute_dispersion(attention_weights)
        error = loss.item() if loss is not None else 0.0

        # Composite difficulty (normalized to [0, 1])
        difficulty = 0.4 * entropy + 0.3 * dispersion + 0.3 * min(error, 10.0) / 10.0

        # Inject if above threshold
        if difficulty > self.threshold:
            self.last_injection = position
            return True

        return False

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute prediction entropy (high = uncertain)."""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        # Normalize to [0, 1]
        max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float32))
        return (entropy / max_entropy).mean().item()

    def _compute_dispersion(self, attention_weights: Optional[torch.Tensor]) -> float:
        """Compute attention dispersion (high = spread out)."""
        if attention_weights is None:
            return 0.5  # Neutral value

        # Compute entropy of attention distribution
        entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1)

        # Normalize
        max_entropy = torch.log(torch.tensor(attention_weights.size(-1), dtype=torch.float32))
        return (entropy / max_entropy).mean().item()

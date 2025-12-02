"""
Memory-Augmented Gate (MAG)

Learns convex combination of current output and memory contribution.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAGGate(nn.Module):
    """
    Memory-Augmented Gate (MAG)

    Learns convex combination of current output (y) and memory (m).
    """

    def __init__(self, d_model: int, hidden: int = 256, entropy_reg: float = 0.001):
        """
        Initialize MAG Gate.

        Args:
            d_model: Model dimension
            hidden: Hidden layer size for gating network
            entropy_reg: Entropy regularization coefficient
        """
        super().__init__()
        self.d_model = d_model
        self.entropy_reg = entropy_reg

        # Gating network
        self.w_concat = nn.Linear(2 * d_model, hidden)
        self.w_gate = nn.Linear(hidden, d_model)

    def forward(self, y: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply memory-augmented gating.

        Args:
            y: Current output [batch, seq_len, d_model]
            m: Memory contribution [batch, seq_len, d_model]

        Returns:
            output: Gated output [batch, seq_len, d_model]
            loss_entropy: Scalar entropy loss for regularization
        """
        # Concatenate y and m
        concat = torch.cat([y, m], dim=-1)

        # Compute gate
        hidden = F.relu(self.w_concat(concat))
        g = torch.sigmoid(self.w_gate(hidden))

        # Convex blend
        output = g * y + (1 - g) * m

        # Entropy regularization (prevent saturation)
        eps = 1e-8
        entropy = -(g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps))
        loss_entropy = -self.entropy_reg * entropy.mean()

        return output, loss_entropy

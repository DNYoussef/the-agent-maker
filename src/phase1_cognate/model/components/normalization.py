"""
RMSNorm - Root Mean Square Layer Normalization

Efficient normalization technique used in Titans-MAG.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.

        Args:
            dim: Dimension of the input
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.

        Args:
            x: Input tensor of shape [batch, seq_len, dim]

        Returns:
            Normalized tensor of same shape
        """
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

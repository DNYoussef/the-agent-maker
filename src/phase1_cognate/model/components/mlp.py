"""
SwiGLU MLP - Swish-Gated Linear Unit Multi-Layer Perceptron

Efficient MLP architecture using SwiGLU activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP with 4x expansion"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize SwiGLU MLP.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4x d_model)
            dropout: Dropout rate
        """
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation: gate * up -> down.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Output tensor of same shape
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))

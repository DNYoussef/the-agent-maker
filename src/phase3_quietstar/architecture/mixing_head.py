"""
Phase 3 Quiet-STaR Mixing Head

Attention-based integration of thoughts with base representation.
Uses 8 attention heads with gating to blend base + thought hiddens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingHead(nn.Module):
    """
    Attention-based integration of thoughts with base representation.

    Uses 8 attention heads with gating to blend base + thought hiddens.
    Includes residual connection and layer normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Multi-head attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        base_hidden: torch.Tensor,
        thought_hiddens: torch.Tensor,
        coherence_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix thoughts with base representation.

        Args:
            base_hidden: (batch, hidden_size)
            thought_hiddens: (batch, num_thoughts, hidden_size)
            coherence_scores: (batch, num_thoughts) - Attention weights

        Returns:
            mixed_hidden: (batch, hidden_size)
        """
        batch_size = base_hidden.size(0)

        # Query from base
        query = self.query(base_hidden).unsqueeze(1)

        # Keys and values from thoughts
        keys = self.key(thought_hiddens)
        values = self.value(thought_hiddens)

        # Reshape for multi-head attention
        query = self._split_heads(query)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scaled dot-product attention
        attention_logits = torch.matmul(query, keys.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply coherence scores as bias
        attention_logits = attention_logits + coherence_scores.unsqueeze(1).unsqueeze(1)

        # Attention weights
        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        context = torch.matmul(attention_weights, values)
        context = self._merge_heads(context).squeeze(1)

        # Output projection
        thought_representation = self.output(context)

        # Gating mechanism
        gate_input = torch.cat([base_hidden, thought_representation], dim=-1)
        gate_value = self.gate(gate_input)

        # Blend base + thoughts
        mixed = gate_value * thought_representation + (1 - gate_value) * base_hidden

        # Residual + layer norm
        output = self.layer_norm(base_hidden + mixed)

        return output

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split into multiple heads."""
        batch_size = x.size(0)
        seq_len = x.size(1)
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge multiple heads."""
        batch_size = x.size(0)
        seq_len = x.size(2)
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

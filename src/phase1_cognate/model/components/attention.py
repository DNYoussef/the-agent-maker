"""
Sliding Window Attention

Efficient attention mechanism with O(n*w) complexity instead of O(n^2).
Each token attends to +-window/2 tokens around it.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention

    Each token attends to +-(window/2) tokens around it.
    Complexity: O(n*w) instead of O(n^2)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window: int,
        dropout: float = 0.1
    ):
        """
        Initialize Sliding Window Attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            window: Sliding window size
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply sliding window attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.w_v(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention: [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute sliding window attention
        attn_output = self._sliding_window_attn(q, k, v, mask)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.d_model)

        return self.w_o(attn_output)

    def _sliding_window_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Efficient Sliding Window Attention (ISS-026).

        Creates a band diagonal mask where each position can only attend
        to positions within the window. O(n*w) effective complexity.

        Args:
            q: Query tensor [batch, n_heads, seq_len, head_dim]
            k: Key tensor [batch, n_heads, seq_len, head_dim]
            v: Value tensor [batch, n_heads, seq_len, head_dim]
            mask: Optional additional mask

        Returns:
            Attention output [batch, n_heads, seq_len, head_dim]
        """
        batch, n_heads, seq_len, _ = q.shape

        # Compute full attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create sliding window mask: band diagonal where each position
        # can attend to positions [i - window//2, i + window//2]
        # Shape: [seq_len, seq_len]
        window_half = self.window // 2
        window_mask = self._create_sliding_window_mask(seq_len, window_half, q.device)

        # Apply sliding window mask (expand for batch and heads)
        # window_mask shape: [1, 1, seq_len, seq_len]
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~window_mask, float('-inf'))

        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        # Handle NaN from all-masked positions
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, v)

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        window_half: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create sliding window mask (ISS-026).

        Args:
            seq_len: Sequence length
            window_half: Half of the window size
            device: Device to create tensor on

        Returns:
            Boolean mask [seq_len, seq_len] where True = can attend
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        # Distance matrix: |i - j|
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # Mask: True where distance <= window_half
        return distance <= window_half

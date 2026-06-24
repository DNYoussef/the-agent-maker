"""
Sliding Window Attention

Efficient attention mechanism with O(n*w) complexity instead of O(n^2).
Each token attends to +-window/2 tokens around it.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention

    Each token attends to +-(window/2) tokens around it.
    Complexity: O(n*w) instead of O(n^2)
    """

    def __init__(
        self, d_model: int, n_heads: int, window: int, dropout: float = 0.1, causal: bool = True
    ):
        """
        Initialize Sliding Window Attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            window: Sliding window size
            dropout: Dropout rate
            causal: If True (default, for LM use), a position attends only to itself and
                the prior window_half tokens - never to the future. False = symmetric
                (bidirectional) window. A causal LM MUST keep this True or it leaks
                future tokens into the next-token prediction.
        """
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]
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
        batch, n_heads, seq_len, head_dim = q.shape
        window_half = self.window // 2
        output = torch.zeros(batch, n_heads, seq_len, head_dim, device=q.device, dtype=q.dtype)

        for pos in range(seq_len):
            start = max(0, pos - window_half)
            # Causal: attend only up to and including the current position (no future).
            # Bidirectional: also attend to the next window_half tokens.
            end = pos + 1 if self.causal else min(seq_len, pos + window_half + 1)

            q_pos = q[:, :, pos : pos + 1, :]
            k_local = k[:, :, start:end, :]
            v_local = v[:, :, start:end, :]

            scores = torch.matmul(q_pos, k_local.transpose(-2, -1)) * self.scale

            if mask is not None:
                local_mask = self._slice_attention_mask(mask, pos, start, end, batch, n_heads)
                scores = scores.masked_fill(local_mask == 0, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)

            output[:, :, pos : pos + 1, :] = torch.matmul(attn_weights, v_local)

        return output

    def _slice_attention_mask(
        self,
        mask: torch.Tensor,
        position: int,
        start: int,
        end: int,
        batch: int,
        n_heads: int,
    ) -> torch.Tensor:
        """Return the mask slice for one query position and local key window."""
        if mask.dim() == 2:
            # A 2-D mask must be [seq, seq]. A non-square 2-D mask is almost certainly a
            # [batch, seq] key-padding mask, which would be silently mis-sliced here;
            # reject it with a clear message instead (Codex #1).
            if mask.shape[0] != mask.shape[1]:
                raise ValueError(
                    f"2-D attention mask must be square [seq, seq]; got {tuple(mask.shape)}. "
                    "Supported shapes: [seq, seq], [batch, seq, seq], [batch, heads, seq, seq] "
                    "(a [batch, seq] key-padding mask is not supported - broadcast it to one of these)."
                )
            local = mask[position : position + 1, start:end]
            return local.view(1, 1, 1, end - start)
        if mask.dim() == 3:
            local = mask[:, position : position + 1, start:end]
            return local.view(batch, 1, 1, end - start)
        if mask.dim() == 4:
            return mask[:, :, position : position + 1, start:end]

        raise ValueError(
            "mask must have shape [seq, seq], [batch, seq, seq], or [batch, heads, seq, seq]"
        )

    def _create_sliding_window_mask(
        self, seq_len: int, window_half: int, device: torch.device
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

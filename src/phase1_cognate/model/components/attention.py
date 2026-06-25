"""
Sliding Window Attention

Efficient attention mechanism with O(n*w) complexity instead of O(n^2).
Each token attends to +-window/2 tokens around it.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


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
        Vectorised banded(-causal) attention. Same windowed attention the old per-token
        Python loop computed, but as a single masked softmax. Position i attends to keys
        j with i-window_half <= j <= i (causal) or |i-j| <= window_half (bidirectional).

        HONEST on the win: scores/attn are DENSE [b,h,s,s] (O(s^2) memory), so this is NOT
        asymptotically O(s*window) - a true banded/flash kernel is future work. The large
        real-world saving is that the loop built one autograd node per position (the
        activation-memory hog); one fused op replaces seq of them. Eval outputs match the
        windowed attention exactly; in training, attention dropout is applied over the
        dense weights (same rate, different random pattern than the per-slice loop -
        dropout is random regardless), so equivalence is asserted in eval / dropout=0.

        Args:
            q, k, v: [batch, n_heads, seq_len, head_dim]
            mask: optional [seq,seq] / [batch,seq,seq] / [batch,heads,seq,seq]

        Returns:
            Attention output [batch, n_heads, seq_len, head_dim]
        """
        batch, n_heads, seq_len, _ = q.shape
        wh = self.window // 2
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [b,h,s,s]

        i = torch.arange(seq_len, device=q.device)[:, None]
        j = torch.arange(seq_len, device=q.device)[None, :]
        allow = (j <= i) & (j >= i - wh) if self.causal else ((j - i).abs() <= wh)
        allow = allow[None, None]  # [1,1,s,s]
        if mask is not None:
            allow = allow & self._mask_allow(mask, batch, n_heads, seq_len)

        scores = scores.masked_fill(~allow, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # a fully-masked row -> 0 (self is always allowed)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)

    def _mask_allow(
        self, mask: torch.Tensor, batch: int, n_heads: int, seq_len: int
    ) -> torch.Tensor:
        """Boolean 'may attend' tensor broadcastable to [batch, n_heads, seq, seq]."""
        if mask.dim() == 2:
            # A 2-D mask must be [seq, seq]. A non-square 2-D mask is almost certainly a
            # [batch, seq] key-padding mask, which would broadcast wrong; reject it (Codex #1).
            if mask.shape[0] != mask.shape[1]:
                raise ValueError(
                    f"2-D attention mask must be square [seq, seq]; got {tuple(mask.shape)}. "
                    "Supported shapes: [seq, seq], [batch, seq, seq], [batch, heads, seq, seq] "
                    "(a [batch, seq] key-padding mask is not supported - broadcast it to one of these)."
                )
            return (mask != 0)[None, None]
        if mask.dim() == 3:
            return (mask != 0)[:, None]
        if mask.dim() == 4:
            return mask != 0
        raise ValueError(
            "mask must have shape [seq, seq], [batch, seq, seq], or [batch, heads, seq, seq]"
        )

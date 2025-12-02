"""
Titans-MAG Backbone Implementation

8-layer transformer with:
- Sliding Window Attention (O(n*w) complexity)
- Long-range Memory Module (LMM) with factorized projections
- MAG Gate for memory-augmented output
- RMSNorm + SwiGLU MLP

Target: ~20M params for backbone (leaves 5M for TRM wrapper + heads)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .components import LongTermMemory, MAGGate, RMSNorm, SlidingWindowAttention, SwiGLUMLP
from .model_config import TitansMAGConfig


class TitansMAGLayer(nn.Module):
    """Single Titans-MAG transformer layer"""

    def __init__(self, config: TitansMAGConfig):
        """
        Initialize Titans-MAG layer.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Pre-LayerNorm
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        # Sliding Window Attention
        self.attention = SlidingWindowAttention(
            config.d_model, config.n_heads, config.sw_window, config.attention_dropout
        )

        # SwiGLU MLP
        self.mlp = SwiGLUMLP(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Titans-MAG layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Pre-LayerNorm attention
        x = x + self.attention(self.norm1(x), mask)

        # Pre-LayerNorm MLP
        x = x + self.mlp(self.norm2(x))

        return x


class TitansMAGBackbone(nn.Module):
    """
    Complete Titans-MAG Backbone

    8-layer transformer + LMM + MAG gate.
    Target: ~20M params
    """

    def __init__(self, config: TitansMAGConfig):
        """
        Initialize Titans-MAG backbone.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings (learned)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # 8 transformer layers
        self.layers = nn.ModuleList([TitansMAGLayer(config) for _ in range(config.n_layers)])

        # Long-term memory
        self.ltm = LongTermMemory(config.d_model, config.d_mem, config.memory_decay)

        # MAG gate
        self.mag_gate = MAGGate(config.d_model, config.mag_hidden, config.mag_entropy_reg)

        # Final norm
        self.norm = RMSNorm(config.d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Titans-MAG backbone.

        Args:
            input_ids: Token IDs [batch, seq_len]
            mask: Optional attention mask

        Returns:
            output: Final hidden states [batch, seq_len, d_model]
            loss_gate: Scalar MAG gate entropy loss
        """
        batch, seq_len = input_ids.shape

        # Clamp sequence length to max_seq_len
        if seq_len > self.config.max_seq_len:
            print(
                f"WARNING: seq_len ({seq_len}) > max_seq_len ({self.config.max_seq_len}), truncating"
            )
            input_ids = input_ids[:, : self.config.max_seq_len]
            if mask is not None:
                mask = mask[:, : self.config.max_seq_len]
            batch, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.token_emb(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.pos_emb(pos_ids).unsqueeze(0).expand(batch, -1, -1)
        x = token_emb + pos_emb

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Normalize
        y = self.norm(x)

        # Get memory contribution
        m = self.ltm(y)

        # MAG gate (blend y and m)
        output, loss_gate = self.mag_gate(y, m)

        return output, loss_gate

    def reset_memory(self) -> None:
        """Reset LTM state (call between batches)"""
        self.ltm.reset_memory()

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""
Long-Term Memory Module (LMM)

Factorized long-range memory with exponential decay.
Compresses state to d_mem dimension, maintains memory, expands back.
"""

import torch
import torch.nn as nn


class LongTermMemory(nn.Module):
    """
    Factorized Long-range Memory Module (LMM)

    Compresses state to d_mem dimension, maintains exponentially-
    decayed memory, expands back to d_model.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        decay: float = 0.99
    ):
        """
        Initialize Long-Term Memory.

        Args:
            d_model: Model dimension
            d_mem: Memory dimension (compression size)
            decay: Exponential decay factor (0-1)
        """
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.decay = decay

        # Factorized projections
        self.w_down = nn.Linear(d_model, d_mem, bias=False)
        self.w_up = nn.Linear(d_mem, d_model, bias=False)

        # Memory state (not a parameter, updated during forward)
        self.register_buffer("memory_state", torch.zeros(1, 1, d_mem))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through long-term memory.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Memory contribution [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # Compress to memory dimension
        x_compressed = self.w_down(x)

        # Initialize local memory state (batch-independent)
        # Start from global memory_state, broadcast to batch size
        memory = self.memory_state.expand(batch, -1, -1).clone()

        # Update memory with exponential decay (local to this forward pass)
        m_list = []
        for t in range(seq_len):
            # Decay previous memory and add current
            memory = (
                self.decay * memory +
                (1 - self.decay) * x_compressed[:, t:t+1, :]
            )
            m_list.append(memory)

        # Update global memory_state (average across batch, detached)
        # This maintains memory across sequences without batch dependency
        self.memory_state = memory.mean(dim=0, keepdim=True).detach()

        # Stack and expand back to d_model
        m_compressed = torch.cat(m_list, dim=1)
        m = self.w_up(m_compressed)

        return m

    def reset_memory(self):
        """Reset memory state (call between batches)"""
        self.memory_state.zero_()

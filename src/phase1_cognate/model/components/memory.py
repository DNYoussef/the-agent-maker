"""
Long-Term Memory Module (LMM)

Factorized long-range memory with exponential decay.
Compresses state to d_mem dimension, maintains memory, expands back.

HONESTY (do not overclaim): this is an EMA-decayed memory pool, NOT the
"Titans" neural long-term memory of Behrouz et al. (2024). Real Titans updates
its memory by gradient descent at test time, gated by a "surprise" signal
(gradient of the loss w.r.t. the memory) with momentum and a forget gate. This
module has none of that - it is a fixed linear recurrence
(memory_t = decay*memory_{t-1} + (1-decay)*W_down x_t) read out through W_up.
The surrounding backbone is still named "Titans-MAG" for back-compat; a real
surprise-based memory is a Wave-2 feature.
"""

import torch
import torch.nn as nn


class LongTermMemory(nn.Module):
    """
    Factorized Long-range Memory Module (LMM)

    Compresses state to d_mem dimension, maintains exponentially-
    decayed memory, expands back to d_model.
    """

    def __init__(self, d_model: int, d_mem: int, decay: float = 0.99):
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

        # STATELESS across forwards: each call starts from zero memory. Carrying the
        # final memory_state into the next forward (the old behavior) leaked one batch's
        # content into the next and made outputs depend on batch order / call history
        # (Codex #10/#11). The within-sequence EMA scan below is unchanged - it is the
        # actual long-range memory; cross-sequence memory is a Wave-2 feature.
        memory = torch.zeros(batch, 1, self.d_mem, device=x.device, dtype=x_compressed.dtype)

        # Update memory with exponential decay (causal scan within this sequence)
        m_list = []
        for t in range(seq_len):
            # Decay previous memory and add current
            memory = self.decay * memory + (1 - self.decay) * x_compressed[:, t : t + 1, :]
            m_list.append(memory)

        # Stack and expand back to d_model
        m_compressed = torch.cat(m_list, dim=1)
        m = self.w_up(m_compressed)

        return m

    def reset_memory(self) -> None:
        """Reset memory state (call between batches)"""
        self.memory_state.zero_()

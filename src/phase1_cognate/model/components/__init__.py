"""
Titans-MAG Components Module

Re-exports all component classes for backward compatibility.
"""

from .normalization import RMSNorm
from .mlp import SwiGLUMLP
from .attention import SlidingWindowAttention
from .memory import LongTermMemory
from .gating import MAGGate

__all__ = [
    'RMSNorm',
    'SwiGLUMLP',
    'SlidingWindowAttention',
    'LongTermMemory',
    'MAGGate',
]

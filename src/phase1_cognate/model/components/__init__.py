"""
Titans-MAG Components Module

Re-exports all component classes for backward compatibility.
"""

from .attention import SlidingWindowAttention
from .gating import MAGGate
from .memory import LongTermMemory
from .mlp import SwiGLUMLP
from .normalization import RMSNorm

__all__ = [
    "RMSNorm",
    "SwiGLUMLP",
    "SlidingWindowAttention",
    "LongTermMemory",
    "MAGGate",
]

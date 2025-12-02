"""
Phase 1 Cognate Model Package

TRM × Titans-MAG Architecture:
- Titans-MAG Backbone: 8-layer transformer with Sliding Window Attention
- TRM Wrapper: Multi-pass recursive reasoning
- ACT Head: Adaptive Computation Time
- Target: ~25M parameters, fits in 6GB VRAM
"""

from .model_config import TitansMAGConfig, TRMConfig, Phase1Config
from .titans_mag import TitansMAGBackbone, MAGGate, LongTermMemory
from .trm_wrapper import TRMWrapper
from .act_head import ACTHead
from .full_model import TRMTitansMAGModel

__all__ = [
    "TitansMAGConfig",
    "TRMConfig",
    "Phase1Config",
    "TitansMAGBackbone",
    "MAGGate",
    "LongTermMemory",
    "TRMWrapper",
    "ACTHead",
    "TRMTitansMAGModel",
]

"""
Phase 1 Cognate Model Package

TRM × Titans-MAG Architecture:
- Titans-MAG Backbone: 8-layer transformer with Sliding Window Attention
- TRM Wrapper: Multi-pass recursive reasoning
- ACT Head: Adaptive Computation Time
- Target: ~25M parameters, fits in 6GB VRAM
"""

from .act_head import ACTHead
from .full_model import TRMTitansMAGModel
from .model_config import Phase1Config, TitansMAGConfig, TRMConfig
from .titans_mag import LongTermMemory, MAGGate, TitansMAGBackbone
from .trm_wrapper import TRMWrapper

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

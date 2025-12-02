"""
MuGrokfast Optimizer Package

Unified optimizer combining Grokfast and Muon techniques.
"""

from .optimizer import MuonGrokfast
from .config import MuGrokConfig

__all__ = ["MuonGrokfast", "MuGrokConfig"]

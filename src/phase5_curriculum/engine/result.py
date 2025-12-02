"""
Result dataclass for Phase 5 curriculum learning.

Contains final model, metrics, and artifacts from curriculum training.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch.nn as nn

from .config import SpecializationType


@dataclass
class Phase5Result:
    """Result of Phase 5 curriculum learning."""
    success: bool
    model: nn.Module
    specialization: SpecializationType
    levels_completed: int
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error: Optional[str] = None

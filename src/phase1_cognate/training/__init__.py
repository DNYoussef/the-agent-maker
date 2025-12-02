"""
Phase 1 Training Pipeline

Complete training system with:
- MuGrokfast optimizer integration
- 3-stage curriculum learning
- W&B logging (37 metrics)
- Model checkpointing
- VRAM monitoring
"""

from .trainer import Phase1Trainer, TrainingConfig
from .wandb_logger import Phase1WandBLogger

__all__ = [
    "Phase1Trainer",
    "TrainingConfig",
    "Phase1WandBLogger",
]

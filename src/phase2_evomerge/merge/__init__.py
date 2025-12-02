"""
Merge Techniques Module

Implements 6 research-validated model merging techniques organized into
3 mutually exclusive pairs for binary pairing strategy (2³ = 8 combinations).

Binary Pairing Strategy:
    Bit 0: Interpolation (0=Linear, 1=SLERP)
    Bit 1: Task Arithmetic (0=DARE, 1=TIES)
    Bit 2: Selection (0=FrankenMerge, 1=DFS)

Example:
    Binary 000 → Linear + DARE + FrankenMerge
    Binary 111 → SLERP + TIES + DFS (typically best performer)

Usage:
    >>> from phase2_evomerge.merge import MergeTechniques
    >>> merger = MergeTechniques()
    >>> models = [model1, model2, model3]
    >>> # Apply binary combo 7 (111)
    >>> result = merger.apply_combo(models, combo_id=7)
"""

from typing import List

import torch.nn as nn

from .dare_merge import DAREMerge
from .dfs_merge import DFSMerge
from .frankenmerge import FrankenMerge
from .linear_merge import LinearMerge
from .slerp_merge import SLERPMerge
from .ties_merge import TIESMerge


class MergeTechniques:
    """
    Unified API for applying merge techniques via binary combinations.

    The binary pairing strategy uses 3 bits to encode 8 unique combinations
    of merge techniques, applied sequentially in a 3-stage pipeline.
    """

    def __init__(self):
        """Initialize all 6 merge technique implementations."""
        self.linear = LinearMerge()
        self.slerp = SLERPMerge()
        self.dare = DAREMerge()
        self.ties = TIESMerge()
        self.frankenmerge = FrankenMerge()
        self.dfs = DFSMerge()

    def apply_combo(self, models: List[nn.Module], combo_id: int) -> nn.Module:
        """
        Apply 3-stage sequential merge pipeline for given binary combo.

        Args:
            models: List of 3 models to merge
            combo_id: Binary combination ID (0-7)

        Returns:
            Merged model

        Raises:
            ValueError: If combo_id not in range [0, 7]
            ValueError: If models list length != 3
        """
        if combo_id < 0 or combo_id > 7:
            raise ValueError(f"combo_id must be 0-7, got {combo_id}")
        if len(models) != 3:
            raise ValueError(f"Expected 3 models, got {len(models)}")

        # Decode binary combination
        bit0 = (combo_id >> 0) & 1  # Interpolation
        bit1 = (combo_id >> 1) & 1  # Task arithmetic
        bit2 = (combo_id >> 2) & 1  # Selection

        # Stage 1: Interpolation (combines 3 → 1)
        stage1 = self.slerp.merge(models) if bit0 else self.linear.merge(models)

        # Stage 2: Task Arithmetic (refines merged model)
        base_model = models[0]  # Use first model as base
        stage2 = self.ties.merge(stage1, models) if bit1 else self.dare.merge(stage1, base_model)

        # Stage 3: Selection (final refinement)
        stage3 = self.dfs.merge(stage2, models) if bit2 else self.frankenmerge.merge(stage2, models)

        # Tag with combo_id for tracking
        stage3.combo_id = combo_id

        return stage3

    def decode_combo(self, combo_id: int) -> str:
        """
        Decode binary combo ID to human-readable technique names.

        Args:
            combo_id: Binary combination ID (0-7)

        Returns:
            String like "SLERP + TIES + DFS"
        """
        bit0 = (combo_id >> 0) & 1
        bit1 = (combo_id >> 1) & 1
        bit2 = (combo_id >> 2) & 1

        interp = "SLERP" if bit0 else "Linear"
        task = "TIES" if bit1 else "DARE"
        select = "DFS" if bit2 else "Franken"

        return f"{interp} + {task} + {select}"


__all__ = [
    "MergeTechniques",
    "LinearMerge",
    "SLERPMerge",
    "DAREMerge",
    "TIESMerge",
    "FrankenMerge",
    "DFSMerge",
]

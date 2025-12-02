"""
TIES (TrIm, Elect Sign, Merge) Technique

TIES merges models by:
    1. TRIM: Keep only top 20% magnitude parameters
    2. ELECT: Vote on sign (+/-) for each parameter across models
    3. MERGE: Average only parameters with matching elected sign

This resolves sign conflicts between models and focuses on important parameters.

Algorithm:
    1. For each parameter:
        - Keep top 20% by magnitude (trim)
        - Vote on sign: sign(mean(deltas))
        - Merge: average params where sign matches elected sign

Benefits:
    - Resolves sign conflicts intelligently
    - Focuses on important (high-magnitude) parameters
    - 5-15% performance improvement over naive merging

Research:
    - Yadav et al., "TIES-Merging: Resolving Interference" (NeurIPS 2023)
"""

from typing import List
import copy
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class TIESMerge:
    """
    TrIm, Elect Sign, and Merge (TIES) technique.

    This technique is designed to handle conflicting updates between models
    by trimming unimportant parameters, voting on signs, and only merging
    parameters that agree with the elected sign.
    """

    def __init__(self, trim_percent: float = 0.2):
        """
        Initialize TIES merge.

        Args:
            trim_percent: Fraction of parameters to keep by magnitude (default: 0.2)
        """
        self.trim_percent = trim_percent

    def merge(
        self, model_target: nn.Module, models_ref: List[nn.Module]
    ) -> nn.Module:
        """
        Merge target model with reference models using TIES.

        Args:
            model_target: Target model (from Stage 2, or base model)
            models_ref: Reference models to merge (typically 3 original models)

        Returns:
            New model with TIES-merged parameters

        Raises:
            ValueError: If models have incompatible architectures
        """
        if not models_ref:
            raise ValueError("models_ref cannot be empty")

        # Verify compatibility
        for model in models_ref:
            if not self._check_compatibility(model_target, model):
                raise ValueError(
                    "All models must have same architecture for TIES merge"
                )

        # Create result as copy of target
        result_model = copy.deepcopy(model_target)

        with torch.no_grad():
            for param_name, target_param in model_target.named_parameters():
                # Get reference parameters
                ref_params = [
                    dict(m.named_parameters())[param_name] for m in models_ref
                ]

                # Compute deltas from target
                deltas = [ref_param - target_param for ref_param in ref_params]

                # Step 1: TRIM - Keep only top k% by magnitude
                trimmed_deltas = self._trim_deltas(deltas, self.trim_percent)

                # Step 2: ELECT - Vote on sign
                elected_sign = self._elect_sign(trimmed_deltas)

                # Step 3: MERGE - Average params with matching sign
                merged_delta = self._merge_with_elected_sign(
                    trimmed_deltas, elected_sign
                )

                # Apply merged delta to result
                result_param = dict(result_model.named_parameters())[param_name]
                result_param.copy_(target_param + merged_delta)

        return result_model

    def _trim_deltas(
        self, deltas: List[torch.Tensor], keep_percent: float
    ) -> List[torch.Tensor]:
        """
        Trim deltas to keep only top k% by magnitude.

        Args:
            deltas: List of delta tensors
            keep_percent: Fraction to keep (e.g., 0.2 = top 20%)

        Returns:
            List of trimmed delta tensors
        """
        trimmed = []
        for delta in deltas:
            # Compute magnitude
            magnitude = torch.abs(delta)

            # Find threshold (top k%)
            flat_mag = magnitude.flatten()
            k = max(1, int(len(flat_mag) * keep_percent))
            threshold = torch.topk(flat_mag, k).values[-1]

            # Create mask for values above threshold
            mask = magnitude >= threshold

            # Apply mask
            trimmed_delta = torch.where(
                mask, delta, torch.zeros_like(delta)
            )
            trimmed.append(trimmed_delta)

        return trimmed

    def _elect_sign(self, deltas: List[torch.Tensor]) -> torch.Tensor:
        """
        Vote on sign (+/-) for each parameter position.

        Args:
            deltas: List of delta tensors (after trimming)

        Returns:
            Tensor of elected signs (-1, 0, or +1)
        """
        # Stack deltas
        stacked = torch.stack(deltas, dim=0)

        # Compute mean sign
        # Note: sign() returns -1, 0, or +1
        # We use mean to get majority vote
        signs = torch.sign(stacked)
        elected = torch.sign(torch.mean(signs, dim=0))

        return elected

    def _merge_with_elected_sign(
        self, deltas: List[torch.Tensor], elected_sign: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge deltas, keeping only those matching elected sign.

        Args:
            deltas: List of delta tensors (after trimming)
            elected_sign: Elected sign tensor (-1, 0, or +1)

        Returns:
            Merged delta tensor
        """
        # Collect deltas that match elected sign
        matching_deltas = []

        for delta in deltas:
            delta_sign = torch.sign(delta)

            # Keep only where signs match
            # (elected_sign == 0 means no consensus, keep nothing)
            match_mask = (delta_sign == elected_sign) & (elected_sign != 0)

            matching_delta = torch.where(
                match_mask, delta, torch.zeros_like(delta)
            )
            matching_deltas.append(matching_delta)

        # Average matching deltas
        if matching_deltas:
            stacked = torch.stack(matching_deltas, dim=0)
            # Average only non-zero values
            sum_deltas = torch.sum(stacked, dim=0)
            count_nonzero = torch.sum(stacked != 0, dim=0).clamp(min=1)
            merged = sum_deltas / count_nonzero
        else:
            merged = torch.zeros_like(deltas[0])

        return merged

    def _check_compatibility(
        self, model1: nn.Module, model2: nn.Module
    ) -> bool:
        """
        Check if two models have compatible architectures.

        Args:
            model1: First model
            model2: Second model

        Returns:
            True if compatible, False otherwise
        """
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())

        # Check same parameter names
        if set(params1.keys()) != set(params2.keys()):
            return False

        # Check same shapes
        for name in params1.keys():
            if params1[name].shape != params2[name].shape:
                return False

        return True


__all__ = ["TIESMerge"]

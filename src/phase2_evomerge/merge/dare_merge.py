"""
DARE (Drop And REscale) Merge Technique

DARE merges models by computing the delta (fine-tuned - base), randomly
dropping 90% of the delta parameters, then rescaling the remaining 10% by 10×.
This creates a sparse update that reduces task interference.

Algorithm:
    1. delta = model_finetuned - model_base
    2. mask = bernoulli(p=0.1)  # Keep 10%
    3. sparse_delta = delta * mask
    4. rescaled_delta = sparse_delta * 10
    5. result = model_base + rescaled_delta

Benefits:
    - Reduces interference between tasks
    - Can eliminate 90-99% of parameters without loss
    - Best for SFT (Supervised Fine-Tuning) models with small deltas

Research:
    - Yu et al., "Language Models are Super Mario" (arXiv 2024)
    - Shows 90% sparsity maintains performance
"""

import copy
import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DAREMerge:
    """
    Drop And REscale (DARE) merge technique.

    This technique is particularly effective when merging fine-tuned models
    back with their base model, as it creates sparse updates that maintain
    performance while reducing parameter interference.
    """

    def __init__(self, drop_rate: float = 0.9, rescale_factor: float = None):
        """
        Initialize DARE merge.

        Args:
            drop_rate: Fraction of delta parameters to drop (default: 0.9)
            rescale_factor: Factor to rescale remaining params. If None,
                uses 1/(1-drop_rate) for unbiased estimate
        """
        self.drop_rate = drop_rate
        self.keep_rate = 1.0 - drop_rate
        self.rescale_factor = rescale_factor if rescale_factor is not None else 1.0 / self.keep_rate

    def merge(self, model_finetuned: nn.Module, model_base: nn.Module) -> nn.Module:
        """
        Merge fine-tuned model with base using DARE.

        Args:
            model_finetuned: Fine-tuned model (or merged model from Stage 1)
            model_base: Base model to compute delta against

        Returns:
            New model with sparse delta applied

        Raises:
            ValueError: If models have incompatible architectures
        """
        # Verify models have same architecture
        if not self._check_compatibility(model_finetuned, model_base):
            raise ValueError("Models must have same architecture for DARE merge")

        # Create result as copy of base
        result_model = copy.deepcopy(model_base)

        with torch.no_grad():
            for param_name, base_param in model_base.named_parameters():
                try:
                    finetuned_param = dict(model_finetuned.named_parameters())[param_name]
                    result_param = dict(result_model.named_parameters())[param_name]
                except KeyError:
                    logger.warning(
                        f"Parameter {param_name} not found in fine-tuned model, skipping"
                    )
                    continue

                # Compute delta
                delta = finetuned_param - base_param

                # Create random mask (keep 10%)
                mask = torch.bernoulli(torch.full_like(delta, self.keep_rate)).bool()

                # Apply mask and rescale
                sparse_delta = torch.where(
                    mask, delta * self.rescale_factor, torch.zeros_like(delta)
                )

                # Apply to result
                result_param.copy_(base_param + sparse_delta)

        return result_model

    def _check_compatibility(self, model1: nn.Module, model2: nn.Module) -> bool:
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


__all__ = ["DAREMerge"]

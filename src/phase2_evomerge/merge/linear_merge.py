"""
Linear Merge Technique

Simple weighted average of model parameters. This is the baseline merging
technique that always works and produces predictable results.

Algorithm:
    merged = (1/n) * sum(models)

For 3 models:
    merged = 0.333 * model1 + 0.333 * model2 + 0.333 * model3

Research:
    - Model Soups (Wortsman et al., 2022)
    - Averaging Weights Leads to Wider Optima (Izmailov et al., 2018)
"""

from typing import List
import copy
import torch
import torch.nn as nn


class LinearMerge:
    """
    Linear interpolation (weighted average) of model parameters.

    This technique merges models by computing a simple average of their
    parameters, which is equivalent to finding a point in the center of
    the model weight space.
    """

    def merge(self, models: List[nn.Module]) -> nn.Module:
        """
        Merge models using weighted average.

        Args:
            models: List of models to merge (typically 3)

        Returns:
            New model with averaged parameters

        Raises:
            ValueError: If models list is empty
        """
        if not models:
            raise ValueError("Cannot merge empty list of models")

        # Create a copy of the first model to use as the base
        merged_model = copy.deepcopy(models[0])
        n_models = len(models)
        weight = 1.0 / n_models

        # Average all parameters
        with torch.no_grad():
            for param_name, merged_param in merged_model.named_parameters():
                # Start with zeros
                merged_param.zero_()

                # Add weighted contribution from each model
                for model in models:
                    param = dict(model.named_parameters())[param_name]
                    merged_param.add_(param * weight)

        return merged_model


__all__ = ["LinearMerge"]

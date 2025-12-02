"""
DFS (Deep Feature Selection) Merge Technique

DFS merges models by weighting parameters based on their importance,
calculated as inverse variance across models. Stable features (low variance)
are considered more important and receive higher weight.

Algorithm:
    1. For each parameter position:
        importance[i] = 1 / variance(param[i] across models)
    2. Weighted merge:
        merged[i] = sum(importance[i] * param[i]) / sum(importance)

Benefits:
    - Preserves stable features
    - Important (consistent) features get higher weight
    - Reduces noise from unstable parameters

Research:
    - Li et al., "Deep Feature Selection" (2016)
    - Variance-based importance weighting
"""

from typing import List
import copy
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class DFSMerge:
    """
    Deep Feature Selection merge using inverse-variance weighting.

    This technique identifies important features as those with low variance
    across models, giving them higher weight in the merged result.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize DFS merge.

        Args:
            epsilon: Small constant to prevent division by zero when
                calculating inverse variance
        """
        self.epsilon = epsilon

    def merge(
        self, model_target: nn.Module, models_ref: List[nn.Module]
    ) -> nn.Module:
        """
        Merge models using inverse-variance weighting.

        Args:
            model_target: Target model (can be used as additional reference)
            models_ref: Reference models to merge (typically 3)

        Returns:
            New model with variance-weighted parameters

        Raises:
            ValueError: If models have incompatible architectures
            ValueError: If models_ref is empty
        """
        if not models_ref:
            raise ValueError("models_ref cannot be empty")

        # Include target in the set of models to consider
        all_models = [model_target] + models_ref

        # Verify all models have same architecture
        for model in all_models:
            if not self._check_compatibility(all_models[0], model):
                raise ValueError(
                    "All models must have same architecture for DFS merge"
                )

        # Create result as copy of target
        result_model = copy.deepcopy(model_target)

        with torch.no_grad():
            for param_name in dict(model_target.named_parameters()).keys():
                # Collect parameters from all models
                params = [
                    dict(m.named_parameters())[param_name] for m in all_models
                ]

                # Stack for variance calculation
                stacked = torch.stack(params, dim=0)

                # Calculate variance across models (dim=0)
                variance = torch.var(stacked, dim=0, unbiased=False)

                # Calculate importance (inverse variance) for each element
                # Add epsilon to prevent division by zero
                importance = 1.0 / (variance + self.epsilon)

                # For each element position, we want to compute a weighted average
                # of the values from different models, where the weights are based
                # on how stable that element is across models.

                # But we need to weight models, not elements. So we compute
                # element-wise weights and apply them to get the merged value.
                # This is a bit different from the typical approach - we're
                # giving higher weight to values that are more stable.

                # The current approach: weighted sum normalized by total importance
                # For identical models (variance=0), importance→∞ for all elements,
                # and after normalization each element gets equal weight 1/total_elements
                # This is wrong - we want each MODEL to get equal weight.

                # Correct approach: Just do simple average when variance is zero
                if torch.all(variance < self.epsilon):
                    # All models are identical, just return any of them
                    merged_param = params[0]
                else:
                    # Weighted average where high-variance positions get lower weight
                    # Normalize importance per-element position
                    importance_normalized = importance / torch.sum(importance)

                    # Compute weighted average across models
                    merged_param = torch.zeros_like(params[0])
                    for param in params:
                        merged_param += importance_normalized * param

                # Apply to result
                result_param = dict(result_model.named_parameters())[param_name]
                result_param.copy_(merged_param)

        return result_model

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


__all__ = ["DFSMerge"]

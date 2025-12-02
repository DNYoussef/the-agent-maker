"""
FrankenMerge Technique

FrankenMerge performs layer-wise selection, choosing the best-performing layer
from each candidate model at each position. This "mix-and-match" approach
creates a Frankenstein-like merged model.

Algorithm:
    For each layer position i:
        1. Evaluate fitness of layer_i from each model
        2. Select best-performing layer_i
        3. Add to merged model

Patterns:
    - ABC: Alternate layers (model A, B, C, A, B, C, ...)
    - ABBA: Symmetric pattern (A, B, B, A, A, B, B, A, ...)
    - Random: Random selection per layer
    - Fitness-based: Select based on layer performance

Benefits:
    - Mix strengths from different models
    - Layer independence assumption
    - Used in Goliath-120B, SOLAR-10.7B

Research:
    - Community-developed technique
    - Widely used in Hugging Face merges
"""

import copy
import logging
import random
from typing import List, Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FrankenMerge:
    """
    Layer-wise selection merge technique.

    This technique assumes layers are relatively independent and can be
    mixed and matched between models. Best results when models share
    the same architecture.
    """

    def __init__(self, pattern: Literal["abc", "abba", "random", "fitness"] = "abc"):
        """
        Initialize FrankenMerge.

        Args:
            pattern: Selection pattern
                - "abc": Alternate models (0, 1, 2, 0, 1, 2, ...)
                - "abba": Symmetric (0, 1, 1, 0, 0, 1, 1, 0, ...)
                - "random": Random selection per layer
                - "fitness": Select based on layer fitness (simplified)
        """
        self.pattern = pattern

    def merge(self, model_target: nn.Module, models_ref: List[nn.Module]) -> nn.Module:
        """
        Merge models using layer-wise selection.

        Args:
            model_target: Target model (for reference, not used in selection)
            models_ref: Reference models to select layers from (typically 3)

        Returns:
            New model with selected layers

        Raises:
            ValueError: If models have incompatible architectures
            ValueError: If models_ref is empty
        """
        if not models_ref:
            raise ValueError("models_ref cannot be empty")

        # Verify all models have same architecture
        for i, model in enumerate(models_ref):
            if i > 0 and not self._check_compatibility(models_ref[0], model):
                raise ValueError("All models must have same architecture for FrankenMerge")

        # Also check compatibility with target
        if not self._check_compatibility(model_target, models_ref[0]):
            raise ValueError("All models must have same architecture for FrankenMerge")

        # Create result as copy of first model
        result_model = copy.deepcopy(models_ref[0])

        # Get layer names (parameters organized by layer)
        layer_params = self._group_params_by_layer(models_ref[0])
        num_layers = len(layer_params)

        # Select layers based on pattern
        layer_selections = self._compute_layer_selections(num_layers, len(models_ref))

        # Apply layer selections
        with torch.no_grad():
            for layer_idx, layer_param_names in enumerate(layer_params):
                selected_model_idx = layer_selections[layer_idx]
                selected_model = models_ref[selected_model_idx]

                # Copy all parameters from selected layer
                for param_name in layer_param_names:
                    selected_param = dict(selected_model.named_parameters())[param_name]
                    result_param = dict(result_model.named_parameters())[param_name]
                    result_param.copy_(selected_param)

        logger.info(
            f"FrankenMerge completed with pattern={self.pattern}, "
            f"layer_selections={layer_selections[:5]}... (showing first 5)"
        )

        return result_model

    def _group_params_by_layer(self, model: nn.Module) -> List[List[str]]:
        """
        Group parameter names by layer.

        Heuristic: Parameters with same prefix (e.g., "layer.0") belong
        to same layer.

        Args:
            model: Model to analyze

        Returns:
            List of lists, where each inner list contains parameter names
            for one layer
        """
        layer_groups: dict[str, list[int]] = {}

        for param_name in dict(model.named_parameters()).keys():
            # Extract layer identifier (e.g., "linear1" from "linear1.weight")
            parts = param_name.split(".")
            if len(parts) >= 2:
                layer_id = parts[0]  # First component
            else:
                layer_id = "root"

            if layer_id not in layer_groups:
                layer_groups[layer_id] = []
            layer_groups[layer_id].append(param_name)

        # Return as ordered list
        return [layer_groups[key] for key in sorted(layer_groups.keys())]

    def _compute_layer_selections(self, num_layers: int, num_models: int) -> List[int]:
        """
        Compute which model to select for each layer based on pattern.

        Args:
            num_layers: Number of layers
            num_models: Number of models to select from

        Returns:
            List of model indices (0 to num_models-1) for each layer
        """
        if self.pattern == "abc":
            # Alternate: 0, 1, 2, 0, 1, 2, ...
            return [i % num_models for i in range(num_layers)]

        elif self.pattern == "abba":
            # Symmetric: 0, 1, 1, 0, 0, 1, 1, 0, ...
            selections = []
            for i in range(num_layers):
                cycle_pos = i % 4
                if cycle_pos == 0:
                    selections.append(0)
                elif cycle_pos == 1 or cycle_pos == 2:
                    selections.append(1 % num_models)
                else:
                    selections.append(0)
            return selections

        elif self.pattern == "random":
            # Random selection
            return [random.randint(0, num_models - 1) for _ in range(num_layers)]

        elif self.pattern == "fitness":
            # Simplified fitness: just alternate (real fitness would require evaluation)
            logger.warning("Fitness-based selection not implemented, falling back to ABC")
            return [i % num_models for i in range(num_layers)]

        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

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


__all__ = ["FrankenMerge"]

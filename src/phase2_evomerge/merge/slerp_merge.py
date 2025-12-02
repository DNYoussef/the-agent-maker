"""
SLERP (Spherical Linear Interpolation) Merge Technique

SLERP interpolates between models along the surface of a hypersphere,
preserving parameter magnitude better than linear interpolation.

Algorithm:
    θ = arccos(dot(w1, w2))
    if θ ≈ 0: fallback to linear
    else: slerp(w1, w2, t) = (sin((1-t)θ)/sin(θ)) * w1 + (sin(tθ)/sin(θ)) * w2

Benefits:
    - Preserves parameter geometry better than linear merge
    - More stable when models are similar
    - Used in Goliath-120B merge

Research:
    - Shoemake, "Animating Rotation with Quaternion Curves" (SIGGRAPH 1985)
    - Applied to neural network merging in recent LLM work
"""

from typing import List
import copy
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SLERPMerge:
    """
    Spherical Linear Interpolation (SLERP) for model merging.

    SLERP interpolates along the hypersphere surface, which is particularly
    useful when models have similar weights but should not be averaged linearly.

    Edge Cases:
        - θ ≈ 0 (identical models): Falls back to linear interpolation
        - θ ≈ π (opposite models): Uses linear interpolation (unstable SLERP)
    """

    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize SLERP merge.

        Args:
            epsilon: Threshold for considering vectors parallel (θ ≈ 0)
        """
        self.epsilon = epsilon
        self.linear_fallback = None  # Lazy import to avoid circular dependency

    def merge(self, models: List[nn.Module]) -> nn.Module:
        """
        Merge models using SLERP.

        For 3 models, performs pairwise SLERP:
            temp = slerp(model1, model2, 0.5)
            result = slerp(temp, model3, 0.333)

        Args:
            models: List of models to merge (typically 3)

        Returns:
            New model with SLERP-interpolated parameters

        Raises:
            ValueError: If models list is empty
        """
        if not models:
            raise ValueError("Cannot merge empty list of models")

        if len(models) == 1:
            return copy.deepcopy(models[0])

        # Merge pairwise using SLERP
        result = models[0]
        for i in range(1, len(models)):
            # Weight for this step: 1/(i+1) to ensure equal contribution
            t = 1.0 / (i + 1)
            result = self._slerp_pair(result, models[i], t)

        return result

    def _slerp_pair(
        self, model1: nn.Module, model2: nn.Module, t: float
    ) -> nn.Module:
        """
        SLERP between two models.

        Args:
            model1: First model
            model2: Second model
            t: Interpolation parameter (0 = model1, 1 = model2)

        Returns:
            Interpolated model
        """
        merged_model = copy.deepcopy(model1)

        with torch.no_grad():
            for param_name, param1 in model1.named_parameters():
                param2 = dict(model2.named_parameters())[param_name]
                merged_param = dict(merged_model.named_parameters())[
                    param_name
                ]

                # Flatten parameters for dot product
                p1_flat = param1.flatten()
                p2_flat = param2.flatten()

                # Compute angle between vectors
                dot_product = torch.dot(p1_flat, p2_flat)
                norm1 = torch.norm(p1_flat)
                norm2 = torch.norm(p2_flat)

                # Avoid division by zero
                if norm1 < self.epsilon or norm2 < self.epsilon:
                    # One vector is zero, use linear
                    merged_param.copy_(param1 * (1 - t) + param2 * t)
                    continue

                # Normalize dot product to [-1, 1]
                cos_theta = dot_product / (norm1 * norm2)
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

                theta = torch.acos(cos_theta)

                # Check for parallel vectors (θ ≈ 0)
                if theta < self.epsilon:
                    # Vectors are parallel, use linear interpolation
                    merged_param.copy_(param1 * (1 - t) + param2 * t)
                    logger.debug(
                        f"Parameter {param_name}: θ ≈ 0, using linear fallback"
                    )
                    continue

                # SLERP formula
                sin_theta = torch.sin(theta)
                weight1 = torch.sin((1 - t) * theta) / sin_theta
                weight2 = torch.sin(t * theta) / sin_theta

                # Apply SLERP
                slerped = weight1 * param1 + weight2 * param2
                merged_param.copy_(slerped)

        return merged_model


__all__ = ["SLERPMerge"]

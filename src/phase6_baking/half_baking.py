"""
Phase 6: Half-Baking Mechanism

Implements partial-strength baking for iterative optimization.
Allows gradual incorporation of baked behaviors.

Research: "Prompt Baking" (arXiv:2409.13697v1)
Key insight: Half-baking (50% strength) enables iterative refinement
without overwriting previous learning.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class HalfBakeConfig:
    """Configuration for half-baking."""

    strength: float = 0.5  # 0.0 = original, 1.0 = fully baked
    layer_wise: bool = False  # Apply different strengths per layer
    preserve_embeddings: bool = True  # Keep embedding layers unchanged
    preserve_layer_norms: bool = True  # Keep layer norms unchanged


class HalfBaker:
    """
    Half-Baking: Partial-strength prompt integration.

    Interpolates between original and baked model weights:
        W_half = (1 - strength) * W_original + strength * W_baked

    Benefits:
    - Prevents catastrophic forgetting
    - Enables iterative refinement
    - Balances multiple baked behaviors
    """

    def __init__(
        self,
        strength: float = 0.5,
        layer_wise: bool = False,
        preserve_embeddings: bool = True,
        preserve_layer_norms: bool = True,
    ):
        """
        Initialize half-baker.

        Args:
            strength: Interpolation strength (0.0-1.0)
            layer_wise: Apply different strengths per layer
            preserve_embeddings: Keep embedding layers unchanged
            preserve_layer_norms: Keep layer norms unchanged
        """
        self.strength = strength
        self.layer_wise = layer_wise
        self.preserve_embeddings = preserve_embeddings
        self.preserve_layer_norms = preserve_layer_norms

        self.metrics = {"total_half_bakes": 0, "layers_interpolated": 0, "layers_preserved": 0}

    def half_bake(
        self,
        original_model: nn.Module,
        baked_model: nn.Module,
        strength_override: Optional[float] = None,
    ) -> nn.Module:
        """
        Create half-baked model by interpolating weights.

        Args:
            original_model: Original model weights
            baked_model: Fully baked model weights
            strength_override: Override default strength

        Returns:
            Half-baked model
        """
        strength = strength_override if strength_override is not None else self.strength

        # Create output model
        half_baked = copy.deepcopy(original_model)

        original_state = original_model.state_dict()
        baked_state = baked_model.state_dict()
        half_baked_state = {}

        layers_interpolated = 0
        layers_preserved = 0

        for name in original_state.keys():
            original_param = original_state[name]
            baked_param = baked_state.get(name, original_param)

            # Check if layer should be preserved
            should_preserve = False

            if self.preserve_embeddings and "embed" in name.lower():
                should_preserve = True
            if self.preserve_layer_norms and any(
                x in name.lower() for x in ["norm", "ln_", "layernorm"]
            ):
                should_preserve = True
            if "bias" in name.lower():
                should_preserve = True

            if should_preserve:
                half_baked_state[name] = original_param.clone()
                layers_preserved += 1
            else:
                # Interpolate: W_half = (1 - s) * W_orig + s * W_baked
                if self.layer_wise:
                    # Layer-wise strength (deeper layers get more baking)
                    layer_strength = self._get_layer_strength(name, strength)
                else:
                    layer_strength = strength

                half_baked_state[name] = (
                    1 - layer_strength
                ) * original_param + layer_strength * baked_param
                layers_interpolated += 1

        # Load interpolated state
        half_baked.load_state_dict(half_baked_state)

        # Update metrics
        self.metrics["total_half_bakes"] += 1
        self.metrics["layers_interpolated"] = layers_interpolated
        self.metrics["layers_preserved"] = layers_preserved

        return half_baked

    def _get_layer_strength(self, layer_name: str, base_strength: float) -> float:
        """
        Calculate layer-wise strength for progressive baking.

        Deeper layers get stronger baking (they encode more high-level behavior).
        """
        # Extract layer number if present
        import re

        match = re.search(r"layer[_.]?(\d+)", layer_name.lower())

        if match:
            layer_num = int(match.group(1))
            # Linear increase: early layers get less baking
            # Assuming ~12 layers typical
            layer_factor = min(1.0, (layer_num + 1) / 12)
            return base_strength * (0.5 + 0.5 * layer_factor)
        else:
            return base_strength

    def progressive_half_bake(
        self, original_model: nn.Module, baked_model: nn.Module, steps: int = 5
    ) -> nn.Module:
        """
        Progressive half-baking in multiple steps.

        Gradually increases strength from 0 to target.

        Args:
            original_model: Original model
            baked_model: Fully baked model
            steps: Number of progressive steps

        Returns:
            Progressively half-baked model
        """
        current_model = original_model

        for step in range(1, steps + 1):
            step_strength = self.strength * (step / steps)
            current_model = self.half_bake(
                original_model=current_model,
                baked_model=baked_model,
                strength_override=step_strength,
            )

        return current_model

    def get_metrics(self) -> Dict:
        """Get half-baking metrics."""
        return self.metrics.copy()


class StrengthScheduler:
    """
    Schedules half-baking strength over iterations.

    Supports:
    - Constant strength
    - Linear warmup
    - Cosine annealing
    - Custom schedules
    """

    def __init__(
        self,
        initial_strength: float = 0.3,
        final_strength: float = 0.7,
        total_iterations: int = 20,
        schedule_type: str = "linear",
    ):
        """
        Initialize strength scheduler.

        Args:
            initial_strength: Starting strength
            final_strength: Ending strength
            total_iterations: Total iterations
            schedule_type: "constant", "linear", "cosine"
        """
        self.initial_strength = initial_strength
        self.final_strength = final_strength
        self.total_iterations = total_iterations
        self.schedule_type = schedule_type
        self.current_iteration = 0

    def get_strength(self) -> float:
        """Get current strength value."""
        if self.schedule_type == "constant":
            return self.initial_strength

        elif self.schedule_type == "linear":
            progress = self.current_iteration / max(1, self.total_iterations - 1)
            return self.initial_strength + progress * (self.final_strength - self.initial_strength)

        elif self.schedule_type == "cosine":
            import math

            progress = self.current_iteration / max(1, self.total_iterations - 1)
            cosine_factor = (1 - math.cos(math.pi * progress)) / 2
            return self.initial_strength + cosine_factor * (
                self.final_strength - self.initial_strength
            )

        else:
            return self.initial_strength

    def step(self):
        """Advance to next iteration."""
        self.current_iteration += 1

    def reset(self):
        """Reset scheduler."""
        self.current_iteration = 0


__all__ = ["HalfBaker", "HalfBakeConfig", "StrengthScheduler"]

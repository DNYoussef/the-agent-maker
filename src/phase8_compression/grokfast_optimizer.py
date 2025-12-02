"""
Phase 8: Grokfast Optimizer

Accelerates "grokking" by amplifying slow-varying (low-frequency) gradient components.
Based on: "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"

Grokking: The phenomenon where models suddenly generalize after appearing to
memorize/overfit during training. This typically happens late in training.

The key insight is that grokking occurs when slow-moving gradient components
finally dominate. Grokfast artificially amplifies these slow components to
accelerate the transition from memorization to generalization.

Filter types:
1. EMA (Exponential Moving Average): Simple low-pass filter
2. MA (Moving Average): Average over fixed window
3. Adaptive: Adjusts alpha based on gradient statistics

Usage:
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    grokfast = GrokfastOptimizer(base_optimizer, alpha=0.98, lamb=2.0)

    for batch in dataloader:
        loss = model(batch)
        grokfast.zero_grad()
        loss.backward()
        grokfast.step()  # Applies Grokfast filtering then steps
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn


@dataclass
class GrokfastConfig:
    """Configuration for Grokfast optimizer wrapper."""

    alpha: float = 0.98  # EMA decay rate (higher = slower adaptation)
    lamb: float = 2.0  # Amplification factor for slow components
    filter_type: Literal["ema", "ma", "adaptive"] = "ema"
    window_size: int = 100  # For MA filter
    warmup_steps: int = 100  # Steps before Grokfast kicks in
    min_alpha: float = 0.90  # Minimum alpha for adaptive filter
    max_alpha: float = 0.999  # Maximum alpha for adaptive filter
    adaptive_target_ratio: float = 0.5  # Target ratio for adaptive alpha


class GrokfastOptimizer:
    """
    Grokfast gradient filter wrapper for any PyTorch optimizer.

    The filter maintains an exponential moving average (EMA) of gradients.
    The EMA captures slow-varying components. We then add this EMA back
    to the gradient, amplifying the slow components:

        grad_new = grad + lambda * EMA(grad)

    This accelerates grokking by artificially boosting the gradient
    components that would otherwise take much longer to dominate.

    Technical Details:
    - EMA update: ema = alpha * ema + (1 - alpha) * grad
    - Filtered grad: grad_filtered = grad + lamb * ema
    - The 'lamb' parameter controls amplification strength
    - The 'alpha' parameter controls filter bandwidth (higher = slower/smoother)

    Mathematical Justification:
    - Grokking occurs when generalizing features overcome memorizing features
    - Slow gradients correspond to generalizing features
    - Amplifying slow gradients speeds up this transition
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: GrokfastConfig = None,
        alpha: float = None,
        lamb: float = None,
        warmup_steps: int = None,
    ):
        """
        Initialize Grokfast wrapper.

        Args:
            optimizer: Base optimizer to wrap (e.g., AdamW)
            config: GrokfastConfig (takes precedence over individual args)
            alpha: EMA decay rate (override config)
            lamb: Amplification factor (override config)
            warmup_steps: Steps before Grokfast activates (override config)
        """
        self.optimizer = optimizer
        self.config = config or GrokfastConfig()

        # Allow individual args to override config
        if alpha is not None:
            self.config.alpha = alpha
        if lamb is not None:
            self.config.lamb = lamb
        if warmup_steps is not None:
            self.config.warmup_steps = warmup_steps

        # EMA state per parameter
        self.ema_grads: Dict[int, torch.Tensor] = {}

        # Moving average buffer for MA filter
        self.grad_history: Dict[int, list] = {}

        # Statistics for adaptive alpha
        self.grad_norms: Dict[int, float] = {}
        self.ema_norms: Dict[int, float] = {}

        self.step_count = 0

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """
        Apply Grokfast filtering then optimizer step.

        Process:
        1. Update EMA/MA of gradients
        2. Add amplified slow component to gradients
        3. Call underlying optimizer step

        Args:
            closure: Optional closure for LBFGS-style optimizers
        """
        self.step_count += 1

        # Skip Grokfast during warmup
        if self.step_count <= self.config.warmup_steps:
            return self.optimizer.step(closure)

        # Apply Grokfast filter to each parameter's gradient
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                param_id = id(p)
                grad = p.grad.data

                if self.config.filter_type == "ema":
                    filtered_grad = self._apply_ema_filter(param_id, grad)
                elif self.config.filter_type == "ma":
                    filtered_grad = self._apply_ma_filter(param_id, grad)
                elif self.config.filter_type == "adaptive":
                    filtered_grad = self._apply_adaptive_filter(param_id, grad)
                else:
                    filtered_grad = self._apply_ema_filter(param_id, grad)

                # Replace gradient with filtered version
                p.grad.data = filtered_grad

        return self.optimizer.step(closure)

    def _apply_ema_filter(self, param_id: int, grad: torch.Tensor) -> torch.Tensor:
        """
        Apply EMA-based Grokfast filter.

        grad_new = grad + lamb * ema
        ema = alpha * ema + (1 - alpha) * grad
        """
        # Initialize EMA if needed
        if param_id not in self.ema_grads:
            self.ema_grads[param_id] = torch.zeros_like(grad)

        ema = self.ema_grads[param_id]

        # Update EMA: ema = alpha * ema + (1 - alpha) * grad
        ema.mul_(self.config.alpha).add_(grad, alpha=1 - self.config.alpha)

        # Grokfast: grad_new = grad + lamb * ema
        filtered = grad + self.config.lamb * ema

        return filtered

    def _apply_ma_filter(self, param_id: int, grad: torch.Tensor) -> torch.Tensor:
        """
        Apply Moving Average based Grokfast filter.

        Uses a sliding window average instead of exponential decay.
        """
        # Initialize history if needed
        if param_id not in self.grad_history:
            self.grad_history[param_id] = []

        history = self.grad_history[param_id]

        # Add current gradient to history
        history.append(grad.clone())

        # Keep only window_size gradients
        if len(history) > self.config.window_size:
            history.pop(0)

        # Compute moving average
        ma = torch.stack(history).mean(dim=0)

        # Grokfast: grad_new = grad + lamb * ma
        filtered = grad + self.config.lamb * ma

        return filtered

    def _apply_adaptive_filter(self, param_id: int, grad: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive Grokfast filter.

        Adjusts alpha based on gradient statistics to maintain
        a target ratio between gradient and EMA norms.
        """
        # Initialize EMA if needed
        if param_id not in self.ema_grads:
            self.ema_grads[param_id] = torch.zeros_like(grad)
            self.grad_norms[param_id] = 0.0
            self.ema_norms[param_id] = 0.0

        ema = self.ema_grads[param_id]

        # Compute norms
        grad_norm = grad.norm().item()
        ema_norm = ema.norm().item()

        # Update running norm estimates
        self.grad_norms[param_id] = 0.9 * self.grad_norms[param_id] + 0.1 * grad_norm
        self.ema_norms[param_id] = 0.9 * self.ema_norms[param_id] + 0.1 * ema_norm

        # Compute adaptive alpha
        # If EMA is too large relative to gradient, increase alpha (more smoothing)
        # If EMA is too small, decrease alpha (less smoothing)
        if self.grad_norms[param_id] > 1e-8:
            ratio = self.ema_norms[param_id] / self.grad_norms[param_id]
            if ratio > self.config.adaptive_target_ratio:
                # EMA too large, increase smoothing
                alpha = min(self.config.max_alpha, self.config.alpha * 1.001)
            else:
                # EMA too small, decrease smoothing
                alpha = max(self.config.min_alpha, self.config.alpha * 0.999)
        else:
            alpha = self.config.alpha

        # Update EMA with adaptive alpha
        ema.mul_(alpha).add_(grad, alpha=1 - alpha)

        # Grokfast: grad_new = grad + lamb * ema
        filtered = grad + self.config.lamb * ema

        return filtered

    @property
    def param_groups(self):
        """Access underlying optimizer's param_groups."""
        return self.optimizer.param_groups

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "ema_grads": {k: v.cpu() for k, v in self.ema_grads.items()},
            "step_count": self.step_count,
            "config": {
                "alpha": self.config.alpha,
                "lamb": self.config.lamb,
                "filter_type": self.config.filter_type,
                "warmup_steps": self.config.warmup_steps,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from checkpoint."""
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        self.step_count = state_dict["step_count"]

        # Restore EMA grads to correct device
        device = next(iter(self.optimizer.param_groups[0]["params"])).device
        self.ema_grads = {k: v.to(device) for k, v in state_dict["ema_grads"].items()}

        # Restore config
        if "config" in state_dict:
            for key, value in state_dict["config"].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

    def get_stats(self) -> Dict[str, float]:
        """Get current Grokfast statistics."""
        if not self.ema_grads:
            return {}

        ema_norms = [v.norm().item() for v in self.ema_grads.values()]
        return {
            "step_count": self.step_count,
            "mean_ema_norm": sum(ema_norms) / len(ema_norms),
            "max_ema_norm": max(ema_norms),
            "min_ema_norm": min(ema_norms),
            "alpha": self.config.alpha,
            "lamb": self.config.lamb,
            "is_active": self.step_count > self.config.warmup_steps,
        }


def create_grokfast_optimizer(
    model: nn.Module,
    base_optimizer: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    grokfast_alpha: float = 0.98,
    grokfast_lamb: float = 2.0,
    grokfast_warmup: int = 100,
) -> GrokfastOptimizer:
    """
    Factory function to create a Grokfast-wrapped optimizer.

    Args:
        model: Model to optimize
        base_optimizer: Base optimizer type ("adamw", "adam", "sgd")
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        grokfast_alpha: Grokfast EMA decay
        grokfast_lamb: Grokfast amplification factor
        grokfast_warmup: Warmup steps before Grokfast activates

    Returns:
        GrokfastOptimizer wrapping the specified base optimizer
    """
    if base_optimizer.lower() == "adamw":
        base = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif base_optimizer.lower() == "adam":
        base = torch.optim.Adam(model.parameters(), lr=lr)
    elif base_optimizer.lower() == "sgd":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {base_optimizer}")

    config = GrokfastConfig(alpha=grokfast_alpha, lamb=grokfast_lamb, warmup_steps=grokfast_warmup)

    return GrokfastOptimizer(base, config)


__all__ = ["GrokfastOptimizer", "GrokfastConfig", "create_grokfast_optimizer"]

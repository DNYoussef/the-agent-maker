"""
MuGrokfast Optimizer (Muon x Grokfast)
Ported from V1 with phase-specific presets

Combines 3 complementary techniques:
- Grokfast: EMA gradient filtering (accelerates grokking)
- Muon: Newton-Schulz orthogonalization (prevents low-rank collapse)
- QK-Clip: Attention safety rails (for RL training) - ISS-025

ISS-025: Added QK-Clip implementation for attention score clipping
"""

import logging
import math
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    # Only needed for the "MuGrokConfig" forward-reference annotation; imported
    # under TYPE_CHECKING to avoid any import cost / cycle at runtime.
    from .config import MuGrokConfig

logger = logging.getLogger(__name__)


class MuonGrokfast(Optimizer):
    """
    Unified optimizer combining Grokfast + Muon + QK-Clip

    Features:
    - Parameter routing: 2-D params -> Muon, 1-D params -> fallback (AdamW)
    - Phase-specific presets
    - STE compatibility (for Phase 5 BitNet)
    - QK-Clip for RL stability (Phases 3, 6)
    """

    def __init__(
        self,
        params,
        config: Optional["MuGrokConfig"] = None,  # ISS-003: Accept config object
        muon_lr: float = 0.01,
        fallback_lr: float = 1e-3,
        grokfast_alpha: float = 0.98,
        grokfast_lambda: float = 2.0,
        qk_clip_threshold: float = 30.0,
        kl_coefficient: float = 0.0,
        muon_ste_mode: bool = False,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,  # Newton-Schulz iterations
        adamw_beta1: float = 0.9,
        adamw_beta2: float = 0.999,
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        # ISS-003: If config provided, use its values (backwards compatible)
        if config is not None:
            muon_lr = config.muon_lr
            fallback_lr = config.fallback_lr
            grokfast_alpha = config.grokfast_alpha
            grokfast_lambda = config.grokfast_lambda
            qk_clip_threshold = config.qk_clip_threshold
            kl_coefficient = config.kl_coefficient
            muon_ste_mode = config.muon_ste_mode
            momentum = config.momentum
            nesterov = config.nesterov
            ns_steps = config.ns_steps
            # getattr fallbacks: older configs predate these AdamW fields.
            adamw_beta1 = getattr(config, "adamw_beta1", adamw_beta1)
            adamw_beta2 = getattr(config, "adamw_beta2", adamw_beta2)
            adamw_eps = getattr(config, "adamw_eps", adamw_eps)
            weight_decay = getattr(config, "weight_decay", weight_decay)

        # ISS-003: Store config for attribute access
        self.config = config

        defaults = dict(
            muon_lr=muon_lr,
            fallback_lr=fallback_lr,
            grokfast_alpha=grokfast_alpha,
            grokfast_lambda=grokfast_lambda,
            qk_clip_threshold=qk_clip_threshold,
            kl_coefficient=kl_coefficient,
            muon_ste_mode=muon_ste_mode,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_beta1=adamw_beta1,
            adamw_beta2=adamw_beta2,
            adamw_eps=adamw_eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # ISS-025: Initialize QK-Clip counter
        self._qk_clip_count = 0

        # Initialize state
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["ema_grad"] = torch.zeros_like(p.data)
                if len(p.shape) == 2:  # Muon params (matrix-only; rank>2 -> fallback)
                    state["momentum_buffer"] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None) -> Any:
        """Perform single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                state["step"] += 1

                # ===== GROKFAST: EMA Gradient Filtering =====
                grad = p.grad.data
                alpha = group["grokfast_alpha"]
                lambda_ = group["grokfast_lambda"]

                # Update EMA
                state["ema_grad"].mul_(alpha).add_(grad, alpha=1 - alpha)

                # Filter gradient
                filtered_grad = grad + lambda_ * (grad - state["ema_grad"])

                # ===== PARAMETER ROUTING =====
                if len(p.shape) == 2:
                    # 2-D params -> Muon (orthogonalization)
                    self._muon_update(p, filtered_grad, state, group)
                else:
                    # 1-D params -> AdamW fallback
                    self._adamw_update(p, filtered_grad, state, group)

        return loss

    def _muon_update(self, param, grad, state, group) -> None:
        """Muon update with Newton-Schulz orthogonalization"""
        lr = group["muon_lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        muon_ste_mode = group["muon_ste_mode"]

        # Newton-Schulz orthogonalization
        if not muon_ste_mode:
            # Standard mode
            G = grad
        else:
            # STE mode (for BitNet - Phase 5)
            # Quantized forward, full-precision backward
            G = grad.clone()

        # Newton-Schulz orthogonalization (canonical Muon form).
        # Operate on a wide matrix (rows <= cols) by transposing tall inputs, then
        # iterate X <- 1.5 X - 0.5 (X X^T) X and transpose back. The previous code
        # used the wrong operand order (G_norm @ A with A=[m,m]) which raised a
        # shape error on every non-square 2D param (e.g. 64x100): A@X is the only
        # shape-correct product. This form works for any 2D shape.
        # ponytail: Wave-0 fix is crash-freedom on any 2D shape; spectral-norm
        # scaling + optimal quintic coefficients are a Wave-1 orthogonality upgrade.
        transposed = False
        X = G
        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True
        scale = X.norm() + 1e-8
        X = X / scale
        for _ in range(ns_steps):
            A = X @ X.T  # [m, m] where m = min(rows, cols)
            X = 1.5 * X - 0.5 * (A @ X)
        X = X * scale
        G = X.T if transposed else X

        # Momentum
        if momentum > 0:
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(G)
            if nesterov:
                G = G + momentum * buf
            else:
                G = buf

        # Update param
        param.add_(G, alpha=-lr)

    def _adamw_update(self, param, grad, state, group) -> None:
        """Real AdamW update for 1-D params (biases, layer-norm weights).

        First/second moment estimates with bias correction and decoupled weight
        decay (Loshchilov & Hutter 2019). Replaces the prior plain-SGD body that
        was mislabeled AdamW. (2-D params, including embeddings, route to Muon.)
        """
        lr = group["fallback_lr"]
        # group.get with defaults: optimizer state restored from a checkpoint
        # saved before these keys existed would otherwise KeyError on resume.
        beta1 = group.get("adamw_beta1", 0.9)
        beta2 = group.get("adamw_beta2", 0.999)
        eps = group.get("adamw_eps", 1e-8)
        weight_decay = group.get("weight_decay", 0.0)

        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(param.data)
            state["exp_avg_sq"] = torch.zeros_like(param.data)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        # Dedicated AdamW timestep, counted from when the moments first exist.
        # Using the shared "step" would mis-bias-correct when upgrading an old
        # SGD checkpoint (nonzero global step but zero-initialized moments).
        state["adamw_step"] = state.get("adamw_step", 0) + 1
        step = state["adamw_step"]

        # Decoupled weight decay: applied to the parameter, not the gradient.
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)

    def get_muon_lr(self) -> float:
        """Get current Muon learning rate"""
        return self.param_groups[0]["muon_lr"]

    def get_mu_norm(self) -> float:
        """Get EMA gradient norm (for logging)"""
        total_norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                total_norm += state["ema_grad"].norm().item() ** 2
        return total_norm**0.5

    def get_qk_clip_count(self) -> int:
        """Get QK-Clip activation count (for logging) - ISS-025."""
        return self._qk_clip_count

    def reset_qk_clip_count(self) -> None:
        """Reset QK-Clip counter (call at start of each epoch)."""
        self._qk_clip_count = 0


class QKClipHook:
    """
    QK-Clip: Attention Score MONITOR (ISS-025)

    Honesty: this is a forward hook, which runs AFTER the module computes its
    output, so it cannot modify pre-softmax attention scores. It only COUNTS how
    often a module output exceeds the threshold (get_clip_count). It does not
    clip and does not prevent gradient explosions.

    Real QK-Clip requires clipping pre-softmax scores INSIDE the attention
    forward via apply_qk_clip(); wiring that into each attention implementation
    is a Wave-2 model change. Use this hook only as a diagnostic.

    Usage:
        monitor = QKClipHook(threshold=25.0)
        monitor.register(model)  # counts threshold exceedances per attention module
    """

    def __init__(self, threshold: float = 25.0):
        """
        Initialize QK-Clip hook.

        Args:
            threshold: Maximum absolute attention score (before softmax)
        """
        self.threshold = threshold
        self.clip_count = 0
        self._handles = []

    def register(self, model: nn.Module) -> None:
        """
        Register QK-Clip hooks on all attention modules.

        Automatically finds modules with 'attention' or 'attn' in name.
        """
        for name, module in model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                if hasattr(module, "forward"):
                    handle = module.register_forward_hook(self._clip_hook)
                    self._handles.append(handle)
                    logger.debug(f"QK-Clip registered on: {name}")

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _clip_hook(self, module, input, output) -> Any:
        """
        Hook function that clips attention scores.

        Note: This is a best-effort approach. For full QK-Clip,
        the attention module itself should implement clipping.
        """
        # This hook runs after forward, so we can't modify pre-softmax scores
        # Instead, we log when output values suggest clipping would be needed
        if isinstance(output, torch.Tensor):
            max_val = output.abs().max().item()
            if max_val > self.threshold:
                self.clip_count += 1

        return output

    def get_clip_count(self) -> int:
        """Get number of times clipping threshold was exceeded."""
        return self.clip_count

    def reset_count(self) -> None:
        """Reset clip counter."""
        self.clip_count = 0


def apply_qk_clip(attention_scores: torch.Tensor, threshold: float = 25.0) -> tuple:
    """
    Apply QK-Clip to attention scores (ISS-025).

    Use this function inside attention modules before softmax:

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores, clip_count = apply_qk_clip(scores, threshold=25.0)
        attn_weights = F.softmax(scores, dim=-1)

    Args:
        attention_scores: Pre-softmax attention scores [batch, heads, seq, seq]
        threshold: Clipping threshold (default: 25.0)

    Returns:
        Tuple of (clipped_scores, num_clipped_values)
    """
    # Count values that exceed threshold
    exceeds = (attention_scores.abs() > threshold).sum().item()

    # Clip scores
    clipped_scores = torch.clamp(attention_scores, -threshold, threshold)

    return clipped_scores, int(exceeds)


def create_optimizer_from_phase(model: nn.Module, phase_num: int) -> MuonGrokfast:
    """
    Create MuGrokfast optimizer with phase-specific preset

    Args:
        model: Model to optimize
        phase_num: Phase number (1, 3, 5, 6, 7)

    Returns:
        MuonGrokfast optimizer
    """
    from .config import MuGrokConfig

    config = MuGrokConfig.from_phase(phase_num)

    # ISS-003: Pass config= so optimizer.config attribute is set
    return MuonGrokfast(
        model.parameters(),
        config=config,
    )

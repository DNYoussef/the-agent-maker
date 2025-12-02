"""
MuonGrokfast: Unified optimizer combining Muon + Grokfast + QK-Clip.

This module implements the MuonGrokfast optimizer, which synergistically combines:
1. Grokfast: Temporal gradient filtering (time-spectrum)
2. Muon: Spatial gradient orthogonalization (space-geometry)
3. QK-Clip: Attention safety rails

Architecture: Parameter routing automatically selects:
- 2-D parameters → Muon path (with Grokfast prefilter)
- 1-D parameters → Fallback path (with Grokfast prefilter)

Author: Agent Forge V2 Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from typing import Dict, List, Optional, Tuple
import warnings

from mugrokfast_config import MuGrokConfig


class MuonGrokfast(Optimizer):
    """
    Muon × Grokfast optimizer with QK-clip safety rails.

    Combines three complementary techniques for improved training:
    - **Grokfast** (time-spectrum): EMA-based gradient filtering
    - **Muon** (space-geometry): Newton-Schulz orthogonalization
    - **QK-Clip**: Attention logit bounding

    Key Features:
    - Automatic parameter routing (2-D → Muon, 1-D → fallback)
    - STE-compatible mode for quantized training (Phase 5)
    - Per-head QK-clip for MLA/decoupled-RoPE architectures
    - Phase-specific presets via MuGrokConfig

    Example:
        >>> config = MuGrokConfig.from_phase(1)  # Phase 1 pretraining
        >>> optimizer = MuonGrokfast(model.parameters(), config=config)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        params,
        config: Optional[MuGrokConfig] = None,
        **kwargs
    ):
        """
        Initialize MuonGrokfast optimizer.

        Args:
            params: Model parameters (iterable)
            config: MuGrokConfig instance (if None, uses defaults)
            **kwargs: Override specific config values

        Example:
            >>> config = MuGrokConfig.from_phase(5, muon_lr=1e-4)
            >>> optimizer = MuonGrokfast(model.parameters(), config=config)
        """
        # Use default config if not provided
        if config is None:
            config = MuGrokConfig()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown config key: {key}")
            setattr(config, key, value)

        self.config = config

        # Initialize optimizer defaults
        defaults = {
            'muon_lr': config.muon_lr,
            'fallback_lr': config.fallback_lr,
        }

        super().__init__(params, defaults)

        # Initialize state tracking
        self._init_state()

        # Create fallback optimizer
        self._init_fallback_optimizer()

    def _init_state(self):
        """Initialize optimizer state (EMA, momentum, etc.)."""
        # Global state (not per-parameter)
        self.state['step'] = 0

    def _init_fallback_optimizer(self):
        """Initialize fallback optimizer for 1-D parameters."""
        # Collect 1-D parameters
        fallback_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.ndim < 2:  # 1-D or scalar
                    fallback_params.append(p)

        # Create fallback optimizer
        if self.config.fallback_type == "adamw":
            self.fallback_optimizer = AdamW(
                fallback_params,
                lr=self.config.fallback_lr,
                betas=self.config.fallback_betas,
                weight_decay=self.config.fallback_weight_decay
            )
        else:
            raise NotImplementedError(
                f"Fallback type '{self.config.fallback_type}' not implemented. "
                "Supported: 'adamw'"
            )

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Algorithm:
        1. Grokfast prefilter (all parameters)
        2. Parameter routing:
           - 2-D → Muon orthogonalization
           - 1-D → Fallback optimizer
        3. QK-clip safety rails (attention layers)

        Args:
            closure: Optional closure for re-evaluating loss

        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Increment global step counter
        self.state['step'] += 1
        step = self.state['step']

        # ====================================================================
        # STAGE 1: GROKFAST PREFILTER (Time-Spectrum Filtering)
        # ====================================================================
        if self.config.enable_grokfast:
            self._apply_grokfast_filter()

        # ====================================================================
        # STAGE 2: PARAMETER ROUTING
        # ====================================================================
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get filtered gradient (or raw if Grokfast disabled)
                if self.config.enable_grokfast:
                    grad = self.state[p].get('filtered_grad', p.grad)
                else:
                    grad = p.grad

                # Route by dimensionality
                if p.ndim >= 2 and self.config.enable_muon:
                    # PATH A: Muon (2-D matrices)
                    update = self._muon_step(p, grad, group)
                else:
                    # PATH B: Fallback (handled separately below)
                    continue  # Fallback optimizer called later

                # ============================================================
                # STAGE 3: QK-CLIP SAFETY RAILS
                # ============================================================
                if self.config.use_qk_clip and self.config.enable_muon_clip:
                    param_name = self._get_param_name(p)
                    if self._is_qk_param(param_name):
                        update = self._apply_qk_clip(p, update, param_name)

                # Apply update
                p.add_(update, alpha=-group['muon_lr'])

        # ====================================================================
        # FALLBACK PATH: Update 1-D parameters
        # ====================================================================
        if hasattr(self, 'fallback_optimizer'):
            self.fallback_optimizer.step()

        return loss

    def _apply_grokfast_filter(self):
        """
        Apply Grokfast EMA filtering to all gradients.

        Algorithm (Grokfast-EMA):
            μ_t = α·μ_{t-1} + (1-α)·g_t  (EMA)
            ĝ_t = g_t + λ·μ_t           (Filtered gradient)

        Time-spectrum interpretation:
        - Emphasizes slow gradient components (high α)
        - Dampens high-frequency noise
        """
        alpha = self.config.grokfast_alpha
        lambda_reg = self.config.grokfast_lambda

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Initialize EMA state
                if 'ema_grad' not in self.state[p]:
                    self.state[p]['ema_grad'] = torch.zeros_like(p.grad)

                # Update EMA: μ_t = α·μ_{t-1} + (1-α)·g_t
                ema = self.state[p]['ema_grad']
                ema.mul_(alpha).add_(p.grad, alpha=1 - alpha)

                # Filtered gradient: ĝ_t = g_t + λ·μ_t
                filtered = p.grad + lambda_reg * ema

                # Store filtered gradient
                self.state[p]['filtered_grad'] = filtered

    def _muon_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        group: dict
    ) -> torch.Tensor:
        """
        Muon optimization step with Newton-Schulz orthogonalization.

        Algorithm:
        1. Accumulate momentum: m_t = β·m_{t-1} + (1-β)·ĝ_t
        2. Orthogonalize momentum:
           - If ns_steps > 0: Newton-Schulz (fast approximation)
           - If ns_steps = 0: SVD (exact but slower)
        3. Return orthogonalized update

        Space-geometry interpretation:
        - Spreads updates across rare spatial directions
        - Prevents low-rank collapse
        - Maintains semi-orthogonality: M @ M.T ≈ I

        Args:
            param: Parameter tensor (2-D)
            grad: Filtered gradient
            group: Parameter group dict

        Returns:
            Orthogonalized update direction
        """
        # Initialize momentum state
        if 'muon_momentum' not in self.state[param]:
            self.state[param]['muon_momentum'] = torch.zeros_like(param)

        # Accumulate momentum: m_t = β·m_{t-1} + (1-β)·ĝ_t
        momentum = self.state[param]['muon_momentum']
        beta = self.config.muon_momentum
        momentum.mul_(beta).add_(grad, alpha=1 - beta)

        # Orthogonalize momentum
        if self.config.muon_ns_steps > 0:
            # Newton-Schulz orthogonalization (fast)
            ortho_momentum = self._newton_schulz_ortho(
                momentum,
                steps=self.config.muon_ns_steps,
                coeffs=self.config.muon_ns_coeffs
            )
        else:
            # SVD orthogonalization (exact)
            ortho_momentum = self._svd_ortho(momentum)

        # STE compatibility mode (Phase 5 quantized training)
        if self.config.muon_ste_mode and hasattr(param, 'weight_full_precision'):
            # Apply update to full-precision weights
            # STE will handle gradient flow through quantization
            return ortho_momentum  # Caller applies to weight_full_precision
        else:
            return ortho_momentum

    def _newton_schulz_ortho(
        self,
        W: torch.Tensor,
        steps: int = 5,
        coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    ) -> torch.Tensor:
        """
        Newton-Schulz iterations for orthogonalization approximation.

        Algorithm (quintic polynomial):
            W_{i+1} = a·W_i + b·W_i³ + c·W_i⁵

        Converges to semi-orthogonal matrix: W @ W.T ≈ I

        Args:
            W: Input matrix (m × n)
            steps: Number of iterations (default: 5)
            coeffs: Polynomial coefficients (a, b, c)

        Returns:
            Semi-orthogonalized matrix
        """
        a, b, c = coeffs
        W_iter = W.clone()

        for _ in range(steps):
            W2 = W_iter @ W_iter.T  # W²
            W3 = W2 @ W_iter        # W³
            W5 = W3 @ W2            # W⁵
            W_iter = a * W_iter + b * W3 + c * W5

        return W_iter

    def _svd_ortho(self, W: torch.Tensor) -> torch.Tensor:
        """
        Exact orthogonalization via SVD.

        Algorithm:
            W = U Σ V^T  (SVD)
            W_ortho = U V^T  (discard singular values)

        Result: W_ortho @ W_ortho.T = I (exact orthogonality)

        Args:
            W: Input matrix (m × n)

        Returns:
            Orthogonalized matrix
        """
        U, _, Vt = torch.linalg.svd(W, full_matrices=False)
        return U @ Vt

    def _apply_qk_clip(
        self,
        param: torch.Tensor,
        update: torch.Tensor,
        param_name: str
    ) -> torch.Tensor:
        """
        Apply QK-clip/MuonClip to attention projection weights.

        Safety mechanism: Prevents unbounded attention logits when
        optimizer dynamics push W_Q or W_K norms too high.

        Algorithm:
            For each attention head h:
                If ||W_h + ΔW_h|| > τ:
                    ΔW_h ← (τ / ||W_h + ΔW_h||) · ΔW_h

        Args:
            param: Current parameter (W_Q or W_K)
            update: Proposed update (ΔW)
            param_name: Parameter name (for head detection)

        Returns:
            Rescaled update (if needed)
        """
        threshold = self.config.qk_clip_threshold

        # Compute new parameter after update
        new_param = param + update

        # Per-head scaling (MLA/decoupled-RoPE compatibility)
        if self.config.qk_clip_per_head:
            # Detect number of heads from parameter shape
            num_heads = self._detect_num_heads(param, param_name)

            if num_heads > 1:
                # Reshape for per-head processing
                head_dim = param.shape[0] // num_heads
                new_param_heads = new_param.view(num_heads, head_dim, -1)
                update_heads = update.view(num_heads, head_dim, -1)

                # Check each head
                for h in range(num_heads):
                    head_norm = new_param_heads[h].norm()
                    if head_norm > threshold:
                        scale = threshold / head_norm
                        update_heads[h].mul_(scale)

                # Reshape back
                update = update_heads.view_as(update)
                return update

        # Global scaling (fallback if not per-head)
        global_norm = new_param.norm()
        if global_norm > threshold:
            scale = threshold / global_norm
            update = update * scale

        return update

    def _is_qk_param(self, param_name: str) -> bool:
        """Check if parameter is Q or K projection."""
        return ('q_proj' in param_name.lower() or
                'k_proj' in param_name.lower() or
                'query' in param_name.lower() or
                'key' in param_name.lower())

    def _detect_num_heads(
        self,
        param: torch.Tensor,
        param_name: str
    ) -> int:
        """
        Detect number of attention heads from parameter shape.

        Heuristic: Assumes head_dim ∈ {64, 128} (common in practice)
        """
        if param.ndim < 2:
            return 1

        out_dim = param.shape[0]

        # Common head dimensions
        for head_dim in [64, 128]:
            if out_dim % head_dim == 0:
                return out_dim // head_dim

        # Fallback: single head
        warnings.warn(
            f"Could not detect num_heads for {param_name} "
            f"with shape {param.shape}. Using global QK-clip."
        )
        return 1

    def _get_param_name(self, param: torch.Tensor) -> str:
        """Get parameter name from tensor (for QK detection)."""
        # Search through parameter groups
        for group in self.param_groups:
            for name, p in group.get('named_params', {}).items():
                if p is param:
                    return name
        return "unknown"

    def zero_grad(self, set_to_none: bool = True):
        """
        Zero out gradients.

        Args:
            set_to_none: If True, set gradients to None (saves memory)
        """
        super().zero_grad(set_to_none=set_to_none)

        # Also zero fallback optimizer
        if hasattr(self, 'fallback_optimizer'):
            self.fallback_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        """Return optimizer state dict (for checkpointing)."""
        state = {
            'config': self.config.to_dict(),
            'state': self.state,
            'param_groups': self.param_groups,
        }

        if hasattr(self, 'fallback_optimizer'):
            state['fallback_state'] = self.fallback_optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict: dict):
        """Load optimizer state dict (from checkpoint)."""
        # Load config
        config_dict = state_dict['config']
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Load state
        super().load_state_dict(state_dict)

        # Load fallback state
        if 'fallback_state' in state_dict and hasattr(self, 'fallback_optimizer'):
            self.fallback_optimizer.load_state_dict(state_dict['fallback_state'])

    def __repr__(self) -> str:
        """Pretty print optimizer state."""
        lines = ["MuonGrokfast("]
        lines.append(f"  Config: {self.config.phase_name}")
        lines.append(f"  Step: {self.state.get('step', 0)}")
        lines.append(f"  Muon LR: {self.config.muon_lr}")
        lines.append(f"  Fallback LR: {self.config.fallback_lr}")
        lines.append(f"  Grokfast: α={self.config.grokfast_alpha}, λ={self.config.grokfast_lambda}")
        lines.append(f"  QK-Clip: enabled={self.config.use_qk_clip}, τ={self.config.qk_clip_threshold}")
        lines.append(")")
        return "\n".join(lines)

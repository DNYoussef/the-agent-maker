"""
MetaGrokfast: Enhanced Grokfast Optimizer with Meta-Calculus

This module enhances the existing Grokfast optimizer with bigeometric
gradient transformation for improved stability and convergence.

Key Enhancements:
    1. Bigeometric gradient transform: g_meta = g * |g|^(2k-1)
    2. Adaptive k via k(L) formula
    3. Layer-wise k scheduling
    4. Log-space EMA for momentum stability
    5. Spectral gap monitoring

Integration Points:
    - Phase 1 (Cognate): Stable pre-training
    - Phase 3 (Quiet-STaR): RL stability with QK-clip
    - Phase 5 (Curriculum): BitNet STE mode
    - Phase 6 (Baking): Fine-tuning stability
    - Phase 7 (Experts): Expert training
    - Phase 8 (Compression): Fine-tuning after compression
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.optim import Optimizer

from .bigeometric import BigeometricConfig, bigeometric_gradient_transform
from .k_formula import KFormulaConfig, compute_k, k_from_gradient, k_from_layer_index


class GrokfastFilterType(Enum):
    """Filter types for gradient EMA."""

    EMA = "ema"  # Exponential moving average
    MA = "ma"  # Simple moving average
    ADAPTIVE = "adaptive"  # Adaptive alpha based on gradient stats
    BIGEOMETRIC = "bigeometric"  # Log-space EMA (new)


@dataclass
class MetaGrokfastConfig:
    """Configuration for MetaGrokfast optimizer."""

    # Base optimizer
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    # Grokfast EMA filtering
    grokfast_alpha: float = 0.98
    grokfast_lambda: float = 2.0
    filter_type: GrokfastFilterType = GrokfastFilterType.BIGEOMETRIC
    warmup_steps: int = 100

    # Bigeometric enhancement
    use_bigeometric: bool = True
    bigeometric_config: BigeometricConfig = field(default_factory=BigeometricConfig)

    # k(L) formula
    use_adaptive_k: bool = True
    k_formula_config: KFormulaConfig = field(default_factory=KFormulaConfig)
    layer_wise_k: bool = True  # Different k per layer

    # QK-Clip (for RL phases)
    use_qk_clip: bool = False
    qk_clip_threshold: float = 25.0

    # Muon orthogonalization
    use_muon: bool = True
    muon_lr: float = 1e-3
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5  # Newton-Schulz iterations

    # STE mode for BitNet
    ste_mode: bool = False

    # Monitoring
    track_stats: bool = True


class GrokfastConfig(MetaGrokfastConfig):
    """
    Backward-compatible facade config.

    Older phase facade modules used GrokfastConfig(alpha=..., lamb=...).
    MetaGrokfastConfig uses explicit grokfast_* field names, so this shim
    keeps those facades importable without weakening the typed core config.
    """

    def __init__(
        self,
        alpha: float = 0.98,
        lamb: float = 2.0,
        filter_type: Union[str, GrokfastFilterType] = GrokfastFilterType.BIGEOMETRIC,
        warmup_steps: int = 100,
        qk_clip: bool = False,
        qk_clip_value: float = 25.0,
        **kwargs,
    ):
        if isinstance(filter_type, str):
            filter_type = GrokfastFilterType(filter_type)

        kwargs.setdefault("grokfast_alpha", alpha)
        kwargs.setdefault("grokfast_lambda", lamb)
        kwargs.setdefault("filter_type", filter_type)
        kwargs.setdefault("warmup_steps", warmup_steps)
        kwargs.setdefault("use_qk_clip", qk_clip)
        kwargs.setdefault("qk_clip_threshold", qk_clip_value)
        super().__init__(**kwargs)


# Pre-defined configs for each phase
PHASE_CONFIGS = {
    "phase1_cognate": MetaGrokfastConfig(
        lr=1e-3,
        grokfast_alpha=0.98,
        grokfast_lambda=0.3,  # Gentle filtering for pretraining
        use_bigeometric=True,
        use_muon=True,
        muon_lr=1e-3,
    ),
    "phase3_quietstar": MetaGrokfastConfig(
        lr=5e-4,
        grokfast_alpha=0.95,
        grokfast_lambda=0.1,  # Lower for RL noise
        use_bigeometric=True,
        use_qk_clip=True,
        qk_clip_threshold=25.0,
        use_muon=True,
        muon_lr=5e-4,
    ),
    "phase5_curriculum": MetaGrokfastConfig(
        lr=1e-3,
        grokfast_alpha=0.99,
        grokfast_lambda=2.0,  # Aggressive for BitNet
        use_bigeometric=True,
        ste_mode=True,
        use_muon=True,
        muon_lr=1e-3,
    ),
    "phase4_bitnet": MetaGrokfastConfig(
        lr=1e-4,
        grokfast_alpha=0.95,
        grokfast_lambda=2.0,
        use_bigeometric=True,
        ste_mode=True,
        use_muon=True,
        muon_lr=1e-4,
    ),
    "phase6_baking": MetaGrokfastConfig(
        lr=1e-4,
        grokfast_alpha=0.98,
        grokfast_lambda=0.2,
        use_bigeometric=True,
        use_muon=True,
        muon_lr=1e-4,
    ),
    "phase7_experts": MetaGrokfastConfig(
        lr=5e-4,
        grokfast_alpha=0.97,
        grokfast_lambda=0.15,
        use_bigeometric=True,
        use_muon=True,
        muon_lr=5e-4,
    ),
    "phase8_compression": MetaGrokfastConfig(
        lr=1e-4,
        grokfast_alpha=0.99,
        grokfast_lambda=1.5,
        use_bigeometric=True,
        use_muon=True,
        muon_lr=1e-4,
    ),
}


class MetaGrokfast(Optimizer):
    """
    Enhanced Grokfast optimizer with meta-calculus integration.

    Combines:
        - Grokfast EMA filtering (time-spectrum)
        - Muon orthogonalization (space-geometry)
        - Bigeometric gradient transform (scale-adaptive)
        - k(L) formula for layer-wise adaptation

    Usage:
        optimizer = MetaGrokfast(model.parameters(), config=PHASE_CONFIGS["phase1_cognate"])
        # or
        optimizer = MetaGrokfast.for_phase("phase1_cognate", model.parameters())
    """

    def __init__(self, params, config: Optional[MetaGrokfastConfig] = None, **kwargs):
        """
        Initialize MetaGrokfast optimizer.

        Args:
            params: Model parameters
            config: MetaGrokfastConfig or None for defaults
            **kwargs: Override config values
        """
        self.config = config or MetaGrokfastConfig()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        defaults = dict(
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
        super().__init__(params, defaults)

        # Initialize state
        self.step_count = 0
        self.stats_history = []

        # Layer indexing for layer-wise k
        self._index_layers()

    @classmethod
    def for_phase(cls, phase: str, params, *args, **kwargs) -> "MetaGrokfast":
        """
        Create optimizer with phase-specific configuration.

        Args:
            phase: Phase name (e.g., "phase1_cognate")
            params: Model parameters
            **kwargs: Override config values

        Returns:
            Configured MetaGrokfast optimizer
        """
        if args:
            if len(args) == 1 and "lr" not in kwargs:
                kwargs["lr"] = args[0]
            else:
                raise TypeError("MetaGrokfast.for_phase accepts at most one positional lr override")

        phase_aliases = {
            "phase1": "phase1_cognate",
            "phase3": "phase3_quietstar",
            "phase4": "phase4_bitnet",
            "phase5": "phase5_curriculum",
            "phase6": "phase6_baking",
            "phase7": "phase7_experts",
            "phase8": "phase8_compression",
        }
        phase = phase_aliases.get(phase, phase)

        if phase not in PHASE_CONFIGS:
            available = list(PHASE_CONFIGS.keys())
            raise ValueError(f"Unknown phase: {phase}. Available: {available}")

        if isinstance(params, torch.nn.Module):
            params = params.parameters()

        config = replace(PHASE_CONFIGS[phase])
        return cls(params, config=config, **kwargs)

    def _index_layers(self):
        """Index parameter groups for layer-wise operations."""
        self.param_to_layer = {}
        layer_idx = 0

        for group in self.param_groups:
            for p in group["params"]:
                self.param_to_layer[id(p)] = layer_idx
                layer_idx += 1

        self.total_layers = layer_idx

    def _get_layer_k(self, param: torch.Tensor) -> float:
        """Get k value for a parameter based on its layer."""
        if not self.config.layer_wise_k:
            return 0.5  # Default

        layer_idx = self.param_to_layer.get(id(param), 0)
        return k_from_layer_index(layer_idx, self.total_layers, self.config.k_formula_config)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform optimization step.

        Args:
            closure: Optional closure for computing loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1
        step_stats = {"step": self.step_count, "params": []}

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["grokfast_ema"] = torch.zeros_like(p.data)
                    state["grokfast_ema_initialized"] = False
                    if self.config.use_muon and len(p.shape) >= 2:
                        state["momentum_buffer"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Track original gradient stats
                orig_norm = torch.norm(grad).item()

                # Step 1: Grokfast EMA filtering (must see raw gradients)
                if self.step_count > self.config.warmup_steps:
                    grad = self._apply_grokfast(grad, state)

                # Step 2: Bigeometric transform (amplify filtered signal)
                if self.config.use_bigeometric and self.step_count > self.config.warmup_steps:
                    if self.config.use_adaptive_k:
                        k = k_from_gradient(grad, self.config.k_formula_config)
                    elif self.config.layer_wise_k:
                        k = self._get_layer_k(p)
                    else:
                        k = 0.5

                    grad = bigeometric_gradient_transform(grad, k, self.config.bigeometric_config)

                # Step 3: QK-Clip (for attention parameters in RL)
                if self.config.use_qk_clip:
                    grad = torch.clamp(
                        grad, -self.config.qk_clip_threshold, self.config.qk_clip_threshold
                    )

                # Step 4: Parameter update
                if self.config.use_muon and len(p.shape) >= 2:
                    self._muon_update(p, grad, state, group)
                else:
                    self._adam_update(p, grad, state, group)

                # Track stats
                if self.config.track_stats:
                    step_stats["params"].append(
                        {
                            "orig_norm": orig_norm,
                            "processed_norm": torch.norm(grad).item(),
                            "param_norm": torch.norm(p.data).item(),
                        }
                    )

        if self.config.track_stats:
            self.stats_history.append(step_stats)

        return loss

    def _apply_grokfast(self, grad: torch.Tensor, state: dict) -> torch.Tensor:
        """
        Apply Grokfast EMA filtering.

        Formula: grad_new = grad + lambda * EMA(grad)

        This amplifies slow-varying gradient components to accelerate grokking.
        """
        ema = state["grokfast_ema"]
        alpha = self.config.grokfast_alpha
        lamb = self.config.grokfast_lambda

        if self.config.filter_type == GrokfastFilterType.BIGEOMETRIC:
            # Log-space EMA over magnitudes with signed EMA state. Initialize
            # from the first active gradient so zero-state epsilon does not bias
            # the early filter toward vanishing values.
            if not state.get("grokfast_ema_initialized", False):
                ema.copy_(grad)
                state["grokfast_ema_initialized"] = True
                return grad + lamb * ema

            sign = torch.sign(grad)
            log_abs_grad = torch.log(torch.abs(grad) + 1e-8)
            log_abs_ema = torch.log(torch.abs(ema) + 1e-8)

            log_abs_ema_new = alpha * log_abs_ema + (1 - alpha) * log_abs_grad
            ema_new = sign * torch.exp(log_abs_ema_new)

            ema.copy_(ema_new)
            return grad + lamb * ema

        elif self.config.filter_type == GrokfastFilterType.EMA:
            # Standard EMA
            ema.mul_(alpha).add_(grad, alpha=1 - alpha)
            return grad + lamb * ema

        elif self.config.filter_type == GrokfastFilterType.ADAPTIVE:
            # Adaptive alpha based on gradient variance
            grad_var = torch.var(grad)
            ema_var = torch.var(ema) + 1e-8
            ratio = (grad_var / ema_var).clamp(0.5, 2.0)
            adaptive_alpha = alpha * ratio.sqrt()

            ema.mul_(adaptive_alpha).add_(grad, alpha=1 - adaptive_alpha)
            return grad + lamb * ema

        else:
            # Fallback to standard
            ema.mul_(alpha).add_(grad, alpha=1 - alpha)
            return grad + lamb * ema

    def _muon_update(
        self, param: torch.Tensor, grad: torch.Tensor, state: dict, group: dict
    ) -> None:
        """Muon update with Newton-Schulz orthogonalization for 2D params."""
        lr = self.config.muon_lr
        momentum = self.config.muon_momentum
        nesterov = self.config.muon_nesterov
        ns_steps = self.config.muon_ns_steps

        G = grad.clone()

        if len(G.shape) == 2 and min(G.shape) >= 2:
            scale = G.norm() + 1e-8
            G_norm = G / scale

            for _ in range(ns_steps):
                if G.shape[0] <= G.shape[1]:
                    A = G_norm @ G_norm.T
                    G_norm = 1.5 * G_norm - 0.5 * A @ G_norm
                else:
                    A = G_norm.T @ G_norm
                    G_norm = 1.5 * G_norm - 0.5 * G_norm @ A

            G = G_norm * scale

        if momentum > 0 and "momentum_buffer" in state:
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(G)
            G = G + momentum * buf if nesterov else buf

        if group["weight_decay"] != 0:
            param.data.mul_(1 - lr * group["weight_decay"])

        param.add_(G, alpha=-lr)

    def _adam_update(
        self, param: torch.Tensor, grad: torch.Tensor, state: dict, group: dict
    ) -> None:
        """Standard Adam update for non-Muon parameters."""
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        step_size = group["lr"] / bias_correction1
        denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group["eps"])

        if group["weight_decay"] != 0:
            param.data.add_(param.data, alpha=-group["lr"] * group["weight_decay"])

        param.data.addcdiv_(exp_avg, denom, value=-step_size)

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.stats_history:
            return {}

        recent = self.stats_history[-100:]  # Last 100 steps

        all_orig_norms = [s["orig_norm"] for step in recent for s in step["params"]]
        all_proc_norms = [s["processed_norm"] for step in recent for s in step["params"]]

        return {
            "steps": self.step_count,
            "avg_orig_grad_norm": sum(all_orig_norms) / len(all_orig_norms)
            if all_orig_norms
            else 0,
            "avg_processed_grad_norm": sum(all_proc_norms) / len(all_proc_norms)
            if all_proc_norms
            else 0,
            "compression_ratio": (sum(all_orig_norms) / (sum(all_proc_norms) + 1e-8))
            if all_proc_norms
            else 1.0,
        }


# =============================================================================
# Utility Functions
# =============================================================================


def create_optimizer_for_model(
    model: torch.nn.Module, phase: str, separate_groups: bool = True, **kwargs
) -> MetaGrokfast:
    """
    Create MetaGrokfast optimizer with intelligent parameter grouping.

    Args:
        model: PyTorch model
        phase: Training phase name
        separate_groups: Whether to create separate groups for different param types
        **kwargs: Additional config overrides

    Returns:
        Configured MetaGrokfast optimizer
    """
    if separate_groups:
        # Separate embeddings, attention, and other parameters
        embedding_params = []
        attention_params = []
        other_params = []

        for name, param in model.named_parameters():
            if "embed" in name.lower():
                embedding_params.append(param)
            elif "attn" in name.lower() or "attention" in name.lower():
                attention_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if embedding_params:
            param_groups.append({"params": embedding_params, "lr_scale": 0.5})
        if attention_params:
            param_groups.append({"params": attention_params, "lr_scale": 1.0})
        if other_params:
            param_groups.append({"params": other_params, "lr_scale": 1.0})

        return MetaGrokfast.for_phase(phase, param_groups, **kwargs)
    else:
        return MetaGrokfast.for_phase(phase, model.parameters(), **kwargs)


def compare_optimizers_demo():
    """Demo comparing standard vs MetaGrokfast."""
    print("\n" + "=" * 60)
    print("METAGROKFAST DEMO")
    print("=" * 60)

    # Create test model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )

    # Create optimizer
    print("\n1. Creating MetaGrokfast for Phase 1 (Cognate)...")
    optimizer = MetaGrokfast.for_phase("phase1_cognate", model.parameters())
    print(f"   Config: lr={optimizer.config.lr}, lambda={optimizer.config.grokfast_lambda}")

    # Simulate training
    print("\n2. Simulating training steps...")
    for step in range(100):
        # Fake forward/backward
        x = torch.randn(32, 100)
        y = model(x)
        loss = y.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    # Show stats
    print("\n3. Optimization Statistics:")
    stats = optimizer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")

    # Show phase configs
    print("\n4. Available Phase Configs:")
    for phase in PHASE_CONFIGS:
        cfg = PHASE_CONFIGS[phase]
        print(
            f"   {phase}: lr={cfg.lr}, lambda={cfg.grokfast_lambda}, bigeometric={cfg.use_bigeometric}"
        )

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    compare_optimizers_demo()

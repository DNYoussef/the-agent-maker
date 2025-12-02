"""
MuGrokConfig: Unified configuration for Muon × GrokFast optimizer system.

This module provides the configuration dataclass for the MuonGrokfast optimizer,
which combines:
- Grokfast: Temporal gradient filtering (time-spectrum)
- Muon: Spatial gradient orthogonalization (space-geometry)
- QK-Clip: Attention safety rails

Author: Agent Forge V2 Team
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass
class MuGrokConfig:
    """
    Unified configuration for Muon × GrokFast optimizer.

    Combines three complementary techniques:
    1. **Grokfast** (time-spectrum): EMA-based gradient filtering
    2. **Muon** (space-geometry): Newton-Schulz orthogonalization
    3. **QK-Clip** (safety): Attention logit bounding

    Synergy: Temporal stability + spatial diversity = faster generalization,
    reduced low-rank collapse, smoother training trajectories.

    Phase-specific presets available via `from_phase()` class method.
    """

    # ========================================================================
    # MUON SETTINGS (Space-Geometry Orthogonalization)
    # ========================================================================

    muon_lr: float = 0.01
    """Learning rate for Muon optimizer (2-D parameters)."""

    muon_momentum: float = 0.95
    """Momentum coefficient for Muon (β in paper)."""

    muon_ns_steps: int = 5
    """
    Number of Newton-Schulz iterations for orthogonalization.
    - ns_steps > 0: Use Newton-Schulz (fast approximation)
    - ns_steps = 0: Use SVD (exact but slower)
    Default: 5 (per Muon paper)
    """

    muon_ns_coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    """
    Newton-Schulz polynomial coefficients (a, b, c).
    W_{i+1} = a·W_i + b·W_i³ + c·W_i⁵
    Default: Tuned values from Muon paper.
    """

    muon_ste_mode: bool = False
    """
    **Phase 5 CRITICAL**: STE (Straight-Through Estimator) compatibility mode.
    When True, applies Muon updates to full-precision weights while preserving
    quantized gradient flow. Required for BitNet quantization-aware training.
    """

    # ========================================================================
    # GROKFAST SETTINGS (Time-Spectrum Filtering)
    # ========================================================================

    grokfast_alpha: float = 0.98
    """
    EMA decay coefficient (α in paper).
    Higher α → more history retention → slower adaptation.
    Range: [0.9, 0.99]
    Default: 0.98 (per Grokfast paper)
    """

    grokfast_lambda: float = 0.05
    """
    Gradient amplification factor (λ in paper).
    Filtered gradient: ĝ_t = g_t + λ·μ_t
    - Low λ (0.05): Gentle filtering for pretraining
    - High λ (2.0): Aggressive filtering for fine-tuning
    Default: 0.05 (conservative)
    """

    # ========================================================================
    # QK-CLIP / MUONCLIP (Attention Safety Rails)
    # ========================================================================

    use_qk_clip: bool = True
    """Enable QK-clip/MuonClip for attention stability."""

    qk_clip_threshold: float = 30.0
    """
    Maximum allowed norm for Q/K projection weights.
    When ||W_Q|| or ||W_K|| exceeds threshold, rescale update to prevent
    unbounded attention logits.
    Default: 30.0 (per user guidance)
    """

    qk_clip_per_head: bool = True
    """
    Apply per-head scaling (respects MLA/decoupled-RoPE).
    When True, checks each attention head independently.
    Required for architectures with Multi-Latent Attention.
    """

    # ========================================================================
    # FALLBACK OPTIMIZER (1-D Parameters)
    # ========================================================================

    fallback_type: Literal["adamw", "lion"] = "adamw"
    """Optimizer for 1-D parameters (bias, LayerNorm, embeddings)."""

    fallback_lr: float = 1e-4
    """Learning rate for fallback optimizer."""

    fallback_betas: Tuple[float, float] = (0.9, 0.999)
    """Beta coefficients for AdamW (β1, β2)."""

    fallback_weight_decay: float = 0.01
    """Weight decay for fallback optimizer."""

    # ========================================================================
    # ABLATION FLAGS
    # ========================================================================

    enable_grokfast: bool = True
    """Enable Grokfast temporal filtering (disable for ablation studies)."""

    enable_muon: bool = True
    """Enable Muon spatial orthogonalization (disable for ablation studies)."""

    enable_muon_clip: bool = True
    """Enable QK-clip/MuonClip safety rails (disable for ablation studies)."""

    # ========================================================================
    # PHASE-SPECIFIC SETTINGS
    # ========================================================================

    kl_coefficient: Optional[float] = None
    """
    KL divergence regularization coefficient (Phase 3, 7 RL training).
    Loss = -reward·log_prob + kl_coefficient·D_KL(π_adapted || π_base)
    Default: None (not used in non-RL phases)
    """

    phase_name: str = "unknown"
    """Phase identifier for logging/debugging."""

    # ========================================================================
    # PHASE-SPECIFIC PRESETS
    # ========================================================================

    @classmethod
    def from_phase(cls, phase: int, **overrides) -> "MuGrokConfig":
        """
        Create phase-specific configuration with tested hyperparameters.

        Args:
            phase: Phase number (1, 3, 5, 6, 7)
            **overrides: Override specific config values

        Returns:
            MuGrokConfig instance with phase-specific defaults

        Examples:
            >>> config = MuGrokConfig.from_phase(1)  # Phase 1 pretraining
            >>> config = MuGrokConfig.from_phase(5, muon_lr=1e-4)  # Override
        """
        configs = {
            1: cls._phase1_cognate(),
            3: cls._phase3_quiet_star(),
            5: cls._phase5_forge(),
            6: cls._phase6_baking(),
            7: cls._phase7_transformer2(),
        }

        if phase not in configs:
            raise ValueError(
                f"Unknown phase: {phase}. "
                f"Supported phases: {list(configs.keys())}"
            )

        config = configs[phase]

        # Apply overrides
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown config key: {key}")
            setattr(config, key, value)

        return config

    @classmethod
    def _phase1_cognate(cls) -> "MuGrokConfig":
        """Phase 1: Cognate (TinyTitan pretraining)."""
        return cls(
            # Muon (2-D matrices: attention + MLP)
            muon_lr=0.01,
            muon_ns_steps=5,
            muon_momentum=0.95,

            # Grokfast (gentle filtering for pretraining)
            grokfast_alpha=0.98,
            grokfast_lambda=0.05,

            # QK-clip (attention stability)
            use_qk_clip=True,
            qk_clip_threshold=30.0,
            qk_clip_per_head=True,

            # Fallback (bias, LayerNorm)
            fallback_type="adamw",
            fallback_lr=1e-4,

            # Ablations
            enable_grokfast=True,
            enable_muon=True,
            enable_muon_clip=True,

            # Metadata
            phase_name="phase1_cognate"
        )

    @classmethod
    def _phase3_quiet_star(cls) -> "MuGrokConfig":
        """Phase 3: Quiet-STaR (thought training with REINFORCE)."""
        return cls(
            # Muon (lower LR for RL stability)
            muon_lr=5e-4,
            muon_ns_steps=3,  # Fewer iterations for speed
            muon_momentum=0.95,

            # Grokfast (lighter filtering for RL)
            grokfast_alpha=0.95,
            grokfast_lambda=0.1,

            # QK-clip (tighter bounds for RL)
            use_qk_clip=True,
            qk_clip_threshold=25.0,
            qk_clip_per_head=True,

            # Fallback (thought embeddings are 1-D)
            fallback_type="adamw",
            fallback_lr=1e-3,

            # RL-specific
            kl_coefficient=0.1,

            # Ablations
            enable_grokfast=True,
            enable_muon=True,
            enable_muon_clip=True,

            # Metadata
            phase_name="phase3_quiet_star"
        )

    @classmethod
    def _phase5_forge(cls) -> "MuGrokConfig":
        """Phase 5: Forge Training (BitNet + Grokfast with STE)."""
        return cls(
            # Muon (STE compatibility mode CRITICAL)
            muon_lr=2e-4,
            muon_ns_steps=5,
            muon_momentum=0.95,
            muon_ste_mode=True,  # **CRITICAL**: Preserves STE gradient flow

            # Grokfast (aggressive filtering per original Phase 5)
            grokfast_alpha=0.98,
            grokfast_lambda=2.0,

            # QK-clip (quantized attention stability)
            use_qk_clip=True,
            qk_clip_threshold=30.0,
            qk_clip_per_head=True,

            # Fallback
            fallback_type="adamw",
            fallback_lr=1e-4,

            # Ablations
            enable_grokfast=True,
            enable_muon=True,
            enable_muon_clip=True,

            # Metadata
            phase_name="phase5_forge"
        )

    @classmethod
    def _phase6_baking(cls) -> "MuGrokConfig":
        """
        Phase 6: Tool & Persona Baking.

        Note: Phase 6 has TWO stages:
        - Stage 1 (prompt tuning): Use this config with enable_muon=False
        - Stage 2 (weight merging): Use this config with enable_muon=True
        """
        return cls(
            # Muon (for stage 2 weight merging)
            muon_lr=1e-4,
            muon_ns_steps=3,
            muon_momentum=0.95,

            # Grokfast
            grokfast_alpha=0.98,
            grokfast_lambda=0.05,

            # QK-clip
            use_qk_clip=True,
            qk_clip_threshold=30.0,
            qk_clip_per_head=True,

            # Fallback (for stage 1 prompt embeddings)
            fallback_type="adamw",
            fallback_lr=5e-3,

            # Ablations (default to stage 2)
            enable_grokfast=True,
            enable_muon=True,
            enable_muon_clip=True,

            # Metadata
            phase_name="phase6_baking_stage2"
        )

    @classmethod
    def _phase7_transformer2(cls) -> "MuGrokConfig":
        """
        Phase 7: Transformer² SVF Training.

        Note: z vectors are 1-D → uses fallback path (no Muon).
        """
        return cls(
            # Muon (not used for 1-D z vectors)
            muon_lr=2e-3,  # Unused
            muon_ns_steps=0,
            enable_muon=False,

            # Grokfast (light filtering for RL stability)
            grokfast_alpha=0.98,
            grokfast_lambda=0.05,

            # QK-clip (applies to attention outputs during RL)
            use_qk_clip=True,
            qk_clip_threshold=30.0,
            qk_clip_per_head=True,
            enable_muon_clip=True,  # QK-clip independent of Muon

            # Fallback (z vectors are 1-D)
            fallback_type="adamw",
            fallback_lr=2e-3,  # Per Transformer² paper

            # RL-specific
            kl_coefficient=0.2,

            # Ablations
            enable_grokfast=True,

            # Metadata
            phase_name="phase7_transformer2"
        )

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate ranges
        if not 0.0 <= self.grokfast_alpha <= 1.0:
            raise ValueError(
                f"grokfast_alpha must be in [0, 1], got {self.grokfast_alpha}"
            )

        if self.muon_ns_steps < 0:
            raise ValueError(
                f"muon_ns_steps must be >= 0, got {self.muon_ns_steps}"
            )

        if self.qk_clip_threshold <= 0:
            raise ValueError(
                f"qk_clip_threshold must be > 0, got {self.qk_clip_threshold}"
            )

        # Warn if conflicting settings
        if self.muon_ste_mode and not self.enable_muon:
            import warnings
            warnings.warn(
                "muon_ste_mode=True but enable_muon=False. "
                "STE mode will have no effect."
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary (for logging/serialization)."""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = [f"MuGrokConfig(phase={self.phase_name})"]
        lines.append("  Muon:")
        lines.append(f"    lr={self.muon_lr}, ns_steps={self.muon_ns_steps}")
        lines.append(f"    ste_mode={self.muon_ste_mode}")
        lines.append("  Grokfast:")
        lines.append(f"    alpha={self.grokfast_alpha}, lambda={self.grokfast_lambda}")
        lines.append("  QK-Clip:")
        lines.append(f"    enabled={self.use_qk_clip}, threshold={self.qk_clip_threshold}")
        lines.append("  Fallback:")
        lines.append(f"    type={self.fallback_type}, lr={self.fallback_lr}")
        lines.append("  Ablations:")
        lines.append(f"    grokfast={self.enable_grokfast}, muon={self.enable_muon}, clip={self.enable_muon_clip}")
        return "\n".join(lines)

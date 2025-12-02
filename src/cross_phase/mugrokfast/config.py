"""
MuGrokfast Configuration
Phase-specific presets for optimal performance
"""

from dataclasses import dataclass


@dataclass
class MuGrokConfig:
    """MuGrokfast optimizer configuration"""

    muon_lr: float
    fallback_lr: float = 1e-3
    grokfast_alpha: float = 0.98
    grokfast_lambda: float = 2.0
    qk_clip_threshold: float = 30.0
    kl_coefficient: float = 0.0
    muon_ste_mode: bool = False
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    weight_decay: float = 0.0

    @classmethod
    def from_phase(cls, phase_num: int) -> "MuGrokConfig":
        """
        Get phase-specific preset configuration

        Args:
            phase_num: Phase number (1, 3, 5, 6, 7)

        Returns:
            MuGrokConfig with optimal settings for that phase
        """
        # Phase-specific presets (from V1 implementation)
        presets = {
            1: cls(  # Phase 1: Cognate (Pretraining)
                muon_lr=1e-3,
                fallback_lr=1e-3,
                grokfast_alpha=0.98,
                grokfast_lambda=0.3,  # Gentle filtering
                qk_clip_threshold=30.0,
                kl_coefficient=0.0,  # No KL (pretraining from scratch)
                muon_ste_mode=False,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                weight_decay=0.0,
            ),
            3: cls(  # Phase 3: Quiet-STaR (RL)
                muon_lr=5e-4,  # HIGHER for RL exploration
                fallback_lr=5e-4,
                grokfast_alpha=0.98,
                grokfast_lambda=0.1,  # LOWER for RL noise
                qk_clip_threshold=25.0,  # TIGHTER for RL attention spikes
                kl_coefficient=0.1,  # NEW: Prevent drift from baked baseline
                muon_ste_mode=False,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                weight_decay=0.0,
            ),
            5: cls(  # Phase 5: Curriculum Learning (BitNet STE)
                muon_lr=1e-3,
                fallback_lr=1e-3,
                grokfast_alpha=0.98,
                grokfast_lambda=2.0,  # AGGRESSIVE filtering
                qk_clip_threshold=30.0,
                kl_coefficient=0.0,
                muon_ste_mode=True,  # CRITICAL: STE mode for BitNet
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                weight_decay=0.0,
            ),
            6: cls(  # Phase 6: Tool & Persona Baking
                muon_lr=1e-4,  # Lower for fine-tuning
                fallback_lr=1e-4,
                grokfast_alpha=0.98,
                grokfast_lambda=0.2,  # Moderate filtering
                qk_clip_threshold=30.0,
                kl_coefficient=0.0,  # No KL (we WANT to change model)
                muon_ste_mode=False,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                weight_decay=0.0,
            ),
            7: cls(  # Phase 7: Self-Guided Experts (SVF Training)
                muon_lr=5e-4,
                fallback_lr=5e-4,
                grokfast_alpha=0.98,
                grokfast_lambda=0.15,
                qk_clip_threshold=28.0,
                kl_coefficient=0.05,
                muon_ste_mode=False,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                weight_decay=0.0,
            ),
        }

        if phase_num not in presets:
            raise ValueError(
                f"Phase {phase_num} not in MuGrokfast presets. "
                f"Available: {list(presets.keys())}"
            )

        return presets[phase_num]

    @classmethod
    def custom(
        cls,
        muon_lr: float = 0.01,
        fallback_lr: float = 1e-3,
        grokfast_alpha: float = 0.98,
        grokfast_lambda: float = 2.0,
        qk_clip_threshold: float = 30.0,
        kl_coefficient: float = 0.0,
        muon_ste_mode: bool = False,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> "MuGrokConfig":
        """Create custom configuration"""
        return cls(
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
            weight_decay=weight_decay,
        )

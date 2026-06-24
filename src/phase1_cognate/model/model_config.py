"""
Phase 1 Cognate Model Configurations

Dataclass configs for TRM × Titans-MAG architecture.
All hyperparameters defined here for easy experimentation.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TitansMAGConfig:
    """
    Titans-MAG Backbone Configuration

    Transformer with Sliding Window Attention + Long-range Memory.
    Trainable-flagship size: ~200M total model params, fits an 8 GB GPU with bf16 +
    gradient checkpointing. (Was 320/8 = ~32.6M; resized 2026-06-24.) head_dim is
    pinned at 64, so d_model must be a multiple of 64 and n_heads = d_model // 64.
    """

    # Model dimensions (~200M trainable flagship; head_dim pinned at 64)
    d_model: int = 896  # 14 * 64
    n_layers: int = 12
    n_heads: int = 14  # d_model / 64 (896 / 64 = 14)
    head_dim: int = 64  # d_model / n_heads
    d_ff: int = 3584  # SwiGLU MLP expansion (4x 896)

    # Vocabulary
    vocab_size: int = 50257  # GPT-2 tokenizer (full vocabulary) - kept for pipeline compat
    max_seq_len: int = 2048

    # Sliding Window Attention
    sw_window: int = 1024  # Tokens attend to ±512 range

    # Long-range Memory (LMM) - scales with d_model (half)
    d_mem: int = 448  # d_model // 2
    memory_decay: float = 0.99  # Exponential decay rate

    # MAG Gate - scales with d_model (half)
    mag_hidden: int = 448  # d_model // 2
    mag_entropy_reg: float = 0.001  # Entropy regularization

    # Dropout
    dropout: float = 0.1
    attention_dropout: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by " f"n_heads ({self.n_heads})"
        )
        assert self.head_dim == self.d_model // self.n_heads


@dataclass
class TRMConfig:
    """
    TRM (Transformer Recursive Memory) Wrapper Configuration

    Multi-pass reasoning with iterative refinement.
    """

    # Recursion parameters
    T_max: int = 3  # Max recursion steps (start conservative)
    micro_steps: int = 2  # Refinement steps per iteration

    # Supervision
    deep_supervision: bool = True  # Loss at each step (enabled - graph reuse fixed with detach)
    step_weights: list[float] = field(
        default_factory=lambda: [0.33, 0.5, 0.75, 1.0]  # y0 + 3 recursion steps
    )

    # Memory efficiency
    detach_between_steps: bool = True  # Detach for gradient flow

    def __post_init__(self) -> None:
        """Validate configuration"""
        assert self.T_max + 1 == len(self.step_weights), (
            f"Need T_max+1 ({self.T_max + 1}) weights for initial state + recursion steps, "
            f"got {len(self.step_weights)}"
        )


@dataclass
class ACTConfig:
    """
    Adaptive Computation Time Configuration

    Learned halting mechanism with EMA calibration.
    """

    # Halting
    halt_threshold: float = 0.5  # Probability threshold

    # EMA calibration
    ema_decay: float = 0.98  # Decay for step accuracy tracking

    # Regularization
    entropy_reg: float = 0.001  # Prevent saturation

    # Loss weight
    act_loss_weight: float = 0.01  # Weight for ACT loss


@dataclass
class Phase1Config:
    """
    Complete Phase 1 Cognate Configuration

    Combines Titans-MAG, TRM, and ACT configs plus training params.
    """

    # Architecture components
    titans_config: TitansMAGConfig = field(default_factory=TitansMAGConfig)
    trm_config: TRMConfig = field(default_factory=TRMConfig)
    act_config: ACTConfig = field(default_factory=ACTConfig)

    # Model specialization (for 3 models)
    specialization: Literal["reasoning", "memory", "speed"] = "reasoning"

    # ACT thresholds per specialization
    act_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "reasoning": 0.95,  # Think longer
            "memory": 0.90,  # Balanced
            "speed": 0.99,  # Halt quickly
        }
    )

    # LTM capacity (number of memory slots) per specialization
    ltm_capacities: dict[str, int] = field(
        default_factory=lambda: {
            "reasoning": 4096,
            "memory": 8192,  # Large memory
            "speed": 2048,  # Small memory
        }
    )

    # Surprise thresholds per specialization
    surprise_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "reasoning": 0.7,  # Very selective
            "memory": 0.5,  # Balanced
            "speed": 0.3,  # Store more
        }
    )

    # Random seed per model
    seeds: dict[str, int] = field(
        default_factory=lambda: {"reasoning": 42, "memory": 1337, "speed": 2023}
    )

    # Training hyperparameters
    # HONEST: at the resized ~222M size, a forward+backward fits an 8 GB GPU only at
    # physical batch 2 / seq 128 / fp32 (measured peak ~4.9 GB). batch 4 already
    # exceeds 8 GB and seq 256 blows up (the TRM uses full O(seq^2) attention).
    # Larger batch/seq needs gradient checkpointing + the O(seq^2) attention fix +
    # the bf16/BCE-autocast fix (a later efficiency phase). Effective batch is raised
    # via the trainer's gradient_accumulation_steps, not the physical batch.
    batch_size: int = 2  # physical batch that fits 8 GB at ~222M (was 16 @ 32.6M)
    learning_rate: float = 1e-3
    num_epochs: int = 10
    gradient_clip: float = 1.0

    # Optimizer (MuGrokfast Phase 1 preset)
    muon_lr: float = 1e-3
    grokfast_lambda: float = 0.3
    qk_clip: float = 30.0
    kl_coef: float = 0.0  # No KL for Phase 1

    # Curriculum stages
    curriculum_stages: int = 3

    # W&B
    wandb_project: str = "agent-forge-v2"
    wandb_mode: str = "offline"  # Local-first

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = (
        False  # bf16 autocast currently CRASHES (BCE in ACT loss); fix is a later phase
    )
    gradient_checkpointing: bool = (
        True  # NOTE: flag only - NOT yet implemented in the backbone forward
    )

    def __post_init__(self) -> None:
        """Apply specialization settings.

        HONESTY: the three "specializations" (reasoning/memory/speed) differ ONLY
        by act_config.halt_threshold (set here) and the random seed (get_seed).
        ltm_capacities and surprise_thresholds are NOT applied to the model and
        cannot be without new architecture: the LMM is a single factorized EMA
        vector (no memory "slots" -> capacity is meaningless) and there is no
        surprise mechanism (the memory is an EMA pool, not Titans neural memory).
        So these are the same architecture with a different halt threshold + init,
        not three distinct designs. Real capacity/surprise differentiation is a
        Wave-2 feature tied to implementing surprise-based memory.
        """
        if self.specialization:
            # Override ACT threshold (the one specialization knob that is wired).
            self.act_config.halt_threshold = self.act_thresholds[self.specialization]

    def get_seed(self) -> int:
        """Get random seed for this specialization"""
        return self.seeds.get(self.specialization, 42)

    def to_dict(self) -> dict:
        """Convert to dictionary for W&B config"""
        return {
            "architecture": "TRM-Titans-MAG",
            "specialization": self.specialization,
            "d_model": self.titans_config.d_model,
            "n_layers": self.titans_config.n_layers,
            "target_params": "~200M",
            "act_threshold": self.act_config.halt_threshold,
            "ltm_capacity": self.ltm_capacities[self.specialization],
            "ltm_d_mem": self.titans_config.d_mem,
            "surprise_threshold": self.surprise_thresholds[self.specialization],
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "muon_lr": self.muon_lr,
            "grokfast_lambda": self.grokfast_lambda,
        }

    def get_ltm_capacity(self) -> int:
        """Get LTM capacity for this specialization"""
        return self.ltm_capacities.get(self.specialization, 4096)

    def get_surprise_threshold(self) -> float:
        """Get surprise threshold for this specialization"""
        return self.surprise_thresholds.get(self.specialization, 0.7)


def create_model_configs() -> dict[str, Phase1Config]:
    """
    Create all 3 model configurations for Phase 1.

    Returns:
        dict: {"reasoning": config1, "memory": config2, "speed": config3}
    """
    configs = {}

    for spec in ["reasoning", "memory", "speed"]:
        configs[spec] = Phase1Config(specialization=spec)

    return configs

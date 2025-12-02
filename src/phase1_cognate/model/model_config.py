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

    8-layer transformer with Sliding Window Attention + Long-range Memory.
    Target: ~20M params for backbone alone.
    """

    # Model dimensions (adjusted for 25M target)
    d_model: int = 320  # Reverted to 320 to match checkpoints
    n_layers: int = 8  # Reverted to 8 to match checkpoints
    n_heads: int = 5  # d_model / 64 (320/64 = 5)
    head_dim: int = 64  # d_model / n_heads
    d_ff: int = 1280  # SwiGLU MLP expansion (4x 320)

    # Vocabulary
    vocab_size: int = 50257  # GPT-2 tokenizer (full vocabulary)
    max_seq_len: int = 2048

    # Sliding Window Attention
    sw_window: int = 1024  # Tokens attend to ±512 range

    # Long-range Memory (LMM)
    d_mem: int = 160  # Reverted to 160 (half of 320)
    memory_decay: float = 0.99  # Exponential decay rate

    # MAG Gate
    mag_hidden: int = 160  # Reverted to 160
    mag_entropy_reg: float = 0.001  # Entropy regularization

    # Dropout
    dropout: float = 0.1
    attention_dropout: float = 0.1

    def __post_init__(self):
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

    def __post_init__(self):
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
    batch_size: int = 16  # Fits in 6GB VRAM
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
    mixed_precision: bool = False  # Optional
    gradient_checkpointing: bool = True  # Memory efficiency

    def __post_init__(self):
        """Apply specialization settings"""
        if self.specialization:
            # Override ACT threshold
            self.act_config.halt_threshold = self.act_thresholds[self.specialization]

            # Note: LTM capacity will be handled in model initialization
            # titans_config.d_mem stays at 256 (factorized dimension)

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
            "target_params": "25M",
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

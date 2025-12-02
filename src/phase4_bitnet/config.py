"""
Phase 4 Configuration
Size-adaptive compression targets with quality gates
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path


@dataclass
class Phase4Config:
    """
    BitNet 1.58-bit compression configuration

    Attributes:
        model_path: Input model from Phase 3
        output_path: Output directory for compressed models
        device: Computation device ('cuda', 'cpu', 'auto')
        quantization_bits: Fixed at 1.58 for BitNet
        preserve_embedding_precision: Keep embeddings in FP16
        preserve_output_precision: Keep LM head in FP16
        sparsity_threshold: Threshold for zero injection
        calibration_samples: Number of calibration samples
        calibration_dataset: Dataset name
        calibration_batch_size: Batch size for calibration
        calibration_sequence_length: Max sequence length
        enable_fine_tuning: Auto fine-tune if needed
        fine_tune_epochs: Number of fine-tuning epochs
        fine_tune_lr: Fine-tuning learning rate
        warmup_steps: Warmup steps for fine-tuning
        enable_grokfast: Use MuGrokfast optimizer
        grokfast_ema_alpha: Grokfast EMA decay
        grokfast_lambda: Grokfast amplification factor
        target_compression_ratio: Compression goal
        max_accuracy_drop: Max acceptable accuracy loss
        fine_tune_threshold: Trigger fine-tuning threshold
        mixed_precision: Use FP16 for compute
        seed: Random seed
        wandb_enabled: Enable W&B logging
        wandb_project: W&B project name
        wandb_tags: W&B tags
    """

    # Paths
    model_path: str = ""
    output_path: str = "phase4_bitnet_output"

    # Hardware
    device: str = "auto"  # auto, cuda, cpu

    # BitNet quantization
    quantization_bits: float = 1.58
    preserve_embedding_precision: bool = True
    preserve_output_precision: bool = True
    sparsity_threshold: float = 0.1  # Adaptive, overridden by size

    # Preserved layer patterns
    preserve_layers: List[str] = field(default_factory=lambda: [
        "embeddings",
        "lm_head",
        "layer_norm",
    ])

    # Calibration
    calibration_samples: int = 1000
    calibration_dataset: str = "openwebtext"
    calibration_batch_size: int = 4
    calibration_sequence_length: int = 512

    # Fine-tuning
    enable_fine_tuning: bool = True
    fine_tune_epochs: int = 2
    fine_tune_lr: float = 1e-5
    warmup_steps: int = 100

    # STE Fine-tuning enhancements (ISS-008)
    use_mixed_precision: bool = True
    max_grad_norm: float = 1.0
    save_every_n_epochs: int = 1
    checkpoint_dir: str = "checkpoints/phase4"

    # Grokfast
    enable_grokfast: bool = True
    grokfast_ema_alpha: float = 0.98
    grokfast_lambda: float = 2.0  # Aggressive for recovery

    # Compression targets (size-adaptive)
    compression_targets: Dict[str, float] = field(default_factory=lambda: {
        "tiny": 6.0,    # <50M params
        "small": 8.0,   # <500M params
        "medium": 10.0, # <2B params
        "large": 12.0,  # >2B params
    })

    # Sparsity thresholds (size-adaptive)
    sparsity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "tiny": 0.05,
        "small": 0.10,
        "medium": 0.15,
        "large": 0.20,
    })

    # Quality gates
    target_compression_ratio: float = 8.0  # Overridden by size
    max_accuracy_drop: float = 0.10  # 10%
    fine_tune_threshold: float = 0.05  # Fine-tune if drop >5%

    # System
    mixed_precision: bool = True
    seed: int = 42
    num_workers: int = 4

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "agent-forge-v2"
    wandb_tags: List[str] = field(default_factory=lambda: [
        "phase4", "bitnet", "compression"
    ])

    # Output options
    save_quantized: bool = True  # Save 1.58-bit model
    save_dequantized_fp16: bool = True  # PRIMARY for Phase 5

    def __post_init__(self):
        """Validate configuration"""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.quantization_bits != 1.58:
            raise ValueError(
                "Phase 4 uses BitNet 1.58-bit quantization only"
            )

        if self.calibration_samples < 100:
            raise ValueError(
                "Calibration requires at least 100 samples"
            )

        if not (0.0 <= self.max_accuracy_drop <= 0.5):
            raise ValueError(
                "max_accuracy_drop must be between 0.0 and 0.5"
            )

    def get_size_category(self, num_params: int) -> str:
        """
        Determine model size category

        Args:
            num_params: Number of model parameters

        Returns:
            Size category: 'tiny', 'small', 'medium', or 'large'
        """
        if num_params < 50_000_000:
            return "tiny"
        elif num_params < 500_000_000:
            return "small"
        elif num_params < 2_000_000_000:
            return "medium"
        else:
            return "large"

    def adapt_to_model_size(self, num_params: int):
        """
        Adapt compression targets to model size

        Args:
            num_params: Number of model parameters
        """
        size_category = self.get_size_category(num_params)

        self.target_compression_ratio = self.compression_targets[
            size_category
        ]
        self.sparsity_threshold = self.sparsity_thresholds[
            size_category
        ]

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "Phase4Config":
        """
        Create config from dictionary

        Args:
            config_dict: Configuration dictionary

        Returns:
            Phase4Config instance
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """
        Convert config to dictionary

        Returns:
            Configuration dictionary
        """
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

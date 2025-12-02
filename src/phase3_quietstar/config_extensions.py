"""
Phase 3 Quiet-STaR Configuration Extensions

Additional configuration parameters for Quiet-STaR paper implementation
that enhance the base config with parallel sampling and teacher forcing.

Usage:
    from .config import QuietSTaRRLConfig
    from .config_extensions import extend_rl_config

    rl_config = QuietSTaRRLConfig()
    rl_config = extend_rl_config(rl_config)
"""

from dataclasses import dataclass
from typing import Optional


def extend_rl_config(config):
    """
    Extend RL config with Quiet-STaR paper hyperparameters.

    Adds:
        - meta_token_grad_scale: Gradient amplification for thinking tokens (paper mentions 100.0)
        - n_true: Number of future tokens for non-myopic loss (paper uses 4)
        - use_parallel_generation: Enable parallel thought sampling with diagonal mask

    Args:
        config: QuietSTaRRLConfig instance

    Returns:
        Extended config with additional attributes
    """
    # Meta-token gradient weighting (Quiet-STaR paper hyperparameter)
    # Paper mentions this amplifies gradients for special thinking tokens
    config.meta_token_grad_scale = getattr(config, "meta_token_grad_scale", 100.0)

    # Non-myopic loss configuration
    # Paper: "We consider n_true=4 future tokens for semantic content"
    config.n_true = getattr(config, "n_true", 4)

    # Parallel generation toggle
    # Uses diagonal attention mask (Section 4.2, Figure 3)
    config.use_parallel_generation = getattr(config, "use_parallel_generation", True)

    return config


@dataclass
class ParallelSamplingConfig:
    """
    Configuration for parallel thought generation.

    Based on Quiet-STaR paper Section 4.2 "Parallel Thought Sampling"
    """

    # Enable parallel sampling (vs sequential)
    enabled: bool = True

    # Diagonal mask configuration
    allow_cross_attention: bool = False  # Thoughts don't attend to each other
    shared_context_window: Optional[int] = None  # All thoughts attend to this (None = all context)

    # Efficiency tuning
    batch_parallel_thoughts: bool = True  # Generate all thoughts in single batch
    recompute_hidden_states: bool = False  # Cache hidden states across thoughts


@dataclass
class TeacherForcingConfig:
    """
    Configuration for teacher-forced non-myopic loss.

    Based on Quiet-STaR paper Section 3.2 "Non-Myopic Loss"
    """

    # Enable teacher forcing
    enabled: bool = True

    # Number of future tokens to consider
    n_true: int = 4  # Paper uses 4 for semantic content

    # Loss weighting
    teacher_forcing_weight: float = 0.5  # Weight relative to REINFORCE loss

    # Future token handling
    use_ground_truth: bool = True  # Use actual future tokens (vs model predictions)
    ignore_padding: bool = True  # Ignore -100 padding tokens in loss


@dataclass
class MetaTokenConfig:
    """
    Configuration for thinking token gradient scaling.

    Quiet-STaR paper mentions amplifying gradients for special thinking tokens
    (e.g., <think>, </think>) to ensure they're properly learned.
    """

    # Enable gradient scaling
    enabled: bool = True

    # Scaling factor (paper mentions 100.0)
    grad_scale: float = 100.0

    # Token IDs to scale (set at runtime after tokenizer adds special tokens)
    token_ids: Optional[list] = None

    # Scaling mode
    mode: str = "multiply"  # "multiply" or "clamp"
    clamp_min: float = -10.0  # If mode="clamp"
    clamp_max: float = 10.0  # If mode="clamp"


def create_complete_rl_config():
    """
    Create a complete RL config with all Quiet-STaR extensions.

    Returns:
        Dict with all configuration components
    """
    from .config import QuietSTaRRLConfig

    base_config = QuietSTaRRLConfig()
    extended_config = extend_rl_config(base_config)

    return {
        "base": extended_config,
        "parallel_sampling": ParallelSamplingConfig(),
        "teacher_forcing": TeacherForcingConfig(),
        "meta_token": MetaTokenConfig(),
    }

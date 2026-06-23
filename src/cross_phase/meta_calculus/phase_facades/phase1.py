"""
Phase 1: Cognate (Pre-training) Facade

Creates 3 diverse 25M parameter models with TRM x Titans-MAG architecture.
This facade provides:
- MetaGrokfast optimizer configured for Phase 1
- Spectral gap monitoring for diversity
- NAS (architecture search) via MOO

Usage:
    from src.cross_phase.meta_calculus.phase_facades import phase1

    # Create optimizer
    optimizer = phase1.create_optimizer(model)

    # Monitor spectral gap during training
    monitor = phase1.create_gap_monitor()
    gap = monitor.compute_gap(embeddings)

    # Architecture search (optional)
    result = phase1.search_architecture(evaluator)
"""

from typing import Any, Callable, Dict, Optional

from ..gap_utils.monitoring import GapHealthStatus, PhaseGapMonitor

# Layer 2 imports (utilities)
from ..k_utils.scheduling import KScheduleConfig, create_k_lr_scheduler, k_learning_rate_schedule

# Layer 1 imports (core)
from ..meta_grokfast import GrokfastConfig, MetaGrokfast
from ..moo_utils.architecture import ArchitectureSearchConfig, ArchitectureSearchProblem
from ..moo_utils.architecture import search_architecture as _search_architecture
from ..spectral_gap import SpectralGapMonitor

# Phase 1 specific defaults
PHASE1_CONFIG = GrokfastConfig(
    alpha=0.98,
    lamb=0.3,  # Gentle for pretraining
    filter_type="bigeometric",
    warmup_steps=500,
)


def create_optimizer(
    model: Any,
    lr: float = 1e-4,
    config: Optional[GrokfastConfig] = None,
    **adamw_kwargs,
) -> MetaGrokfast:
    """
    Create Phase 1 optimizer with k(L) adaptive gradient filtering.

    Args:
        model: PyTorch model
        lr: Learning rate
        config: Optional custom GrokfastConfig
        **adamw_kwargs: Additional kwargs for AdamW

    Returns:
        MetaGrokfast optimizer configured for Phase 1
    """
    if config is None:
        return MetaGrokfast.for_phase("phase1_cognate", model, lr, **adamw_kwargs)

    return MetaGrokfast(model, config=config, lr=lr, **adamw_kwargs)


def create_gap_monitor(
    warning_threshold: float = 0.1,
    critical_threshold: float = 0.05,
) -> PhaseGapMonitor:
    """
    Create spectral gap monitor for Phase 1 training.

    Monitors diversity collapse during pretraining.

    Args:
        warning_threshold: Gap below this triggers warning
        critical_threshold: Gap below this is critical

    Returns:
        PhaseGapMonitor configured for Phase 1
    """
    return PhaseGapMonitor("phase1_cognate")


def create_lr_scheduler(
    optimizer: Any,
    total_steps: int,
    warmup_steps: int = 500,
) -> Callable[[int], float]:
    """
    Create k(L)-adaptive learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total training steps
        warmup_steps: Warmup steps

    Returns:
        LR scheduler function
    """
    config = KScheduleConfig(warmup_steps=warmup_steps)
    return create_k_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        config=config,
    )


def search_architecture(
    evaluator: Callable,
    n_generations: int = 50,
    pop_size: int = 20,
    config: Optional[ArchitectureSearchConfig] = None,
) -> Any:
    """
    Run neural architecture search for Phase 1 model.

    Args:
        evaluator: Function (x) -> (param_count, latency, perplexity, memory)
        n_generations: Number of evolution generations
        pop_size: Population size
        config: Optional architecture search config

    Returns:
        pymoo Result with Pareto-optimal architectures
    """
    config = config or ArchitectureSearchConfig(
        n_vars=6,  # hidden_dim, n_layers, n_heads, ff_mult, dropout, activation
        n_objectives=4,  # param_count, latency, perplexity, memory
        xl=[128, 4, 4, 2.0, 0.0, 0],  # Lower bounds
        xu=[1024, 12, 16, 8.0, 0.3, 3],  # Upper bounds
    )
    return _search_architecture(
        evaluator,
        n_generations=n_generations,
        pop_size=pop_size,
        config=config,
    )


def get_k_value(layer_idx: int, total_layers: int) -> float:
    """
    Get k(L) value for a specific layer.

    Args:
        layer_idx: Layer index (0-indexed)
        total_layers: Total number of layers

    Returns:
        k value for this layer
    """
    from ..k_formula import compute_k

    L = (layer_idx + 1) / total_layers  # Normalize to [0, 1]
    return compute_k(L)


__all__ = [
    "create_optimizer",
    "create_gap_monitor",
    "create_lr_scheduler",
    "search_architecture",
    "get_k_value",
    "PHASE1_CONFIG",
]

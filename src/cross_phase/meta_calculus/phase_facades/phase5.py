"""
Phase 5: Curriculum Facade

7-stage adaptive curriculum with virtue baking (Honesty, Empathy, Growth, Respect).
This facade provides:
- k(L) difficulty scheduling
- Stage advancement via spectral gap
- Virtue weight optimization via MOO
- Curriculum schedule optimization

Usage:
    from src.cross_phase.meta_calculus.phase_facades import phase5

    # Get current difficulty
    difficulty = phase5.get_stage_difficulty(stage=3, total_stages=7)

    # Check if ready to advance
    should_advance = phase5.should_advance_stage(gap_history, current_stage)

    # Optimize curriculum schedule
    result = phase5.optimize_curriculum(evaluator)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

# Layer 1 imports (core)
from ..meta_grokfast import MetaGrokfast, GrokfastConfig

# Layer 2 imports (utilities)
from ..k_utils.scheduling import (
    k_difficulty_schedule,
    k_warmup_schedule,
    KScheduleConfig,
    get_stage_difficulties,
)
from ..gap_utils.monitoring import (
    PhaseGapMonitor,
    should_advance_stage as _should_advance_stage,
    get_stage_advancement_info,
    GapHealthStatus,
)
from ..moo_utils.hyperparams import (
    optimize_curriculum_schedule,
    CurriculumScheduleProblem,
)

# Phase 5 specific defaults
PHASE5_CONFIG = GrokfastConfig(
    alpha=0.97,
    lamb=1.0,
    filter_type="bigeometric",
    warmup_steps=100,
)

# Virtue definitions
VIRTUES = ["honesty", "empathy", "growth", "respect"]


def create_optimizer(
    model: Any,
    lr: float = 1e-4,
    config: Optional[GrokfastConfig] = None,
    **adamw_kwargs,
) -> MetaGrokfast:
    """
    Create Phase 5 optimizer for curriculum learning.

    Args:
        model: PyTorch model
        lr: Learning rate
        config: Optional custom config
        **adamw_kwargs: Additional AdamW kwargs

    Returns:
        MetaGrokfast configured for Phase 5
    """
    if config is None:
        return MetaGrokfast.for_phase("phase5_curriculum", model, lr, **adamw_kwargs)

    return MetaGrokfast(model, config=config, lr=lr, **adamw_kwargs)


def get_stage_difficulty(
    stage: int,
    total_stages: int = 7,
    base_difficulty: float = 0.3,
) -> float:
    """
    Get k(L)-adaptive difficulty for curriculum stage.

    Early stages: lower difficulty (foundational)
    Later stages: higher difficulty (challenging)

    Args:
        stage: Current stage (1-indexed)
        total_stages: Total number of stages
        base_difficulty: Base difficulty level

    Returns:
        Difficulty value [0.0-1.0]
    """
    return k_difficulty_schedule(
        stage=stage,
        total_stages=total_stages,
        base_difficulty=base_difficulty,
    )


def get_all_difficulties(
    total_stages: int = 7,
    base_difficulty: float = 0.3,
) -> List[float]:
    """
    Get difficulties for all curriculum stages.

    Args:
        total_stages: Number of stages
        base_difficulty: Base difficulty

    Returns:
        List of difficulties for each stage
    """
    return get_stage_difficulties(total_stages, base_difficulty)


def create_gap_monitor() -> PhaseGapMonitor:
    """
    Create spectral gap monitor for curriculum progression.

    Returns:
        PhaseGapMonitor configured for Phase 5
    """
    return PhaseGapMonitor("phase5_curriculum")


def should_advance_stage(
    gap_history: List[float],
    current_stage: int,
    total_stages: int = 7,
    stability_window: int = 10,
) -> bool:
    """
    Check if curriculum should advance to next stage.

    Advances when spectral gap is stable (model has learned current stage).

    Args:
        gap_history: Recent spectral gap values
        current_stage: Current curriculum stage
        total_stages: Total stages
        stability_window: Window for stability check

    Returns:
        True if should advance
    """
    if current_stage >= total_stages:
        return False
    return _should_advance_stage(
        gap_history,
        patience=stability_window,
    )


def get_advancement_info(
    gap_history: List[float],
    current_stage: int,
) -> Dict[str, Any]:
    """
    Get detailed info about stage advancement decision.

    Args:
        gap_history: Recent gap values
        current_stage: Current stage

    Returns:
        Dict with stability metrics, readiness score, etc.
    """
    return get_stage_advancement_info(gap_history)


def get_virtue_weights(
    stage: int,
    total_stages: int = 7,
) -> Dict[str, float]:
    """
    Get virtue weights for current curriculum stage.

    Early stages emphasize foundational virtues (honesty).
    Later stages balance all virtues.

    Args:
        stage: Current stage
        total_stages: Total stages

    Returns:
        Dict mapping virtue name to weight
    """
    # k(L) determines how much to emphasize later virtues
    from ..k_formula import compute_k
    L = stage / total_stages
    k = compute_k(L)

    # Early (high k): focus on honesty/empathy
    # Later (low k): balance all virtues
    weights = {
        "honesty": 1.0,  # Always important
        "empathy": 0.5 + 0.5 * (1 - k),  # Grows with stage
        "growth": 0.3 + 0.7 * (1 - k),  # Grows with stage
        "respect": 0.3 + 0.7 * (1 - k),  # Grows with stage
    }

    # Normalize
    total = sum(weights.values())
    return {v: w / total for v, w in weights.items()}


def optimize_curriculum(
    evaluator: Callable,
    n_stages: int = 7,
    n_generations: int = 30,
) -> Any:
    """
    Optimize curriculum schedule via MOO.

    Objectives: learning_efficiency, final_performance, stability

    Args:
        evaluator: Function (schedule_config) -> objectives
        n_stages: Number of curriculum stages
        n_generations: Evolution generations

    Returns:
        Pareto-optimal curriculum schedules
    """
    from ..moo_utils.hyperparams import HyperparamSearchConfig

    config = HyperparamSearchConfig(n_generations=n_generations)
    return optimize_curriculum_schedule(
        n_stages=n_stages,
        curriculum_evaluator=evaluator,
        config=config,
    )


def get_eudaimonia_score(
    virtue_scores: Dict[str, float],
    stage: int,
    total_stages: int = 7,
) -> float:
    """
    Compute eudaimonia (flourishing) score.

    Weighted combination of virtue scores adjusted by stage.

    Args:
        virtue_scores: Dict of virtue name to score [0-1]
        stage: Current stage
        total_stages: Total stages

    Returns:
        Eudaimonia score [0-1]
    """
    weights = get_virtue_weights(stage, total_stages)

    score = 0.0
    for virtue, weight in weights.items():
        if virtue in virtue_scores:
            score += weight * virtue_scores[virtue]

    return min(1.0, max(0.0, score))


__all__ = [
    "create_optimizer",
    "get_stage_difficulty",
    "get_all_difficulties",
    "create_gap_monitor",
    "should_advance_stage",
    "get_advancement_info",
    "get_virtue_weights",
    "optimize_curriculum",
    "get_eudaimonia_score",
    "VIRTUES",
    "PHASE5_CONFIG",
]

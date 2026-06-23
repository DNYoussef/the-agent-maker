"""
k-Formula Scheduling Utilities

Learning rate schedules, difficulty curves, and warmup schedules
based on the k(L) = -0.0137*log10(L) + 0.1593 formula.

Phase Applications:
    - Phase 1 (Cognate): k-based LR scheduling during pretraining
    - Phase 5 (Curriculum): k-based difficulty curve for 7-stage curriculum
    - All phases: Warmup schedules with k-adaptation
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

# Layer 1 import only
from ..k_formula import K_MAX, K_MIN, compute_k, normalize_k_value


@dataclass
class KScheduleConfig:
    """Configuration for k-based schedules."""

    # Base values
    base_value: float = 1.0

    # k-formula parameters (can override defaults)
    k_min: float = K_MIN
    k_max: float = K_MAX

    # Schedule modifiers
    scale_factor: float = 1.0
    offset: float = 0.0

    # Warmup
    warmup_steps: int = 0
    warmup_start_factor: float = 0.1


# =============================================================================
# LEARNING RATE SCHEDULES (Phase 1, all phases)
# =============================================================================


def k_learning_rate_schedule(
    base_lr: float,
    step: int,
    total_steps: int,
    config: Optional[KScheduleConfig] = None,
) -> float:
    """
    k(L)-based learning rate schedule.

    LR decreases as training progresses because k decreases with progress.
    Formula: lr = base_lr * k(step/total_steps)

    Args:
        base_lr: Base learning rate
        step: Current training step
        total_steps: Total training steps
        config: Optional configuration

    Returns:
        Adjusted learning rate

    Example:
        >>> for step in range(1000):
        ...     lr = k_learning_rate_schedule(1e-3, step, 1000)
        ...     optimizer.param_groups[0]['lr'] = lr
    """
    if total_steps <= 0:
        return base_lr

    config = config or KScheduleConfig()

    # Handle warmup
    if step < config.warmup_steps:
        warmup_factor = config.warmup_start_factor + (1 - config.warmup_start_factor) * (
            step / config.warmup_steps
        )
        return base_lr * warmup_factor

    # Compute progress ratio (avoid log(0))
    progress = max(step / total_steps, 1e-10)

    # Get k value for this progress
    k = compute_k(progress)

    # Apply scale and offset
    k_adjusted = k * config.scale_factor + config.offset

    # Clamp to valid range
    k_adjusted = max(config.k_min, min(config.k_max, k_adjusted))

    return base_lr * k_adjusted


def create_k_lr_scheduler(
    optimizer,
    total_steps: int,
    config: Optional[KScheduleConfig] = None,
):
    """
    Create a PyTorch LR scheduler using k(L) formula.

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total training steps
        config: Optional configuration

    Returns:
        LambdaLR scheduler

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = create_k_lr_scheduler(optimizer, 10000)
        >>> for step in range(10000):
        ...     train_step()
        ...     scheduler.step()
    """
    try:
        from torch.optim.lr_scheduler import LambdaLR
    except ImportError:
        raise ImportError("PyTorch required for create_k_lr_scheduler")

    config = config or KScheduleConfig()
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(step: int) -> float:
        # Return multiplier (not absolute LR)
        adjusted_lr = k_learning_rate_schedule(base_lr, step, total_steps, config)
        return adjusted_lr / base_lr

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# DIFFICULTY SCHEDULES (Phase 5 Curriculum)
# =============================================================================


def k_difficulty_schedule(
    stage: int,
    total_stages: int,
    base_difficulty: float = 1.0,
    min_difficulty: float = 0.1,
    max_difficulty: float = 2.0,
) -> float:
    """
    k(L)-based difficulty schedule for curriculum learning.

    Early stages (high k) = easier, later stages (low k) = harder.
    Formula: difficulty = base * (1 + scale * (k_max - k) / (k_max - k_min))

    Args:
        stage: Current curriculum stage (0-indexed)
        total_stages: Total number of stages
        base_difficulty: Starting difficulty
        min_difficulty: Minimum difficulty clamp
        max_difficulty: Maximum difficulty clamp

    Returns:
        Difficulty value for the stage

    Example:
        >>> for stage in range(7):  # 7-stage curriculum
        ...     difficulty = k_difficulty_schedule(stage, 7)
        ...     print(f"Stage {stage}: difficulty = {difficulty:.2f}")
    """
    if total_stages <= 1:
        return base_difficulty

    # Compute progress (0 to 1)
    progress = stage / (total_stages - 1) if total_stages > 1 else 0

    # Avoid log(0)
    progress = max(progress, 1e-10)

    # Get k value
    k = compute_k(progress)

    # Invert: high k (early) = low difficulty, low k (late) = high difficulty
    # Normalize k to [0, 1] range
    k_normalized = normalize_k_value(k)

    # Invert: early stages get lower difficulty
    difficulty_factor = 1 + (1 - k_normalized)

    difficulty = base_difficulty * difficulty_factor

    return max(min_difficulty, min(max_difficulty, difficulty))


def get_stage_difficulties(
    total_stages: int,
    base_difficulty: float = 1.0,
) -> List[float]:
    """
    Get difficulty values for all curriculum stages.

    Args:
        total_stages: Number of curriculum stages
        base_difficulty: Starting difficulty

    Returns:
        List of difficulty values per stage

    Example:
        >>> difficulties = get_stage_difficulties(7)
        >>> print(difficulties)
        [1.0, 1.14, 1.29, 1.43, 1.57, 1.71, 1.86]
    """
    return [
        k_difficulty_schedule(stage, total_stages, base_difficulty) for stage in range(total_stages)
    ]


# =============================================================================
# WARMUP SCHEDULES
# =============================================================================


def k_warmup_schedule(
    step: int,
    warmup_steps: int,
    base_value: float = 1.0,
    start_factor: float = 0.1,
) -> float:
    """
    k(L)-based warmup schedule.

    Smoother than linear warmup, uses k formula for acceleration.

    Args:
        step: Current step
        warmup_steps: Total warmup steps
        base_value: Target value after warmup
        start_factor: Starting multiplier (0.1 = start at 10%)

    Returns:
        Warmed-up value

    Example:
        >>> for step in range(100):
        ...     factor = k_warmup_schedule(step, 100)
        ...     effective_lr = base_lr * factor
    """
    if warmup_steps <= 0:
        return base_value

    if step >= warmup_steps:
        return base_value

    # Progress through warmup (0 to 1)
    progress = step / warmup_steps

    # Use k formula for smooth acceleration
    # Invert progress so k is HIGH at start (more dampening)
    k = compute_k(1 - progress + 1e-10)

    # Normalize k effect
    k_normalized = normalize_k_value(k)

    # Interpolate from start_factor to 1.0
    factor = start_factor + (1 - start_factor) * (1 - k_normalized)

    return base_value * factor


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_schedule_preview(
    schedule_fn: Callable,
    total_steps: int,
    sample_points: int = 10,
    **kwargs,
) -> List[tuple]:
    """
    Preview a schedule by sampling at regular intervals.

    Args:
        schedule_fn: Schedule function to preview
        total_steps: Total steps for the schedule
        sample_points: Number of points to sample
        **kwargs: Additional arguments for schedule function

    Returns:
        List of (step, value) tuples

    Example:
        >>> preview = get_schedule_preview(k_learning_rate_schedule, 1000, base_lr=1e-3)
        >>> for step, lr in preview:
        ...     print(f"Step {step}: lr = {lr:.6f}")
    """
    points = []
    for i in range(sample_points):
        step = int(i * total_steps / (sample_points - 1)) if sample_points > 1 else 0
        value = schedule_fn(step=step, total_steps=total_steps, **kwargs)
        points.append((step, value))
    return points


def print_schedule_table(
    schedule_fn: Callable,
    total_steps: int,
    sample_points: int = 10,
    **kwargs,
) -> None:
    """
    Print a formatted table of schedule values.

    Args:
        schedule_fn: Schedule function
        total_steps: Total steps
        sample_points: Number of samples
        **kwargs: Additional arguments
    """
    preview = get_schedule_preview(schedule_fn, total_steps, sample_points, **kwargs)

    print(f"\nSchedule Preview ({schedule_fn.__name__})")
    print("-" * 40)
    print(f"{'Step':>10} | {'Value':>15} | {'Progress':>10}")
    print("-" * 40)

    for step, value in preview:
        progress = step / total_steps if total_steps > 0 else 0
        print(f"{step:>10} | {value:>15.6f} | {progress:>10.1%}")

    print("-" * 40)

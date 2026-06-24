"""
Phase-Specific Spectral Gap Monitoring

Enhanced monitoring with phase-specific thresholds and health status tracking.

Phase Applications:
    - Phase 1-8: All phases benefit from diversity monitoring
    - Phase 5: Stage advancement detection based on gap stability
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

# Layer 1 import only
from ..spectral_gap import (
    SpectralGapConfig,
    SpectralGapMonitor,
    compute_expert_diversity,
    compute_thought_diversity,
)


class GapHealthStatus(Enum):
    """Health status based on spectral gap."""

    HEALTHY = "healthy"  # Gap >= threshold
    WARNING = "warning"  # Gap near threshold
    COLLAPSED = "collapsed"  # Gap critically low
    UNKNOWN = "unknown"  # Not enough data


@dataclass
class PhaseGapConfig:
    """Configuration for phase-specific gap monitoring."""

    # Phase identifier
    phase: str = "generic"

    # Thresholds
    healthy_threshold: float = 0.1
    warning_threshold: float = 0.05
    collapse_threshold: float = 0.01

    # History tracking
    history_size: int = 100

    # Stage advancement (Phase 5)
    stage_stability_patience: int = 10
    stage_stability_threshold: float = 0.01

    # Alert callbacks
    on_warning: Optional[callable] = None
    on_collapse: Optional[callable] = None


# Phase-specific default configurations
PHASE_GAP_CONFIGS = {
    "phase1": PhaseGapConfig(
        phase="phase1_cognate",
        healthy_threshold=0.1,
        warning_threshold=0.05,
    ),
    "phase2": PhaseGapConfig(
        phase="phase2_evomerge",
        healthy_threshold=0.15,  # Higher threshold for merging
        warning_threshold=0.08,
    ),
    "phase3": PhaseGapConfig(
        phase="phase3_quietstar",
        healthy_threshold=0.08,  # Thoughts can be more similar
        warning_threshold=0.04,
    ),
    "phase4": PhaseGapConfig(
        phase="phase4_bitnet",
        healthy_threshold=0.1,
        warning_threshold=0.05,
    ),
    "phase5": PhaseGapConfig(
        phase="phase5_curriculum",
        healthy_threshold=0.1,
        warning_threshold=0.05,
        stage_stability_patience=10,
    ),
    "phase6": PhaseGapConfig(
        phase="phase6_baking",
        healthy_threshold=0.1,
        warning_threshold=0.05,
    ),
    "phase7": PhaseGapConfig(
        phase="phase7_experts",
        healthy_threshold=0.12,  # Experts should be diverse
        warning_threshold=0.06,
    ),
    "phase8": PhaseGapConfig(
        phase="phase8_compression",
        healthy_threshold=0.08,  # Some collapse OK after compression
        warning_threshold=0.04,
    ),
}


class PhaseGapMonitor:
    """
    Phase-specific spectral gap monitor with health tracking.

    Example:
        >>> monitor = PhaseGapMonitor("phase3")
        >>> for step in range(1000):
        ...     thoughts = model.generate_thoughts(batch)
        ...     status = monitor.check(thoughts)
        ...     if status == GapHealthStatus.COLLAPSED:
        ...         print("WARNING: Thought diversity collapsed!")
    """

    def __init__(
        self,
        phase: str,
        config: Optional[PhaseGapConfig] = None,
    ):
        """
        Initialize phase-specific gap monitor.

        Args:
            phase: Phase identifier (e.g., "phase3", "phase7")
            config: Optional custom configuration
        """
        # Normalize phase name
        phase_key = phase.lower().replace("_", "").replace("-", "")
        if not phase_key.startswith("phase"):
            phase_key = f"phase{phase_key}"

        # Get default config or use provided
        if config is None:
            # Try to match phase
            for key, default_config in PHASE_GAP_CONFIGS.items():
                if key in phase_key or phase_key in key:
                    config = default_config
                    break
            if config is None:
                config = PhaseGapConfig(phase=phase)

        self.config = config
        self.phase = phase

        # Core monitor
        self._monitor = SpectralGapMonitor(
            SpectralGapConfig(
                healthy_threshold=config.healthy_threshold,
                collapse_threshold=config.collapse_threshold,
            )
        )

        # History tracking
        self.gap_history: List[float] = []
        self.status_history: List[GapHealthStatus] = []

    def check(
        self,
        embeddings,
        record: bool = True,
    ) -> GapHealthStatus:
        """
        Check spectral gap and return health status.

        Args:
            embeddings: Embedding tensor [N, D] or list of tensors
            record: Whether to record in history

        Returns:
            GapHealthStatus enum value
        """
        # Compute gap
        result = self._monitor.compute_gap(embeddings)
        gap = result.get("gap", 0.0)

        # Determine status
        if gap >= self.config.healthy_threshold:
            status = GapHealthStatus.HEALTHY
        elif gap >= self.config.warning_threshold:
            status = GapHealthStatus.WARNING
        elif gap >= self.config.collapse_threshold:
            status = GapHealthStatus.WARNING
        else:
            status = GapHealthStatus.COLLAPSED

        # Record history
        if record:
            self.gap_history.append(gap)
            self.status_history.append(status)

            # Trim history if needed
            if len(self.gap_history) > self.config.history_size:
                self.gap_history = self.gap_history[-self.config.history_size :]
                self.status_history = self.status_history[-self.config.history_size :]

        # Trigger callbacks
        if status == GapHealthStatus.WARNING and self.config.on_warning:
            self.config.on_warning(gap, self)
        elif status == GapHealthStatus.COLLAPSED and self.config.on_collapse:
            self.config.on_collapse(gap, self)

        return status

    def check_thoughts(self, thought_embeddings) -> Dict[str, Any]:
        """
        Check thought diversity (Phase 3 specific).

        Args:
            thought_embeddings: Thought embedding tensor [N_thoughts, D]

        Returns:
            Dictionary with diversity metrics and status
        """
        result = compute_thought_diversity(thought_embeddings)
        status = self.check(thought_embeddings)

        return {
            **result,
            "status": status,
            "is_healthy": status == GapHealthStatus.HEALTHY,
        }

    def check_experts(self, expert_weights: List) -> Dict[str, Any]:
        """
        Check expert diversity (Phase 7 specific).

        Args:
            expert_weights: List of expert weight tensors

        Returns:
            Dictionary with diversity metrics and status
        """
        result = compute_expert_diversity(expert_weights)

        # Compute gap for status
        if expert_weights:
            try:
                import torch

                stacked = torch.stack([w.flatten() for w in expert_weights])
                status = self.check(stacked)
            except Exception:
                status = GapHealthStatus.UNKNOWN
        else:
            status = GapHealthStatus.UNKNOWN

        return {
            **result,
            "status": status,
            "is_healthy": status == GapHealthStatus.HEALTHY,
        }

    def get_current_gap(self) -> float:
        """Get most recent gap value."""
        return self.gap_history[-1] if self.gap_history else 0.0

    def get_gap_trend(self, window: int = 10) -> float:
        """
        Get gap trend over recent history.

        Returns:
            Positive = improving, Negative = declining, 0 = stable
        """
        if len(self.gap_history) < window:
            return 0.0

        recent = self.gap_history[-window:]
        first_half = sum(recent[: window // 2]) / (window // 2)
        second_half = sum(recent[window // 2 :]) / (window - window // 2)

        return second_half - first_half

    def get_statistics(self) -> Dict[str, float]:
        """Get gap statistics from history."""
        if not self.gap_history:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "current": 0}

        gaps = self.gap_history
        mean = sum(gaps) / len(gaps)
        variance = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        std = math.sqrt(variance)

        return {
            "mean": mean,
            "std": std,
            "min": min(gaps),
            "max": max(gaps),
            "current": gaps[-1],
            "trend": self.get_gap_trend(),
        }

    def reset_history(self) -> None:
        """Clear history."""
        self.gap_history.clear()
        self.status_history.clear()


# =============================================================================
# STAGE ADVANCEMENT (Phase 5 Curriculum)
# =============================================================================


def should_advance_stage(
    gap_history: List[float],
    patience: int = 10,
    stability_threshold: float = 0.01,
) -> bool:
    """
    Determine if curriculum should advance to next stage based on gap stability.

    Advances when gap variance stabilizes (learning plateau detected).

    Args:
        gap_history: Recent gap values
        patience: Minimum observations before advancing
        stability_threshold: Maximum variance for stability

    Returns:
        True if should advance to next stage

    Example:
        >>> monitor = PhaseGapMonitor("phase5")
        >>> for step in range(1000):
        ...     status = monitor.check(embeddings)
        ...     if should_advance_stage(monitor.gap_history):
        ...         advance_curriculum_stage()
    """
    if len(gap_history) < patience:
        return False

    recent = gap_history[-patience:]

    # Compute variance
    mean = sum(recent) / len(recent)
    variance = sum((g - mean) ** 2 for g in recent) / len(recent)

    return variance < stability_threshold


def get_stage_advancement_info(
    gap_history: List[float],
    patience: int = 10,
    stability_threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    Get detailed stage advancement information.

    Args:
        gap_history: Recent gap values
        patience: Minimum observations
        stability_threshold: Stability threshold

    Returns:
        Dictionary with advancement recommendation and metrics
    """
    if len(gap_history) < patience:
        return {
            "should_advance": False,
            "reason": f"Not enough data ({len(gap_history)}/{patience})",
            "variance": None,
            "mean_gap": None,
        }

    recent = gap_history[-patience:]
    mean = sum(recent) / len(recent)
    variance = sum((g - mean) ** 2 for g in recent) / len(recent)
    should_advance = variance < stability_threshold

    return {
        "should_advance": should_advance,
        "reason": "Gap stabilized" if should_advance else "Gap still changing",
        "variance": variance,
        "mean_gap": mean,
        "threshold": stability_threshold,
        "observations": len(recent),
    }


# =============================================================================
# HEALTH STATUS UTILITIES
# =============================================================================


def get_gap_health_status(
    gap: float,
    healthy_threshold: float = 0.1,
    warning_threshold: float = 0.05,
    collapse_threshold: float = 0.01,
) -> GapHealthStatus:
    """
    Get health status from a gap value.

    Args:
        gap: Spectral gap value
        healthy_threshold: Minimum for healthy
        warning_threshold: Minimum for warning
        collapse_threshold: Minimum before collapse

    Returns:
        GapHealthStatus enum value
    """
    if gap >= healthy_threshold:
        return GapHealthStatus.HEALTHY
    elif gap >= warning_threshold:
        return GapHealthStatus.WARNING
    elif gap >= collapse_threshold:
        return GapHealthStatus.WARNING
    else:
        return GapHealthStatus.COLLAPSED


def format_health_status(status: GapHealthStatus) -> str:
    """Format health status for display."""
    icons = {
        GapHealthStatus.HEALTHY: "[OK]",
        GapHealthStatus.WARNING: "[WARN]",
        GapHealthStatus.COLLAPSED: "[CRITICAL]",
        GapHealthStatus.UNKNOWN: "[?]",
    }
    return f"{icons.get(status, '[?]')} {status.value}"

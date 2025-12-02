"""
Phase 6: Plateau Detector

Detects convergence in A/B baking cycles.
Triggers automatic cycle switching when improvement plateaus.

Research: "Prompt Baking" (arXiv:2409.13697v1)
Key insight: Plateaus indicate diminishing returns, time to switch cycles.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PlateauConfig:
    """Configuration for plateau detection."""
    window_size: int = 3  # Number of iterations to consider
    threshold: float = 0.01  # Minimum improvement to not be plateau
    min_iterations: int = 2  # Minimum iterations before detecting plateau
    patience: int = 2  # How many plateaus before declaring convergence


class PlateauDetector:
    """
    Plateau Detection for A/B Cycle Switching.

    Monitors improvement in each cycle and detects when:
    1. A single cycle has plateaued (switch to other cycle)
    2. Both cycles have plateaued (stop optimization)

    Uses rolling window average comparison.
    """

    def __init__(
        self,
        window_size: int = 3,
        threshold: float = 0.01,
        min_iterations: int = 2,
        patience: int = 2
    ):
        """
        Initialize plateau detector.

        Args:
            window_size: Rolling window size for comparison
            threshold: Minimum improvement to avoid plateau
            min_iterations: Minimum iterations before detection
            patience: Plateaus before declaring convergence
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_iterations = min_iterations
        self.patience = patience

        # Track scores for each cycle
        self.history = {
            'a_cycle': [],
            'b_cycle': []
        }

        # Track plateau counts
        self.plateau_counts = {
            'a_cycle': 0,
            'b_cycle': 0
        }

        # Track plateau events
        self.plateau_events = []

    def check(self, score: float, cycle: str) -> bool:
        """
        Check if the given cycle has plateaued.

        Args:
            score: Latest score for the cycle
            cycle: 'a_cycle' or 'b_cycle'

        Returns:
            True if cycle has plateaued
        """
        if cycle not in self.history:
            self.history[cycle] = []
            self.plateau_counts[cycle] = 0

        self.history[cycle].append(score)

        # Need minimum iterations
        if len(self.history[cycle]) < self.min_iterations:
            return False

        # Calculate improvement over window
        scores = self.history[cycle]

        if len(scores) < self.window_size + 1:
            # Compare to first score
            improvement = scores[-1] - scores[0]
        else:
            # Compare rolling windows
            recent_avg = sum(scores[-self.window_size:]) / self.window_size
            previous_avg = sum(scores[-(self.window_size*2):-self.window_size]) / self.window_size
            improvement = recent_avg - previous_avg

        # Check if plateaued
        is_plateau = improvement < self.threshold

        if is_plateau:
            self.plateau_counts[cycle] += 1
            self.plateau_events.append({
                'cycle': cycle,
                'iteration': len(scores),
                'score': score,
                'improvement': improvement
            })
            return True

        return False

    def both_plateaued(self) -> bool:
        """
        Check if both A and B cycles have plateaued.

        Returns:
            True if both cycles have hit patience limit
        """
        a_exhausted = self.plateau_counts.get('a_cycle', 0) >= self.patience
        b_exhausted = self.plateau_counts.get('b_cycle', 0) >= self.patience

        return a_exhausted and b_exhausted

    def get_recommendation(self) -> str:
        """
        Get recommendation for next action.

        Returns:
            'a_cycle', 'b_cycle', or 'stop'
        """
        if self.both_plateaued():
            return 'stop'

        a_plateau = self.plateau_counts.get('a_cycle', 0)
        b_plateau = self.plateau_counts.get('b_cycle', 0)

        # Prefer the cycle with fewer plateaus
        if a_plateau < b_plateau:
            return 'a_cycle'
        elif b_plateau < a_plateau:
            return 'b_cycle'
        else:
            # Equal plateaus, prefer A-cycle (tool use typically more important)
            return 'a_cycle'

    def reset_cycle(self, cycle: str):
        """
        Reset plateau count for a cycle (e.g., after significant improvement).

        Args:
            cycle: 'a_cycle' or 'b_cycle'
        """
        if cycle in self.plateau_counts:
            self.plateau_counts[cycle] = 0

    def get_history(self) -> Dict:
        """Get full detection history."""
        return {
            'scores': self.history.copy(),
            'plateau_counts': self.plateau_counts.copy(),
            'plateau_events': self.plateau_events.copy()
        }

    def get_statistics(self) -> Dict:
        """Get plateau detection statistics."""
        stats = {}

        for cycle, scores in self.history.items():
            if scores:
                stats[cycle] = {
                    'iterations': len(scores),
                    'initial_score': scores[0],
                    'final_score': scores[-1],
                    'best_score': max(scores),
                    'worst_score': min(scores),
                    'total_improvement': scores[-1] - scores[0],
                    'plateau_count': self.plateau_counts.get(cycle, 0)
                }

        return stats


class AdaptivePlateauDetector(PlateauDetector):
    """
    Adaptive plateau detection with dynamic threshold.

    Adjusts threshold based on:
    - Score variance
    - Iteration count
    - Cycle history
    """

    def __init__(
        self,
        initial_threshold: float = 0.02,
        min_threshold: float = 0.005,
        decay_rate: float = 0.9,
        **kwargs
    ):
        """
        Initialize adaptive detector.

        Args:
            initial_threshold: Starting threshold
            min_threshold: Minimum threshold
            decay_rate: Threshold decay per iteration
        """
        super().__init__(threshold=initial_threshold, **kwargs)
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.decay_rate = decay_rate

    def _get_adaptive_threshold(self, cycle: str) -> float:
        """Calculate adaptive threshold for cycle."""
        scores = self.history.get(cycle, [])

        if len(scores) < 2:
            return self.initial_threshold

        # Calculate variance
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5

        # Threshold based on standard deviation
        variance_threshold = std_dev * 0.5

        # Decay threshold over iterations
        iteration_factor = self.decay_rate ** len(scores)
        decayed_threshold = self.initial_threshold * iteration_factor

        # Use minimum of variance-based and decayed threshold
        adaptive_threshold = max(
            self.min_threshold,
            min(variance_threshold, decayed_threshold)
        )

        return adaptive_threshold

    def check(self, score: float, cycle: str) -> bool:
        """Check with adaptive threshold."""
        # Update threshold before check
        self.threshold = self._get_adaptive_threshold(cycle)

        return super().check(score, cycle)


__all__ = ['PlateauDetector', 'PlateauConfig', 'AdaptivePlateauDetector']

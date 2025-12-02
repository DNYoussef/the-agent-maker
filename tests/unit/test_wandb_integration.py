"""
Unit tests for W&B Integration
Tests metrics tracking, artifact versioning, and continuity tracking
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cross_phase.monitoring.wandb_integration import (
    WandBIntegration,
    METRICS_COUNT,
    MetricContinuityTracker
)


class TestWandBIntegration:
    """Test WandBIntegration"""

    def test_wandb_creation(self):
        """Test W&B integration creation"""
        wandb = WandBIntegration(mode="offline", project="test-project")

        assert wandb.mode == "offline"
        assert wandb.project == "test-project"

    def test_metrics_count_total(self):
        """Test total metrics count"""
        total = sum(METRICS_COUNT.values())

        assert total == 676

    def test_metrics_count_breakdown(self):
        """Test metrics breakdown per phase"""
        assert METRICS_COUNT["phase1"] == 37
        assert METRICS_COUNT["phase2"] == 370
        assert METRICS_COUNT["phase3"] == 17
        assert METRICS_COUNT["phase4"] == 19
        assert METRICS_COUNT["phase5"] == 78
        assert METRICS_COUNT["phase6"] == 32
        assert METRICS_COUNT["phase7"] == 28
        assert METRICS_COUNT["phase8"] == 95

    def test_offline_mode(self):
        """Test offline mode setting"""
        wandb = WandBIntegration(mode="offline")

        assert wandb.mode == "offline"

    def test_online_mode(self):
        """Test online mode setting"""
        wandb = WandBIntegration(mode="online", project="test")

        assert wandb.mode == "online"


class TestMetricContinuityTracker:
    """Test MetricContinuityTracker"""

    def test_tracker_creation(self):
        """Test tracker creation"""
        tracker = MetricContinuityTracker()

        assert tracker is not None

    def test_add_phase_metrics(self):
        """Test adding phase metrics"""
        tracker = MetricContinuityTracker()

        tracker.add_phase_metrics(
            phase="phase1",
            metrics={"loss": 2.34, "accuracy": 45.2}
        )

        assert "phase1" in tracker.history

    def test_get_trend(self):
        """Test getting metric trend"""
        tracker = MetricContinuityTracker()

        tracker.add_phase_metrics("phase1", {"loss": 3.5})
        tracker.add_phase_metrics("phase2", {"loss": 2.8})
        tracker.add_phase_metrics("phase3", {"loss": 2.1})

        trend = tracker.get_trend("loss")

        assert len(trend) == 3
        assert trend[0] > trend[1] > trend[2]  # Loss decreasing

    def test_detect_degradation(self):
        """Test detecting metric degradation"""
        tracker = MetricContinuityTracker()

        # Stable metrics
        tracker.add_phase_metrics("phase1", {"accuracy": 45.0})
        tracker.add_phase_metrics("phase2", {"accuracy": 56.0})
        tracker.add_phase_metrics("phase3", {"accuracy": 62.0})

        is_degraded = tracker.detect_degradation("accuracy", threshold=0.1)

        assert is_degraded is False

        # Degraded metrics
        tracker.add_phase_metrics("phase4", {"accuracy": 30.0})

        is_degraded = tracker.detect_degradation("accuracy", threshold=0.1)

        assert is_degraded is True

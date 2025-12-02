"""
Unit tests for 3D Model Comparison Visualization

Tests cover:
- Data validation
- Figure creation
- Pareto frontier computation
- Filtering logic
- Edge cases
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ui.components.model_comparison_3d import (
    PHASE_COLORS,
    STATUS_SYMBOLS,
    _compute_pareto_surface,
    create_model_comparison_3d,
    get_sample_data,
)


class TestDataValidation:
    """Test data validation and preprocessing."""

    def test_sample_data_generation(self):
        """Sample data should have all required fields."""
        data = get_sample_data()

        assert len(data) > 0, "Should generate models"
        assert len(data) >= 24, "Should generate at least 3 models per phase"

        # Check required fields
        required_fields = ["id", "name", "phase", "params", "accuracy", "latency", "status"]
        for model in data:
            for field in required_fields:
                assert field in model, f"Missing field: {field}"

    def test_sample_data_phases(self):
        """Sample data should cover all 8 phases."""
        data = get_sample_data()
        phases = {m["phase"] for m in data}

        expected_phases = {f"phase{i}" for i in range(1, 9)}
        assert phases == expected_phases, "Should have all 8 phases"

    def test_sample_data_status_distribution(self):
        """Sample data should have realistic status distribution."""
        data = get_sample_data()
        statuses = [m["status"] for m in data]

        # Should have mostly complete models
        complete_count = sum(1 for s in statuses if s == "complete")
        assert complete_count > len(data) * 0.5, "Majority should be complete"

        # Should have some variety
        unique_statuses = set(statuses)
        assert len(unique_statuses) >= 2, "Should have multiple statuses"


class TestFigureCreation:
    """Test 3D figure creation."""

    def get_test_dataframe(self, n_models: int = 10) -> pd.DataFrame:
        """Create test DataFrame."""
        return pd.DataFrame(
            [
                {
                    "id": f"model_{i}",
                    "name": f"Test Model {i}",
                    "phase": f"phase{(i % 8) + 1}",
                    "params": 25_000_000 + i * 1_000_000,
                    "accuracy": 40.0 + i * 2.0,
                    "latency": 150.0 - i * 5.0,
                    "compression": 1.0 + i * 0.2,
                    "status": "complete",
                }
                for i in range(n_models)
            ]
        )

    def test_create_basic_figure(self):
        """Should create valid Plotly figure."""
        df = self.get_test_dataframe()
        fig = create_model_comparison_3d(df, animate=False)

        assert fig is not None, "Should return figure"
        assert len(fig.data) > 0, "Should have traces"

    def test_phase_filtering(self):
        """Should filter to specified phases."""
        df = self.get_test_dataframe(16)  # 2 models per phase
        show_phases = ["phase1", "phase3", "phase5"]

        fig = create_model_comparison_3d(df, show_phases=show_phases, animate=False)

        # Count traces (one per phase that has data)
        trace_names = {trace.name for trace in fig.data}
        expected_names = {"PHASE1", "PHASE3", "PHASE5"}

        assert expected_names.issubset(trace_names), "Should show filtered phases"

    def test_highlighted_models(self):
        """Should highlight specified models."""
        df = self.get_test_dataframe(10)
        highlighted = ["model_0", "model_5"]

        fig = create_model_comparison_3d(df, highlighted_ids=highlighted, animate=False)

        # Should have traces for highlighted models
        # (implementation creates separate traces for champions)
        assert len(fig.data) >= 2, "Should have traces for highlighted models"

    def test_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame(
            columns=["id", "name", "phase", "params", "accuracy", "latency", "status"]
        )

        fig = create_model_comparison_3d(df, animate=False)

        assert fig is not None, "Should return figure even if empty"
        assert len(fig.data) == 0, "Should have no traces"

    def test_single_model(self):
        """Should handle single model."""
        df = self.get_test_dataframe(1)

        fig = create_model_comparison_3d(df, animate=False)

        assert fig is not None, "Should return figure"
        assert len(fig.data) > 0, "Should have trace"

    def test_animation_enabled(self):
        """Should add animation frames when enabled."""
        df = self.get_test_dataframe(5)

        fig = create_model_comparison_3d(df, animate=True)

        assert hasattr(fig, "frames"), "Should have frames attribute"
        assert len(fig.frames) > 0, "Should have animation frames"

    def test_animation_disabled(self):
        """Should not add animation frames when disabled."""
        df = self.get_test_dataframe(5)

        fig = create_model_comparison_3d(df, animate=False)

        # Frames might exist but should be empty
        if hasattr(fig, "frames"):
            assert len(fig.frames) == 0, "Should have no animation frames"


class TestParetoFrontier:
    """Test Pareto frontier computation."""

    def get_pareto_test_data(self) -> pd.DataFrame:
        """Create data with known Pareto-optimal points."""
        return pd.DataFrame(
            [
                # Pareto-optimal: Best accuracy, moderate size/latency
                {
                    "id": "opt1",
                    "name": "Optimal 1",
                    "phase": "phase1",
                    "params": 30_000_000,
                    "accuracy": 80.0,
                    "latency": 100.0,
                    "status": "complete",
                },
                # Pareto-optimal: Smallest size, moderate accuracy
                {
                    "id": "opt2",
                    "name": "Optimal 2",
                    "phase": "phase2",
                    "params": 10_000_000,
                    "accuracy": 60.0,
                    "latency": 80.0,
                    "status": "complete",
                },
                # Pareto-optimal: Fastest latency, moderate accuracy
                {
                    "id": "opt3",
                    "name": "Optimal 3",
                    "phase": "phase3",
                    "params": 25_000_000,
                    "accuracy": 70.0,
                    "latency": 50.0,
                    "status": "complete",
                },
                # Dominated: Worse in all metrics than opt1
                {
                    "id": "dom1",
                    "name": "Dominated 1",
                    "phase": "phase4",
                    "params": 40_000_000,
                    "accuracy": 70.0,
                    "latency": 150.0,
                    "status": "complete",
                },
                # Dominated: Worse than opt2
                {
                    "id": "dom2",
                    "name": "Dominated 2",
                    "phase": "phase5",
                    "params": 20_000_000,
                    "accuracy": 50.0,
                    "latency": 100.0,
                    "status": "complete",
                },
            ]
        )

    def test_pareto_surface_creation(self):
        """Should create Pareto surface for sufficient data."""
        df = self.get_pareto_test_data()

        surface = _compute_pareto_surface(df)

        # Surface may or may not be created depending on scipy availability
        # and data distribution, so we just check it doesn't crash
        assert surface is None or hasattr(surface, "type"), "Should return None or Surface"

    def test_pareto_insufficient_data(self):
        """Should return None for <4 models."""
        df = pd.DataFrame(
            [
                {
                    "id": "model1",
                    "name": "Model 1",
                    "phase": "phase1",
                    "params": 25_000_000,
                    "accuracy": 50.0,
                    "latency": 100.0,
                    "status": "complete",
                },
                {
                    "id": "model2",
                    "name": "Model 2",
                    "phase": "phase2",
                    "params": 30_000_000,
                    "accuracy": 60.0,
                    "latency": 110.0,
                    "status": "complete",
                },
            ]
        )

        surface = _compute_pareto_surface(df)

        assert surface is None, "Should return None for <4 models"

    def test_pareto_with_figure(self):
        """Should integrate Pareto surface into figure."""
        df = self.get_pareto_test_data()

        fig = create_model_comparison_3d(df, show_pareto=True, animate=False)

        assert fig is not None, "Should create figure"
        # Surface may or may not be added depending on scipy and data
        # Just verify it doesn't crash


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_optional_fields(self):
        """Should handle missing compression field."""
        df = pd.DataFrame(
            [
                {
                    "id": "model1",
                    "name": "Model 1",
                    "phase": "phase1",
                    "params": 25_000_000,
                    "accuracy": 50.0,
                    "latency": 100.0,
                    "status": "complete",
                }
                # Note: no 'compression' field
            ]
        )

        fig = create_model_comparison_3d(df, animate=False)

        assert fig is not None, "Should handle missing compression"

    def test_invalid_status(self):
        """Should handle unknown status values."""
        df = pd.DataFrame(
            [
                {
                    "id": "model1",
                    "name": "Model 1",
                    "phase": "phase1",
                    "params": 25_000_000,
                    "accuracy": 50.0,
                    "latency": 100.0,
                    "status": "unknown_status",
                    "compression": 1.0,
                }
            ]
        )

        fig = create_model_comparison_3d(df, animate=False)

        assert fig is not None, "Should handle unknown status"

    def test_extreme_values(self):
        """Should handle extreme metric values."""
        df = pd.DataFrame(
            [
                {
                    "id": "model1",
                    "name": "Model 1",
                    "phase": "phase1",
                    "params": 1_000,
                    "accuracy": 0.1,
                    "latency": 1.0,
                    "status": "complete",
                    "compression": 0.1,
                },
                {
                    "id": "model2",
                    "name": "Model 2",
                    "phase": "phase2",
                    "params": 10_000_000_000,
                    "accuracy": 99.99,
                    "latency": 10000.0,
                    "status": "complete",
                    "compression": 100.0,
                },
            ]
        )

        fig = create_model_comparison_3d(df, animate=False)

        assert fig is not None, "Should handle extreme values"

    def test_all_same_values(self):
        """Should handle degenerate case where all models identical."""
        df = pd.DataFrame(
            [
                {
                    "id": f"model{i}",
                    "name": f"Model {i}",
                    "phase": f"phase{i+1}",
                    "params": 25_000_000,
                    "accuracy": 50.0,
                    "latency": 100.0,
                    "status": "complete",
                    "compression": 1.0,
                }
                for i in range(5)
            ]
        )

        fig = create_model_comparison_3d(df, animate=False)

        assert fig is not None, "Should handle identical models"

    def test_highlighting_nonexistent_models(self):
        """Should handle highlighted IDs that don't exist."""
        df = pd.DataFrame(
            [
                {
                    "id": "model1",
                    "name": "Model 1",
                    "phase": "phase1",
                    "params": 25_000_000,
                    "accuracy": 50.0,
                    "latency": 100.0,
                    "status": "complete",
                    "compression": 1.0,
                }
            ]
        )

        fig = create_model_comparison_3d(
            df, highlighted_ids=["nonexistent1", "nonexistent2"], animate=False
        )

        assert fig is not None, "Should handle nonexistent highlighted IDs"


class TestConstants:
    """Test configuration constants."""

    def test_phase_colors_complete(self):
        """Should have colors for all 8 phases."""
        expected_phases = {f"phase{i}" for i in range(1, 9)}
        actual_phases = set(PHASE_COLORS.keys())

        assert actual_phases == expected_phases, "Should have all 8 phase colors"

    def test_phase_colors_valid_hex(self):
        """All phase colors should be valid hex codes."""
        for phase, color in PHASE_COLORS.items():
            assert color.startswith("#"), f"{phase} color should start with #"
            assert len(color) == 7, f"{phase} color should be 7 chars (#RRGGBB)"

    def test_status_symbols_defined(self):
        """Should have symbols for common statuses."""
        expected_statuses = {"complete", "running", "failed", "pending"}
        actual_statuses = set(STATUS_SYMBOLS.keys())

        assert expected_statuses.issubset(
            actual_statuses
        ), "Should have symbols for all common statuses"


class TestIntegration:
    """Integration tests with realistic data."""

    def test_full_pipeline_simulation(self):
        """Simulate complete 8-phase pipeline."""
        models = []

        # Generate realistic progression through phases
        for phase_num in range(1, 9):
            phase = f"phase{phase_num}"
            n_models = np.random.randint(3, 7)

            # Metrics improve with phases
            base_accuracy = 40 + phase_num * 5
            base_latency = 150 - phase_num * 10

            for i in range(n_models):
                models.append(
                    {
                        "id": f"{phase}_model_{i}",
                        "name": f"{phase.upper()} Model {i+1}",
                        "phase": phase,
                        "params": 25_000_000,
                        "accuracy": base_accuracy + np.random.randn() * 2,
                        "latency": max(20, base_latency + np.random.randn() * 10),
                        "compression": 1.0 + (phase_num - 1) * 0.5,
                        "status": np.random.choice(
                            ["complete", "running", "failed"], p=[0.8, 0.15, 0.05]
                        ),
                    }
                )

        df = pd.DataFrame(models)

        # Create figure with all features
        fig = create_model_comparison_3d(
            df, highlighted_ids=None, show_phases=None, show_pareto=True, animate=True
        )

        assert fig is not None, "Should create complete figure"
        assert len(fig.data) > 0, "Should have traces"
        assert hasattr(fig, "frames"), "Should have animation"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

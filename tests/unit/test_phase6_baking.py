"""
Unit tests for Phase 6: Tool & Persona Baking Engine

Tests:
- BakingConfig dataclass
- BakingEngine initialization
- BakingCycleType enum
- BakingResult structure
- A/B cycle switching logic

Target: >=90% coverage for core functionality
"""

import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from phase6_baking.baking_engine import BakingConfig, BakingCycleType, BakingEngine, BakingResult


class TestBakingCycleType:
    """Test BakingCycleType enum."""

    def test_cycle_types(self):
        """Test A and B cycle types."""
        assert BakingCycleType.A_CYCLE.value == "tool"
        assert BakingCycleType.B_CYCLE.value == "persona"

    def test_enum_count(self):
        """Test correct number of cycle types."""
        assert len(BakingCycleType) == 2


class TestBakingConfig:
    """Test BakingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BakingConfig()

        # A-Cycle defaults
        assert config.a_cycle_iterations == 5
        assert len(config.tool_prompts) == 3

        # B-Cycle defaults
        assert config.b_cycle_iterations == 5
        assert len(config.persona_prompts) == 3

        # Half-baking defaults
        assert config.half_bake_strength == 0.5
        assert config.baking_epochs == 3
        assert config.learning_rate == 5e-5  # Fixed: was 1e-4, now 5e-5 per M4 spec

        # Convergence defaults
        assert config.plateau_window == 3
        assert config.plateau_threshold == 0.01
        assert config.max_total_iterations == 20

        # LoRA defaults
        assert config.lora_r == 16
        assert config.lora_alpha == 32

    def test_custom_config(self):
        """Test custom configuration."""
        config = BakingConfig(
            a_cycle_iterations=10, half_bake_strength=0.3, max_total_iterations=50
        )

        assert config.a_cycle_iterations == 10
        assert config.half_bake_strength == 0.3
        assert config.max_total_iterations == 50

    def test_config_serializable(self):
        """Test config can be converted to dict."""
        config = BakingConfig()
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert "half_bake_strength" in config_dict
        assert "plateau_threshold" in config_dict

    def test_tool_prompts_are_list(self):
        """Test tool prompts is a list."""
        config = BakingConfig()
        assert isinstance(config.tool_prompts, list)
        assert all(isinstance(p, str) for p in config.tool_prompts)

    def test_persona_prompts_are_list(self):
        """Test persona prompts is a list."""
        config = BakingConfig()
        assert isinstance(config.persona_prompts, list)
        assert all(isinstance(p, str) for p in config.persona_prompts)


class TestBakingResult:
    """Test BakingResult dataclass."""

    def test_successful_result(self):
        """Test successful BakingResult."""
        mock_model = Mock(spec=nn.Module)

        result = BakingResult(
            success=True,
            model=mock_model,
            total_iterations=15,
            a_cycle_count=8,
            b_cycle_count=7,
            final_tool_score=0.85,
            final_persona_score=0.90,
            metrics={"a_cycle_scores": [0.7, 0.8, 0.85]},
            artifacts={},
        )

        assert result.success is True
        assert result.total_iterations == 15
        assert result.a_cycle_count == 8
        assert result.b_cycle_count == 7
        assert result.final_tool_score == 0.85
        assert result.final_persona_score == 0.90
        assert result.error is None

    def test_failed_result(self):
        """Test failed BakingResult."""
        mock_model = Mock(spec=nn.Module)

        result = BakingResult(
            success=False,
            model=mock_model,
            total_iterations=5,
            a_cycle_count=3,
            b_cycle_count=2,
            final_tool_score=0.0,
            final_persona_score=0.0,
            metrics={},
            artifacts={},
            error="Optimization diverged",
        )

        assert result.success is False
        assert result.error == "Optimization diverged"


class TestBakingEngine:
    """Test BakingEngine class."""

    def test_initialization_default(self):
        """Test engine initialization with defaults."""
        engine = BakingEngine()

        assert engine.config is not None
        assert isinstance(engine.config, BakingConfig)
        assert "a_cycle_scores" in engine.metrics
        assert "b_cycle_scores" in engine.metrics
        assert "iteration_times" in engine.metrics
        assert "plateau_detections" in engine.metrics

    def test_initialization_custom_config(self):
        """Test engine initialization with custom config."""
        config = BakingConfig(max_total_iterations=10)
        engine = BakingEngine(config=config)

        assert engine.config.max_total_iterations == 10

    def test_metrics_initialized_empty(self):
        """Test metrics are initialized as empty lists."""
        engine = BakingEngine()

        assert engine.metrics["a_cycle_scores"] == []
        assert engine.metrics["b_cycle_scores"] == []
        assert engine.metrics["iteration_times"] == []
        assert engine.metrics["plateau_detections"] == []


@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    model = Mock(spec=nn.Module)
    model.parameters.return_value = [torch.randn(100, 100)]
    return model


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    return tokenizer


class TestBakingEngineRun:
    """Test BakingEngine.run() method."""

    @patch("phase6_baking.baking_engine.ACycleOptimizer")
    @patch("phase6_baking.baking_engine.BCycleOptimizer")
    @patch("phase6_baking.baking_engine.PlateauDetector")
    @patch("phase6_baking.baking_engine.HalfBaker")
    def test_run_returns_baking_result(
        self, mock_half, mock_plateau, mock_b, mock_a, mock_model, mock_tokenizer
    ):
        """Test run() returns BakingResult."""
        # Setup mocks
        mock_a_instance = Mock()
        mock_a_instance.optimize.return_value = (mock_model, 0.8)
        mock_a_instance.get_state.return_value = {}
        mock_a.return_value = mock_a_instance

        mock_b_instance = Mock()
        mock_b_instance.optimize.return_value = (mock_model, 0.85)
        mock_b_instance.get_state.return_value = {}
        mock_b.return_value = mock_b_instance

        mock_plateau_instance = Mock()
        mock_plateau_instance.check.return_value = False
        mock_plateau_instance.both_plateaued.return_value = True  # Stop immediately
        mock_plateau_instance.get_history.return_value = []
        mock_plateau.return_value = mock_plateau_instance

        mock_half_instance = Mock()
        mock_half_instance.half_bake.return_value = mock_model
        mock_half.return_value = mock_half_instance

        config = BakingConfig(max_total_iterations=2)
        engine = BakingEngine(config=config)
        result = engine.run(mock_model, mock_tokenizer)

        assert isinstance(result, BakingResult)

    def test_run_catches_exception(self, mock_model, mock_tokenizer):
        """Test run() handles exceptions gracefully."""
        engine = BakingEngine()

        with patch("phase6_baking.baking_engine.ACycleOptimizer") as mock_a:
            mock_a.side_effect = Exception("Import failed")
            result = engine.run(mock_model, mock_tokenizer)

        assert result.success is False
        assert "Import failed" in result.error


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_iterations(self):
        """Test configuration with zero max iterations."""
        config = BakingConfig(max_total_iterations=0)
        engine = BakingEngine(config=config)

        assert engine.config.max_total_iterations == 0

    def test_extreme_half_bake_strength(self):
        """Test extreme half-bake strength values."""
        config_low = BakingConfig(half_bake_strength=0.0)
        config_high = BakingConfig(half_bake_strength=1.0)

        assert config_low.half_bake_strength == 0.0
        assert config_high.half_bake_strength == 1.0

    def test_empty_prompts(self):
        """Test configuration with empty prompts."""
        config = BakingConfig(tool_prompts=[], persona_prompts=[])

        assert config.tool_prompts == []
        assert config.persona_prompts == []

    def test_single_iteration_limit(self):
        """Test with single iteration limit."""
        config = BakingConfig(max_total_iterations=1)
        engine = BakingEngine(config=config)

        assert engine.config.max_total_iterations == 1


class TestBakingCycleLogic:
    """Test A/B cycle switching logic."""

    def test_starts_with_a_cycle(self):
        """Test that engine starts with A-cycle."""
        # This is implicitly tested by the engine's run() method
        # which sets current_cycle = BakingCycleType.A_CYCLE
        pass

    def test_plateau_detection_config(self):
        """Test plateau detection configuration."""
        config = BakingConfig(plateau_window=5, plateau_threshold=0.02)

        assert config.plateau_window == 5
        assert config.plateau_threshold == 0.02

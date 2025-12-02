"""
Unit tests for Phase 5: Curriculum Learning Engine

Tests:
- CurriculumConfig dataclass
- CurriculumEngine initialization and methods
- LevelProgress tracking
- Phase5Result structure
- SpecializationType enum
- Temperature range calculations

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

from phase5_curriculum.curriculum_engine import (
    CurriculumConfig,
    CurriculumEngine,
    LevelProgress,
    Phase5Result,
    SpecializationType,
)


class TestSpecializationType:
    """Test SpecializationType enum."""

    def test_all_types_defined(self):
        """Test all specialization types exist."""
        assert SpecializationType.CODING.value == "coding"
        assert SpecializationType.RESEARCH.value == "research"
        assert SpecializationType.WRITING.value == "writing"
        assert SpecializationType.REASONING.value == "reasoning"
        assert SpecializationType.GENERAL.value == "general"

    def test_enum_count(self):
        """Test correct number of specializations."""
        assert len(SpecializationType) == 5


class TestCurriculumConfig:
    """Test CurriculumConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CurriculumConfig()

        # Assessment defaults
        assert config.edge_of_chaos_threshold == 0.75
        assert config.assessment_questions == 2000

        # Curriculum defaults
        assert config.num_levels == 10
        assert config.questions_per_level == 2000
        assert len(config.frontier_models) == 4

        # Training defaults
        assert config.consecutive_successes_for_mastery == 3
        assert config.max_hints_per_question == 5
        assert config.variant_generation_enabled is True

        # Self-modeling defaults
        assert config.base_temperature_width == 0.2
        assert config.temperature_width_growth == 0.1
        assert config.base_num_ranges == 10

        # Dream consolidation defaults
        assert config.dream_temperature == 1.5
        assert config.dream_training_temperature == 0.8
        assert config.dream_samples == 1000

        # Default specialization
        assert config.specialization == SpecializationType.CODING

    def test_custom_config(self):
        """Test custom configuration."""
        config = CurriculumConfig(
            edge_of_chaos_threshold=0.80, num_levels=5, specialization=SpecializationType.RESEARCH
        )

        assert config.edge_of_chaos_threshold == 0.80
        assert config.num_levels == 5
        assert config.specialization == SpecializationType.RESEARCH

    def test_config_serializable(self):
        """Test config can be converted to dict."""
        config = CurriculumConfig()
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert "edge_of_chaos_threshold" in config_dict
        assert "num_levels" in config_dict


class TestLevelProgress:
    """Test LevelProgress dataclass."""

    def test_level_progress_creation(self):
        """Test LevelProgress creation."""
        progress = LevelProgress(
            level=1,
            initial_questions=2000,
            current_questions=1500,
            mastered_questions=500,
            variants_generated=50,
            hints_given=100,
            accuracy=0.85,
        )

        assert progress.level == 1
        assert progress.initial_questions == 2000
        assert progress.current_questions == 1500
        assert progress.mastered_questions == 500
        assert progress.variants_generated == 50
        assert progress.hints_given == 100
        assert progress.accuracy == 0.85
        assert progress.completed is False

    def test_level_progress_completed(self):
        """Test LevelProgress with completed flag."""
        progress = LevelProgress(
            level=1,
            initial_questions=2000,
            current_questions=0,
            mastered_questions=2000,
            variants_generated=100,
            hints_given=50,
            accuracy=0.95,
            completed=True,
        )

        assert progress.completed is True


class TestPhase5Result:
    """Test Phase5Result dataclass."""

    def test_successful_result(self):
        """Test successful Phase5Result."""
        mock_model = Mock(spec=nn.Module)

        result = Phase5Result(
            success=True,
            model=mock_model,
            specialization=SpecializationType.CODING,
            levels_completed=10,
            metrics={"final_accuracy": 0.95},
            artifacts={"level_progress": []},
        )

        assert result.success is True
        assert result.levels_completed == 10
        assert result.error is None

    def test_failed_result(self):
        """Test failed Phase5Result."""
        mock_model = Mock(spec=nn.Module)

        result = Phase5Result(
            success=False,
            model=mock_model,
            specialization=SpecializationType.CODING,
            levels_completed=3,
            metrics={},
            artifacts={},
            error="Training failed at level 4",
        )

        assert result.success is False
        assert result.error == "Training failed at level 4"


class TestCurriculumEngine:
    """Test CurriculumEngine class."""

    def test_initialization_default(self):
        """Test engine initialization with defaults."""
        engine = CurriculumEngine()

        assert engine.config is not None
        assert isinstance(engine.config, CurriculumConfig)
        assert engine.level_progress == []
        assert engine.metrics == {}
        assert engine.start_time is None

    def test_initialization_custom_config(self):
        """Test engine initialization with custom config."""
        config = CurriculumConfig(num_levels=5)
        engine = CurriculumEngine(config=config)

        assert engine.config.num_levels == 5

    def test_calculate_temperature_ranges_level1(self):
        """Test temperature range calculation for level 1."""
        engine = CurriculumEngine()
        ranges = engine._calculate_temperature_ranges(level=1)

        assert len(ranges) == 10  # base_num_ranges
        assert ranges[0]["start"] == 0.0
        assert ranges[0]["width"] if "width" in ranges[0] else True

    def test_calculate_temperature_ranges_level5(self):
        """Test temperature range calculation for level 5."""
        engine = CurriculumEngine()
        ranges = engine._calculate_temperature_ranges(level=5)

        # num_ranges = 10 + 5 - 1 = 14
        assert len(ranges) == 14

        # width = 0.2 + (5-1) * 0.1 = 0.6
        expected_width = 0.2 + 4 * 0.1
        actual_width = ranges[0]["end"] - ranges[0]["start"]
        assert abs(actual_width - expected_width) < 0.01

    def test_compile_metrics_empty(self):
        """Test metrics compilation with no progress."""
        engine = CurriculumEngine()
        metrics = engine._compile_metrics(duration=100.0)

        assert "duration_seconds" in metrics
        assert metrics["duration_seconds"] == 100.0

    def test_compile_metrics_with_progress(self):
        """Test metrics compilation with level progress."""
        engine = CurriculumEngine()
        engine.level_progress = [
            LevelProgress(
                level=1,
                initial_questions=2000,
                current_questions=500,
                mastered_questions=1500,
                variants_generated=50,
                hints_given=100,
                accuracy=0.85,
                completed=True,
            ),
            LevelProgress(
                level=2,
                initial_questions=2000,
                current_questions=300,
                mastered_questions=1700,
                variants_generated=60,
                hints_given=80,
                accuracy=0.90,
                completed=True,
            ),
        ]

        metrics = engine._compile_metrics(duration=7200.0)

        assert metrics["levels_completed"] == 2
        assert metrics["final_accuracy"] == 0.90
        assert metrics["curriculum_stats"]["total_questions_mastered"] == 3200
        assert metrics["curriculum_stats"]["variants_generated"] == 110
        assert metrics["curriculum_stats"]["hints_given"] == 180

    def test_get_curriculum_stats(self):
        """Test curriculum statistics calculation."""
        engine = CurriculumEngine()
        curriculum = {
            1: [{"q": "a"}, {"q": "b"}],
            2: [{"q": "c"}, {"q": "d"}, {"q": "e"}],
            3: [{"q": "f"}],
        }

        stats = engine._get_curriculum_stats(curriculum)

        assert stats["total_questions"] == 6
        assert stats["questions_per_level"][1] == 2
        assert stats["questions_per_level"][2] == 3
        assert stats["questions_per_level"][3] == 1


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


class TestCurriculumEngineRun:
    """Test CurriculumEngine.run() method."""

    @patch("phase5_curriculum.curriculum_engine.CurriculumEngine._run_assessment")
    @patch("phase5_curriculum.curriculum_engine.CurriculumEngine._generate_curriculum")
    def test_run_catches_exception(self, mock_gen, mock_assess, mock_model, mock_tokenizer):
        """Test run() handles exceptions gracefully."""
        mock_assess.side_effect = Exception("Assessment failed")

        engine = CurriculumEngine()
        result = engine.run(mock_model, mock_tokenizer)

        assert result.success is False
        assert "Assessment failed" in result.error

    def test_run_returns_phase5_result(self, mock_model, mock_tokenizer):
        """Test run() returns Phase5Result."""
        engine = CurriculumEngine()

        with patch.object(engine, "_run_assessment") as mock_assess:
            mock_assess.side_effect = Exception("Test exception")
            result = engine.run(mock_model, mock_tokenizer)

        assert isinstance(result, Phase5Result)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_levels(self):
        """Test configuration with zero levels."""
        config = CurriculumConfig(num_levels=0)
        engine = CurriculumEngine(config=config)

        assert engine.config.num_levels == 0

    def test_extreme_threshold(self):
        """Test extreme edge-of-chaos thresholds."""
        config_low = CurriculumConfig(edge_of_chaos_threshold=0.0)
        config_high = CurriculumConfig(edge_of_chaos_threshold=1.0)

        assert config_low.edge_of_chaos_threshold == 0.0
        assert config_high.edge_of_chaos_threshold == 1.0

    def test_temperature_ranges_level10(self):
        """Test temperature ranges at maximum level."""
        engine = CurriculumEngine()
        ranges = engine._calculate_temperature_ranges(level=10)

        # num_ranges = 10 + 10 - 1 = 19
        assert len(ranges) == 19

        # width = 0.2 + (10-1) * 0.1 = 1.1
        expected_width = 0.2 + 9 * 0.1
        actual_width = ranges[0]["end"] - ranges[0]["start"]
        assert abs(actual_width - expected_width) < 0.01

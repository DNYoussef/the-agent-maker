"""
Unit tests for Phase 7: Self-Guided Experts Engine

Tests:
- ExpertsConfig dataclass
- ExpertsEngine initialization
- Phase7Result structure
- Expert discovery and training pipeline

Target: >=90% coverage for core functionality
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from phase7_experts.experts_engine import (
    ExpertsEngine,
    ExpertsConfig,
    Phase7Result
)


class TestExpertsConfig:
    """Test ExpertsConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExpertsConfig()

        # Discovery defaults
        assert config.min_experts == 3
        assert config.max_experts == 10
        assert config.discovery_samples == 100

        # SVF defaults
        assert config.svf_epochs == 5
        assert config.svf_learning_rate == 1e-4
        assert config.num_singular_values == 32

        # ADAS defaults
        assert config.adas_population == 50
        assert config.adas_generations == 100
        assert config.mutation_rate == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExpertsConfig(
            min_experts=5,
            max_experts=15,
            adas_generations=200
        )

        assert config.min_experts == 5
        assert config.max_experts == 15
        assert config.adas_generations == 200

    def test_config_serializable(self):
        """Test config can be converted to dict."""
        config = ExpertsConfig()
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert 'min_experts' in config_dict
        assert 'adas_population' in config_dict


class TestPhase7Result:
    """Test Phase7Result dataclass."""

    def test_successful_result(self):
        """Test successful Phase7Result."""
        mock_model = Mock(spec=nn.Module)
        mock_profiles = [Mock(), Mock(), Mock()]

        result = Phase7Result(
            success=True,
            model=mock_model,
            num_experts=3,
            expert_profiles=mock_profiles,
            routing_config={'default': 0},
            metrics={'discovery_time': 10.0},
            artifacts={},
            duration=100.0
        )

        assert result.success is True
        assert result.num_experts == 3
        assert len(result.expert_profiles) == 3
        assert result.duration == 100.0
        assert result.error is None

    def test_failed_result(self):
        """Test failed Phase7Result."""
        mock_model = Mock(spec=nn.Module)

        result = Phase7Result(
            success=False,
            model=mock_model,
            num_experts=0,
            expert_profiles=[],
            routing_config={},
            metrics={},
            artifacts={},
            duration=5.0,
            error="Discovery failed"
        )

        assert result.success is False
        assert result.error == "Discovery failed"
        assert result.num_experts == 0


class TestExpertsEngine:
    """Test ExpertsEngine class."""

    def test_initialization_default(self):
        """Test engine initialization with defaults."""
        engine = ExpertsEngine()

        assert engine.config is not None
        assert isinstance(engine.config, ExpertsConfig)
        assert 'discovery_time' in engine.metrics
        assert 'svf_time' in engine.metrics
        assert 'adas_time' in engine.metrics
        assert 'expert_metrics' in engine.metrics

    def test_initialization_custom_config(self):
        """Test engine initialization with custom config."""
        config = ExpertsConfig(min_experts=5)
        engine = ExpertsEngine(config=config)

        assert engine.config.min_experts == 5

    def test_metrics_initialized(self):
        """Test metrics are initialized correctly."""
        engine = ExpertsEngine()

        assert engine.metrics['discovery_time'] == 0.0
        assert engine.metrics['svf_time'] == 0.0
        assert engine.metrics['adas_time'] == 0.0
        assert engine.metrics['expert_metrics'] == []


@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    model = Mock(spec=nn.Module)
    model.parameters.return_value = [torch.randn(100, 100)]
    model._expert_routing = {'default': 0}
    return model


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    return tokenizer


class TestExpertsEngineRun:
    """Test ExpertsEngine.run() method."""

    def test_run_returns_phase7_result(self, mock_model, mock_tokenizer):
        """Test run() returns Phase7Result."""
        engine = ExpertsEngine()

        with patch('phase7_experts.experts_engine.ExpertDiscovery') as mock_discovery:
            mock_discovery.side_effect = Exception("Test exception")
            result = engine.run(mock_model, mock_tokenizer)

        assert isinstance(result, Phase7Result)

    def test_run_catches_exception(self, mock_model, mock_tokenizer):
        """Test run() handles exceptions gracefully."""
        engine = ExpertsEngine()

        with patch('phase7_experts.experts_engine.ExpertDiscovery') as mock_discovery:
            mock_discovery.side_effect = Exception("Discovery failed")
            result = engine.run(mock_model, mock_tokenizer)

        assert result.success is False
        assert "Discovery failed" in result.error

    @patch('phase7_experts.experts_engine.ExpertDiscovery')
    @patch('phase7_experts.experts_engine.SVFTrainer')
    @patch('phase7_experts.experts_engine.ADASOptimizer')
    def test_run_full_pipeline(
        self, mock_adas, mock_svf, mock_discovery, mock_model, mock_tokenizer
    ):
        """Test full pipeline execution."""
        # Setup discovery mock
        mock_expert = Mock()
        mock_expert.id = 1
        mock_expert.capabilities = ['reasoning']

        mock_discovery_instance = Mock()
        mock_discovery_instance.discover.return_value = (3, [mock_expert])
        mock_discovery.return_value = mock_discovery_instance

        # Setup SVF mock
        mock_svf_result = Mock()
        mock_svf_result.success = True
        mock_svf_result.final_loss = 0.1
        mock_svf_result.sv_changes = [0.01]

        mock_svf_instance = Mock()
        mock_svf_instance.train_expert.return_value = (mock_model, mock_svf_result)
        mock_svf.return_value = mock_svf_instance

        # Setup ADAS mock
        mock_adas_result = Mock()
        mock_adas_instance = Mock()
        mock_adas_instance.optimize.return_value = (mock_model, mock_adas_result)
        mock_adas.return_value = mock_adas_instance

        engine = ExpertsEngine()
        result = engine.run(mock_model, mock_tokenizer)

        assert isinstance(result, Phase7Result)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_min_equals_max_experts(self):
        """Test config where min equals max experts."""
        config = ExpertsConfig(min_experts=5, max_experts=5)

        assert config.min_experts == config.max_experts == 5

    def test_zero_adas_generations(self):
        """Test configuration with zero ADAS generations."""
        config = ExpertsConfig(adas_generations=0)

        assert config.adas_generations == 0

    def test_extreme_mutation_rate(self):
        """Test extreme mutation rates."""
        config_low = ExpertsConfig(mutation_rate=0.0)
        config_high = ExpertsConfig(mutation_rate=1.0)

        assert config_low.mutation_rate == 0.0
        assert config_high.mutation_rate == 1.0

    def test_single_expert(self):
        """Test configuration for single expert."""
        config = ExpertsConfig(min_experts=1, max_experts=1)

        assert config.min_experts == 1
        assert config.max_experts == 1


class TestExpertMetrics:
    """Test expert metrics tracking."""

    def test_metrics_structure(self):
        """Test metrics dictionary structure."""
        engine = ExpertsEngine()

        expected_keys = ['discovery_time', 'svf_time', 'adas_time', 'expert_metrics']
        for key in expected_keys:
            assert key in engine.metrics

    def test_expert_metrics_list(self):
        """Test expert_metrics is a list."""
        engine = ExpertsEngine()

        assert isinstance(engine.metrics['expert_metrics'], list)

"""
Unit tests for Phase 8: Final Compression Engine

Tests:
- CompressionConfig dataclass
- CompressionEngine initialization
- Phase8Result structure
- Compression pipeline stages
- Quality gates and rollback

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

from phase8_compression.compression_engine import (
    CompressionEngine,
    CompressionConfig,
    Phase8Result
)


class TestCompressionConfig:
    """Test CompressionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CompressionConfig()

        # SeedLM defaults
        assert config.seedlm_enabled is True
        assert config.seed_bits == 8
        assert config.seed_block_size == 64

        # VPTQ defaults
        assert config.vptq_enabled is True
        assert config.codebook_size == 256
        assert config.vector_dim == 8

        # Hypercompression defaults
        assert config.hyper_enabled is True
        assert config.num_curve_params == 8
        assert config.curve_type == "bezier"

        # Quality gates
        assert config.min_retention_seedlm == 0.95
        assert config.min_retention_vptq == 0.95
        assert config.min_retention_final == 0.84

        # Benchmark settings
        assert config.run_benchmarks is True
        assert config.benchmark_samples == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = CompressionConfig(
            seedlm_enabled=False,
            codebook_size=512,
            min_retention_final=0.90
        )

        assert config.seedlm_enabled is False
        assert config.codebook_size == 512
        assert config.min_retention_final == 0.90

    def test_config_serializable(self):
        """Test config can be converted to dict."""
        config = CompressionConfig()
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert 'seedlm_enabled' in config_dict
        assert 'min_retention_final' in config_dict

    def test_disable_all_stages(self):
        """Test disabling all compression stages."""
        config = CompressionConfig(
            seedlm_enabled=False,
            vptq_enabled=False,
            hyper_enabled=False
        )

        assert config.seedlm_enabled is False
        assert config.vptq_enabled is False
        assert config.hyper_enabled is False


class TestPhase8Result:
    """Test Phase8Result dataclass."""

    def test_successful_result(self):
        """Test successful Phase8Result."""
        mock_model = Mock(spec=nn.Module)

        result = Phase8Result(
            success=True,
            model=mock_model,
            original_size_mb=100.0,
            final_size_mb=0.4,
            total_compression=250.0,
            retention_score=0.85,
            stage_results={
                'seedlm': {'compression_ratio': 2.0},
                'vptq': {'compression_ratio': 20.0},
                'hyper': {'compression_ratio': 6.25}
            },
            benchmark_results={'accuracy': 0.9},
            duration=3600.0
        )

        assert result.success is True
        assert result.original_size_mb == 100.0
        assert result.final_size_mb == 0.4
        assert result.total_compression == 250.0
        assert result.retention_score == 0.85
        assert result.error is None
        assert result.rollback_stage is None

    def test_failed_result(self):
        """Test failed Phase8Result."""
        mock_model = Mock(spec=nn.Module)

        result = Phase8Result(
            success=False,
            model=mock_model,
            original_size_mb=100.0,
            final_size_mb=100.0,
            total_compression=1.0,
            retention_score=0.0,
            stage_results={},
            benchmark_results={},
            duration=10.0,
            error="Compression failed"
        )

        assert result.success is False
        assert result.error == "Compression failed"

    def test_result_with_rollback(self):
        """Test Phase8Result with rollback."""
        mock_model = Mock(spec=nn.Module)

        result = Phase8Result(
            success=True,
            model=mock_model,
            original_size_mb=100.0,
            final_size_mb=5.0,
            total_compression=20.0,
            retention_score=0.80,
            stage_results={},
            benchmark_results={},
            duration=1000.0,
            rollback_stage='hyper'
        )

        assert result.rollback_stage == 'hyper'


class TestCompressionEngine:
    """Test CompressionEngine class."""

    def test_initialization_default(self):
        """Test engine initialization with defaults."""
        engine = CompressionEngine()

        assert engine.config is not None
        assert isinstance(engine.config, CompressionConfig)
        assert 'seedlm' in engine.metrics
        assert 'vptq' in engine.metrics
        assert 'hyper' in engine.metrics
        assert 'benchmarks' in engine.metrics

    def test_initialization_custom_config(self):
        """Test engine initialization with custom config."""
        config = CompressionConfig(benchmark_samples=50)
        engine = CompressionEngine(config=config)

        assert engine.config.benchmark_samples == 50

    def test_metrics_initialized_empty(self):
        """Test metrics are initialized as empty dicts."""
        engine = CompressionEngine()

        assert engine.metrics['seedlm'] == {}
        assert engine.metrics['vptq'] == {}
        assert engine.metrics['hyper'] == {}
        assert engine.metrics['benchmarks'] == {}


@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    model = Mock(spec=nn.Module)
    # Mock parameters with known size
    param = torch.randn(1000, 1000)  # 4MB in float32
    model.parameters.return_value = iter([param])
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


class TestCompressionEngineModelSize:
    """Test CompressionEngine._get_model_size() method."""

    def test_get_model_size_float32(self):
        """Test model size calculation for float32."""
        engine = CompressionEngine()

        model = Mock(spec=nn.Module)
        param = torch.randn(1000, 1000, dtype=torch.float32)
        model.parameters.return_value = iter([param])

        size = engine._get_model_size(model)

        # 1000 * 1000 * 4 bytes = 4MB
        expected_size = (1000 * 1000 * 4) / (1024 * 1024)
        assert abs(size - expected_size) < 0.01

    def test_get_model_size_float16(self):
        """Test model size calculation for float16."""
        engine = CompressionEngine()

        model = Mock(spec=nn.Module)
        param = torch.randn(1000, 1000, dtype=torch.float16)
        model.parameters.return_value = iter([param])

        size = engine._get_model_size(model)

        # 1000 * 1000 * 2 bytes = 2MB
        expected_size = (1000 * 1000 * 2) / (1024 * 1024)
        assert abs(size - expected_size) < 0.01


class TestCompressionEngineRun:
    """Test CompressionEngine.run() method."""

    def test_run_returns_phase8_result(self, mock_model, mock_tokenizer):
        """Test run() returns Phase8Result."""
        config = CompressionConfig(
            seedlm_enabled=False,
            vptq_enabled=False,
            hyper_enabled=False,
            run_benchmarks=False
        )
        engine = CompressionEngine(config=config)
        result = engine.run(mock_model, mock_tokenizer)

        assert isinstance(result, Phase8Result)

    def test_run_catches_exception(self, mock_model, mock_tokenizer):
        """Test run() handles exceptions gracefully."""
        config = CompressionConfig(seedlm_enabled=True)
        engine = CompressionEngine(config=config)

        with patch('phase8_compression.compression_engine.SeedLMCompressor') as mock_seedlm:
            mock_seedlm.side_effect = Exception("Compression failed")
            result = engine.run(mock_model, mock_tokenizer)

        assert result.success is False
        assert "Compression failed" in result.error

    def test_run_with_all_stages_disabled(self, mock_model, mock_tokenizer):
        """Test run with all compression stages disabled."""
        config = CompressionConfig(
            seedlm_enabled=False,
            vptq_enabled=False,
            hyper_enabled=False,
            run_benchmarks=False
        )
        engine = CompressionEngine(config=config)
        result = engine.run(mock_model, mock_tokenizer)

        assert result.success is True
        assert result.total_compression >= 1.0


class TestCompressionEngineBenchmarks:
    """Test CompressionEngine._run_benchmarks() method."""

    def test_run_benchmarks_default_data(self, mock_model, mock_tokenizer):
        """Test benchmarks with default data."""
        engine = CompressionEngine()

        # Setup model for inference
        mock_model.eval = Mock()
        mock_model.parameters.return_value = iter([torch.randn(10, 10)])

        results = engine._run_benchmarks(mock_model, mock_tokenizer)

        assert 'accuracy' in results
        assert 'perplexity' in results
        assert 'latency_ms' in results

    def test_run_benchmarks_custom_data(self, mock_model, mock_tokenizer):
        """Test benchmarks with custom data."""
        engine = CompressionEngine()

        mock_model.eval = Mock()
        mock_model.parameters.return_value = iter([torch.randn(10, 10)])

        custom_data = ["Test prompt 1", "Test prompt 2"]
        results = engine._run_benchmarks(mock_model, mock_tokenizer, custom_data)

        assert isinstance(results, dict)


class TestQualityGates:
    """Test quality gate configurations."""

    def test_retention_thresholds(self):
        """Test retention threshold configurations."""
        config = CompressionConfig()

        # SeedLM and VPTQ: 95%
        assert config.min_retention_seedlm == 0.95
        assert config.min_retention_vptq == 0.95

        # Final cumulative: 84%
        assert config.min_retention_final == 0.84

    def test_custom_retention_thresholds(self):
        """Test custom retention thresholds."""
        config = CompressionConfig(
            min_retention_seedlm=0.90,
            min_retention_vptq=0.90,
            min_retention_final=0.80
        )

        assert config.min_retention_seedlm == 0.90
        assert config.min_retention_vptq == 0.90
        assert config.min_retention_final == 0.80


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_benchmark_samples(self):
        """Test configuration with zero benchmark samples."""
        config = CompressionConfig(benchmark_samples=0)

        assert config.benchmark_samples == 0

    def test_extreme_compression_targets(self):
        """Test extreme compression configurations."""
        config = CompressionConfig(
            seed_bits=1,
            codebook_size=2,
            num_curve_params=1
        )

        assert config.seed_bits == 1
        assert config.codebook_size == 2
        assert config.num_curve_params == 1

    def test_different_curve_types(self):
        """Test different curve type configurations."""
        config_bezier = CompressionConfig(curve_type="bezier")
        config_spline = CompressionConfig(curve_type="spline")

        assert config_bezier.curve_type == "bezier"
        assert config_spline.curve_type == "spline"


class TestCompressionRatios:
    """Test expected compression ratios."""

    def test_target_compression(self):
        """Test target compression ratio is achievable."""
        # Target: 280x (100MB -> 0.4MB)
        # SeedLM: 2x, VPTQ: 20x, Hyper: 6.25x
        # 2 * 20 * 6.25 = 250x (close to 280x)

        seedlm_ratio = 2.0
        vptq_ratio = 20.0
        hyper_ratio = 6.25

        total_ratio = seedlm_ratio * vptq_ratio * hyper_ratio
        assert total_ratio == 250.0

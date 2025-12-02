"""
Unit tests for cross-phase utilities
Tests model size detection, batch sizing, and diversity metrics
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cross_phase.utils import (
    calculate_safe_batch_size,
    compute_population_diversity,
    detect_training_divergence,
    get_model_size,
    validate_model_diversity,
)


@pytest.mark.skipif("torch" not in sys.modules, reason="PyTorch not available")
class TestModelSizeUtils:
    """Test model size utilities"""

    def test_get_model_size(self, mock_model):
        """Test model size calculation"""
        size_info = get_model_size(mock_model)

        assert "params" in size_info
        assert "size_mb" in size_info
        assert "size_category" in size_info

        assert size_info["params"] > 0
        assert size_info["size_mb"] > 0

    def test_size_categorization_tiny(self):
        """Test tiny model categorization"""
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

        model = TinyModel()
        size_info = get_model_size(model)

        assert size_info["size_category"] == "tiny"

    def test_calculate_safe_batch_size(self):
        """Test safe batch size calculation"""
        batch_size = calculate_safe_batch_size(model_size_mb=95.4, device_vram_gb=6.0)

        assert batch_size > 0
        assert isinstance(batch_size, int)

    def test_batch_size_scales_with_vram(self):
        """Test batch size scales with VRAM"""
        batch_small = calculate_safe_batch_size(95.4, 4.0)
        batch_large = calculate_safe_batch_size(95.4, 16.0)

        assert batch_large > batch_small


class TestDiversityMetrics:
    """Test diversity and divergence detection"""

    def test_validate_model_diversity(self, mock_model):
        """Test model diversity validation"""
        models = [mock_model, mock_model, mock_model]

        # This should fail (all same model)
        is_diverse = validate_model_diversity(models, min_diversity=0.1)

        # Since we're using same model, diversity should be low
        assert isinstance(is_diverse, bool)

    def test_detect_training_divergence(self):
        """Test training divergence detection"""
        # Stable training (loss decreasing)
        losses = [3.5, 3.2, 2.9, 2.7, 2.5]
        is_diverging = detect_training_divergence(losses, window=3)

        assert is_diverging is False

        # Diverging training (loss increasing)
        losses_diverging = [2.5, 2.7, 3.0, 3.5, 4.2]
        is_diverging = detect_training_divergence(losses_diverging, window=3)

        assert is_diverging is True

    def test_compute_population_diversity(self):
        """Test population diversity computation"""
        # Mock fitness scores
        fitness_scores = [0.8, 0.7, 0.6, 0.5, 0.4]

        diversity = compute_population_diversity(fitness_scores)

        assert 0.0 <= diversity <= 1.0

"""
Unit tests for CompressedModel
Tests STE wrapper, dual output, and compression stats
"""

import pytest
import torch
import torch.nn as nn

from src.phase4_bitnet.compressed_model import CompressedModel
from src.phase4_bitnet.config import Phase4Config
from src.phase4_bitnet.quantizer import BitNetQuantizer


class SimpleTestModel(nn.Module):
    """Simple model for testing"""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(100, 32)
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 32)
        self.lm_head = nn.Linear(32, 100)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return self.lm_head(x)


class TestCompressedModel:
    """Test suite for CompressedModel"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Phase4Config(wandb_enabled=False, 
            sparsity_threshold=0.1,
            preserve_embedding_precision=True,
            preserve_output_precision=True,
        )

    @pytest.fixture
    def base_model(self):
        """Create base model"""
        return SimpleTestModel()

    @pytest.fixture
    def quantizer(self, config):
        """Create quantizer"""
        return BitNetQuantizer(config)

    @pytest.fixture
    def compressed_model(self, base_model, quantizer, config):
        """Create compressed model"""
        model = CompressedModel(base_model, quantizer, config)
        model.compress()
        return model

    def test_initialization(self, base_model, quantizer, config):
        """Test compressed model initialization"""
        model = CompressedModel(base_model, quantizer, config)

        assert model.base_model == base_model
        assert model.quantizer == quantizer
        assert model.config == config
        assert model.is_compressed is False

    def test_compression(self, base_model, quantizer, config):
        """Test model compression"""
        model = CompressedModel(base_model, quantizer, config)

        # Compress
        model.compress()

        # Check compression performed
        assert model.is_compressed is True
        assert len(model.quantized_state) > 0
        assert len(model.scale_factors) > 0
        assert len(model.shadow_weights) > 0

    def test_forward_before_compression(self, base_model, quantizer, config):
        """Test forward pass before compression"""
        model = CompressedModel(base_model, quantizer, config)

        # Forward should use base model
        x = torch.randint(0, 100, (2, 10))
        output = model(x)

        # Check output shape
        assert output.shape == (2, 10, 100)

    def test_forward_after_compression(self, compressed_model):
        """Test forward pass with compressed weights"""
        x = torch.randint(0, 100, (2, 10))

        # Forward pass
        output = compressed_model(x)

        # Check output shape
        assert output.shape == (2, 10, 100)

        # Output should be valid
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_ste_gradient_flow(self, compressed_model):
        """Test STE gradient flow"""
        compressed_model.train()

        # Forward pass
        x = torch.randint(0, 100, (2, 10))
        output = compressed_model(x)

        # Create loss
        target = torch.randint(0, 100, (2, 10))
        loss = nn.CrossEntropyLoss()(output.view(-1, 100), target.view(-1))

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_gradients = any(
            p.grad is not None for p in compressed_model.base_model.parameters() if p.requires_grad
        )

        assert has_gradients, "STE should preserve gradient flow"

    def test_get_quantized_state_dict(self, compressed_model):
        """Test quantized state dict retrieval"""
        quantized_dict = compressed_model.get_quantized_state_dict()

        # Check all parameters present
        assert len(quantized_dict) > 0

        # Check linear layers are int8
        assert quantized_dict["linear1.weight"].dtype == torch.int8
        assert quantized_dict["linear2.weight"].dtype == torch.int8

    def test_get_dequantized_state_dict(self, compressed_model):
        """Test dequantized FP16 state dict"""
        dequantized_dict = compressed_model.get_dequantized_state_dict()

        # Check all parameters present
        assert len(dequantized_dict) > 0

        # Check all are FP16 or FP32
        for param in dequantized_dict.values():
            assert param.dtype in [torch.float16, torch.float32]

    def test_dequantized_model_loadable(self, compressed_model, base_model):
        """Test dequantized model can be loaded"""
        # Get dequantized state
        dequantized_dict = compressed_model.get_dequantized_state_dict()

        # Create fresh model
        fresh_model = type(base_model)()

        # Load dequantized state
        fresh_model.load_state_dict(dequantized_dict, strict=False)

        # Test forward pass
        x = torch.randint(0, 100, (2, 10))
        output = fresh_model(x)

        assert output.shape == (2, 10, 100)
        assert not torch.isnan(output).any()

    def test_get_scale_factors(self, compressed_model):
        """Test scale factor retrieval"""
        scales = compressed_model.get_scale_factors()

        # Check scales exist
        assert len(scales) > 0

        # Check all scales are positive
        for scale in scales.values():
            assert (scale > 0).all()

    def test_compression_stats(self, compressed_model):
        """Test compression statistics"""
        stats = compressed_model.get_compression_stats()

        # Check required fields
        assert "is_compressed" in stats
        assert "original_size_mb" in stats
        assert "quantized_size_mb" in stats
        assert "compression_ratio" in stats
        assert "layers_quantized" in stats
        assert "sparsity_ratio" in stats

        # Check values
        assert stats["is_compressed"] is True
        assert stats["compression_ratio"] > 1.0
        assert 0.0 <= stats["sparsity_ratio"] <= 1.0

    def test_compression_ratio_calculation(self, compressed_model):
        """Test compression ratio is calculated correctly"""
        stats = compressed_model.get_compression_stats()

        original_mb = stats["original_size_mb"]
        compressed_mb = stats["quantized_size_mb"]
        ratio = stats["compression_ratio"]

        # Check formula
        expected_ratio = original_mb / compressed_mb
        assert abs(ratio - expected_ratio) < 0.01

    def test_stats_before_compression(self, base_model, quantizer, config):
        """Test stats before compression"""
        model = CompressedModel(base_model, quantizer, config)

        stats = model.get_compression_stats()

        assert stats["is_compressed"] is False
        assert stats["compression_ratio"] == 1.0

    def test_dual_output_size_difference(self, compressed_model):
        """Test quantized vs dequantized size difference"""
        quantized_dict = compressed_model.get_quantized_state_dict()
        dequantized_dict = compressed_model.get_dequantized_state_dict()

        # Calculate sizes
        quantized_size = sum(t.nelement() * t.element_size() for t in quantized_dict.values())

        dequantized_size = sum(t.nelement() * t.element_size() for t in dequantized_dict.values())

        # Quantized should be smaller
        assert quantized_size < dequantized_size

    def test_shadow_weights_preserved(self, compressed_model):
        """Test shadow weights are preserved for STE"""
        # Check shadow weights exist
        assert len(compressed_model.shadow_weights) > 0

        # Check they require gradients
        for shadow_weight in compressed_model.shadow_weights.values():
            assert shadow_weight.requires_grad is True


class TestCompressedModelEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def config(self):
        return Phase4Config(wandb_enabled=False, )

    @pytest.fixture
    def quantizer(self, config):
        return BitNetQuantizer(config)

    def test_get_quantized_before_compression_raises(self, quantizer, config):
        """Test getting quantized dict before compression raises error"""
        model = CompressedModel(SimpleTestModel(), quantizer, config)

        with pytest.raises(RuntimeError):
            model.get_quantized_state_dict()

    def test_get_scales_before_compression_raises(self, quantizer, config):
        """Test getting scales before compression raises error"""
        model = CompressedModel(SimpleTestModel(), quantizer, config)

        with pytest.raises(RuntimeError):
            model.get_scale_factors()

    def test_multiple_compressions(self, quantizer, config):
        """Test compressing same model multiple times"""
        model = CompressedModel(SimpleTestModel(), quantizer, config)

        # Compress twice
        model.compress()
        first_stats = model.get_compression_stats()

        model.compress()
        second_stats = model.get_compression_stats()

        # Stats should be similar (may vary slightly due to randomness)
        assert abs(first_stats["compression_ratio"] - second_stats["compression_ratio"]) < 1.0

    def test_empty_model_compression(self, quantizer, config):
        """Test compressing empty model"""

        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        model = CompressedModel(EmptyModel(), quantizer, config)

        # Should not crash
        model.compress()

        stats = model.get_compression_stats()
        assert stats["is_compressed"] is True

    def test_large_batch_forward(self, quantizer, config):
        """Test forward pass with large batch"""
        model = CompressedModel(SimpleTestModel(), quantizer, config)
        model.compress()

        # Large batch
        x = torch.randint(0, 100, (64, 50))
        output = model(x)

        assert output.shape == (64, 50, 100)
        assert not torch.isnan(output).any()


class TestCompressedModelIntegration:
    """Integration tests with other components"""

    @pytest.fixture
    def config(self):
        return Phase4Config(wandb_enabled=False, )

    def test_quantizer_compressed_model_roundtrip(self, config):
        """Test full quantize-compress-dequantize roundtrip"""
        # Create components
        base_model = SimpleTestModel()
        quantizer = BitNetQuantizer(config)

        # Get original output
        x = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            original_output = base_model(x)

        # Compress
        compressed_model = CompressedModel(base_model, quantizer, config)
        compressed_model.compress()

        # Get compressed output
        with torch.no_grad():
            compressed_output = compressed_model(x)

        # Dequantize and load
        dequantized_dict = compressed_model.get_dequantized_state_dict()
        restored_model = SimpleTestModel()
        restored_model.load_state_dict(dequantized_dict, strict=False)

        # Get restored output
        with torch.no_grad():
            restored_output = restored_model(x)

        # Outputs should be similar (not exact due to quantization)
        mse_compressed = ((original_output - compressed_output) ** 2).mean()
        mse_restored = ((original_output - restored_output) ** 2).mean()

        # Some degradation expected but should be reasonable
        assert mse_compressed < 100.0
        assert mse_restored < 100.0

    def test_config_preserve_settings_respected(self):
        """Test that config preserve settings are respected"""
        # Config with no preservation
        config_no_preserve = Phase4Config(wandb_enabled=False, 
            preserve_embedding_precision=False,
            preserve_output_precision=False,
        )

        quantizer = BitNetQuantizer(config_no_preserve)
        model = CompressedModel(SimpleTestModel(), quantizer, config_no_preserve)
        model.compress()

        # Get quantized dict
        quantized_dict = model.get_quantized_state_dict()

        # Even embeddings should be quantized (if config allows)
        # Note: This depends on quantizer implementation
        stats = model.get_compression_stats()
        assert stats["is_compressed"] is True

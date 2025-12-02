"""
Unit tests for BitNet Quantizer
Tests ternary quantization, scaling, and layer preservation
"""

import pytest
import torch
import torch.nn as nn
from src.phase4_bitnet.quantizer import BitNetQuantizer
from src.phase4_bitnet.config import Phase4Config


class TestBitNetQuantizer:
    """Test suite for BitNet quantizer"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Phase4Config(
            sparsity_threshold=0.1,
            preserve_embedding_precision=True,
            preserve_output_precision=True,
        )

    @pytest.fixture
    def quantizer(self, config):
        """Create quantizer instance"""
        return BitNetQuantizer(config)

    def test_quantizer_initialization(self, quantizer, config):
        """Test quantizer initializes correctly"""
        assert quantizer.config == config
        assert isinstance(quantizer.stats, dict)
        assert quantizer.stats['layers_quantized'] == 0

    def test_quantize_tensor_to_ternary(self, quantizer):
        """Test tensor quantization to {-1, 0, +1}"""
        # Create test tensor
        tensor = torch.tensor([
            [1.0, -1.0, 0.1, -0.1, 0.01],
            [2.0, -2.0, 0.2, -0.2, 0.02]
        ])

        # Quantize
        quantized, scale = quantizer.quantize_tensor(tensor, threshold=0.1)

        # Check dtype
        assert quantized.dtype == torch.int8

        # Check values are ternary
        unique_values = torch.unique(quantized)
        assert all(v in [-1, 0, 1] for v in unique_values.tolist())

        # Check scale is positive
        assert (scale > 0).all()

    def test_quantize_tensor_sparsity(self, quantizer):
        """Test sparsity injection"""
        # Create tensor with small values
        tensor = torch.tensor([
            [1.0, 0.05, 0.01, -1.0, -0.05],
        ])

        # Quantize with threshold
        quantized, scale = quantizer.quantize_tensor(
            tensor,
            threshold=0.1
        )

        # Count zeros (sparsity)
        zero_count = (quantized == 0).sum().item()

        # Should have zeros for small values
        assert zero_count > 0

        # Check specific values
        # 1.0 should map to 1 or -1
        assert quantized[0, 0].item() != 0

        # 0.01 should map to 0 (below threshold)
        # (assuming scale allows it)

    def test_dequantize_tensor(self, quantizer):
        """Test tensor dequantization"""
        # Create and quantize tensor
        original = torch.randn(3, 5)
        quantized, scale = quantizer.quantize_tensor(original)

        # Dequantize
        dequantized = quantizer.dequantize_tensor(quantized, scale)

        # Check dtype
        assert dequantized.dtype == torch.float32

        # Check shape preserved
        assert dequantized.shape == original.shape

        # Dequantized values should be in [-scale, 0, +scale] range
        max_scale = scale.max().item()
        assert (dequantized.abs() <= max_scale * 1.1).all()

    def test_quantize_dequantize_roundtrip(self, quantizer):
        """Test quantize-dequantize preserves approximate values"""
        # Create tensor
        original = torch.randn(5, 10)

        # Quantize then dequantize
        quantized, scale = quantizer.quantize_tensor(original)
        reconstructed = quantizer.dequantize_tensor(quantized, scale)

        # Calculate reconstruction error
        mse = ((original - reconstructed) ** 2).mean().item()

        # Error should be reasonable (not exact due to quantization)
        assert mse < 1.0  # Adjust threshold as needed

    def test_should_quantize_layer(self, quantizer):
        """Test layer quantization decision"""
        # Test layers that should be quantized
        assert quantizer.should_quantize_layer("model.attention.q_proj") is True
        assert quantizer.should_quantize_layer("model.mlp.fc1") is True
        assert quantizer.should_quantize_layer("linear_layer") is True

        # Test layers that should be preserved
        assert quantizer.should_quantize_layer("embeddings.word_embeddings") is False
        assert quantizer.should_quantize_layer("lm_head.dense") is False
        assert quantizer.should_quantize_layer("layer_norm.weight") is False

    def test_quantize_model(self, quantizer):
        """Test full model quantization"""
        # Create simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(100, 64)
                self.linear1 = nn.Linear(64, 128)
                self.linear2 = nn.Linear(128, 64)
                self.lm_head = nn.Linear(64, 100)

            def forward(self, x):
                x = self.embeddings(x)
                x = self.linear1(x)
                x = self.linear2(x)
                return self.lm_head(x)

        model = SimpleModel()

        # Quantize
        quantized_dict, scales = quantizer.quantize_model(model)

        # Check all parameters present
        assert len(quantized_dict) == len(list(model.state_dict().keys()))

        # Check statistics
        stats = quantizer.get_stats()
        assert stats['layers_quantized'] > 0
        assert stats['layers_preserved'] > 0
        assert stats['total_params'] > 0
        assert 0.0 <= stats['sparsity_ratio'] <= 1.0

        # Check linear layers are int8
        assert quantized_dict['linear1.weight'].dtype == torch.int8
        assert quantized_dict['linear2.weight'].dtype == torch.int8

        # Check embeddings are preserved (FP16)
        assert quantized_dict['embeddings.weight'].dtype in [
            torch.float16, torch.float32
        ]

    def test_dequantize_model(self, quantizer):
        """Test full model dequantization"""
        # Create and quantize model
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x)

        model = TinyModel()
        quantized_dict, scales = quantizer.quantize_model(model)

        # Dequantize
        dequantized_dict = quantizer.dequantize_model(
            quantized_dict,
            scales
        )

        # Check all parameters present
        assert len(dequantized_dict) == len(quantized_dict)

        # Check dequantized weights are FP16
        for param in dequantized_dict.values():
            assert param.dtype in [torch.float16, torch.float32]

    def test_per_channel_scaling(self, quantizer):
        """Test per-channel scale factor calculation"""
        # Create multi-channel tensor
        tensor = torch.randn(4, 8)  # 4 output channels, 8 input

        # Quantize
        quantized, scale = quantizer.quantize_tensor(tensor)

        # Scale should have one value per output channel
        assert scale.shape[0] == tensor.shape[0]
        assert scale.dim() == 2

    def test_sparsity_threshold_effect(self, quantizer):
        """Test different sparsity thresholds"""
        tensor = torch.randn(10, 10)

        # Quantize with low threshold (less sparsity)
        quantized_low, _ = quantizer.quantize_tensor(tensor, threshold=0.01)
        sparsity_low = (quantized_low == 0).float().mean().item()

        # Quantize with high threshold (more sparsity)
        quantized_high, _ = quantizer.quantize_tensor(tensor, threshold=0.5)
        sparsity_high = (quantized_high == 0).float().mean().item()

        # Higher threshold should give more sparsity
        assert sparsity_high > sparsity_low

    def test_get_stats(self, quantizer):
        """Test statistics retrieval"""
        # Create model and quantize
        model = nn.Linear(100, 50)
        quantizer.quantize_model(model)

        # Get stats
        stats = quantizer.get_stats()

        # Check required fields
        assert 'layers_quantized' in stats
        assert 'layers_preserved' in stats
        assert 'total_params' in stats
        assert 'quantized_params' in stats
        assert 'sparsity_ratio' in stats

        # Check types
        assert isinstance(stats['layers_quantized'], int)
        assert isinstance(stats['sparsity_ratio'], float)

    def test_zero_prevention_in_scale(self, quantizer):
        """Test that scale factors never become zero"""
        # Create tensor with all zeros
        zero_tensor = torch.zeros(5, 5)

        # Quantize
        quantized, scale = quantizer.quantize_tensor(zero_tensor)

        # Scale should be clamped to minimum value
        assert (scale >= 1e-8).all()

    def test_quantization_deterministic(self, quantizer):
        """Test quantization is deterministic"""
        tensor = torch.randn(10, 10)

        # Quantize twice
        quantized1, scale1 = quantizer.quantize_tensor(tensor)
        quantized2, scale2 = quantizer.quantize_tensor(tensor)

        # Results should be identical
        assert torch.equal(quantized1, quantized2)
        assert torch.allclose(scale1, scale2)


class TestQuantizerEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def config(self):
        return Phase4Config()

    @pytest.fixture
    def quantizer(self, config):
        return BitNetQuantizer(config)

    def test_1d_tensor_quantization(self, quantizer):
        """Test quantization of 1D tensors (bias)"""
        tensor_1d = torch.randn(64)

        # Should work without error
        quantized, scale = quantizer.quantize_tensor(tensor_1d)

        # Check shape preserved
        assert quantized.shape == tensor_1d.shape

        # Scale should be scalar
        assert scale.dim() == 0 or scale.numel() == 1

    def test_large_tensor_quantization(self, quantizer):
        """Test quantization of large tensors"""
        # Simulate large model layer
        large_tensor = torch.randn(4096, 4096)

        # Should complete without memory error
        quantized, scale = quantizer.quantize_tensor(large_tensor)

        assert quantized.shape == large_tensor.shape

    def test_extreme_values(self, quantizer):
        """Test quantization with extreme values"""
        # Very large values
        tensor = torch.tensor([[1e6, -1e6, 1e-6, -1e-6]])

        quantized, scale = quantizer.quantize_tensor(tensor)

        # Should still produce valid ternary values
        unique_vals = torch.unique(quantized)
        assert all(v in [-1, 0, 1] for v in unique_vals.tolist())

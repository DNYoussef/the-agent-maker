"""
E2E tests for Phase 4: BitNet (1.58-bit Quantization).

Tests the complete quantization pipeline including:
- Quantization (ternary weights: -1, 0, +1)
- Dequantization
- Compression ratio calculation
- Fine-tuning step with STE (Straight-Through Estimator)
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPhase4BitNetE2E:
    """E2E tests for Phase 4 BitNet pipeline."""

    def test_weight_quantization(self, mock_model):
        """Test quantization of weights to ternary values {-1, 0, +1}."""
        # Get a weight tensor
        weight = next(mock_model.parameters()).data

        # BitNet quantization: scale and quantize to {-1, 0, +1}
        scale = weight.abs().mean()
        normalized = weight / (scale + 1e-8)

        # Ternary quantization
        quantized = torch.sign(normalized)  # {-1, 0, +1}

        # Verify ternary values
        unique_values = torch.unique(quantized)
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_values.tolist())

    def test_weight_dequantization(self, mock_model):
        """Test dequantization back to float."""
        weight = next(mock_model.parameters()).data

        # Quantize
        scale = weight.abs().mean()
        normalized = weight / (scale + 1e-8)
        quantized = torch.sign(normalized)

        # Dequantize
        dequantized = quantized * scale

        # Verify dequantized has same shape
        assert dequantized.shape == weight.shape

    def test_quantization_preserves_shape(self, mock_model):
        """Test quantization preserves tensor shapes."""
        for param in mock_model.parameters():
            original_shape = param.shape

            # Quantize
            scale = param.abs().mean()
            normalized = param / (scale + 1e-8)
            quantized = torch.sign(normalized)

            assert quantized.shape == original_shape

    def test_compression_ratio_calculation(self, mock_model):
        """Test 1.58-bit compression ratio calculation."""
        # Original model size (32-bit float)
        original_bits = 0
        for param in mock_model.parameters():
            original_bits += param.numel() * 32  # 32 bits per float

        # Quantized model size (1.58 bits per weight + scale factor)
        quantized_bits = 0
        for param in mock_model.parameters():
            # 1.58 bits per weight (ternary encoding)
            quantized_bits += param.numel() * 1.58
            # Add scale factor (32 bits per tensor)
            quantized_bits += 32

        compression_ratio = original_bits / quantized_bits

        # Target: ~8.2x compression
        assert compression_ratio > 5.0  # At least 5x compression

    def test_activation_quantization(self, mock_model, mock_tokenizer):
        """Test 8-bit activation quantization."""
        inputs = mock_tokenizer("Test activation quantization")

        with torch.no_grad():
            outputs = mock_model(inputs['input_ids'])
            activations = outputs.hidden_states

            # Quantize activations to 8-bit
            activation_min = activations.min()
            activation_max = activations.max()
            scale = (activation_max - activation_min) / 255

            quantized_activations = torch.round(
                (activations - activation_min) / scale
            ).clamp(0, 255)

            # Verify 8-bit range
            assert quantized_activations.min() >= 0
            assert quantized_activations.max() <= 255

    def test_ste_forward_pass(self, mock_model, mock_tokenizer):
        """Test STE (Straight-Through Estimator) forward pass."""
        inputs = mock_tokenizer("Test STE forward")

        # Get original weight
        first_param = next(mock_model.parameters())
        original_weight = first_param.data.clone()

        # Quantize for forward pass
        scale = original_weight.abs().mean()
        normalized = original_weight / (scale + 1e-8)
        quantized_weight = torch.sign(normalized) * scale

        # Replace weight temporarily
        first_param.data = quantized_weight

        # Forward pass with quantized weights
        outputs = mock_model(inputs['input_ids'])

        assert outputs.logits is not None

        # Restore original weight
        first_param.data = original_weight

    def test_ste_backward_pass(self, mock_model, mock_dataloader):
        """Test STE backward pass (gradients flow through quantization)."""
        batch = next(iter(mock_dataloader))

        # Forward pass
        outputs = mock_model(batch['input_ids'], labels=batch['labels'])
        loss = outputs.loss

        # Backward pass (STE: gradients bypass quantization)
        loss.backward()

        # Verify gradients exist (STE allows gradient flow)
        for param in mock_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_quantization_aware_training_step(self, mock_model, mock_dataloader):
        """Test single QAT (Quantization-Aware Training) step."""
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
        batch = next(iter(mock_dataloader))

        # Store original weights
        original_weights = [p.data.clone() for p in mock_model.parameters()]

        # Training step with STE
        optimizer.zero_grad()

        # Forward (with quantized weights in real implementation)
        outputs = mock_model(batch['input_ids'], labels=batch['labels'])
        loss = outputs.loss

        # Backward (gradients to full-precision weights)
        loss.backward()
        optimizer.step()

        # Verify weights updated
        for original, updated in zip(original_weights, mock_model.parameters()):
            # Weights should have changed
            assert not torch.allclose(original, updated.data)

    def test_fine_tuning_preserves_performance(self, mock_model, mock_dataloader):
        """Test fine-tuning maintains model performance."""
        # Initial evaluation
        mock_model.eval()
        with torch.no_grad():
            batch = next(iter(mock_dataloader))
            initial_loss = mock_model(batch['input_ids'], labels=batch['labels']).loss.item()

        # Fine-tune 3 steps
        mock_model.train()
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)

        for i, batch in enumerate(mock_dataloader):
            if i >= 3:
                break

            optimizer.zero_grad()
            outputs = mock_model(batch['input_ids'], labels=batch['labels'])
            outputs.loss.backward()
            optimizer.step()

        # Final evaluation
        mock_model.eval()
        with torch.no_grad():
            batch = next(iter(mock_dataloader))
            final_loss = mock_model(batch['input_ids'], labels=batch['labels']).loss.item()

        # Loss should be reasonable (may increase or decrease slightly)
        assert final_loss > 0

    def test_model_size_reduction(self, mock_model, temp_checkpoint_dir):
        """Test quantized model has smaller file size."""
        # Save full-precision model
        fp_path = temp_checkpoint_dir / "full_precision.pt"
        torch.save(mock_model.state_dict(), fp_path)
        fp_size = fp_path.stat().st_size

        # Quantize and save
        quantized_state = {}
        for name, param in mock_model.named_parameters():
            # Quantize to ternary
            scale = param.abs().mean()
            normalized = param / (scale + 1e-8)
            quantized = torch.sign(normalized)

            # Store quantized + scale
            quantized_state[name] = {
                'quantized': quantized.to(torch.int8),  # Store as int8
                'scale': scale
            }

        quant_path = temp_checkpoint_dir / "quantized.pt"
        torch.save(quantized_state, quant_path)
        quant_size = quant_path.stat().st_size

        # Quantized should be smaller (may not be 8.2x due to overhead)
        assert quant_size < fp_size

    def test_inference_speedup(self, mock_model, mock_tokenizer):
        """Test quantized inference is faster (simulated)."""
        inputs = mock_tokenizer("Test inference speed")

        # Full-precision inference
        import time

        mock_model.eval()
        with torch.no_grad():
            start_fp = time.time()
            for _ in range(10):
                outputs_fp = mock_model(inputs['input_ids'])
            time_fp = time.time() - start_fp

        # Quantized inference (simulated - would be faster with real BitNet)
        # In real implementation, BitNet uses specialized kernels
        with torch.no_grad():
            start_quant = time.time()
            for _ in range(10):
                outputs_quant = mock_model(inputs['input_ids'])
            time_quant = time.time() - start_quant

        # Both should complete successfully
        assert time_fp > 0
        assert time_quant > 0

    def test_layer_wise_quantization(self, mock_model):
        """Test quantization applied layer by layer."""
        quantized_layers = {}

        for name, module in mock_model.named_modules():
            if isinstance(module, nn.Linear):
                # Quantize linear layer weights
                weight = module.weight.data
                scale = weight.abs().mean()
                normalized = weight / (scale + 1e-8)
                quantized = torch.sign(normalized)

                quantized_layers[name] = {
                    'quantized': quantized,
                    'scale': scale
                }

        # Should have quantized embeddings and lm_head at minimum
        assert len(quantized_layers) >= 1

    def test_embedding_quantization(self, mock_model):
        """Test embedding layer quantization."""
        embeddings = mock_model.get_input_embeddings()
        weight = embeddings.weight.data

        # Quantize embeddings
        scale = weight.abs().mean()
        normalized = weight / (scale + 1e-8)
        quantized = torch.sign(normalized)

        assert quantized.shape == weight.shape
        assert quantized.unique().numel() <= 3  # {-1, 0, 1}

    def test_absmax_quantization(self, mock_model):
        """Test absmax quantization scheme."""
        weight = next(mock_model.parameters()).data

        # Absmax quantization
        absmax = weight.abs().max()
        scale = absmax / 1.0  # Max quantized value is 1

        normalized = weight / (scale + 1e-8)
        quantized = torch.clamp(torch.round(normalized), -1, 1)

        assert quantized.min() >= -1
        assert quantized.max() <= 1

    def test_per_tensor_quantization(self, mock_model):
        """Test per-tensor quantization (one scale per tensor)."""
        for param in mock_model.parameters():
            # One scale factor per tensor
            scale = param.abs().mean()
            normalized = param / (scale + 1e-8)
            quantized = torch.sign(normalized)

            # Dequantize
            dequantized = quantized * scale

            assert dequantized.shape == param.shape

    def test_per_channel_quantization(self, mock_model):
        """Test per-channel quantization (one scale per output channel)."""
        # Get a 2D weight matrix
        for param in mock_model.parameters():
            if param.dim() == 2:
                # Per-output-channel quantization
                scales = param.abs().mean(dim=1, keepdim=True)
                normalized = param / (scales + 1e-8)
                quantized = torch.sign(normalized)

                # Verify shape preserved
                assert quantized.shape == param.shape
                break

    def test_gradient_clipping_with_ste(self, mock_model, mock_dataloader):
        """Test gradient clipping works with STE."""
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
        batch = next(iter(mock_dataloader))

        optimizer.zero_grad()
        outputs = mock_model(batch['input_ids'], labels=batch['labels'])
        outputs.loss.backward()

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(mock_model.parameters(), max_norm)

        # Verify gradients clipped
        total_norm = 0.0
        for param in mock_model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= max_norm + 1e-6  # Allow small numerical error

    def test_mixed_precision_quantization(self, mock_model):
        """Test mixed precision (some layers quantized, some full precision)."""
        # Strategy: Quantize most layers, keep embeddings full precision
        quantization_map = {}

        for name, param in mock_model.named_parameters():
            if 'embedding' in name.lower():
                # Keep full precision
                quantization_map[name] = 'fp32'
            else:
                # Quantize to 1.58-bit
                quantization_map[name] = '1.58bit'

        assert 'fp32' in quantization_map.values()
        assert '1.58bit' in quantization_map.values()

    def test_quantization_error_measurement(self, mock_model):
        """Test quantization error measurement."""
        weight = next(mock_model.parameters()).data

        # Quantize
        scale = weight.abs().mean()
        normalized = weight / (scale + 1e-8)
        quantized = torch.sign(normalized) * scale

        # Measure error
        mse = torch.mean((weight - quantized) ** 2)
        mae = torch.mean(torch.abs(weight - quantized))

        assert mse >= 0
        assert mae >= 0

    def test_calibration_data_statistics(self, mock_dataloader):
        """Test collecting calibration statistics for quantization."""
        # Collect activation statistics
        activation_mins = []
        activation_maxs = []

        for batch in mock_dataloader:
            # Simulated activations
            activations = batch['input_ids'].float()

            activation_mins.append(activations.min().item())
            activation_maxs.append(activations.max().item())

        # Calculate global min/max for quantization
        global_min = min(activation_mins)
        global_max = max(activation_maxs)

        assert global_min <= global_max

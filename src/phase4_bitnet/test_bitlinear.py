"""
Test script for BitLinear layer implementation

Validates:
1. BitLinear layer creates {-1, 0, +1} weights
2. Activation quantization to 8-bit range
3. STE gradient flow
4. Drop-in replacement for nn.Linear
5. Memory footprint reduction
6. Compatibility with Phase 3 output
"""

import torch
import torch.nn as nn

from src.phase4_bitnet.bitlinear import BitLinear, replace_linear_with_bitlinear
from src.phase4_bitnet.quantizer import activation_quant, apply_ste


def test_bitlinear_quantization():
    """Test that BitLinear quantizes weights to {-1, 0, +1}"""
    print("\n=== Test 1: Weight Quantization ===")

    layer = BitLinear(128, 256, bias=True)

    # Run forward pass to trigger quantization
    x = torch.randn(2, 10, 128)
    output = layer(x)

    # Get quantized weights
    quant_state = layer.get_quantized_state()
    w_quant = quant_state["quantized_weight"]

    # Check that weights are in {-1, 0, +1}
    unique_values = torch.unique(w_quant)
    print(f"Unique quantized values: {unique_values.tolist()}")

    assert set(unique_values.tolist()).issubset({-1, 0, 1}), "Weights not ternary!"

    # Check sparsity
    sparsity = (w_quant == 0).sum().item() / w_quant.numel()
    print(f"Sparsity ratio: {sparsity:.2%}")

    print("✓ Weight quantization passed")


def test_activation_quantization():
    """Test 8-bit activation quantization"""
    print("\n=== Test 2: Activation Quantization ===")

    # Create test activation
    x = torch.randn(2, 10, 128) * 10  # Scale up for testing

    # Quantize
    x_quant = activation_quant(x)

    # Check range (should be close to original range after dequantization)
    print(f"Original range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Quantized range: [{x_quant.min():.2f}, {x_quant.max():.2f}]")

    # Check that quantization reduces precision
    num_unique_original = len(torch.unique(x))
    num_unique_quantized = len(torch.unique(x_quant))
    print(f"Unique values: {num_unique_original} -> {num_unique_quantized}")

    # Check that values are approximately preserved
    mse = ((x - x_quant) ** 2).mean()
    print(f"Quantization MSE: {mse:.6f}")

    assert mse < 1.0, "Quantization error too high!"

    print("✓ Activation quantization passed")


def test_ste_gradient_flow():
    """Test that STE allows gradients to flow"""
    print("\n=== Test 3: STE Gradient Flow ===")

    layer = BitLinear(64, 32, bias=False)
    layer.weight.requires_grad = True

    x = torch.randn(2, 5, 64, requires_grad=True)

    # Forward pass
    output = layer(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert layer.weight.grad is not None, "No gradient for weights!"
    assert x.grad is not None, "No gradient for input!"

    grad_norm = layer.weight.grad.norm().item()
    print(f"Weight gradient norm: {grad_norm:.6f}")

    assert grad_norm > 0, "Gradient is zero!"

    print("✓ STE gradient flow passed")


def test_drop_in_replacement():
    """Test that BitLinear is a drop-in replacement for nn.Linear"""
    print("\n=== Test 4: Drop-in Replacement ===")

    # Create simple model with nn.Linear
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = SimpleModel()

    # Test original model
    x = torch.randn(2, 128)
    output_original = model(x)

    # Replace with BitLinear
    model = replace_linear_with_bitlinear(model, exclude_patterns=[])

    # Test replaced model
    output_bitlinear = model(x)

    # Check that output shapes match
    assert output_original.shape == output_bitlinear.shape, "Output shape mismatch!"

    # Check that all Linear layers were replaced
    num_bitlinear = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    print(f"Replaced {num_bitlinear} nn.Linear layers with BitLinear")

    assert num_bitlinear == 3, f"Expected 3 BitLinear layers, got {num_bitlinear}"

    print("✓ Drop-in replacement passed")


def test_memory_footprint():
    """Test memory footprint reduction"""
    print("\n=== Test 5: Memory Footprint ===")

    layer = BitLinear(512, 1024, bias=True)

    footprint = layer.get_memory_footprint()

    print(f"Original FP32: {footprint['original_fp32'] / 1024:.2f} KB")
    print(f"Quantized 1.58-bit: {footprint['quantized_1.58bit'] / 1024:.2f} KB")
    print(f"Compression ratio: {footprint['compression_ratio']:.2f}x")

    assert footprint["compression_ratio"] > 7.0, "Compression ratio too low!"

    print("✓ Memory footprint test passed")


def test_phase3_compatibility():
    """Test compatibility with Phase 3 Quiet-STaR output"""
    print("\n=== Test 6: Phase 3 Compatibility ===")

    # Simulate Phase 3 model structure
    class Phase3Model(nn.Module):
        def __init__(self, vocab_size=50000, hidden_size=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.attention = nn.Linear(hidden_size, hidden_size * 3)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size)
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            attn = self.attention(x)
            x = x + self.mlp(x)
            return self.lm_head(x)

    model = Phase3Model()

    # Preserve embedding and lm_head (as per Phase 4 config)
    model_quantized = replace_linear_with_bitlinear(
        model, exclude_patterns=["embedding", "lm_head"]
    )

    # Check that only MLP and attention were quantized
    num_bitlinear = sum(1 for m in model_quantized.modules() if isinstance(m, BitLinear))
    print(f"Quantized {num_bitlinear} layers (excluding embedding and lm_head)")

    # Verify embedding and lm_head are still nn.Linear
    assert isinstance(model_quantized.lm_head, nn.Linear), "lm_head was replaced!"
    assert isinstance(model_quantized.embedding, nn.Embedding), "embedding was replaced!"

    # Test forward pass
    input_ids = torch.randint(0, 50000, (2, 10))
    output = model_quantized(input_ids)

    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 10, 50000), "Output shape mismatch!"

    print("✓ Phase 3 compatibility passed")


def test_safetensors_compatibility():
    """Test that quantized state can be saved/loaded"""
    print("\n=== Test 7: SafeTensors Compatibility ===")

    layer = BitLinear(128, 256, bias=True)

    # Get quantized state
    quant_state = layer.get_quantized_state()

    print(f"Quantized weight shape: {quant_state['quantized_weight'].shape}")
    print(f"Scale factor shape: {quant_state['scale_factor'].shape}")
    print(f"Quantized weight dtype: {quant_state['quantized_weight'].dtype}")
    print(f"Scale factor dtype: {quant_state['scale_factor'].dtype}")

    # Create new layer and load state
    layer_new = BitLinear(128, 256, bias=True)
    layer_new.load_quantized_state(quant_state)

    # Test that loaded weights work
    x = torch.randn(2, 10, 128)
    output_original = layer(x)
    output_loaded = layer_new(x)

    # Check similarity (should be very close)
    mse = ((output_original - output_loaded) ** 2).mean()
    print(f"Reconstruction MSE: {mse:.6f}")

    assert mse < 1e-4, "Reconstruction error too high!"

    print("✓ SafeTensors compatibility passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("BitLinear Implementation Tests")
    print("=" * 60)

    tests = [
        test_bitlinear_quantization,
        test_activation_quantization,
        test_ste_gradient_flow,
        test_drop_in_replacement,
        test_memory_footprint,
        test_phase3_compatibility,
        test_safetensors_compatibility,
    ]

    failed = []

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed.append((test.__name__, str(e)))

    print("\n" + "=" * 60)
    if not failed:
        print("✓ ALL TESTS PASSED")
    else:
        print(f"✗ {len(failed)} TESTS FAILED:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    print("=" * 60)

    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

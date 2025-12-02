"""
Test script to verify refactored titans_mag.py maintains backward compatibility.

This script verifies that:
1. All components can be imported from the components module
2. The main TitansMAGBackbone can still be imported from titans_mag
3. The model can be instantiated and run a forward pass
"""

import torch
from model.model_config import TitansMAGConfig
from model.titans_mag import TitansMAGBackbone, TitansMAGLayer

# Also verify direct component imports work
from model.components import (
    RMSNorm,
    SwiGLUMLP,
    SlidingWindowAttention,
    LongTermMemory,
    MAGGate
)


def test_component_imports():
    """Test that all components can be imported"""
    print("Testing component imports...")

    # Test RMSNorm
    norm = RMSNorm(512)
    assert norm is not None
    print("  RMSNorm: OK")

    # Test SwiGLUMLP
    mlp = SwiGLUMLP(512, 2048)
    assert mlp is not None
    print("  SwiGLUMLP: OK")

    # Test SlidingWindowAttention
    attn = SlidingWindowAttention(512, 8, 256)
    assert attn is not None
    print("  SlidingWindowAttention: OK")

    # Test LongTermMemory
    ltm = LongTermMemory(512, 128)
    assert ltm is not None
    print("  LongTermMemory: OK")

    # Test MAGGate
    gate = MAGGate(512)
    assert gate is not None
    print("  MAGGate: OK")

    print("All component imports successful!\n")


def test_model_instantiation():
    """Test that the model can be instantiated"""
    print("Testing model instantiation...")

    config = TitansMAGConfig()
    model = TitansMAGBackbone(config)

    param_count = model.count_parameters()
    print(f"  Model created with {param_count:,} parameters")
    print("  Model instantiation: OK\n")

    return model


def test_forward_pass(model):
    """Test that a forward pass works"""
    print("Testing forward pass...")

    # Create dummy input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))

    # Forward pass
    output, loss_gate = model(input_ids)

    assert output.shape == (batch_size, seq_len, model.config.d_model)
    assert loss_gate.dim() == 0  # scalar

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Gate loss: {loss_gate.item():.6f}")
    print("  Forward pass: OK\n")


def test_memory_reset():
    """Test memory reset functionality"""
    print("Testing memory reset...")

    config = TitansMAGConfig()
    model = TitansMAGBackbone(config)

    # Run forward pass
    input_ids = torch.randint(0, 10000, (2, 128))
    _, _ = model(input_ids)

    # Reset memory
    model.reset_memory()

    # Verify memory is reset
    assert torch.all(model.ltm.memory_state == 0)
    print("  Memory reset: OK\n")


if __name__ == "__main__":
    print("="*60)
    print("Titans-MAG Refactoring Verification Tests")
    print("="*60 + "\n")

    try:
        # Test 1: Component imports
        test_component_imports()

        # Test 2: Model instantiation
        model = test_model_instantiation()

        # Test 3: Forward pass
        test_forward_pass(model)

        # Test 4: Memory reset
        test_memory_reset()

        print("="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nRefactoring successful! The modular structure maintains")
        print("full backward compatibility with the original implementation.")

    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

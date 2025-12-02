"""
Quick test training with synthetic data (no datasets library needed)

Usage:
    python src/phase1_cognate/test_training_synthetic.py
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config

def generate_synthetic_batch(batch_size=4, seq_len=64, vocab_size=50257):
    """Generate synthetic training batch"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels

def test_training_loop():
    """Test training loop with synthetic data"""

    print("\n" + "="*70)
    print("QUICK TRAINING TEST (Synthetic Data)")
    print("="*70 + "\n")

    # Create model
    config = Phase1Config(specialization='reasoning')
    model = TRMTitansMAGModel(config)

    print("Model created:")
    counts = model.count_parameters()
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    print()

    # Create optimizer (simple AdamW for quick test)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    print("[OK] Optimizer created (AdamW)\n")

    # Training loop (10 steps)
    model.train()
    print("Training for 10 steps...")

    for step in range(10):
        # Generate batch
        input_ids, labels = generate_synthetic_batch(batch_size=4, seq_len=64)

        # Forward pass
        output = model(input_ids, labels=labels)
        loss = output['loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Log
        if step % 2 == 0:
            halt_mean = output['halting_steps'].float().mean().item()
            print(f"Step {step:2d}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}, "
                  f"halt_steps={halt_mean:.2f}")

    print("\n[OK] Training loop successful!\n")

    # Test inference
    model.eval()
    print("Testing inference...")

    with torch.no_grad():
        test_input = torch.randint(0, 50257, (1, 32))
        output = model(test_input)

    print(f"[OK] Inference successful!")
    print(f"  Output shape: {output['logits'].shape}")
    print(f"  Halting steps: {output['halting_steps'].item():.1f}")

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70 + "\n")

    print("Next steps:")
    print("1. Install datasets: pip install datasets")
    print("2. Run full test: python src/phase1_cognate/train_phase1.py --model reasoning --test")
    print("3. Full training: python src/phase1_cognate/train_phase1.py --model reasoning --epochs 10")

if __name__ == "__main__":
    test_training_loop()

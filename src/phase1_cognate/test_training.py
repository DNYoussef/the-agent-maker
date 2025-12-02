"""
Quick test of Phase 1 training pipeline

Tests model creation, dataset processing, and training loop
without downloading actual datasets (uses synthetic data).
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from phase1_cognate.data.curriculum_loader import CurriculumLoader
from phase1_cognate.training.trainer import Phase1Trainer, TrainingConfig


def create_synthetic_datasets(num_samples=100):
    """Create synthetic datasets for testing"""
    datasets = {}

    dataset_names = ["gsm8k", "svamp", "mbpp", "arc_easy", "piqa", "wikitext"]

    for name in dataset_names:
        samples = []
        for i in range(num_samples):
            samples.append({
                "text": f"Q: Test question {i} for {name}?\nA: Test answer {i}",
                "input": f"Test question {i}",
                "output": f"Test answer {i}",
                "metadata": {"dataset": name, "type": "test"}
            })
        datasets[name] = samples

    return datasets


class SimpleTokenizer:
    """Simple mock tokenizer"""
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text, max_length=512, **kwargs):
        # Simple character-based tokenization
        tokens = [ord(c) % 32768 for c in text[:max_length]]

        # Pad
        tokens = tokens + [0] * (max_length - len(tokens))

        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.tensor([[1] * min(len(text), max_length) +
                                           [0] * max(0, max_length - len(text))])
        }


def test_model_creation():
    """Test creating all 3 models"""
    print("\n" + "="*70)
    print("TEST 1: MODEL CREATION")
    print("="*70 + "\n")

    for spec in ["reasoning", "memory", "speed"]:
        print(f"Creating {spec} model...")
        config = Phase1Config(specialization=spec)
        model = TRMTitansMAGModel(config)

        param_counts = model.count_parameters()
        print(f"  OK {spec}: {param_counts['total']:,} params")

        # Test forward pass
        test_input = torch.randint(0, 32768, (2, 64))
        with torch.no_grad():
            output = model(test_input)
        print(f"  OK Forward pass: logits shape = {output['logits'].shape}")

    print("\nOK Model creation test passed!\n")


def test_curriculum():
    """Test curriculum loader"""
    print("\n" + "="*70)
    print("TEST 2: CURRICULUM LOADER")
    print("="*70 + "\n")

    curriculum = CurriculumLoader()

    # Test epoch mapping
    for epoch in [1, 4, 7]:
        stage = curriculum.get_stage_for_epoch(epoch)
        datasets = curriculum.get_datasets_for_epoch(epoch)
        print(f"Epoch {epoch}: {stage.name} ({len(datasets)} datasets)")

    print("\nOK Curriculum loader test passed!\n")


def test_training_loop():
    """Test training loop with synthetic data"""
    print("\n" + "="*70)
    print("TEST 3: TRAINING LOOP (1 epoch, synthetic data)")
    print("="*70 + "\n")

    # Create model
    model_config = Phase1Config(specialization="reasoning")
    model = TRMTitansMAGModel(model_config)

    # Create synthetic datasets
    datasets = create_synthetic_datasets(num_samples=50)

    # Create tokenizer
    tokenizer = SimpleTokenizer()

    # Create training config (minimal)
    train_config = TrainingConfig(
        model_config=model_config,
        num_epochs=1,  # Just 1 epoch
        batch_size=4,
        checkpoint_dir=Path("tests/artifacts/checkpoints"),
        wandb_mode="disabled",  # Disable W&B for test
        device="cpu",
        log_every_n_steps=10
    )

    # Create trainer
    trainer = Phase1Trainer(
        model=model,
        config=train_config,
        train_datasets=datasets,
        val_datasets=None,  # No validation
        tokenizer=tokenizer
    )

    # Train (just 1 epoch)
    try:
        print("Starting training...")
        trainer.train()
        print("\nOK Training loop test passed!\n")
        return True
    except Exception as e:
        print(f"\nFAILED Training loop test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PHASE 1 TRAINING PIPELINE TEST")
    print("="*70)

    # Test 1: Model creation
    try:
        test_model_creation()
    except Exception as e:
        print(f"FAILED Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 2: Curriculum
    try:
        test_curriculum()
    except Exception as e:
        print(f"FAILED Curriculum test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 3: Training loop
    success = test_training_loop()

    if success:
        print("\n" + "="*70)
        print("ALL TESTS PASSED - OK")
        print("="*70 + "\n")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())

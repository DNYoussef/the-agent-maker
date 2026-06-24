"""
Quick test of Phase 1 training pipeline

Tests model creation, dataset processing, and training loop
without downloading actual datasets (uses synthetic data).
"""

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repo src to path when run directly from tests/sandbox.
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from phase1_cognate.data.curriculum_loader import CurriculumLoader  # noqa: E402
from phase1_cognate.model.full_model import TRMTitansMAGModel  # noqa: E402
from phase1_cognate.model.model_config import Phase1Config  # noqa: E402
from phase1_cognate.training.trainer import Phase1Trainer, TrainingConfig  # noqa: E402


def create_synthetic_datasets(num_samples=100) -> dict:
    """Create synthetic datasets for testing"""
    datasets = {}

    dataset_names = ["gsm8k", "svamp", "mbpp", "arc_easy", "piqa", "wikitext"]

    for name in dataset_names:
        samples = []
        for i in range(num_samples):
            samples.append(
                {
                    "text": f"Q: Test question {i} for {name}?\nA: Test answer {i}",
                    "input": f"Test question {i}",
                    "output": f"Test answer {i}",
                    "metadata": {"dataset": name, "type": "test"},
                }
            )
        datasets[name] = samples

    return datasets


class SimpleTokenizer:
    """Simple mock tokenizer"""

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text, max_length=512, **kwargs) -> None:
        # Simple character-based tokenization
        tokens = [ord(c) % 32768 for c in text[:max_length]]

        # Pad
        tokens = tokens + [0] * (max_length - len(tokens))

        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.tensor(
                [[1] * min(len(text), max_length) + [0] * max(0, max_length - len(text))]
            ),
        }


class TinyTrainingModel(nn.Module):
    """Small deterministic model for exercising the trainer loop safely."""

    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.output = nn.Linear(vocab_size, vocab_size, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        bounded_ids = input_ids.remainder(self.vocab_size)
        one_hot = F.one_hot(bounded_ids, num_classes=self.vocab_size).float()
        logits = self.output(one_hot)
        result = {
            "logits": logits,
            "halting_steps": torch.ones(input_ids.shape[0], device=input_ids.device),
        }

        if labels is not None:
            bounded_labels = labels.remainder(self.vocab_size)
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                bounded_labels.reshape(-1),
            )
            zero = torch.zeros((), device=input_ids.device)
            result.update(
                {
                    "loss": loss,
                    "loss_ce": loss,
                    "loss_act": zero,
                    "loss_gate": zero,
                }
            )

        return result

    def count_parameters(self) -> dict[str, int]:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {"total": total, "trainable": trainable}


def test_model_creation() -> None:
    """Test creating all 3 models"""
    print("\n" + "=" * 70)
    print("TEST 1: MODEL CREATION")
    print("=" * 70 + "\n")

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


def test_curriculum() -> None:
    """Test curriculum loader"""
    print("\n" + "=" * 70)
    print("TEST 2: CURRICULUM LOADER")
    print("=" * 70 + "\n")

    curriculum = CurriculumLoader()

    # Test epoch mapping
    for epoch in [1, 4, 7]:
        stage = curriculum.get_stage_for_epoch(epoch)
        datasets = curriculum.get_datasets_for_epoch(epoch)
        print(f"Epoch {epoch}: {stage.name} ({len(datasets)} datasets)")

    print("\nOK Curriculum loader test passed!\n")


def test_training_loop(tmp_path: Path) -> None:
    """Test training loop with synthetic data"""
    print("\n" + "=" * 70)
    print("TEST 3: TRAINING LOOP (1 epoch, synthetic data)")
    print("=" * 70 + "\n")

    model_config = Phase1Config(specialization="reasoning")
    model = TinyTrainingModel()

    # Create synthetic datasets
    datasets = create_synthetic_datasets(num_samples=50)

    # Create tokenizer
    tokenizer = SimpleTokenizer()

    # Create training config (minimal)
    train_config = TrainingConfig(
        model_config=model_config,
        num_epochs=1,  # Just 1 epoch
        batch_size=4,
        checkpoint_dir=tmp_path / "checkpoints",
        wandb_mode="disabled",  # Disable W&B for test
        device="cpu",
        log_every_n_steps=10,
        use_curriculum=False,
        use_lr_scheduler=False,
        use_ema=False,
        gradient_accumulation_steps=1,
        save_every_n_epochs=99,
    )

    # Create trainer
    trainer = Phase1Trainer(
        model=model,
        config=train_config,
        train_datasets=datasets,
        val_datasets=None,  # No validation
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    assert trainer.global_step > 0
    assert trainer.current_epoch == 1
    print("\nOK Training loop test passed!\n")


def main() -> None:
    """Run all tests"""
    print("\n" + "=" * 70)
    print("PHASE 1 TRAINING PIPELINE TEST")
    print("=" * 70)

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
    try:
        with tempfile.TemporaryDirectory() as tmp:
            test_training_loop(Path(tmp))
    except Exception as e:
        print(f"FAILED Training loop test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED - OK")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    exit(main())

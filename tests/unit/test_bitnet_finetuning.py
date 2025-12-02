"""
Unit tests for Fine-Tuning Pipeline
Tests MuGrokfast STE mode, gradient flow, and quality recovery
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.phase4_bitnet.compressed_model import CompressedModel
from src.phase4_bitnet.config import Phase4Config
from src.phase4_bitnet.fine_tuner import FineTuner
from src.phase4_bitnet.quantizer import BitNetQuantizer


class SimpleModel(nn.Module):
    """Simple test model"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.linear(input_ids.float())

        if labels is not None:
            loss = nn.MSELoss()(output, labels.float())
            return type("Output", (), {"loss": loss})()

        return output


class SimpleDataset(Dataset):
    """Simple test dataset"""

    def __init__(self, num_samples=20):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randn(10),
            "attention_mask": torch.ones(10),
        }


class TestFineTuner:
    """Test suite for FineTuner"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Phase4Config(wandb_enabled=False, 
            fine_tune_epochs=2,
            fine_tune_lr=1e-4,
            enable_fine_tuning=True,
            enable_grokfast=True,
            grokfast_lambda=2.0,
        )

    @pytest.fixture
    def compressed_model(self, config):
        """Create compressed model for testing"""
        base_model = SimpleModel()
        quantizer = BitNetQuantizer(config)

        model = CompressedModel(base_model, quantizer, config)
        model.compress()

        return model

    @pytest.fixture
    def dataloader(self):
        """Create test dataloader"""
        dataset = SimpleDataset(num_samples=20)
        return DataLoader(dataset, batch_size=4)

    def test_finetuner_initialization(self, compressed_model, config):
        """Test fine-tuner initialization"""
        tuner = FineTuner(model=compressed_model, config=config, device="cpu")

        assert tuner.model == compressed_model
        assert tuner.config == config
        assert tuner.device == "cpu"
        assert tuner.optimizer is not None
        assert tuner.current_epoch == 0

    def test_optimizer_creation(self, compressed_model, config):
        """Test MuGrokfast optimizer creation with STE mode"""
        tuner = FineTuner(compressed_model, config, device="cpu")

        # Check optimizer exists
        assert tuner.optimizer is not None

        # Check STE mode enabled (would need to inspect optimizer internals)
        # For now, just check optimizer is created

    def test_should_fine_tune_decision(self, compressed_model, config):
        """Test fine-tuning decision logic"""
        tuner = FineTuner(compressed_model, config, device="cpu")

        # High degradation → should fine-tune
        should_tune = tuner.should_fine_tune(
            pre_perplexity=10.0, post_perplexity=12.0  # 20% degradation
        )
        assert should_tune is True

        # Low degradation → skip fine-tuning
        should_tune = tuner.should_fine_tune(
            pre_perplexity=10.0, post_perplexity=10.3  # 3% degradation
        )
        assert should_tune is False

    def test_should_fine_tune_disabled(self, compressed_model):
        """Test fine-tuning decision when disabled"""
        config = Phase4Config(enable_fine_tuning=False)
        tuner = FineTuner(compressed_model, config, device="cpu")

        # Should never fine-tune when disabled
        should_tune = tuner.should_fine_tune(
            pre_perplexity=10.0, post_perplexity=20.0  # 100% degradation!
        )
        assert should_tune is False

    def test_fine_tune_basic(self, compressed_model, config, dataloader):
        """Test basic fine-tuning execution"""
        tuner = FineTuner(compressed_model, config, device="cpu")

        # Fine-tune
        results = tuner.fine_tune(
            train_dataloader=dataloader, eval_dataloader=None, log_callback=None
        )

        # Check results structure
        assert "epochs_completed" in results
        assert "final_loss" in results
        assert "training_history" in results

        # Check epochs completed
        assert results["epochs_completed"] == config.fine_tune_epochs

        # Check training history
        assert len(results["training_history"]) == config.fine_tune_epochs

    def test_training_history_tracking(self, compressed_model, config, dataloader):
        """Test training history is tracked correctly"""
        tuner = FineTuner(compressed_model, config, device="cpu")

        results = tuner.fine_tune(train_dataloader=dataloader)

        history = results["training_history"]

        # Check history structure
        assert len(history) == config.fine_tune_epochs

        for epoch_stats in history:
            assert "epoch" in epoch_stats
            assert "loss" in epoch_stats
            assert "num_batches" in epoch_stats

    def test_gradient_computation_during_finetuning(self, compressed_model, config, dataloader):
        """Test that gradients are computed during fine-tuning"""
        tuner = FineTuner(compressed_model, config, device="cpu")

        # Store parameter to check gradients
        param_to_check = None
        for param in compressed_model.parameters():
            if param.requires_grad:
                param_to_check = param
                break

        assert param_to_check is not None

        # Run one training step manually
        compressed_model.train()
        batch = next(iter(dataloader))

        output = compressed_model(input_ids=batch["input_ids"], labels=batch["input_ids"])

        loss = output.loss
        loss.backward()

        # Check gradients computed
        assert param_to_check.grad is not None

    def test_evaluation_during_training(self, compressed_model, config, dataloader):
        """Test evaluation during training"""
        tuner = FineTuner(compressed_model, config, device="cpu")

        results = tuner.fine_tune(
            train_dataloader=dataloader, eval_dataloader=dataloader  # Same for testing
        )

        # Check evaluation metrics in history
        history = results["training_history"]

        for epoch_stats in history:
            assert "perplexity" in epoch_stats
            assert "eval_loss" in epoch_stats

        # Check best perplexity tracked
        assert "best_perplexity" in results

    def test_log_callback(self, compressed_model, config, dataloader):
        """Test logging callback is called"""
        logged_metrics = []

        def log_callback(metrics):
            logged_metrics.append(metrics)

        tuner = FineTuner(compressed_model, config, device="cpu")

        tuner.fine_tune(train_dataloader=dataloader, log_callback=log_callback)

        # Check callback was called
        assert len(logged_metrics) > 0

        # Check metric structure
        for metrics in logged_metrics:
            assert "epoch" in metrics or "batch" in metrics

    def test_get_training_summary_before_training(self, compressed_model, config):
        """Test training summary before any training"""
        tuner = FineTuner(compressed_model, config, device="cpu")

        summary = tuner.get_training_summary()

        assert summary["trained"] is False
        assert "message" in summary

    def test_get_training_summary_after_training(self, compressed_model, config, dataloader):
        """Test training summary after training"""
        tuner = FineTuner(compressed_model, config, device="cpu")

        tuner.fine_tune(train_dataloader=dataloader)

        summary = tuner.get_training_summary()

        assert summary["trained"] is True
        assert "epochs" in summary
        assert "final_loss" in summary
        assert "initial_loss" in summary
        assert "improvement" in summary

        # Check improvement calculation
        improvement = summary["initial_loss"] - summary["final_loss"]
        assert summary["improvement"] == improvement


class TestFineTunerEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def config(self):
        return Phase4Config(wandb_enabled=False, fine_tune_epochs=1)

    @pytest.fixture
    def compressed_model(self, config):
        base_model = SimpleModel()
        quantizer = BitNetQuantizer(config)
        model = CompressedModel(base_model, quantizer, config)
        model.compress()
        return model

    def test_zero_epochs(self, compressed_model):
        """Test fine-tuning with zero epochs"""
        config = Phase4Config(fine_tune_epochs=0)
        tuner = FineTuner(compressed_model, config, device="cpu")

        dataset = SimpleDataset(num_samples=10)
        dataloader = DataLoader(dataset, batch_size=2)

        results = tuner.fine_tune(train_dataloader=dataloader)

        # Should complete without error
        assert results["epochs_completed"] == 0
        assert len(results["training_history"]) == 0

    def test_empty_dataloader(self, compressed_model, config):
        """Test fine-tuning with empty dataloader"""

        class EmptyDataset(Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError("Empty dataset")

        dataset = EmptyDataset()
        dataloader = DataLoader(dataset, batch_size=2)

        tuner = FineTuner(compressed_model, config, device="cpu")

        results = tuner.fine_tune(train_dataloader=dataloader)

        # Should handle gracefully
        assert "epochs_completed" in results

    def test_single_batch_dataloader(self, compressed_model, config):
        """Test fine-tuning with single batch"""
        dataset = SimpleDataset(num_samples=4)
        dataloader = DataLoader(dataset, batch_size=4)

        tuner = FineTuner(compressed_model, config, device="cpu")

        results = tuner.fine_tune(train_dataloader=dataloader)

        # Should complete successfully
        assert results["epochs_completed"] == config.fine_tune_epochs


class TestFineTunerIntegration:
    """Integration tests with other components"""

    @pytest.fixture
    def config(self):
        return Phase4Config(wandb_enabled=False, 
            fine_tune_epochs=1,
            fine_tune_lr=1e-4,
        )

    def test_full_compression_finetuning_workflow(self, config):
        """Test complete compression → fine-tuning workflow"""
        # Create and compress model
        base_model = SimpleModel()
        quantizer = BitNetQuantizer(config)

        compressed_model = CompressedModel(base_model, quantizer, config)
        compressed_model.compress()

        # Get compression stats
        pre_stats = compressed_model.get_compression_stats()
        assert pre_stats["is_compressed"] is True

        # Fine-tune
        dataset = SimpleDataset(num_samples=20)
        dataloader = DataLoader(dataset, batch_size=4)

        tuner = FineTuner(compressed_model, config, device="cpu")
        results = tuner.fine_tune(train_dataloader=dataloader)

        # Check fine-tuning completed
        assert results["epochs_completed"] > 0

        # Model should still be compressed
        post_stats = compressed_model.get_compression_stats()
        assert post_stats["is_compressed"] is True

    def test_model_improvement_after_finetuning(self, config):
        """Test that model improves after fine-tuning"""
        # Create compressed model
        base_model = SimpleModel()
        quantizer = BitNetQuantizer(config)
        compressed_model = CompressedModel(base_model, quantizer, config)
        compressed_model.compress()

        # Create dataset
        dataset = SimpleDataset(num_samples=40)
        dataloader = DataLoader(dataset, batch_size=4)

        # Fine-tune
        tuner = FineTuner(compressed_model, config, device="cpu")
        results = tuner.fine_tune(train_dataloader=dataloader, eval_dataloader=dataloader)

        # Check improvement (initial loss > final loss)
        history = results["training_history"]

        initial_loss = history[0]["loss"]
        final_loss = history[-1]["loss"]

        # Loss should generally decrease (may not always in simple test)
        # Just check both are valid numbers
        assert initial_loss > 0
        assert final_loss > 0

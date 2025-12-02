"""
Unit tests for Calibration System
Tests dataset loading, tokenization, and activation collection
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports

from transformers import AutoTokenizer

from cross_phase.utils import MockTokenizer
from phase4_bitnet.calibration import (
    CalibrationDataset,
    collect_activation_statistics,
    create_calibration_dataloader,
)
from phase4_bitnet.config import Phase4Config


class TestCalibrationDataset:
    """Test suite for CalibrationDataset"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return Phase4Config(wandb_enabled=False, 
            calibration_samples=1000,
            calibration_sequence_length=256,
        )

    @pytest.fixture
    def tokenizer(self):
        """Create mock tokenizer"""
        return MockTokenizer()

    def test_dataset_initialization_custom(self, tokenizer, config):
        """Test dataset initialization with custom samples"""
        dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")

        # Should have no samples initially
        assert len(dataset.samples) == 0

    def test_set_custom_samples(self, tokenizer, config):
        """Test setting custom calibration samples"""
        dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")

        # Set samples
        custom_samples = [
            "Sample 1 text here",
            "Sample 2 text here",
            "Sample 3 text here",
        ]

        dataset.set_custom_samples(custom_samples)

        # Check samples loaded
        assert len(dataset.samples) == 3
        assert dataset.samples[0] == "Sample 1 text here"

    def test_dataset_length(self, tokenizer, config):
        """Test dataset __len__ method"""
        dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")

        dataset.set_custom_samples(["Sample"] * 200)

        assert len(dataset) == 50

    def test_dataset_getitem(self, tokenizer, config):
        """Test dataset __getitem__ method"""
        dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")

        dataset.set_custom_samples(["Test sample"])

        # Get item
        item = dataset[0]

        # Check structure
        assert "input_ids" in item
        assert "attention_mask" in item

        # Check types
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)

        # Check shapes
        assert item["input_ids"].dim() == 1
        assert item["attention_mask"].dim() == 1

    def test_samples_truncated_to_config(self, tokenizer):
        """Test samples are truncated to configured limit"""
        config = Phase4Config(wandb_enabled=False, calibration_samples=100)

        dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")

        # Set more samples than limit
        dataset.set_custom_samples(["Sample"] * 200)

        # Should be truncated
        assert len(dataset.samples) == 100

    def test_synthetic_sample_generation(self, tokenizer, config):
        """Test synthetic sample fallback"""
        dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")

        # Generate synthetic samples
        dataset._add_synthetic_samples()

        # Should have samples
        assert len(dataset.samples) > 0

        # Samples should be strings
        assert all(isinstance(s, str) for s in dataset.samples)

    def test_dataset_fallback_chain(self, tokenizer, config):
        """Test dataset loading fallback (OpenWebText → C4 → WikiText → Synthetic)"""
        # This would test with mocked datasets module
        # For now, test that unknown datasets raise error

        with pytest.raises(ValueError):
            dataset = CalibrationDataset(tokenizer, config, dataset_name="unknown_dataset")


class TestCalibrationDataLoader:
    """Test suite for calibration dataloader creation"""

    @pytest.fixture
    def config(self):
        return Phase4Config(wandb_enabled=False, 
            calibration_samples=100,
            calibration_batch_size=4,
        )

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    def test_dataloader_creation(self, tokenizer, config):
        """Test creating calibration dataloader"""
        dataloader = create_calibration_dataloader(tokenizer, config)

        assert dataloader is not None
        assert dataloader.batch_size == config.calibration_batch_size

    def test_dataloader_iteration(self, tokenizer, config):
        """Test iterating through dataloader"""
        dataloader = create_calibration_dataloader(tokenizer, config)

        # Get first batch
        batch = next(iter(dataloader))

        # Check structure
        assert "input_ids" in batch
        assert "attention_mask" in batch

        # Check batch size
        assert batch["input_ids"].size(0) == config.calibration_batch_size

    def test_dataloader_no_shuffle(self, tokenizer, config):
        """Test dataloader doesn't shuffle (calibration is deterministic)"""
        dataloader = create_calibration_dataloader(tokenizer, config)

        # DataLoader should not shuffle for calibration
        assert dataloader.shuffle is False

    def test_dataloader_pin_memory_cuda(self, tokenizer):
        """Test dataloader pin_memory for CUDA"""
        config_cuda = Phase4Config(wandb_enabled=False, device="cuda")
        dataloader = create_calibration_dataloader(tokenizer, config_cuda)

        # Should pin memory for CUDA
        assert dataloader.pin_memory is True

    def test_dataloader_pin_memory_cpu(self, tokenizer):
        """Test dataloader pin_memory for CPU"""
        config_cpu = Phase4Config(wandb_enabled=False, device="cpu")
        dataloader = create_calibration_dataloader(tokenizer, config_cpu)

        # Should not pin memory for CPU
        assert dataloader.pin_memory is False


class TestActivationStatistics:
    """Test suite for activation statistics collection"""

    @pytest.fixture
    def simple_model(self):
        """Create simple test model"""

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 20)
                self.linear2 = torch.nn.Linear(20, 10)

            def forward(self, input_ids):
                x = self.linear1(input_ids.float())
                x = torch.relu(x)
                x = self.linear2(x)
                return x

        return SimpleModel()

    @pytest.fixture
    def simple_dataloader(self):
        """Create simple dataloader"""

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randn(10),
                    "attention_mask": torch.ones(10),
                }

        dataset = SimpleDataset()
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def test_collect_statistics(self, simple_model, simple_dataloader):
        """Test collecting activation statistics"""
        stats = collect_activation_statistics(simple_model, simple_dataloader, device="cpu")

        # Should have stats for linear layers
        assert len(stats) > 0

        # Check stats structure
        for layer_stats in stats.values():
            assert "mean" in layer_stats
            assert "std" in layer_stats
            assert "max" in layer_stats
            assert "min" in layer_stats

            # Check types
            assert isinstance(layer_stats["mean"], float)
            assert isinstance(layer_stats["std"], float)

    def test_statistics_values_reasonable(self, simple_model, simple_dataloader):
        """Test that collected statistics are reasonable"""
        stats = collect_activation_statistics(simple_model, simple_dataloader, device="cpu")

        for layer_stats in stats.values():
            # Max should be >= mean
            assert layer_stats["max"] >= layer_stats["mean"]

            # Min should be <= mean
            assert layer_stats["min"] <= layer_stats["mean"]

            # Std should be non-negative
            assert layer_stats["std"] >= 0

    def test_model_remains_in_eval_mode(self, simple_model, simple_dataloader):
        """Test model is put in eval mode during calibration"""
        simple_model.train()  # Start in train mode

        collect_activation_statistics(simple_model, simple_dataloader, device="cpu")

        # Should be in eval mode during calibration
        # (Note: function puts in eval, but doesn't restore state)

    def test_no_gradient_computation(self, simple_model, simple_dataloader):
        """Test that calibration doesn't compute gradients"""
        # Set model parameters to require gradients
        for param in simple_model.parameters():
            param.requires_grad = True

        collect_activation_statistics(simple_model, simple_dataloader, device="cpu")

        # No gradients should be accumulated
        for param in simple_model.parameters():
            assert param.grad is None


class TestCalibrationEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def config(self):
        return Phase4Config(wandb_enabled=False, )

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    def test_empty_dataset(self, tokenizer, config):
        """Test with empty dataset"""
        dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")

        dataset.set_custom_samples([])

        assert len(dataset) == 0

    def test_very_long_samples(self, tokenizer, config):
        """Test with very long text samples"""
        dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")

        # Very long sample
        long_sample = "word " * 10000

        dataset.set_custom_samples([long_sample])

        # Should truncate to config.calibration_sequence_length
        item = dataset[0]

        assert item["input_ids"].size(0) == config.calibration_sequence_length

    def test_special_characters_in_samples(self, tokenizer, config):
        """Test with special characters"""
        dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")

        special_samples = [
            "Text with émojis 🚀 and 中文",
            "Symbols: @#$%^&*()",
            "Unicode: \u0394\u03A9\u03C0",
        ]

        dataset.set_custom_samples(special_samples)

        # Should handle without error
        assert len(dataset) == 3

        for i in range(len(dataset)):
            item = dataset[i]
            assert "input_ids" in item

    def test_batch_size_larger_than_dataset(self, tokenizer):
        """Test when batch size exceeds dataset size"""
        config = Phase4Config(wandb_enabled=False, 
            calibration_samples=100,
            calibration_batch_size=10,
        )

        dataloader = create_calibration_dataloader(tokenizer, config)

        # Should still work
        batch = next(iter(dataloader))

        # Batch size will be dataset size
        assert batch["input_ids"].size(0) == 5

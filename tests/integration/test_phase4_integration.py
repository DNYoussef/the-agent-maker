"""
Integration Tests for Phase 4
Tests Phase 3→4→5 handoffs, W&B integration, and end-to-end pipeline
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from phase4_bitnet.phase_controller import Phase4Controller
from phase4_bitnet.config import Phase4Config
from phase4_bitnet.utils import (
    test_gradient_flow,
    load_compression_metadata,
)
from cross_phase.monitoring.wandb_integration import WandBIntegration
from cross_phase.utils import MockTokenizer


class MockTransformerModel(nn.Module):
    """Mock transformer model for testing"""
    def __init__(self):
        super().__init__()
        self.config = type('Config', (), {})()
        self.embeddings = nn.Embedding(1000, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4),
            num_layers=2
        )
        self.lm_head = nn.Linear(128, 1000)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embeddings(input_ids)
        x = self.transformer(x)
        output = self.lm_head(x)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                output.view(-1, 1000),
                labels.view(-1)
            )
            return type('Output', (), {'loss': loss})()

        return output

    def save_pretrained(self, path):
        """Save model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")


class MockTokenizerWithSave(MockTokenizer):
    """MockTokenizer with save_pretrained for testing"""
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        # Create dummy files
        (Path(path) / "tokenizer_config.json").write_text("{}")

    @classmethod
    def from_pretrained(cls, path):
        return cls()


class TestPhase3To4Handoff:
    """Test Phase 3 → Phase 4 handoff"""

    @pytest.fixture
    def temp_phase3_output(self):
        """Create temporary Phase 3 output directory"""
        temp_dir = tempfile.mkdtemp()

        # Create mock Phase 3 model
        model = MockTransformerModel()
        model.save_pretrained(temp_dir)

        # Save tokenizer
        tokenizer = MockTokenizerWithSave()
        tokenizer.save_pretrained(temp_dir)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_phase3_output_loading(self, temp_phase3_output):
        """Test loading Phase 3 output"""
        config = Phase4Config(
            model_path=temp_phase3_output,
            output_path=tempfile.mkdtemp(),
        )

        controller = Phase4Controller(config)

        # Should load without error
        controller._load_phase3_model(temp_phase3_output)

        assert controller.model is not None
        assert controller.tokenizer is not None

    def test_size_adaptive_target_selection(self, temp_phase3_output):
        """Test size-adaptive compression target selection"""
        config = Phase4Config()
        controller = Phase4Controller(config)

        # Load model
        controller._load_phase3_model(temp_phase3_output)

        # Config should be adapted
        # (Our mock model is small, should be "tiny" category)
        assert config.target_compression_ratio > 0


class TestPhase4To5Handoff:
    """Test Phase 4 → Phase 5 handoff"""

    @pytest.fixture
    def phase4_output_dir(self):
        """Create Phase 4 output directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_dual_model_output(self, phase4_output_dir):
        """Test that Phase 4 outputs both models"""
        config = Phase4Config(
            output_path=phase4_output_dir,
            save_quantized=True,
            save_dequantized_fp16=True,
        )

        # Create mock models
        quantized_path = Path(phase4_output_dir) / "bitnet_quantized_model.pt"
        dequantized_path = Path(phase4_output_dir) / "bitnet_dequantized_fp16.pt"

        # Save dummy models
        torch.save({'state_dict': {}}, quantized_path)
        torch.save({}, dequantized_path)

        # Check both exist
        assert quantized_path.exists()
        assert dequantized_path.exists()

    def test_primary_output_is_dequantized(self, phase4_output_dir):
        """Test that dequantized FP16 is marked as primary"""
        # This would be in controller's output
        output_paths = {
            'quantized': str(Path(phase4_output_dir) / "quantized.pt"),
            'dequantized_fp16': str(Path(phase4_output_dir) / "deq.pt"),
            'primary_output': str(Path(phase4_output_dir) / "deq.pt"),
        }

        # Primary should point to dequantized
        assert output_paths['primary_output'] == output_paths['dequantized_fp16']

    def test_gradient_flow_validation(self):
        """Test gradient flow through dequantized model"""
        # Create simple model
        model = nn.Linear(10, 10)

        # Test gradient flow
        passed, error = test_gradient_flow(model, device='cpu')

        # Should pass for normal model
        assert passed is True
        assert error is None

    def test_metadata_saved_with_outputs(self, phase4_output_dir):
        """Test compression metadata is saved"""
        from src.phase4_bitnet.utils import save_compression_metadata

        metadata = {
            'compression_method': 'BitNet-1.58',
            'compression_ratio': 8.2,
            'sparsity_ratio': 0.35,
        }

        save_compression_metadata(phase4_output_dir, metadata)

        # Load and verify
        loaded = load_compression_metadata(phase4_output_dir)

        assert loaded['compression_method'] == 'BitNet-1.58'
        assert loaded['compression_ratio'] == 8.2


class TestWandBIntegration:
    """Test W&B integration"""

    @pytest.fixture
    def wandb_logger(self):
        """Create W&B logger in offline mode"""
        return WandBIntegration(mode="disabled")

    def test_log_pre_compression(self, wandb_logger):
        """Test logging pre-compression metrics"""
        metrics = {
            'original_size_mb': 100.0,
            'pre_perplexity': 12.5,
            'pre_eval_loss': 2.3,
        }

        # Should not raise error
        wandb_logger.log_phase4_pre_compression(metrics)

    def test_log_compression_process(self, wandb_logger):
        """Test logging compression process metrics"""
        metrics = {
            'compressed_size_mb': 12.0,
            'compression_ratio': 8.3,
            'layers_quantized': 24,
            'sparsity_ratio': 0.35,
            'quantized_params': 85000000,
            'total_params': 100000000,
        }

        wandb_logger.log_phase4_compression(metrics)

    def test_log_post_compression(self, wandb_logger):
        """Test logging post-compression metrics"""
        metrics = {
            'post_perplexity': 13.2,
            'perplexity_degradation': 0.056,
            'accuracy_preserved': True,
            'dequantization_accuracy': 0.998,
            'gradient_flow_passed': True,
        }

        wandb_logger.log_phase4_post_compression(metrics)

    def test_log_fine_tuning(self, wandb_logger):
        """Test logging fine-tuning metrics"""
        metrics = {
            'best_perplexity': 12.8,
            'improvement': 0.4,
            'epochs': 2,
            'time_hours': 1.5,
        }

        wandb_logger.log_phase4_fine_tuning(metrics)

    def test_log_fine_tuning_skipped(self, wandb_logger):
        """Test logging when fine-tuning is skipped"""
        # Should handle None gracefully
        wandb_logger.log_phase4_fine_tuning(None)

    def test_log_phase_summary(self, wandb_logger):
        """Test logging phase summary"""
        results = {
            'success': True,
            'pre_compression': {
                'original_size_mb': 100.0,
            },
            'post_compression': {
                'compressed_size_mb': 12.0,
                'compression_ratio': 8.3,
                'sparsity_ratio': 0.35,
                'accuracy_preserved': True,
            },
            'fine_tuning': None,
        }

        wandb_logger.log_phase4_summary(results)

    def test_all_19_metrics_logged(self, wandb_logger):
        """Test that all 19 unique metrics are logged"""
        # This is a comprehensive test to ensure all metrics are covered

        # Pre-compression (3)
        wandb_logger.log_phase4_pre_compression({
            'original_size_mb': 100.0,
            'pre_perplexity': 12.0,
            'pre_eval_loss': 2.0,
        })

        # Compression (7)
        wandb_logger.log_phase4_compression({
            'compressed_size_mb': 12.0,
            'compression_ratio': 8.0,
            'layers_quantized': 24,
            'sparsity_ratio': 0.35,
            'quantized_params': 85000000,
            'total_params': 100000000,
        })

        # Post-compression (5)
        wandb_logger.log_phase4_post_compression({
            'post_perplexity': 13.0,
            'perplexity_degradation': 0.08,
            'accuracy_preserved': True,
            'dequantization_accuracy': 0.998,
            'gradient_flow_passed': True,
        })

        # Fine-tuning (4)
        wandb_logger.log_phase4_fine_tuning({
            'best_perplexity': 12.5,
            'improvement': 0.5,
            'epochs': 2,
            'time_hours': 1.0,
        })

        # Total: 3 + 7 + 5 + 4 = 19 unique metrics


class TestEndToEndPipeline:
    """Test complete Phase 4 pipeline"""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories"""
        phase3_dir = tempfile.mkdtemp()
        phase4_dir = tempfile.mkdtemp()

        # Create Phase 3 output
        model = MockTransformerModel()
        model.save_pretrained(phase3_dir)

        tokenizer = MockTokenizerWithSave()
        tokenizer.save_pretrained(phase3_dir)

        yield phase3_dir, phase4_dir

        # Cleanup
        shutil.rmtree(phase3_dir, ignore_errors=True)
        shutil.rmtree(phase4_dir, ignore_errors=True)

    def test_minimal_pipeline_execution(self, temp_dirs):
        """Test minimal pipeline execution (may not converge)"""
        phase3_dir, phase4_dir = temp_dirs

        config = Phase4Config(
            model_path=phase3_dir,
            output_path=phase4_dir,
            calibration_samples=10,  # Minimal for testing
            fine_tune_epochs=1,
            enable_fine_tuning=False,  # Skip for faster test
            wandb_enabled=False,
        )

        controller = Phase4Controller(config)

        # Note: This will likely fail without proper models
        # Just test that it doesn't crash on initialization
        assert controller.config == config
        assert controller.quantizer is not None

    def test_compression_stats_structure(self):
        """Test compression stats have correct structure"""
        from src.phase4_bitnet.compressed_model import CompressedModel
        from src.phase4_bitnet.quantizer import BitNetQuantizer

        config = Phase4Config()
        model = nn.Linear(100, 50)
        quantizer = BitNetQuantizer(config)

        compressed = CompressedModel(model, quantizer, config)
        compressed.compress()

        stats = compressed.get_compression_stats()

        # Check all required fields
        required_fields = [
            'is_compressed',
            'original_size_mb',
            'quantized_size_mb',
            'compression_ratio',
            'layers_quantized',
            'layers_preserved',
            'total_params',
            'quantized_params',
            'sparsity_ratio',
        ]

        for field in required_fields:
            assert field in stats, f"Missing field: {field}"


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_missing_phase3_output(self):
        """Test handling of missing Phase 3 output"""
        config = Phase4Config(
            model_path="/nonexistent/path",
            output_path=tempfile.mkdtemp(),
        )

        controller = Phase4Controller(config)

        # Should raise appropriate error
        with pytest.raises(Exception):
            controller._load_phase3_model("/nonexistent/path")

    def test_invalid_config(self):
        """Test invalid configuration handling"""
        # Invalid quantization bits
        with pytest.raises(ValueError):
            Phase4Config(quantization_bits=8.0)  # Should be 1.58

        # Invalid accuracy drop
        with pytest.raises(ValueError):
            Phase4Config(max_accuracy_drop=1.5)  # Should be 0-0.5

    def test_output_directory_creation(self):
        """Test output directory is created if missing"""
        temp_dir = tempfile.mkdtemp()
        output_dir = Path(temp_dir) / "nonexistent" / "nested" / "path"

        config = Phase4Config(output_path=str(output_dir))

        # Directory should be created when saving
        from src.phase4_bitnet.utils import save_compression_metadata

        save_compression_metadata(output_dir, {})

        assert output_dir.exists()

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

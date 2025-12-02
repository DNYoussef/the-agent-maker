"""
E2E tests for Phase 1: Cognate (Model Creation).

Tests the complete model creation pipeline including:
- Model initialization with TRM architecture
- Forward pass computation
- Training step execution
- Checkpoint save/load operations
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file



class TestPhase1CognateE2E:
    """E2E tests for Phase 1 Cognate pipeline."""

    def test_model_creation(self, mock_model):
        """Test model can be created with expected architecture."""
        assert hasattr(mock_model, "config")
        assert mock_model.config.hidden_size == 256
        assert mock_model.config.num_layers == 2
        assert mock_model.config.vocab_size == 1000

    def test_model_has_required_components(self, mock_model):
        """Test model has all required components."""
        assert hasattr(mock_model, "embeddings")
        assert hasattr(mock_model, "layers")
        assert hasattr(mock_model, "lm_head")
        assert isinstance(mock_model.embeddings, nn.Embedding)
        assert isinstance(mock_model.layers, nn.ModuleList)
        assert isinstance(mock_model.lm_head, nn.Linear)

    def test_forward_pass(self, mock_model, mock_tokenizer):
        """Test model forward pass works."""
        inputs = mock_tokenizer("Test input text")
        outputs = mock_model(inputs["input_ids"])

        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1  # batch size
        assert outputs.logits.shape[2] == mock_model.config.vocab_size

    def test_forward_pass_with_attention_mask(self, mock_model, mock_tokenizer):
        """Test forward pass with attention mask."""
        inputs = mock_tokenizer("Test input text with multiple tokens")
        outputs = mock_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

        assert outputs.logits is not None
        assert outputs.hidden_states is not None

    def test_training_step(self, mock_model, mock_dataloader):
        """Test single training step completes."""
        batch = next(iter(mock_dataloader))
        outputs = mock_model(batch["input_ids"], labels=batch["labels"])

        assert outputs.loss is not None
        assert outputs.loss.requires_grad

        # Test backward pass
        outputs.loss.backward()

        # Check gradients exist
        for param in mock_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_multiple_training_steps(self, mock_model, mock_dataloader):
        """Test multiple training steps complete."""
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-3)
        losses = []

        for i, batch in enumerate(mock_dataloader):
            if i >= 3:  # Test 3 steps
                break

            optimizer.zero_grad()
            outputs = mock_model(batch["input_ids"], labels=batch["labels"])
            outputs.loss.backward()
            optimizer.step()

            losses.append(outputs.loss.item())

        assert len(losses) == 3
        assert all(loss > 0 for loss in losses)

    def test_checkpoint_save(self, mock_model, temp_checkpoint_dir):
        """Test model checkpoint can be saved."""
        safetensors_path = temp_checkpoint_dir / "phase1_test.safetensors"
        json_path = temp_checkpoint_dir / "phase1_test.json"

        # Save state dict using SafeTensors
        state_dict = mock_model.state_dict()
        safe_save_file(state_dict, str(safetensors_path))

        # Save metadata separately as JSON
        metadata = {
            "config": {
                "vocab_size": mock_model.config.vocab_size,
                "hidden_size": mock_model.config.hidden_size,
                "num_layers": mock_model.config.num_layers,
            }
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        assert safetensors_path.exists()
        assert safetensors_path.stat().st_size > 0
        assert json_path.exists()

    def test_checkpoint_load(self, mock_model, temp_checkpoint_dir):
        """Test model can be loaded from checkpoint."""
        safetensors_path = temp_checkpoint_dir / "phase1_load_test.safetensors"
        json_path = temp_checkpoint_dir / "phase1_load_test.json"

        # Save checkpoint
        state_dict = mock_model.state_dict()
        safe_save_file(state_dict, str(safetensors_path))

        metadata = {
            "config": {
                "vocab_size": mock_model.config.vocab_size,
                "hidden_size": mock_model.config.hidden_size,
                "num_layers": mock_model.config.num_layers,
            }
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Load checkpoint
        loaded_state_dict = safe_load_file(str(safetensors_path), device="cpu")
        mock_model.load_state_dict(loaded_state_dict)

        # Load metadata
        with open(json_path, "r") as f:
            loaded_config = json.load(f)

        assert loaded_config["config"]["vocab_size"] == 1000
        assert loaded_config["config"]["hidden_size"] == 256

    def test_checkpoint_save_load_preserves_weights(self, mock_model, temp_checkpoint_dir):
        """Test checkpoint save/load preserves model weights."""
        safetensors_path = temp_checkpoint_dir / "phase1_weights_test.safetensors"

        # Save original weights
        original_weights = {k: v.clone() for k, v in mock_model.state_dict().items()}
        safe_save_file(mock_model.state_dict(), str(safetensors_path))

        # Modify model weights
        with torch.no_grad():
            for param in mock_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Load original weights
        loaded_state_dict = safe_load_file(str(safetensors_path), device="cpu")
        mock_model.load_state_dict(loaded_state_dict)

        # Verify weights match
        for key, original_value in original_weights.items():
            loaded_value = mock_model.state_dict()[key]
            assert torch.allclose(original_value, loaded_value)

    def test_batch_processing(self, mock_model, mock_tokenizer):
        """Test model can process batches."""
        texts = ["Text one", "Text two", "Text three"]
        inputs = mock_tokenizer(texts)

        outputs = mock_model(inputs["input_ids"])

        assert outputs.logits.shape[0] == len(texts)
        assert outputs.logits.shape[2] == mock_model.config.vocab_size

    def test_get_input_embeddings(self, mock_model):
        """Test get_input_embeddings method."""
        embeddings = mock_model.get_input_embeddings()

        assert embeddings is not None
        assert isinstance(embeddings, nn.Embedding)
        assert embeddings.num_embeddings == mock_model.config.vocab_size
        assert embeddings.embedding_dim == mock_model.config.hidden_size

    def test_model_device_placement(self, mock_model):
        """Test model can be moved to different devices."""
        # Test CPU placement (default)
        assert next(mock_model.parameters()).device.type == "cpu"

        # Test moving to CPU explicitly
        mock_model.cpu()
        assert next(mock_model.parameters()).device.type == "cpu"

    def test_gradient_flow(self, mock_model, mock_dataloader):
        """Test gradients flow through all layers."""
        batch = next(iter(mock_dataloader))
        outputs = mock_model(batch["input_ids"], labels=batch["labels"])
        outputs.loss.backward()

        # Check all parameters have gradients
        for name, param in mock_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_inference_mode(self, mock_model, mock_tokenizer):
        """Test model works in inference mode."""
        mock_model.eval()

        with torch.no_grad():
            inputs = mock_tokenizer("Inference test")
            outputs = mock_model(inputs["input_ids"])

        assert outputs.logits is not None
        assert not outputs.logits.requires_grad

        mock_model.train()

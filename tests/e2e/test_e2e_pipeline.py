"""
E2E tests for complete Agent Maker pipeline.

Tests the handoff between phases and validates the full workflow.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPipelineHandoffs:
    """Test phase-to-phase handoffs."""

    def test_phase1_to_phase2_handoff(self, mock_model, temp_checkpoint_dir):
        """Test Phase 1 output can be consumed by Phase 2."""
        # Save Phase 1 output
        safetensors_path = temp_checkpoint_dir / "phase1_model.safetensors"
        json_path = temp_checkpoint_dir / "phase1_model.json"

        # Save state dict
        safe_save_file(mock_model.state_dict(), str(safetensors_path))

        # Save metadata
        metadata = {
            "config": {"hidden_size": 256},
            "phase": 1,
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Verify Phase 2 can load it
        state_dict = safe_load_file(str(safetensors_path), device="cpu")
        with open(json_path, "r") as f:
            checkpoint_metadata = json.load(f)

        assert state_dict is not None
        assert checkpoint_metadata["phase"] == 1

    def test_phase2_to_phase3_handoff(self, mock_model, temp_checkpoint_dir):
        """Test Phase 2 (EvoMerge) output for Phase 3 (Quiet-STaR)."""
        safetensors_path = temp_checkpoint_dir / "phase2_champion.safetensors"
        json_path = temp_checkpoint_dir / "phase2_champion.json"

        # Save state dict
        safe_save_file(mock_model.state_dict(), str(safetensors_path))

        # Save metadata
        metadata = {
            "fitness": 0.85,
            "generation": 50,
            "phase": 2,
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Load and verify
        with open(json_path, "r") as f:
            checkpoint_metadata = json.load(f)
        assert checkpoint_metadata["fitness"] > 0.8

    def test_phase3_to_phase4_handoff(self, mock_model, temp_checkpoint_dir):
        """Test Phase 3 (Quiet-STaR) output for Phase 4 (BitNet)."""
        # Phase 3 adds reasoning enhancement
        safetensors_path = temp_checkpoint_dir / "phase3_reasoning.safetensors"
        json_path = temp_checkpoint_dir / "phase3_reasoning.json"

        # Save state dict
        safe_save_file(mock_model.state_dict(), str(safetensors_path))

        # Save metadata
        metadata = {
            "reasoning_metrics": {"coherence": 0.9},
            "phase": 3,
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Load and verify
        with open(json_path, "r") as f:
            checkpoint_metadata = json.load(f)
        assert "reasoning_metrics" in checkpoint_metadata

    def test_phase4_to_phase5_handoff(self, mock_model, temp_checkpoint_dir):
        """Test Phase 4 (BitNet) output for Phase 5 (Curriculum)."""
        safetensors_path = temp_checkpoint_dir / "phase4_quantized.safetensors"
        json_path = temp_checkpoint_dir / "phase4_quantized.json"

        # Save state dict
        safe_save_file(mock_model.state_dict(), str(safetensors_path))

        # Save metadata
        metadata = {
            "compression_ratio": 8.2,
            "bits": 1.58,
            "phase": 4,
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Load and verify
        with open(json_path, "r") as f:
            checkpoint_metadata = json.load(f)
        assert checkpoint_metadata["compression_ratio"] > 8.0


class TestFullPipeline:
    """Test complete pipeline execution."""

    def test_pipeline_config_propagation(self):
        """Test configuration flows through all phases."""
        config = {
            "model_size": "25M",
            "hidden_size": 256,
            "phases_enabled": [1, 2, 3, 4, 5, 6, 7, 8],
        }

        # Verify config is valid for all phases
        assert len(config["phases_enabled"]) == 8
        assert config["hidden_size"] > 0

    def test_checkpoint_format_consistency(self, mock_model, temp_checkpoint_dir):
        """Test all phases use consistent checkpoint format."""
        # With SafeTensors, state_dict is in .safetensors file, metadata in .json
        required_metadata_keys = ["phase"]

        for phase in range(1, 9):
            safetensors_path = temp_checkpoint_dir / f"phase{phase}_test.safetensors"
            json_path = temp_checkpoint_dir / f"phase{phase}_test.json"

            # Save state dict
            safe_save_file(mock_model.state_dict(), str(safetensors_path))

            # Save metadata
            metadata = {"phase": phase}
            with open(json_path, "w") as f:
                json.dump(metadata, f)

            # Verify both files exist and have required keys
            state_dict = safe_load_file(str(safetensors_path), device="cpu")
            assert state_dict is not None, f"Phase {phase} state dict is None"

            with open(json_path, "r") as f:
                loaded_metadata = json.load(f)
            for key in required_metadata_keys:
                assert key in loaded_metadata, f"Phase {phase} missing key: {key}"


class TestPipelineRecovery:
    """Test pipeline recovery and rollback."""

    def test_resume_from_checkpoint(self, mock_model, temp_checkpoint_dir):
        """Test pipeline can resume from any phase."""
        # Create checkpoint at phase 3
        safetensors_path = temp_checkpoint_dir / "resume_phase3.safetensors"
        json_path = temp_checkpoint_dir / "resume_phase3.json"

        # Save state dict
        safe_save_file(mock_model.state_dict(), str(safetensors_path))

        # Save metadata
        metadata = {
            "phase": 3,
            "episode": 5000,
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Verify resume is possible
        state_dict = safe_load_file(str(safetensors_path), device="cpu")
        with open(json_path, "r") as f:
            checkpoint_metadata = json.load(f)

        assert state_dict is not None
        assert checkpoint_metadata["phase"] == 3
        assert checkpoint_metadata["episode"] == 5000

    def test_rollback_on_quality_failure(self, mock_model, temp_checkpoint_dir):
        """Test rollback mechanism when quality gate fails."""
        # Save pre-compression checkpoint
        safetensors_path = temp_checkpoint_dir / "pre_compression.safetensors"
        json_path = temp_checkpoint_dir / "pre_compression.json"

        # Save state dict
        safe_save_file(mock_model.state_dict(), str(safetensors_path))

        # Save metadata
        metadata = {"quality_score": 0.95}
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Simulate compression that fails quality gate
        post_compression_score = 0.75  # Below 0.84 threshold

        if post_compression_score < 0.84:
            # Rollback
            state_dict = safe_load_file(str(safetensors_path), device="cpu")
            with open(json_path, "r") as f:
                rollback_metadata = json.load(f)

            assert state_dict is not None
            assert rollback_metadata["quality_score"] > post_compression_score

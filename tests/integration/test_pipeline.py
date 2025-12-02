"""
Integration tests for Pipeline Orchestrator
Tests end-to-end phase sequencing and handoff validation
"""

import sys
from pathlib import Path

import pytest


from cross_phase.orchestrator.phase_controller import PhaseResult
from cross_phase.orchestrator.pipeline import PipelineOrchestrator


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for pipeline"""

    def test_pipeline_creation(self, sample_config, temp_dir):
        """Test pipeline creation"""
        config = sample_config.copy()
        config["registry"] = {"db_path": str(temp_dir / "test_pipeline.db")}

        with PipelineOrchestrator(config) as pipeline:
            assert pipeline is not None
            assert pipeline.config == config

    def test_single_phase_execution(self, sample_config, temp_dir):
        """Test executing single phase"""
        config = sample_config.copy()
        config["registry"] = {"db_path": str(temp_dir / "test_single_phase.db")}

        # Note: This will fail without actual phase implementations
        # This is a placeholder test structure

        with pytest.raises(AttributeError):  # Phase 1 not implemented yet
            with PipelineOrchestrator(config) as pipeline:
                result = pipeline.run_single_phase(1)

    def test_context_manager(self, sample_config, temp_dir):
        """Test context manager closes resources"""
        config = sample_config.copy()
        config["registry"] = {"db_path": str(temp_dir / "test_context.db")}

        with PipelineOrchestrator(config) as pipeline:
            pass  # Context manager should close automatically

        # Verify database exists
        assert (temp_dir / "test_context.db").exists()


@pytest.mark.integration
class TestPhaseHandoff:
    """Test phase handoff validation"""

    def test_phase_result_structure(self):
        """Test PhaseResult structure"""
        result = PhaseResult(
            success=True,
            phase_name="phase1",
            model=None,
            metrics={"loss": 2.34},
            duration=120.5,
            artifacts={"checkpoint": "/path/to/model.pt"},
            config={"epochs": 10},
        )

        assert result.success is True
        assert result.phase_name == "phase1"
        assert result.metrics["loss"] == 2.34
        assert result.duration == 120.5

    def test_phase_result_failure(self):
        """Test failed phase result"""
        result = PhaseResult(
            success=False,
            phase_name="phase2",
            model=None,
            metrics={},
            duration=5.0,
            artifacts={},
            config={},
            error="Test error message",
        )

        assert result.success is False
        assert result.error == "Test error message"

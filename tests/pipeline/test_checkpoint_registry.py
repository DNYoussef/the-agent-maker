"""E1 gate - CRUCIBLE: no phantom checkpoint registration.

The synthesis found every phase registers ./checkpoints/phaseN/model.safetensors that no
phase ever writes, so rollback_to_phase points at a non-existent file. The pipeline is
in-memory; E1 makes registration require a REAL on-disk model_path and makes rollback
fail loudly (clear error) instead of returning a phantom path.
"""

import pytest

from cross_phase.orchestrator.base_controller import PhaseResult
from cross_phase.orchestrator.pipeline import PipelineOrchestrator


def _orch(tmp_path):
    cfg = {"registry": {"db_path": str(tmp_path / "reg.db")}, "wandb": {"enabled": False}}
    return PipelineOrchestrator(cfg, session_id="t")


def _result(model_path=None):
    artifacts = {"model_path": model_path} if model_path else {}
    return PhaseResult(True, "phase2", object(), {}, 1.0, artifacts, {})


def test_no_phantom_registration_when_no_disk_model(tmp_path):
    orch = _orch(tmp_path)
    orch._register_phase_model("phase2", _result(model_path=None), 1.0)
    # Nothing real on disk -> nothing registered -> get_model raises (no phantom row).
    with pytest.raises(FileNotFoundError):
        orch.registry.get_model(session_id="t", phase_name="phase2")


def test_registers_real_checkpoint_and_rollback_finds_it(tmp_path):
    orch = _orch(tmp_path)
    ckpt = tmp_path / "phase2.safetensors"
    ckpt.write_bytes(b"\x00\x01\x02")  # a real file on disk
    orch._register_phase_model("phase2", _result(model_path=str(ckpt)), 1.0)
    info = orch.rollback_to_phase(2)
    assert info["model_path"] == str(ckpt)


def test_rollback_without_checkpoint_raises_clearly(tmp_path):
    orch = _orch(tmp_path)
    with pytest.raises(FileNotFoundError):
        orch.rollback_to_phase(2)  # nothing registered -> clear error, never a phantom path


def test_registration_skips_stale_path(tmp_path):
    # A model_path that does not exist on disk must NOT be registered (it would be a
    # phantom rollback target).
    orch = _orch(tmp_path)
    orch._register_phase_model(
        "phase3", _result(model_path=str(tmp_path / "gone.safetensors")), 1.0
    )
    with pytest.raises(FileNotFoundError):
        orch.registry.get_model(session_id="t", phase_name="phase3")

"""Phase 4 security regressions."""

import importlib.util
from pathlib import Path

import pytest

from cross_phase.storage.model_registry import ModelRegistry


def _load_evaluator_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "evaluate_phase1.py"
    spec = importlib.util.spec_from_file_location("evaluate_phase1", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase1_evaluator_rejects_legacy_pickle_checkpoint(tmp_path):
    evaluator = _load_evaluator_module()
    checkpoint_dir = tmp_path / "reasoning"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best_model.pt").write_bytes(b"malicious pickle payload")

    with pytest.raises(RuntimeError, match="Refusing to load legacy pickle checkpoint"):
        evaluator.secure_checkpoint_base("reasoning", tmp_path)


def test_phase1_evaluator_prefers_safetensors_without_touching_legacy_pt(tmp_path):
    evaluator = _load_evaluator_module()
    checkpoint_dir = tmp_path / "reasoning"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best_model.safetensors").write_bytes(b"safe tensor bytes")
    (checkpoint_dir / "best_model.pt").write_bytes(b"legacy pickle payload")

    assert evaluator.secure_checkpoint_base("reasoning", tmp_path) == checkpoint_dir / "best_model"


def test_model_registry_incremental_vacuum_rejects_sql_injection(tmp_path):
    registry = ModelRegistry(str(tmp_path / "registry.db"))

    with pytest.raises(ValueError, match="Vacuum pages must be an integer"):
        registry.vacuum_incremental("1); DROP TABLE models;--")

    row = registry.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'models'"
    ).fetchone()
    assert row == ("models",)
    registry.close()


def test_model_registry_incremental_vacuum_accepts_bounded_integer_string(tmp_path):
    registry = ModelRegistry(str(tmp_path / "registry.db"))

    registry.vacuum_incremental("0")

    registry.close()

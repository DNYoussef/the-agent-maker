"""Tier2-P3 gate - CRUCIBLE: Phase 3 RL is wired (dataloaders + correct return).

Synthesis: the controller called trainer.train() with NO dataloaders (train() requires
train_dl, val_dl) -> TypeError -> silent fallback, so RL never ran; and it assigned the
RETURN of train() (a metrics dict) as the model. P3 builds dataloaders and reads the
in-place trained model. (A full multi-episode run is compute-bound; this proves the path.)
"""

import torch

from cross_phase.orchestrator.phase3_controller import Phase3Controller


def _tok(text, **kw):
    return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}


def test_build_rl_dataloaders_yields_stackable_batches():
    ctrl = Phase3Controller(config={}, session_id="t")
    train_dl, val_dl = ctrl._build_rl_dataloaders(_tok)
    batch = next(iter(train_dl))
    assert "input_ids" in batch and "labels" in batch
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape[0] == 2  # batch collated (fixed-length padding)


def test_run_quietstar_rl_invokes_trainer_with_dataloaders(monkeypatch):
    import phase3_quietstar.step2_rl as rl_mod

    captured = {}

    class _SpyTrainer:
        def __init__(self, **kw):
            self.model = "TRAINED_MODEL"

        def train(self, train_dl, val_dl, num_episodes=None):
            captured.update(train_dl=train_dl, val_dl=val_dl, episodes=num_episodes)
            return {"loss": 0.0}  # train() returns METRICS, not a model

    monkeypatch.setattr(rl_mod, "REINFORCETrainer", _SpyTrainer)
    ctrl = Phase3Controller(config={"enable_full_rl": True, "rl_episodes": 3}, session_id="t")

    model, completed = ctrl._run_quietstar_rl(object(), object(), _tok)

    assert completed is True, "RL must run (was falling back due to missing dataloaders)"
    assert model == "TRAINED_MODEL", "must return trainer.model, not the metrics dict"
    assert captured["train_dl"] is not None and captured["val_dl"] is not None
    assert captured["episodes"] == 3

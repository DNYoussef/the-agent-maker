"""
Wave 2 - Class B: Phase1 trainer must report MEASURED summary metrics, not
hardcoded placeholders.

Before the fix:
  - log_final_metrics streamed final_loss=2.5, final_perplexity=12.2 and a
    diversity dict of constants (avg_halting_steps=7.5, ltm_usage=0.45,
    inference_time_ms=85) straight to W&B regardless of training.
  - validate() returned a canned (2.5, {}) when there was no val data.

These tests build a bare Phase1Trainer (via __new__, no heavy model/optimizer
init), inject real "last loss" state, and assert the logged values TRACK that
state and are SENSITIVE to it. They fail if any value is a hardcoded constant
(removing the source literal alone is not enough to make them pass).
"""

import math

import torch.nn as nn

from src.phase1_cognate.training.trainer import Phase1Trainer


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(4, 4)

    def count_parameters(self):
        return {"total": sum(p.numel() for p in self.parameters())}


class _CaptureLogger:
    """Captures the kwargs passed to log_final instead of hitting W&B."""

    def __init__(self):
        self.final = None

    def log_final(self, **kwargs):
        self.final = kwargs


def _bare_trainer():
    t = Phase1Trainer.__new__(Phase1Trainer)
    t.model = _FakeModel()
    t.logger = _CaptureLogger()
    t.ema = None
    t.last_train_loss = None
    t.last_val_loss = None
    return t


def test_validate_returns_no_loss_when_no_val_data():
    """Honest contract: no val data => no validation loss (None), never 2.5."""
    t = _bare_trainer()
    t.val_datasets = {}
    val_loss, val_accs = t.validate()
    assert val_loss is None, f"expected None for no-val-data, got {val_loss!r}"
    assert val_accs == {}


def test_log_final_reports_measured_loss_not_constant():
    t = _bare_trainer()
    t.last_val_loss = 1.2345
    t.log_final_metrics(training_time_hours=0.1)

    cap = t.logger.final
    assert cap is not None, "log_final was never called"
    # Would be 2.5 / 12.2 if still hardcoded.
    assert abs(cap["final_loss"] - 1.2345) < 1e-9
    assert abs(cap["final_perplexity"] - math.exp(1.2345)) < 1e-6


def test_log_final_drops_unmeasurable_diversity_constants():
    t = _bare_trainer()
    t.last_val_loss = 1.0
    t.log_final_metrics(training_time_hours=0.1)

    dm = t.logger.final["diversity_metrics"]
    # The fabricated constants must not be emitted at all.
    assert dm.get("avg_halting_steps") != 7.5
    assert "ltm_usage" not in dm, "ltm_usage is unmeasurable; must not be logged"
    assert "inference_time_ms" not in dm, "inference time is not measured; must not be logged"


def test_log_final_is_sensitive_to_actual_loss():
    """Two different losses must produce two different logged values."""
    seen = []
    for loss_value in (0.5, 3.7):
        t = _bare_trainer()
        t.last_val_loss = loss_value
        t.log_final_metrics(training_time_hours=0.1)
        seen.append(t.logger.final["final_loss"])
    assert seen == [0.5, 3.7], f"final_loss not tracking input: {seen}"
    # perplexity must move too (exp is monotonic, so strictly different)
    assert seen[0] != seen[1]


def test_log_final_prefers_val_then_falls_back_to_train_loss():
    t = _bare_trainer()
    t.last_train_loss = 2.0
    t.last_val_loss = None
    t.log_final_metrics(training_time_hours=0.1)
    assert t.logger.final["final_loss"] == 2.0
    assert abs(t.logger.final["final_perplexity"] - math.exp(2.0)) < 1e-6

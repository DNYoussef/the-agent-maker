"""E4 gate - CRUCIBLE: Phase 3 anti-theater validation is real, and best-state is cloned.

Synthesis: _validate_anti_theater hardcoded consistency_test and ablation_test to True and
fake-passed divergence via hash() when the model had no generate(); best_model_state was
captured with state_dict() (live refs), so restore-best restored the LATEST weights. E4
makes divergence + consistency real, drops the unimplemented ablation claim, fails honestly
when a model can't generate, and deep-copies the best state.
"""

import copy

import torch
import torch.nn as nn

from cross_phase.orchestrator.phase3_controller import Phase3Controller


def _tok(text, **kw):
    # distinct ids per text so a genuine model can diverge
    return {"input_ids": torch.tensor([[len(text), abs(hash(text)) % 7 + 1]])}


class _Divergent(nn.Module):
    """Deterministic, input-dependent generate() - a genuine model."""

    def generate(self, input_ids=None, **kw):
        return input_ids * 10


class _Constant(nn.Module):
    """Same output for every input - theatrical (no real divergence)."""

    def generate(self, input_ids=None, **kw):
        return torch.tensor([[7, 7]])


class _NoGenerate(nn.Module):
    pass


def _ctrl():
    return Phase3Controller(config={}, session_id="t")


def test_genuine_model_passes():
    r = _ctrl()._validate_anti_theater(_Divergent(), _tok)
    assert r["all_passed"] is True
    assert r["divergence_test"] is True and r["consistency_test"] is True


def test_constant_model_fails_divergence():
    r = _ctrl()._validate_anti_theater(_Constant(), _tok)
    assert r["all_passed"] is False, "a model with identical outputs must not pass anti-theater"


def test_model_without_generate_fails_honestly():
    r = _ctrl()._validate_anti_theater(_NoGenerate(), _tok)
    assert r["all_passed"] is False, "no generate() -> cannot prove genuineness -> must fail"


def test_no_unimplemented_ablation_claim():
    # ablation_test was hardcoded True; an unimplemented test must not be reported as passing.
    r = _ctrl()._validate_anti_theater(_Divergent(), _tok)
    assert r.get("ablation_test") is not True, "must not report a hardcoded-pass ablation test"


def test_best_state_capture_is_decoupled_from_later_mutation():
    # The fix pattern: deepcopy the state_dict so later in-place training doesn't corrupt the
    # captured 'best'. Raw state_dict() aliases the live tensors (the bug).
    m = nn.Linear(4, 4)
    aliased = m.state_dict()
    cloned = copy.deepcopy(m.state_dict())
    with torch.no_grad():
        m.weight.add_(1.0)
    assert torch.equal(
        aliased["weight"], m.state_dict()["weight"]
    ), "raw state_dict aliases (the bug)"
    assert not torch.equal(cloned["weight"], m.state_dict()["weight"]), "deepcopy must decouple"

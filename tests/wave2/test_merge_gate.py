"""Wave-2 merge gate test - CRUCIBLE: Codex's #1 correction, proven FIRST.

The 8 phases run in sequence, so a change that improves ONE phase's local metric
can still hurt the end-to-end output. The merge gate must therefore key on the
END-TO-END LOCKED score and treat per-phase diagnostics as advisory only:

1. a candidate that improves every per-phase diagnostic but does NOT improve the
   end-to-end LOCKED score is REJECTED,
2. a candidate that improves the end-to-end LOCKED score is ACCEPTED,
3. a tie is rejected (strict greedy ratchet - keep only on real improvement), and
4. evaluating a candidate spends LOCKED budget through the walled harness (the gate
   is the one place LOCKED is queried).

This file must FAIL until cross_phase/evaluation/merge_gate.py exists.
"""

import math

import pytest

from cross_phase.evaluation.benchmark_suite import BenchmarkSuite, Split
from cross_phase.evaluation.merge_gate import decide_merge, evaluate_candidate
from cross_phase.evaluation.walled_harness import BudgetExhausted, WalledHarness


def test_gate_keys_on_end_to_end_not_per_phase():
    # Candidate beats incumbent on every phase diagnostic but not end-to-end.
    d = decide_merge(
        incumbent_score=0.80,
        candidate_score=0.80,
        per_phase={"phase1": (0.5, 0.9), "phase3": (0.4, 0.95)},  # (incumbent, candidate)
    )
    assert not d.accepted
    assert "end-to-end" in d.reason.lower()


def test_gate_accepts_real_end_to_end_improvement():
    d = decide_merge(incumbent_score=0.80, candidate_score=0.85)
    assert d.accepted


def test_gate_rejects_tie():
    d = decide_merge(incumbent_score=0.80, candidate_score=0.80)
    assert not d.accepted


def test_evaluate_candidate_spends_locked_budget():
    s = BenchmarkSuite()
    h = WalledHarness(s, locked_budget=5)
    answers = {it.prompt: it.answer for it in s.items(Split.LOCKED)}

    def incumbent(prompt):
        return "wrong"

    def candidate(prompt):
        return answers.get(prompt, "wrong")

    before = h.locked_budget_remaining()
    d = evaluate_candidate(h, incumbent, candidate)
    assert d.accepted, "oracle candidate must beat the wrong incumbent end-to-end"
    assert h.locked_budget_remaining() == before - 2, "scored both fns once on LOCKED"


def test_evaluate_candidate_preflights_budget_atomically():
    # Budget=1 must NOT half-spend: raise before scoring, leaving budget intact.
    s = BenchmarkSuite()
    h = WalledHarness(s, locked_budget=1)
    with pytest.raises(BudgetExhausted):
        evaluate_candidate(h, lambda p: "a", lambda p: "b")
    assert h.locked_budget_remaining() == 1, "preflight must not spend the incumbent query"


def test_decide_merge_rejects_bad_inputs():
    with pytest.raises(ValueError):
        decide_merge(0.8, 0.9, min_improvement=-0.1)  # negative bar accepts regressions
    with pytest.raises(ValueError):
        decide_merge(0.8, 0.9, min_improvement=math.inf)
    with pytest.raises(ValueError):
        decide_merge(float("nan"), 0.9)

"""Wave-2 walling test - CRUCIBLE: the wall is the safety property, proven FIRST.

Codex's load-bearing correction: the LOCKED split is only meaningful if the
editing agent can NEVER see its answers, and if querying it is budget-accounted
(so the loop can't launder LOCKED into a second VAL by hammering it). This test
nails both BEFORE the loop exists:

1. the agent-facing surface exposes VAL prompts/answers ONLY - no LOCKED/BLIND
   prompt or answer is reachable through any agent method,
2. scoring against LOCKED is metered: it decrements a finite budget and HARD-STOPS
   (raises) once exhausted, and
3. scoring against VAL is unmetered (the cheap signal the ratchet optimizes).

This file must FAIL until cross_phase/evaluation/walled_harness.py exists.
"""

import pytest

from cross_phase.evaluation.benchmark_suite import BenchmarkSuite, Split
from cross_phase.evaluation.walled_harness import BudgetExhausted, WalledHarness


def _suite_locked_prompts():
    s = BenchmarkSuite()
    return {it.prompt for it in s.items(Split.LOCKED)} | {it.prompt for it in s.items(Split.BLIND)}


def test_agent_surface_exposes_val_only():
    s = BenchmarkSuite()
    h = WalledHarness(s, locked_budget=3)
    walled_off = _suite_locked_prompts()

    # The agent may legitimately learn VAL answers; nothing else.
    answers = h.agent_answers()
    val_prompts = {it.prompt for it in s.items(Split.VAL)}
    assert set(answers.keys()) == val_prompts
    assert set(h.val_prompts()) == val_prompts
    # No locked/blind prompt leaks into anything the agent can read.
    assert walled_off.isdisjoint(set(answers.keys()))
    assert walled_off.isdisjoint(set(h.val_prompts()))


def test_locked_scoring_is_budget_metered_and_hard_stops():
    s = BenchmarkSuite()
    h = WalledHarness(s, locked_budget=2)
    answers = {it.prompt: it.answer for it in s.items(Split.LOCKED)}

    def oracle(prompt):
        return answers.get(prompt, "x")

    assert h.locked_budget_remaining() == 2
    h.score_locked(oracle)
    assert h.locked_budget_remaining() == 1
    h.score_locked(oracle)
    assert h.locked_budget_remaining() == 0
    with pytest.raises(BudgetExhausted):
        h.score_locked(oracle)


def test_val_scoring_is_unmetered():
    s = BenchmarkSuite()
    h = WalledHarness(s, locked_budget=1)
    answers = h.agent_answers()

    def oracle(prompt):
        return answers.get(prompt, "x")

    for _ in range(10):
        assert h.score_val(oracle) > 0.9
    assert h.locked_budget_remaining() == 1, "VAL scoring must not spend the LOCKED budget"

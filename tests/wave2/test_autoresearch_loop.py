"""Wave-2 autoresearch loop test - CRUCIBLE: the greedy ratchet, GPU-free core.

Karpathy's autoresearch loop, made safe by the Wave-2 gates. The GPU-free core is
the ORCHESTRATION: propose -> tiered filter -> end-to-end gate -> ratchet -> journal.
The heavy, non-deterministic step (edit phase code, TRAIN on a GPU, wrap the result)
is injected as the `proposer`, exactly like evaluator.generate_fn - so the loop's
logic is fully testable with stubs and no GPU.

The properties proven here (Codex's load-bearing corrections):
1. TIERED FILTER: a candidate that does not beat VAL is rejected WITHOUT spending
   the expensive LOCKED budget (cheap tier first),
2. END-TO-END RATCHET: a candidate that improves VAL but NOT end-to-end LOCKED is
   rejected and the incumbent is unchanged (a local gain that hurts the whole),
3. a candidate that improves end-to-end LOCKED is accepted and becomes incumbent,
4. every attempt is journaled, and
5. the loop stops cleanly when the LOCKED budget is exhausted.

This file must FAIL until cross_phase/evaluation/autoresearch_loop.py exists.
"""

import sys
import textwrap

import pytest

from cross_phase.evaluation.autoresearch_loop import Candidate, journal_tsv, run_loop
from cross_phase.evaluation.benchmark_suite import BenchmarkSuite, Split
from cross_phase.evaluation.sealed_scorer import SealedScorer, seal

ORACLE = textwrap.dedent(
    """
    import json, re, sys
    prompts = json.load(sys.stdin)
    out = []
    for p in prompts:
        m = re.search(r"What is (\\d+) (.) (\\d+)", p)
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        out.append(str({"+": a + b, "-": a - b, "*": a * b}[op]))
    json.dump(out, sys.stdout)
    """
)


def _seal(tmp_path, budget=10):
    seal(str(tmp_path), locked_budget=budget)
    return SealedScorer(sealed_dir=str(tmp_path))


def _oracle_cmd(tmp_path):
    p = tmp_path / "oracle.py"
    p.write_text(ORACLE)
    return [sys.executable, str(p)]


def _wrong_generate(_prompt):
    return "nope"


def _oracle_generate_factory(suite):
    answers = {it.prompt: it.answer for it in suite.items(Split.VAL)}
    return lambda prompt: answers.get(prompt, "nope")


def test_val_gate_skips_locked_spend(tmp_path):
    suite = BenchmarkSuite()
    sealed = _seal(tmp_path)
    incumbent = Candidate("base", _oracle_generate_factory(suite), _oracle_cmd(tmp_path))
    # Proposer always returns a candidate that is WORSE on VAL than the oracle incumbent.
    weak = Candidate("weak", _wrong_generate, _oracle_cmd(tmp_path))
    before = sealed.budget_remaining("locked")

    def proposer(_i, _inc):
        return weak

    result = run_loop(suite=suite, sealed=sealed, incumbent=incumbent, proposer=proposer, rounds=3)
    # The loop measures the incumbent's LOCKED baseline once (spends 1); the 3
    # VAL-failing candidates are rejected by the cheap tier and spend 0 more.
    assert sealed.budget_remaining("locked") == before - 1
    assert all(a.locked is None for a in result.attempts)
    assert all(not a.accepted for a in result.attempts)
    assert len(result.attempts) == 3


def test_rejects_val_up_locked_down(tmp_path):
    suite = BenchmarkSuite()
    sealed = _seal(tmp_path)
    # Incumbent: mediocre on VAL, but PERFECT on LOCKED (oracle cmd).
    n_val = len(suite.items(Split.VAL))

    def mediocre_gen(prompt):
        return "nope"  # 0 on VAL

    incumbent = Candidate("base", mediocre_gen, _oracle_cmd(tmp_path))

    # Candidate: better on VAL (oracle), but WORSE on LOCKED (all wrong).
    wrong_cmd_path = tmp_path / "wrong.py"
    wrong_cmd_path.write_text("import json,sys;print(json.dumps(['x']*len(json.load(sys.stdin))))")
    cand = Candidate(
        "vup_ldown", _oracle_generate_factory(suite), [sys.executable, str(wrong_cmd_path)]
    )

    locked_before = sealed.budget_remaining("locked")

    def proposer(_i, _inc):
        return cand

    result = run_loop(suite=suite, sealed=sealed, incumbent=incumbent, proposer=proposer, rounds=1)
    a = result.attempts[0]
    assert a.val > 0.0 and a.locked == 0.0  # VAL improved, LOCKED collapsed
    assert not a.accepted, "a VAL gain that hurts end-to-end LOCKED must be rejected"
    assert result.incumbent_id == "base"  # incumbent unchanged
    assert sealed.budget_remaining("locked") == locked_before - 2  # baseline + this candidate
    assert n_val > 0


def test_accepts_end_to_end_improvement(tmp_path):
    suite = BenchmarkSuite()
    sealed = _seal(tmp_path)

    def zero_gen(prompt):
        return "nope"

    wrong_cmd_path = tmp_path / "wrong.py"
    wrong_cmd_path.write_text("import json,sys;print(json.dumps(['x']*len(json.load(sys.stdin))))")
    incumbent = Candidate("base", zero_gen, [sys.executable, str(wrong_cmd_path)])  # 0 / 0

    better = Candidate(
        "better", _oracle_generate_factory(suite), _oracle_cmd(tmp_path)
    )  # high / high

    def proposer(_i, _inc):
        return better

    result = run_loop(suite=suite, sealed=sealed, incumbent=incumbent, proposer=proposer, rounds=1)
    assert result.attempts[0].accepted
    assert result.incumbent_id == "better"
    assert result.incumbent_locked > 0.9


def test_journal_records_all_and_stops_on_budget(tmp_path):
    suite = BenchmarkSuite()
    sealed = _seal(tmp_path, budget=2)  # baseline takes 1, leaving 1 for candidates
    incumbent = Candidate("base", _wrong_generate, _oracle_cmd(tmp_path))  # 0 VAL, high LOCKED

    # Every candidate beats VAL (passes the cheap tier) so each tries to spend LOCKED.
    cand = Candidate("c", _oracle_generate_factory(suite), _oracle_cmd(tmp_path))

    def proposer(_i, _inc):
        return cand

    result = run_loop(suite=suite, sealed=sealed, incumbent=incumbent, proposer=proposer, rounds=5)
    # 1 candidate scored LOCKED (budget hit 0), then the loop stopped.
    assert any("budget" in a.reason.lower() for a in result.attempts)
    assert sealed.budget_remaining("locked") == 0
    tsv = journal_tsv(result)
    assert tsv.splitlines()[0].startswith("candidate_id\t")
    assert len(tsv.splitlines()) == len(result.attempts) + 1  # header + one row per attempt


def test_invalid_args_raise_before_any_spend(tmp_path):
    suite = BenchmarkSuite()
    sealed = _seal(tmp_path)
    incumbent = Candidate("base", _wrong_generate, _oracle_cmd(tmp_path))
    before = sealed.budget_remaining("locked")

    def proposer(_i, _inc):
        return incumbent

    for bad in ({"min_improvement": -0.1}, {"min_improvement": float("nan")}, {"rounds": -1}):
        kwargs = {"rounds": 1, **bad}
        with pytest.raises(ValueError):
            run_loop(suite=suite, sealed=sealed, incumbent=incumbent, proposer=proposer, **kwargs)
    assert sealed.budget_remaining("locked") == before  # not even the baseline was scored


def test_candidate_crash_is_journaled_and_loop_continues(tmp_path):
    suite = BenchmarkSuite()
    sealed = _seal(tmp_path, budget=10)
    wrong_cmd = tmp_path / "low.py"
    wrong_cmd.write_text("import json,sys;print(json.dumps(['x']*len(json.load(sys.stdin))))")
    incumbent = Candidate("base", _wrong_generate, [sys.executable, str(wrong_cmd)])  # 0 / 0

    broken = tmp_path / "broken.py"
    broken.write_text("import sys; sys.stdout.write('not json')")
    crasher = Candidate("crash", _oracle_generate_factory(suite), [sys.executable, str(broken)])

    def proposer(_i, _inc):
        return crasher

    before = sealed.budget_remaining("locked")  # after _seal, before run_loop
    result = run_loop(suite=suite, sealed=sealed, incumbent=incumbent, proposer=proposer, rounds=2)
    assert len(result.attempts) == 2, "a crashing candidate must not kill the campaign"
    assert all("failed" in a.reason for a in result.attempts)
    assert all(not a.accepted for a in result.attempts)
    assert result.incumbent_id == "base"
    # baseline (1) + two crashing candidates charged on exposure (2) = 3 spent
    assert sealed.budget_remaining("locked") == before - 3


def test_proposer_receives_current_incumbent(tmp_path):
    suite = BenchmarkSuite()
    sealed = _seal(tmp_path, budget=10)
    wrong_cmd = tmp_path / "low.py"
    wrong_cmd.write_text("import json,sys;print(json.dumps(['x']*len(json.load(sys.stdin))))")
    incumbent = Candidate("base", _wrong_generate, [sys.executable, str(wrong_cmd)])  # 0 / 0
    winner = Candidate("winner", _oracle_generate_factory(suite), _oracle_cmd(tmp_path))
    seen = []

    def proposer(i, inc):
        seen.append(inc.id)
        return winner if i == 0 else incumbent

    run_loop(suite=suite, sealed=sealed, incumbent=incumbent, proposer=proposer, rounds=2)
    # Round 0 proposes from "base"; after it is accepted, round 1 sees "winner".
    assert seen == ["base", "winner"]

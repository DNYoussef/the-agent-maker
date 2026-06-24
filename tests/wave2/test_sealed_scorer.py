"""Wave-2 sealed scorer test - CRUCIBLE: the out-of-process wall, audited HARD.

An adversarial multi-lens audit (4 Claude lenses + Codex) of the first cut showed
several tests asserted properties BY CONSTRUCTION (theater). These tests assert the
REAL properties the redesign must hold:

1. unsealed store -> fail CLOSED,
2. the loop-side client process NEVER holds the answers (scoring is out-of-process),
3. seal() is CREATE-ONLY - re-sealing cannot silently reset a live budget,
4. the budget ledger is persisted, atomically charged, and concurrency-safe
   (N concurrent evaluations on budget=1 spend at most 1),
5. the candidate cannot read the sealed dir via inherited env,
6. a hung candidate fails closed via timeout (and the eval is still charged),
7. malformed / non-string completions fail closed, and the eval is charged,
8. an empty split fails closed WITHOUT charging (no spend on a bad store).

This file must FAIL until cross_phase/evaluation/sealed_scorer.py exists.
"""

import json
import sys
import textwrap
import threading

import pytest

from cross_phase.evaluation.sealed_scorer import (
    BudgetExhausted,
    SealedScorer,
    budget_remaining,
    seal,
)

# Candidates read a JSON list of prompts on stdin, write a JSON list on stdout.
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
WRONG = "import json,sys; print(json.dumps(['nope']*len(json.load(sys.stdin))))"
INTS = "import json,sys; print(json.dumps(list(range(len(json.load(sys.stdin))))))"
BROKEN = "import sys; sys.stdout.write('not json')"
HANG = "import time; time.sleep(60)"  # never returns -> must be killed by the timeout
ENV_PROBE = textwrap.dedent(
    """
    import json, os, sys
    prompts = json.load(sys.stdin)
    open(sys.argv[1], "w").write(os.environ.get("AGENT_MAKER_SEALED_DIR", "ABSENT"))
    json.dump(["nope"] * len(prompts), sys.stdout)
    """
)


def _script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return [sys.executable, str(p)]


def test_unsealed_dir_fails_closed(tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        SealedScorer(sealed_dir=str(tmp_path / "nope"))


def test_seal_then_oracle_beats_wrong(tmp_path):
    seal(str(tmp_path), locked_budget=10)
    sc = SealedScorer(sealed_dir=str(tmp_path))
    assert sc.score(_script(tmp_path, "o.py", ORACLE), "locked") > 0.9
    assert sc.score(_script(tmp_path, "w.py", WRONG), "locked") < 0.1


def test_client_process_never_holds_answers(tmp_path):
    seal(str(tmp_path), locked_budget=5)
    sc = SealedScorer(sealed_dir=str(tmp_path))
    # The real wall: the client holds ONLY the dir path - no items, prompts, or
    # answer key. (Answers are single digits, so a substring scan of a tmp path is
    # meaningless; the load-bearing fact is the client never loaded splits.json.)
    assert set(vars(sc)) == {"_dir"}
    blob = json.dumps(vars(sc))
    assert "What is" not in blob and "answer" not in blob


def test_seal_is_create_only(tmp_path):
    seal(str(tmp_path), locked_budget=2)
    sc = SealedScorer(sealed_dir=str(tmp_path))
    sc.score(_script(tmp_path, "o.py", ORACLE), "locked")
    assert sc.budget_remaining("locked") == 1
    with pytest.raises(FileExistsError):  # cannot silently reset a live budget
        seal(str(tmp_path), locked_budget=99)
    assert sc.budget_remaining("locked") == 1
    seal(str(tmp_path), locked_budget=2, force=True)  # explicit reset only
    assert sc.budget_remaining("locked") == 2


def test_budget_persists_across_instances_and_hard_stops(tmp_path):
    seal(str(tmp_path), locked_budget=2)
    oracle = _script(tmp_path, "o.py", ORACLE)
    SealedScorer(sealed_dir=str(tmp_path)).score(oracle, "locked")
    assert SealedScorer(sealed_dir=str(tmp_path)).budget_remaining("locked") == 1
    SealedScorer(sealed_dir=str(tmp_path)).score(oracle, "locked")
    last = SealedScorer(sealed_dir=str(tmp_path))
    assert last.budget_remaining("locked") == 0
    with pytest.raises(BudgetExhausted):
        last.score(oracle, "locked")


def test_concurrent_evaluations_do_not_overspend(tmp_path):
    seal(str(tmp_path), locked_budget=1)
    oracle = _script(tmp_path, "o.py", ORACLE)
    results = []

    def run():
        sc = SealedScorer(sealed_dir=str(tmp_path))
        try:
            sc.score(oracle, "locked")
            results.append("ok")
        except BudgetExhausted:
            results.append("exhausted")
        except Exception as exc:  # noqa: BLE001 - any other outcome is a failure
            results.append(f"error:{exc}")

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Every thread must have a clean outcome; exactly one wins, the rest are refused.
    assert sorted(results) == ["exhausted", "exhausted", "exhausted", "ok"], results
    assert budget_remaining(str(tmp_path), "locked") == 0


def test_blind_metered_once(tmp_path):
    seal(str(tmp_path), locked_budget=5)
    sc = SealedScorer(sealed_dir=str(tmp_path))
    oracle = _script(tmp_path, "o.py", ORACLE)
    assert sc.budget_remaining("blind") == 1
    sc.score(oracle, "blind")
    with pytest.raises(BudgetExhausted):
        sc.score(oracle, "blind")


def test_candidate_cannot_read_sealed_dir_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MAKER_SEALED_DIR", str(tmp_path))
    seal(str(tmp_path), locked_budget=5)
    sc = SealedScorer(sealed_dir=str(tmp_path))
    probe = tmp_path / "seen.txt"
    cmd = _script(tmp_path, "env.py", ENV_PROBE) + [str(probe)]
    sc.score(cmd, "locked")
    assert probe.read_text() == "ABSENT", "candidate must not inherit the sealed-dir path"


def test_hung_candidate_fails_closed_and_charges(tmp_path):
    seal(str(tmp_path), locked_budget=3)
    sc = SealedScorer(sealed_dir=str(tmp_path))
    before = sc.budget_remaining("locked")
    with pytest.raises(RuntimeError):
        sc.score(_script(tmp_path, "h.py", HANG), "locked", timeout=2)
    assert sc.budget_remaining("locked") == before - 1  # charge-on-exposure


def test_malformed_and_nonstring_fail_closed_and_charge(tmp_path):
    seal(str(tmp_path), locked_budget=5)
    sc = SealedScorer(sealed_dir=str(tmp_path))
    before = sc.budget_remaining("locked")
    with pytest.raises(RuntimeError):
        sc.score(_script(tmp_path, "b.py", BROKEN), "locked")
    with pytest.raises(RuntimeError):
        sc.score(_script(tmp_path, "i.py", INTS), "locked")
    assert sc.budget_remaining("locked") == before - 2


def test_empty_split_fails_closed_without_charge(tmp_path):
    seal(str(tmp_path), locked_budget=3)
    # Corrupt the store to an empty locked split; scoring must refuse and NOT charge.
    data = json.loads((tmp_path / "splits.json").read_text())
    data["locked"] = []
    (tmp_path / "splits.json").write_text(json.dumps(data))
    sc = SealedScorer(sealed_dir=str(tmp_path))
    before = sc.budget_remaining("locked")
    with pytest.raises(RuntimeError):
        sc.score(_script(tmp_path, "o.py", ORACLE), "locked")
    assert sc.budget_remaining("locked") == before  # no spend on a bad store


def test_rejects_val_split(tmp_path):
    seal(str(tmp_path), locked_budget=3)
    sc = SealedScorer(sealed_dir=str(tmp_path))
    with pytest.raises(ValueError):
        sc.score(_script(tmp_path, "o.py", ORACLE), "val")

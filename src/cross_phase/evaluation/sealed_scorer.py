"""Wave-2 sealed scorer - the out-of-process LOCKED/BLIND gate (Layer 2 prereq).

Layer 1's WalledHarness was an in-process PROTOCOL boundary; Codex's audit showed
it was not a SECURITY one. An adversarial multi-lens audit of the FIRST cut of this
module showed it was only HALF sealed - it ran the candidate out-of-process but
scored IN the loop process, loading the whole answer key into an attribute. This
version fixes that: SCORING runs in a separate process, so the loop/agent process
never holds the answers.

ARCHITECTURE
  - seal(dir): materialize held-out LOCKED/BLIND items + budgets into an out-of-tree
    dir. CREATE-ONLY: refuses to clobber a sealed dir (no silent budget reset).
  - SealedScorer (loop-side client): holds ONLY the dir path. score() shells out to
    a short-lived scorer subprocess; the answers are loaded only in THAT process and
    die with it. budget_remaining() reads the ledger (counts, no answers).
  - the scorer subprocess: charges the budget atomically (OS file lock + fsync),
    then runs the candidate as its own child - timeout'd, utf-8, with an env that
    strips the sealed-dir path and a throwaway cwd, so the candidate is not HANDED
    the path to the store (a determined filesystem search is still the OS ceiling
    below) - then prints only {score, remaining}.

WHAT THIS BUYS over Layer 1 (each was a real audit finding):
  - answers are never in the loop/agent process (scoring is out-of-process),
  - the budget is a persisted, atomically-charged, append-only ledger that a fresh
    client or a concurrent call cannot overspend, and re-seal cannot silently reset,
  - the candidate runs isolated (timeout / sanitized env / throwaway cwd).

HONEST LIMITS (documented, not solved here):
  - SCORE IS A TRUSTED-GATE SIGNAL. The returned float is for the merge gate. If it
    is relayed back to the editing agent, the agent can recover the answer key by
    submitting per-item guesses and reading score deltas (a label oracle). The loop
    MUST consume locked/blind scores in the gate and never hand them to the agent.
  - BUDGET UNIT = EVALUATIONS. One score() call exposes the WHOLE split's prompts to
    the candidate; the budget caps the number of evaluations, not prompts.
  - NO REAL ANSWER SECRECY FOR A COMMITTED DETERMINISTIC GENERATOR. seal() builds
    items via BenchmarkSuite, which the candidate can import and recompute; and the
    toy arithmetic answers are computable from the prompts anyway. Real secrecy needs
    held-out items from an EXTERNAL secret artifact, not derivable from repo source.
    This module delivers the MECHANISM; production must supply sealed data.
  - OS CEILING. A process with full filesystem access can still read splits.json or
    delete a ledger. True sealing needs OS-level isolation (separate user/container).

Seal a store once:
  python -m cross_phase.evaluation.sealed_scorer seal <dir> --locked-budget 50
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import List, Tuple

from .benchmark_suite import BenchmarkSuite, Split
from .evaluator import matches

try:
    import fcntl  # POSIX

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows path
    import msvcrt

    _HAVE_FCNTL = False

_METERED = ("locked", "blind")
_DEFAULT_TIMEOUT = 120
_SEALED_ENV = "AGENT_MAKER_SEALED_DIR"


class BudgetExhausted(RuntimeError):
    """Raised when a metered split's evaluation budget is spent."""


# --------------------------------------------------------------------------- #
# sealing
# --------------------------------------------------------------------------- #
def _atomic_write_json(path: str, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def seal(sealed_dir: str, locked_budget: int, blind_budget: int = 1, force: bool = False) -> None:
    """Materialize held-out splits + budgets out-of-tree. CREATE-ONLY by default.

    Refuses to overwrite an existing sealed store (which would silently reset the
    budget) unless force=True. The dir must live OUTSIDE the editable workspace.
    """
    splits_path = os.path.join(sealed_dir, "splits.json")
    meta_path = os.path.join(sealed_dir, "meta.json")
    if (os.path.exists(splits_path) or os.path.exists(meta_path)) and not force:
        raise FileExistsError(f"{sealed_dir} already sealed; pass force=True to overwrite")
    os.makedirs(sealed_dir, exist_ok=True)
    suite = BenchmarkSuite()
    splits = {
        name: [
            {"id": it.id, "prompt": it.prompt, "answer": it.answer, "family": it.family}
            for it in suite.items(split)
        ]
        for name, split in (("locked", Split.LOCKED), ("blind", Split.BLIND))
    }
    _atomic_write_json(splits_path, splits)
    _atomic_write_json(
        meta_path,
        {"locked": {"budget": locked_budget}, "blind": {"budget": blind_budget}},
    )
    for name in _METERED:  # fresh empty ledgers (reached only on fresh or forced seal)
        open(os.path.join(sealed_dir, f"{name}.jsonl"), "w", encoding="utf-8").close()


# --------------------------------------------------------------------------- #
# budget ledger (read in either process; charge only in the scorer process)
# --------------------------------------------------------------------------- #
def _ledger(sealed_dir: str, split: str) -> str:
    return os.path.join(sealed_dir, f"{split}.jsonl")


def _count_spent(sealed_dir: str, split: str) -> int:
    path = _ledger(sealed_dir, split)
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if json.loads(line).get("spent") != 1:
                raise RuntimeError(f"corrupt ledger record in {path}: {line[:80]}")
            n += 1
    return n


def budget_remaining(sealed_dir: str, split: str) -> int:
    if split not in _METERED:
        raise ValueError(f"{split} is not a metered split")
    with open(os.path.join(sealed_dir, "meta.json"), encoding="utf-8") as fh:
        total = json.load(fh)[split]["budget"]
    return total - _count_spent(sealed_dir, split)


class _FileLock:
    """Cross-process exclusive lock around the budget critical section."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._fh = None

    def __enter__(self) -> "_FileLock":
        self._fh = open(self._path, "a+", encoding="utf-8")
        if _HAVE_FCNTL:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        else:  # pragma: no cover - Windows path
            self._fh.seek(0)
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
        return self

    def __exit__(self, *exc) -> None:
        if _HAVE_FCNTL:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        else:  # pragma: no cover - Windows path
            self._fh.seek(0)
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        self._fh.close()


def _charge(sealed_dir: str, split: str) -> int:
    """Spend one evaluation. Caller MUST hold the split lock. Returns remaining."""
    if budget_remaining(sealed_dir, split) <= 0:
        raise BudgetExhausted(f"{split} evaluation budget exhausted")
    with open(_ledger(sealed_dir, split), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"spent": 1}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return budget_remaining(sealed_dir, split)


# --------------------------------------------------------------------------- #
# scoring core (runs ONLY in the scorer subprocess - holds the answers)
# --------------------------------------------------------------------------- #
def _run_candidate(candidate_cmd: List[str], prompts: List[str], timeout: int) -> str:
    if not candidate_cmd or not all(isinstance(a, str) for a in candidate_cmd):
        raise RuntimeError("candidate_cmd must be a non-empty list of strings")
    env = {k: v for k, v in os.environ.items() if k != _SEALED_ENV}
    with tempfile.TemporaryDirectory() as cwd:
        try:
            proc = subprocess.run(
                candidate_cmd,
                input=json.dumps(prompts),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                env=env,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"candidate timed out after {timeout}s") from exc
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(f"candidate not runnable: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"candidate exited {proc.returncode}: {(proc.stderr or '')[:500]}")
    return proc.stdout


def _parse_completions(stdout: str, n: int) -> List[str]:
    try:
        completions = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"candidate stdout was not JSON: {stdout[:200]}") from exc
    if not isinstance(completions, list):
        raise RuntimeError(f"candidate output is not a JSON list: {type(completions).__name__}")
    if len(completions) != n:
        raise RuntimeError(f"candidate returned {len(completions)} of {n} completions")
    if not all(isinstance(c, str) for c in completions):
        raise RuntimeError("candidate completions must all be strings")
    return completions


def _score_split(
    sealed_dir: str, split: str, candidate_cmd: List[str], timeout: int = _DEFAULT_TIMEOUT
) -> Tuple[float, int]:
    if split not in _METERED:
        raise ValueError(f"{split} is not a sealed split; score val via BenchmarkSuite")
    with open(os.path.join(sealed_dir, "splits.json"), encoding="utf-8") as fh:
        items = json.load(fh)[split]
    if not items:
        raise RuntimeError(f"{split} split is empty")  # before charge - no spend on a bad store
    with _FileLock(os.path.join(sealed_dir, f"{split}.lock")):
        remaining = _charge(sealed_dir, split)  # charge-on-exposure, atomic
    prompts = [it["prompt"] for it in items]
    completions = _parse_completions(_run_candidate(candidate_cmd, prompts, timeout), len(items))
    correct = sum(1 for it, c in zip(items, completions) if matches(c, it["answer"]))
    return correct / len(items), remaining


# --------------------------------------------------------------------------- #
# loop-side client - holds ONLY the dir path; never loads answers
# --------------------------------------------------------------------------- #
_ERRORS = {
    "BudgetExhausted": BudgetExhausted,
    "ValueError": ValueError,
    "FileNotFoundError": FileNotFoundError,
    "RuntimeError": RuntimeError,
}


class SealedScorer:
    def __init__(self, sealed_dir: str = None) -> None:
        self._dir = sealed_dir or os.environ.get(_SEALED_ENV)
        if not self._dir:
            raise RuntimeError(f"set {_SEALED_ENV} or pass sealed_dir; seal first")
        if not os.path.exists(os.path.join(self._dir, "meta.json")):
            raise FileNotFoundError(f"{self._dir} is not sealed; run seal() first")
        # Deliberately does NOT load splits.json - answers never enter this process.

    def budget_remaining(self, split: str) -> int:
        return budget_remaining(self._dir, split)

    def score(self, candidate_cmd: List[str], split: str, timeout: int = _DEFAULT_TIMEOUT) -> float:
        """Score the candidate on a sealed split in a separate scorer process.

        The float is a TRUSTED-GATE signal - consume it in the merge gate; never
        relay a locked/blind score to the editing agent (see module docstring).
        """
        cmd = [
            sys.executable,
            "-m",
            "cross_phase.evaluation.sealed_scorer",
            "__score__",
            "--dir",
            self._dir,
            "--split",
            split,
            "--timeout",
            str(timeout),
            "--",
            *candidate_cmd,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        out = json.loads(proc.stdout) if proc.stdout.strip() else {}
        if proc.returncode != 0:
            cls = _ERRORS.get(out.get("error"), RuntimeError)
            raise cls(out.get("msg") or (proc.stderr or "scorer subprocess failed")[:500])
        return out["score"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="sealed_scorer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_seal = sub.add_parser("seal")
    p_seal.add_argument("dir")
    p_seal.add_argument("--locked-budget", type=int, default=50)
    p_seal.add_argument("--blind-budget", type=int, default=1)
    p_seal.add_argument("--force", action="store_true")
    p_score = sub.add_parser("__score__")
    p_score.add_argument("--dir", required=True)
    p_score.add_argument("--split", required=True)
    p_score.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT)
    p_score.add_argument("candidate", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv[1:])

    if args.cmd == "seal":
        seal(args.dir, args.locked_budget, args.blind_budget, args.force)
        print(f"sealed {args.dir} (locked={args.locked_budget}, blind={args.blind_budget})")
        return 0

    candidate = args.candidate[1:] if args.candidate[:1] == ["--"] else args.candidate
    try:
        score, remaining = _score_split(args.dir, args.split, candidate, args.timeout)
    except Exception as exc:  # noqa: BLE001 - relay typed error across the process boundary
        print(json.dumps({"error": type(exc).__name__, "msg": str(exc)}))
        return 1
    print(json.dumps({"score": score, "remaining": remaining}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))

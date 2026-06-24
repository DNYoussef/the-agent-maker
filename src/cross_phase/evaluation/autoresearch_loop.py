"""Wave-2 autoresearch loop - the greedy ratchet, made safe by the gates.

Karpathy's autoresearch loop (propose a change, keep it if a held-out metric
improves) with the Wave-2 corrections wired in:

  - TIERED FILTER (cheap before expensive): a candidate is first scored on the
    public VAL split in-process (free). Only if it beats the incumbent on VAL (by
    ANY margin) does the loop spend the metered LOCKED budget on it. The end-to-end
    margin (min_improvement) applies only to the LOCKED gate, so a small VAL gain
    that hides a large LOCKED gain is not filtered out. More tiers (multi-seed,
    second-model audit) belong with the real GPU training and are not stubbed here.
  - END-TO-END RATCHET: the keep/drop decision keys on the end-to-end LOCKED score
    (decide_merge), never on a per-phase or VAL gain - a local win that hurts the
    whole is rejected.
  - WALLED LOCKED: LOCKED is scored only through the out-of-process SealedScorer,
    so the loop never holds the held-out answers and the budget is enforced.

GPU BOUNDARY: `proposer(i, incumbent)` is the injected heavy step - edit the phase
code, TRAIN on a GPU, wrap the result as a Candidate, given the round index and the
current incumbent to propose from. The loop's orchestration is GPU-free and fully
tested with stub proposers. A Candidate exposes the trained model two ways:
val_generate (in-process, for the public VAL split) and locked_cmd (a subprocess
command, for the sealed LOCKED split - see sealed_scorer).

TRUSTED LOG: the journal carries LOCKED scores. It is a trusted-side artifact - do
NOT feed the LOCKED column back to the editing agent, or LOCKED becomes a metered
score oracle (same contract as sealed_scorer).
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .benchmark_suite import BenchmarkSuite, Split
from .merge_gate import decide_merge
from .sealed_scorer import BudgetExhausted, SealedScorer


@dataclass(frozen=True)
class Candidate:
    id: str
    val_generate: Callable[[str], str]  # in-process scorer for the public VAL split
    locked_cmd: Tuple[str, ...]  # subprocess command for the sealed LOCKED split


@dataclass(frozen=True)
class Attempt:
    candidate_id: str
    val: float
    locked: Optional[float]  # None when the cheap VAL tier rejected it (no LOCKED spend)
    accepted: bool
    reason: str


@dataclass(frozen=True)
class LoopResult:
    incumbent_id: str
    incumbent_val: float
    incumbent_locked: float
    attempts: List[Attempt]


def run_loop(
    *,
    suite: BenchmarkSuite,
    sealed: SealedScorer,
    incumbent: Candidate,
    proposer: Callable[[int, Candidate], Candidate],
    rounds: int,
    min_improvement: float = 0.0,
) -> LoopResult:
    """Run the greedy ratchet for `rounds` proposals. Returns the final incumbent
    and the full journal of attempts. Validates args BEFORE spending any budget."""
    if min_improvement < 0 or not math.isfinite(min_improvement):
        raise ValueError("min_improvement must be finite and non-negative")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")

    inc = incumbent
    inc_val = suite.score(inc.val_generate, Split.VAL)
    inc_locked = sealed.score(inc.locked_cmd, "locked")  # one-time baseline
    attempts: List[Attempt] = []

    for i in range(rounds):
        cand = proposer(i, inc)
        val = suite.score(cand.val_generate, Split.VAL)
        if val <= inc_val:  # cheap tier: any VAL improvement required, no margin
            attempts.append(Attempt(cand.id, val, None, False, "val-gate: no VAL improvement"))
            continue
        try:
            locked = sealed.score(cand.locked_cmd, "locked")  # expensive tier
        except BudgetExhausted:
            attempts.append(Attempt(cand.id, val, None, False, "locked budget exhausted"))
            break
        except RuntimeError as exc:  # candidate crashed/timed out; budget already charged
            attempts.append(Attempt(cand.id, val, None, False, f"locked scoring failed: {exc}"))
            continue
        decision = decide_merge(inc_locked, locked, min_improvement=min_improvement)
        attempts.append(Attempt(cand.id, val, locked, decision.accepted, decision.reason))
        if decision.accepted:
            inc, inc_val, inc_locked = cand, val, locked

    return LoopResult(inc.id, inc_val, inc_locked, attempts)


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def journal_tsv(result: LoopResult) -> str:
    """Serialize the journal as TSV (Karpathy's results.tsv). One row per attempt;
    floats are full-precision (replayable) and fields escape tab/newline. The final
    incumbent is in LoopResult, not necessarily the last row (none may be accepted)."""
    header = "candidate_id\tval\tlocked\taccepted\treason"
    rows = [
        "\t".join(
            [
                _esc(a.candidate_id),
                repr(a.val),
                "" if a.locked is None else repr(a.locked),
                str(a.accepted),
                _esc(a.reason),
            ]
        )
        for a in result.attempts
    ]
    return "\n".join([header, *rows])

"""Wave-2 merge gate - the END-TO-END accept/reject decision.

Codex's load-bearing correction: with 8 sequential phases, a per-phase gain can
hurt the end-to-end result. So the decision keys ONLY on the end-to-end LOCKED
score; per-phase numbers are carried for diagnostics but never vote.

The decision (decide_merge) is a PURE function of two scores - testable with no
budget. evaluate_candidate is the thin metered orchestrator that scores both the
incumbent and the candidate on LOCKED (spending budget) and applies the decision.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from .walled_harness import BudgetExhausted, WalledHarness


@dataclass(frozen=True)
class MergeDecision:
    accepted: bool
    reason: str
    incumbent_score: float
    candidate_score: float
    # advisory only: phase -> (incumbent, candidate); never affects `accepted`.
    per_phase: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def decide_merge(
    incumbent_score: float,
    candidate_score: float,
    per_phase: Optional[Dict[str, Tuple[float, float]]] = None,
    min_improvement: float = 0.0,
) -> MergeDecision:
    """Accept iff the candidate strictly improves the END-TO-END score.

    min_improvement raises the bar above noise; the default (0.0) is the strict
    greedy ratchet - ties and regressions are rejected.
    """
    if min_improvement < 0 or not math.isfinite(min_improvement):
        raise ValueError("min_improvement must be finite and non-negative")
    if not (math.isfinite(incumbent_score) and math.isfinite(candidate_score)):
        raise ValueError("scores must be finite")
    improved = candidate_score > incumbent_score + min_improvement
    reason = (
        f"end-to-end {candidate_score:.4f} > {incumbent_score:.4f} (+>{min_improvement:.4f})"
        if improved
        else f"end-to-end {candidate_score:.4f} <= {incumbent_score:.4f}; per-phase gains do not vote"
    )
    return MergeDecision(
        accepted=improved,
        reason=reason,
        incumbent_score=incumbent_score,
        candidate_score=candidate_score,
        per_phase=dict(per_phase or {}),
    )


def evaluate_candidate(
    harness: WalledHarness,
    incumbent_fn: Callable[[str], str],
    candidate_fn: Callable[[str], str],
    per_phase: Optional[Dict[str, Tuple[float, float]]] = None,
    min_improvement: float = 0.0,
) -> MergeDecision:
    """Score both on LOCKED (spends 2 budget) and apply the end-to-end decision.

    Preflights the budget so a pair is scored atomically - never spend the
    incumbent query then fail half-way when only one query remained.
    """
    if harness.locked_budget_remaining() < 2:
        raise BudgetExhausted("need 2 LOCKED queries to evaluate a candidate pair")
    incumbent_score = harness.score_locked(incumbent_fn)
    candidate_score = harness.score_locked(candidate_fn)
    return decide_merge(incumbent_score, candidate_score, per_phase, min_improvement)

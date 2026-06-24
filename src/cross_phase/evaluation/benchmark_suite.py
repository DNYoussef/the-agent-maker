"""Wave-2 benchmark suite - the load-bearing measurement object.

Per the Wave-2 plan (and Codex's audit: "the loop is only as good as the
evaluator"), this is the single most important object. It holds deterministic,
verifiable tasks across families, partitioned into three DISJOINT splits drawn
from the same distribution:

  Split.VAL    - the cheap signal the nightly ratchet optimizes (agent may see).
  Split.LOCKED - the qualification gate: queried rarely, budget-accounted, and
                 NEVER optimized against. Exposes val-overfitting.
  Split.BLIND  - touched only at campaign end (final, outside the loop).

Because the splits are disjoint draws from one distribution, a scorer that overfit
VAL scores high on VAL but low on LOCKED/BLIND - which is how the merge gate
rejects the autoresearch single-metric overfit failure mode.

score(generate_fn, split) -> accuracy in [0, 1], fully deterministic (no RNG; the
items and their order are fixed). generate_fn(prompt: str) -> continuation text.

NOTE on walling: this class exposes full items (incl. answers) for TRUSTED callers
(the meta-gate test, the baseline runner). The restricted, answer-free, budget-
accounted view used by the editing agent in the loop is a separate wrapper
(Layer 1.4) - it must never hand the LOCKED/BLIND answers to the agent path.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List

from .evaluator import matches


class Split(Enum):
    VAL = "val"
    LOCKED = "locked"
    BLIND = "blind"


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    answer: str
    family: str


def _hash_key(item_id: str) -> str:
    """Stable pseudo-random ordering key (Python's hash() is per-process salted,
    so it cannot be used for a reproducible split)."""
    return hashlib.md5(item_id.encode("utf-8")).hexdigest()


def _generate_items() -> List[Item]:
    """Deterministic, verifiable arithmetic across three families. Breadth keeps a
    val-memorizer from generalizing; exact-answer keeps scoring objective."""
    items: List[Item] = []

    def add(family: str, a: int, b: int, result: int, op: str) -> None:
        item_id = f"{family}-{a}-{b}"
        items.append(
            Item(
                id=item_id,
                prompt=f"What is {a} {op} {b}? Answer: ",
                answer=str(result),
                family=family,
            )
        )

    for a in range(1, 9):
        for b in range(1, 9):
            add("add", a, b, a + b, "+")
    for a in range(2, 10):
        for b in range(1, a):  # keep results non-negative
            add("sub", a, b, a - b, "-")
    for a in range(2, 8):
        for b in range(2, 8):
            add("mul", a, b, a * b, "*")
    return items


class BenchmarkSuite:
    """A frozen, deterministic benchmark with disjoint val/locked/blind splits."""

    def __init__(self) -> None:
        self._by_split: Dict[Split, List[Item]] = {s: [] for s in Split}
        # Stratify PER FAMILY so each split is a same-distribution sample (an
        # item-level hash skews small families, e.g. 72/14/14 instead of 60/20/20).
        for family in sorted({it.family for it in _generate_items()}):
            members = [it for it in _generate_items() if it.family == family]
            # Pseudo-random but reproducible order, then a 60/20/20 slice.
            members.sort(key=lambda it: _hash_key(it.id))
            n = len(members)
            n_val = (n * 6) // 10
            n_locked = (n * 2) // 10
            self._by_split[Split.VAL].extend(members[:n_val])
            self._by_split[Split.LOCKED].extend(members[n_val : n_val + n_locked])
            self._by_split[Split.BLIND].extend(members[n_val + n_locked :])
        # Stable order for deterministic scoring.
        for s in Split:
            self._by_split[s].sort(key=lambda it: it.id)

    def items(self, split: Split) -> List[Item]:
        return list(self._by_split[split])

    def score(self, generate_fn: Callable[[str], str], split: Split) -> float:
        """Exact-answer accuracy in [0, 1] over the split. Deterministic."""
        items = self._by_split[split]
        if not items:
            raise ValueError(f"{split} split is empty")
        correct = sum(1 for it in items if matches(generate_fn(it.prompt), it.answer))
        return correct / len(items)

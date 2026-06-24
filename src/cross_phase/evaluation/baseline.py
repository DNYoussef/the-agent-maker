"""Wave-2 baseline runner - records the incumbent's starting score.

The greedy ratchet (Layer 2) keeps a candidate only if it beats the incumbent.
This records that incumbent line on VAL and LOCKED - the two splits the loop
reasons about. BLIND is deliberately NOT scored here: it is touched only once, at
campaign end (a final, decision-free number), so scoring it at startup would risk
contaminating it. The campaign-end BLIND eval lives in Layer 3.

It uses the trusted suite path directly: establishing the starting line is
infrastructure, not the editing agent, so the wall (which guards the AGENT from
LOCKED/BLIND answers) does not apply here.

ponytail: a plain dataclass, no persistence. The loop (Layer 2) owns the journal
file - record this and serialize it there, don't build a store the loop will own.
"""

from dataclasses import dataclass
from typing import Callable

from .benchmark_suite import BenchmarkSuite, Split


@dataclass(frozen=True)
class BaselineRecord:
    val: float
    locked: float


def record_baseline(suite: BenchmarkSuite, generate_fn: Callable[[str], str]) -> BaselineRecord:
    return BaselineRecord(
        val=suite.score(generate_fn, Split.VAL),
        locked=suite.score(generate_fn, Split.LOCKED),
    )

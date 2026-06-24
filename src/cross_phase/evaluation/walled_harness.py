"""Wave-2 walled harness - the boundary between the agent and the LOCKED gate.

Wraps a BenchmarkSuite so the autoresearch editing agent can optimize the cheap
VAL signal freely while the LOCKED qualification split stays walled off:

  - agent_answers() / val_prompts() / score_val()  -> VAL only, unmetered.
  - score_locked()                                 -> metered; HARD-STOPS at budget.

HONEST SCOPE (do not overclaim - this is an in-process Python object):
This enforces the *protocol* boundary, not a *security* boundary. It guarantees
the agent-facing methods return VAL only, and that LOCKED scoring is budget-
counted. It does NOT defend against an agent that reads source or pokes private
attributes: `_suite.items(Split.LOCKED)` and `_locked_remaining = 999` both work
from inside the process. The benchmark generator is plain repo source, so any
agent that can read files can derive the answers. The real isolation - run the
gate in a separate trusted process whose answers/budget the agent path cannot
import or mutate - is a LAYER 2 requirement (tracked), not built here.

So this object stops the HONEST agent path from trivially reading answers or
overspending the LOCKED budget; it is not a sandbox against a hostile agent.

ponytail: the budget is a plain counter, not a rate limiter. A counter is the
whole requirement (queries per campaign), upgrade to time-windowed only if the
loop ever needs LOCKED access faster than once per merge gate.
"""

from typing import Callable, Dict, List

from .benchmark_suite import BenchmarkSuite, Split


class BudgetExhausted(RuntimeError):
    """Raised when the LOCKED query budget for the campaign is spent."""


class WalledHarness:
    def __init__(self, suite: BenchmarkSuite, locked_budget: int) -> None:
        if locked_budget < 0:
            raise ValueError("locked_budget must be non-negative")
        self._suite = suite  # private: never exposed to the agent path
        self._locked_remaining = locked_budget
        self._val = suite.items(Split.VAL)

    # --- agent-facing (VAL only, unmetered) ---------------------------------
    def val_prompts(self) -> List[str]:
        return [it.prompt for it in self._val]

    def agent_answers(self) -> Dict[str, str]:
        """The only answers the agent may legitimately learn: VAL."""
        return {it.prompt: it.answer for it in self._val}

    def score_val(self, generate_fn: Callable[[str], str]) -> float:
        return self._suite.score(generate_fn, Split.VAL)

    # --- gate-facing (LOCKED, metered) --------------------------------------
    def locked_budget_remaining(self) -> int:
        return self._locked_remaining

    def score_locked(self, generate_fn: Callable[[str], str]) -> float:
        if self._locked_remaining <= 0:
            raise BudgetExhausted("LOCKED query budget exhausted for this campaign")
        self._locked_remaining -= 1
        return self._suite.score(generate_fn, Split.LOCKED)

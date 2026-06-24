"""Wave-2 baseline runner test - CRUCIBLE: the ratchet's starting line, FIRST.

The greedy ratchet needs a recorded incumbent score on VAL and LOCKED before the
loop can ask "did this candidate improve?". record_baseline measures a scorer on
those two splits through the trusted suite path (infrastructure establishing the
starting line, not the agent path - the wall guards the agent). BLIND is NOT
recorded here: it is campaign-end-only to avoid contamination.

This file must FAIL until cross_phase/evaluation/baseline.py exists.
"""

from cross_phase.evaluation.baseline import BaselineRecord, record_baseline
from cross_phase.evaluation.benchmark_suite import BenchmarkSuite, Split


def test_baseline_records_val_and_locked():
    s = BenchmarkSuite()
    answers = {}
    for sp in (Split.VAL, Split.LOCKED):
        for it in s.items(sp):
            answers[it.prompt] = it.answer

    def oracle(prompt):
        return answers[prompt]

    def wrong(prompt):
        return "nope"

    good = record_baseline(s, oracle)
    bad = record_baseline(s, wrong)

    assert isinstance(good, BaselineRecord)
    assert good.val > 0.9 and good.locked > 0.9
    assert bad.val < 0.1 and bad.locked < 0.1
    # BLIND must not be a field the baseline records.
    assert not hasattr(good, "blind")

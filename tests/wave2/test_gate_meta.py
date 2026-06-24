"""Wave-2 gate META-test - CRUCIBLE: the gate for the gate, built FIRST.

The whole autoresearch loop's safety rests on the benchmark suite. autoresearch's
documented failure mode is overfitting a single metric (the flagged Shopify case).
This test proves the suite defeats that BEFORE any loop is built:

1. the three splits (val / locked / blind) are disjoint and non-empty,
2. scoring is deterministic (same scorer + split -> same number),
3. it discriminates (an oracle beats a wrong scorer), and
4. THE KEY PROPERTY: a scorer that has overfit the VAL split (knows only val
   answers) scores high on VAL but LOW on the walled-off LOCKED split. So a merge
   gate keyed on LOCKED rejects val-overfitting - exactly the guard autoresearch
   lacks.

This file must FAIL until cross_phase/evaluation/benchmark_suite.py exists.
"""

from cross_phase.evaluation.benchmark_suite import BenchmarkSuite, Split


def _answer_map(suite, *splits):
    m = {}
    for sp in splits:
        for item in suite.items(sp):
            m[item.prompt] = item.answer
    return m


def test_splits_are_disjoint_and_nonempty():
    s = BenchmarkSuite()
    ids = {sp: {it.id for it in s.items(sp)} for sp in (Split.VAL, Split.LOCKED, Split.BLIND)}
    for sp, idset in ids.items():
        assert idset, f"{sp} split is empty"
    assert ids[Split.VAL].isdisjoint(ids[Split.LOCKED])
    assert ids[Split.VAL].isdisjoint(ids[Split.BLIND])
    assert ids[Split.LOCKED].isdisjoint(ids[Split.BLIND])


def test_every_family_present_in_every_split():
    # Per Codex: if a family is missing from LOCKED, the held-out guard is vacuous
    # for that family. Each split must be a same-distribution sample.
    s = BenchmarkSuite()
    families = {it.family for it in s.items(Split.VAL)}
    assert families, "no families found"
    for sp in (Split.VAL, Split.LOCKED, Split.BLIND):
        present = {it.family for it in s.items(sp)}
        assert present == families, f"{sp} is missing families {families - present}"


def test_numeric_spam_scorer_scores_zero():
    # The catastrophic-matcher guard: an output that lists many numbers must NOT
    # score by accident. Defeats the substring-match failure mode at the gate.
    s = BenchmarkSuite()

    def spam(prompt):
        return " ".join(str(n) for n in range(100))

    for sp in (Split.VAL, Split.LOCKED, Split.BLIND):
        assert s.score(spam, sp) == 0.0, f"numeric spam scored >0 on {sp}"


def test_scoring_is_deterministic():
    s = BenchmarkSuite()
    answers = _answer_map(s, Split.VAL, Split.LOCKED, Split.BLIND)

    def oracle(prompt):
        return answers[prompt]

    a = s.score(oracle, Split.LOCKED)
    b = s.score(oracle, Split.LOCKED)
    assert a == b, "same scorer + split must yield the same number"


def test_oracle_beats_wrong():
    s = BenchmarkSuite()
    answers = _answer_map(s, Split.VAL, Split.LOCKED, Split.BLIND)

    def oracle(prompt):
        return answers[prompt]

    def wrong(prompt):
        return "not-an-answer"

    assert s.score(oracle, Split.LOCKED) > 0.9
    assert s.score(wrong, Split.LOCKED) < 0.1
    assert s.score(oracle, Split.LOCKED) > s.score(wrong, Split.LOCKED)


def test_locked_split_catches_val_overfit():
    """The load-bearing property: val-overfitting is exposed by the locked split."""
    s = BenchmarkSuite()
    val_answers = _answer_map(s, Split.VAL)

    def val_memorizer(prompt):
        # Knows ONLY the val answers - the shape of a scorer that gamed the metric
        # the nightly ratchet optimizes. Returns garbage on anything unseen.
        return val_answers.get(prompt, "not-an-answer")

    val_score = s.score(val_memorizer, Split.VAL)
    locked_score = s.score(val_memorizer, Split.LOCKED)
    assert val_score > 0.9, "memorizer should ace the split it overfit"
    assert locked_score < 0.2, "the walled-off locked split must expose the overfit"
    assert val_score - locked_score > 0.7, "the val/locked gap is the overfit signal"

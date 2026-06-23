"""Gate for the shared evaluator: it must MEASURE (discriminate correct from wrong,
deterministically) and fail closed. This is the property the proxy/random scorers
it replaces do NOT have.
"""
import pytest

from cross_phase.evaluation.evaluator import DEFAULT_TASKS, evaluate


def test_evaluator_measures_correctness_not_noise():
    answers = {t.prompt: t.answer for t in DEFAULT_TASKS}

    def oracle(model, tok, prompt, n):
        return answers[prompt]

    def wrong(model, tok, prompt, n):
        return "0"  # none of the answers is 0

    # discriminates: oracle scores perfect, a wrong model scores low
    assert evaluate(None, None, generate_fn=oracle) == 1.0
    assert evaluate(None, None, generate_fn=wrong) < 0.5

    # deterministic (unlike random.random()-based scoring): same in -> same out
    assert evaluate(None, None, generate_fn=wrong) == evaluate(None, None, generate_fn=wrong)


def test_evaluator_partial_credit_is_proportional():
    answers = {t.prompt: t.answer for t in DEFAULT_TASKS}
    # right on exactly the first 2 of 5 tasks
    first_two = list(answers.items())[:2]

    def half(model, tok, prompt, n):
        return answers[prompt] if prompt in dict(first_two) else "nope"

    assert evaluate(None, None, generate_fn=half) == pytest.approx(2 / 5)


def test_evaluator_fail_closed_without_generate():
    class NoGen:  # no .generate and no generate_fn -> must refuse, not fabricate
        pass

    with pytest.raises(RuntimeError):
        evaluate(NoGen(), tokenizer=None)


def test_phase5_check_correctness_is_real_not_random():
    # Integration: phase5 assessment now scores via the shared evaluator instead of
    # random.random(). Must be deterministic and discriminating.
    from phase5_curriculum.assessment import EdgeOfChaosAssessment

    a = EdgeOfChaosAssessment()
    q = {"level": 5, "test_cases": [{"input": "x", "expected": "42"}]}

    assert a._check_correctness(q, "the answer is 42") is True
    assert a._check_correctness(q, "the answer is 7") is False
    # deterministic: repeat gives the same verdict (the old random code did not)
    assert a._check_correctness(q, "the answer is 42") is True
    # no ground truth -> fail-closed False, not an invented success probability
    assert a._check_correctness({"level": 5, "test_cases": []}, "anything") is False

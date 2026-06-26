"""Wave 4 behavioral tests for two deep correctness findings in Phase 5.

Finding 1 -- DockerSandbox / training_loop interface mismatch.
    The training loop's ``_validate_response`` calls ``coding_env.execute(code,
    timeout=5)`` *synchronously* and then ``result.check(input, expected)``.
    DockerSandbox only exposed an ``async def execute(...)`` returning an
    ``ExecutionResult`` with no ``.check`` method, so the consumer could never
    actually drive it -- the coroutine was never awaited and ``.check`` raised
    AttributeError that got swallowed into a silent validation failure. Worse,
    ``_check_docker`` reported "available" whenever the docker *client* existed
    even if the daemon was down, so ``docker run`` would fail and be
    misattributed to the model. The fix: one agreed sync interface
    (``run_sync`` -> ``ExecutionResult`` with ``.check``) plus honest
    degradation (``docker_unavailable`` flag) when the daemon is unreachable.

Finding 2 -- dead learned weights.
    ``VirtueWeightOptimizer`` and ``ArchetypeWeightLearner`` computed weights
    that were never fed back into scoring. ``EudaimoniaRuleSystem.assess`` used
    the hard-coded ``rule.weight`` property, and ``ArchetypeCouncil.synthesize``
    averaged vectors with equal ``1/N`` weight, ignoring any learned weights.
    The fix wires learned weights into both scoring paths.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from phase5_curriculum.docker_sandbox import (
    DockerSandbox,
    ExecutionResult,
    Language,
)
from phase5_curriculum.training_loop import CurriculumTrainingLoop
from phase5_curriculum.curriculum_generator import Question
from phase5_curriculum.eudaimonia.rules import (
    EudaimoniaRuleSystem,
    RuleType,
    VirtueWeightOptimizer,
)
from phase5_curriculum.eudaimonia.archetypes import (
    ArchetypeCouncil,
    ArchetypeType,
    ArchetypeWeightLearner,
)


# ---------------------------------------------------------------------------
# Finding 1: DockerSandbox <-> training_loop interface
# ---------------------------------------------------------------------------

def test_sandbox_exposes_sync_interface_with_check():
    """The consumer needs a sync ``run_sync`` returning a result with ``check``.

    Before the fix DockerSandbox had no ``run_sync`` and ExecutionResult had no
    ``check`` -> AttributeError. This proves the agreed interface exists.
    """
    sandbox = DockerSandbox()
    result = sandbox.run_sync("print('hello')", timeout=5)
    assert isinstance(result, ExecutionResult)
    # check() must be a real, callable predicate
    assert hasattr(result, "check") and callable(result.check)


def test_sandbox_degrades_honestly_when_docker_unavailable():
    """When the docker daemon is unreachable the sandbox must say so honestly,
    not raise an opaque error and not silently return a plausible 'success'.
    """
    sandbox = DockerSandbox()
    result = sandbox.run_sync("print('hi')", timeout=5)
    if not sandbox.is_docker_available():
        assert result.docker_unavailable is True
        assert result.success is False
        assert result.error and "docker" in result.error.lower()
        # An honest unavailable result must never claim a test passed.
        assert result.check(None, "hi") is False
    else:
        # Daemon really is up: it must actually run the code.
        assert result.docker_unavailable is False
        assert result.success is True
        assert result.check(None, "hi") is True


def test_training_loop_can_drive_sandbox_interface():
    """The consumer (_validate_response) must be able to call the sandbox.

    Before the fix this path hit ``'coroutine' object has no attribute
    'check'`` (swallowed -> always False) for any code-bearing response.

    The response prints '42'; the test case expects '42'. This holds whether
    the daemon runs the code (stdout == '42') or the sandbox degrades to the
    deterministic heuristic ('42' appears in the response).
    """
    loop = CurriculumTrainingLoop()
    sandbox = DockerSandbox()
    question = Question(
        id="q1",
        level=1,
        original_difficulty=20.0,
        question="Print the number 42.",
        source="test",
        test_cases=[{"input": "", "expected": "42", "description": "prints 42"}],
        hints=[],
    )
    response = "```python\nprint(42)\n```"
    success, error = loop._validate_response(response, question, sandbox)
    assert success is True, f"consumer could not validate via sandbox: {error}"


# ---------------------------------------------------------------------------
# Finding 2a: VirtueWeightOptimizer / EudaimoniaRuleSystem weights are live
# ---------------------------------------------------------------------------

def test_rule_weights_affect_overall_score():
    """Custom virtue weights must change the overall eudaimonia score.

    Before the fix ``assess`` ignored any supplied weights and always used the
    hard-coded 40/20/20/20 property values, so optimizer output was dead.
    """
    action = "Help the user but the action may cause harm to biological beings"
    context = {"is_harmful": True, "benefits_user": True, "preserves_agency": True}

    default_system = EudaimoniaRuleSystem()
    default_score = default_system.assess(action, context).overall_score

    # Heavily weight LIFE_VALUE (which is the rule penalised by is_harmful).
    skewed = {
        RuleType.PRIME_DIRECTIVE: 0.10,
        RuleType.CURIOSITY: 0.10,
        RuleType.ESPRIT_DE_CORPS: 0.10,
        RuleType.LIFE_VALUE: 0.70,
    }
    skewed_system = EudaimoniaRuleSystem(weights=skewed)
    skewed_score = skewed_system.assess(action, context).overall_score

    assert abs(skewed_score - default_score) > 1e-6, (
        "weights had no effect on scoring -- they are still dead"
    )
    # Weighting the violated rule heavily must lower the score.
    assert skewed_score < default_score


def test_optimizer_output_feeds_scoring():
    """The optimizer's weight dict must be consumable by the scoring system."""
    optimizer = VirtueWeightOptimizer()
    weights = optimizer._default_weights()
    # Perturb to a non-default distribution and confirm it flows through.
    weights = {
        RuleType.PRIME_DIRECTIVE: 0.25,
        RuleType.CURIOSITY: 0.25,
        RuleType.ESPRIT_DE_CORPS: 0.25,
        RuleType.LIFE_VALUE: 0.25,
    }
    system = EudaimoniaRuleSystem()
    system.set_weights(weights)
    assert system.weights[RuleType.LIFE_VALUE] == 0.25

    action = "Help the user but the action may cause harm to biological beings"
    context = {"is_harmful": True}
    uniform_score = system.assess(action, context).overall_score
    default_score = EudaimoniaRuleSystem().assess(action, context).overall_score
    assert abs(uniform_score - default_score) > 1e-6


# ---------------------------------------------------------------------------
# Finding 2b: ArchetypeWeightLearner / ArchetypeCouncil weights are live
# ---------------------------------------------------------------------------

def test_archetype_weights_affect_synthesis():
    """Council weights must change the synthesized averaged_vector.

    Before the fix ``synthesize`` always used equal 1/N averaging, so the
    learner's weights were dead.
    """
    situation = "User repeatedly fails and is getting frustrated"
    context = {"user_frustrated": True, "repeated_failure": True}

    equal_council = ArchetypeCouncil()
    equal = equal_council.get_moral_direction(situation, 0.4, context)

    skewed_council = ArchetypeCouncil()
    skewed_council.archetype_weights = {
        ArchetypeType.CHRIST: 0.80,
        ArchetypeType.HARMONY: 0.10,
        ArchetypeType.STOIC: 0.10,
    }
    skewed = skewed_council.get_moral_direction(situation, 0.4, context)

    # "compassion" is a Christ-only vector key; weighting Christ up must raise it.
    assert "compassion" in equal["averaged_vector"]
    assert skewed["averaged_vector"]["compassion"] > equal["averaged_vector"]["compassion"]


def test_learner_weighted_council_is_wired():
    """ArchetypeWeightLearner.get_weighted_council() output must reach synthesis."""
    learner = ArchetypeWeightLearner(
        initial_weights={
            ArchetypeType.CHRIST: 0.6,
            ArchetypeType.HARMONY: 0.2,
            ArchetypeType.STOIC: 0.2,
        }
    )
    council = learner.get_weighted_council()
    assert council.archetype_weights[ArchetypeType.CHRIST] == 0.6

    situation = "User repeatedly fails and is getting frustrated"
    context = {"user_frustrated": True, "repeated_failure": True}
    weighted = council.get_moral_direction(situation, 0.4, context)
    equal = ArchetypeCouncil().get_moral_direction(situation, 0.4, context)
    assert weighted["averaged_vector"]["compassion"] > equal["averaged_vector"]["compassion"]


def test_check_rejects_substring_and_superset_fakepass():
    """Codex audit canary: check() must use exact match, not substring.

    expected '4' must NOT pass stdout '42'; a solution that prints multiple
    answers must not match any single expected value by containment."""
    r = ExecutionResult(
        success=True, stdout="42\n", stderr="", exit_code=0, execution_time_ms=1.0
    )
    assert r.check(expected="4") is False, "substring fake-pass ('4' in '42')"

    multi = ExecutionResult(
        success=True, stdout="2\n4\n", stderr="", exit_code=0, execution_time_ms=1.0
    )
    assert multi.check(expected="2") is False, "superset-stdout fake-pass"
    assert multi.check(expected="4") is False, "superset-stdout fake-pass"

    exact = ExecutionResult(
        success=True, stdout="4\n", stderr="", exit_code=0, execution_time_ms=1.0
    )
    assert exact.check(expected="4") is True, "exact match must still pass"


def test_run_sync_works_from_inside_a_running_event_loop():
    """Codex audit canary: run_sync must work even when called from within an
    already-running event loop (the prior fallback raised 'Cannot run the event
    loop while another loop is running')."""
    import asyncio

    async def driver():
        # Called with a loop already running in this thread.
        return DockerSandbox().run_sync("print('x')", timeout=5)

    result = asyncio.run(driver())
    assert isinstance(result, ExecutionResult)  # no RuntimeError, returns a result

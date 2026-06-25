"""Phase 5 flow-unblock gate - CRUCIBLE.

The synthesis found Phase 5's controller always builds a truthy DockerSandbox, so
_validate_response takes the broken sandbox path (async execute() called sync + a
nonexistent ExecutionResult.check) -> every question fails -> hard wall at level 1,
yet validate_output returns True so Phases 6-8 run on an untrained model. The unblock:
the controller defaults coding_env=None (config-gated), routing to the working
deterministic heuristic validator. Full sandbox correctness is E6.
"""

from phase5_curriculum.training_loop import CurriculumTrainingLoop


def _question():
    from phase5_curriculum.curriculum_generator import Question

    return Question(
        id="q1",
        level=1,
        original_difficulty=10,
        question="Write a function double(x) that returns x*2.",
        source="test",
        test_cases=[{"input": "2", "expected": "4", "description": "double"}],
        hints=[],
    )


def test_heuristic_validation_can_succeed_with_coding_env_none():
    tl = object.__new__(CurriculumTrainingLoop)  # bypass heavy __init__; method uses only helpers
    resp = "Here is the solution:\n```python\ndef double(x):\n    return x * 2\n```\nIt doubles the input."
    ok, err = tl._validate_response(resp, _question(), coding_env=None)
    assert ok is True, f"heuristic should accept a correct, substantive answer; got {err}"


def test_truthy_sandbox_takes_the_broken_path():
    # Demonstrates WHY the default must be None: a sandbox whose execute() is async/broken
    # makes _validate_response fail closed - exactly the level-1 wall.
    tl = object.__new__(CurriculumTrainingLoop)

    class _BrokenSandbox:
        def execute(self, code, timeout=5):
            raise RuntimeError("async execute called sync")

    ok, _ = tl._validate_response(
        "def double(x): return x*2", _question(), coding_env=_BrokenSandbox()
    )
    assert ok is False


def test_phase5_controller_defaults_to_no_sandbox(monkeypatch):
    import phase5_curriculum as p5
    import phase5_curriculum.docker_sandbox as p5docker
    from cross_phase.orchestrator.phase5_controller import Phase5Controller

    captured = {}

    class _Engine:
        def __init__(self, *a, **k):
            pass

        def run(self, *a, **k):
            captured["coding_env"] = k.get("coding_env", "MISSING")
            from cross_phase.orchestrator.base_controller import PhaseResult

            return PhaseResult(True, "phase5", object(), {}, 0.0, {}, {})

    monkeypatch.setattr(p5, "CurriculumEngine", _Engine, raising=False)
    monkeypatch.setattr(p5, "OpenRouterClient", lambda *a, **k: object(), raising=False)
    monkeypatch.setattr(p5, "CurriculumConfig", lambda *a, **k: object(), raising=False)

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("DockerSandbox must NOT be constructed by default")

    monkeypatch.setattr(p5docker, "DockerSandbox", _Boom, raising=False)

    ctrl = Phase5Controller(config={}, session_id="t")
    monkeypatch.setattr(ctrl, "_get_tokenizer", lambda: object(), raising=False)
    monkeypatch.setattr(ctrl, "_build_wandb_integration", lambda: None, raising=False)
    monkeypatch.setattr(ctrl, "validate_input", lambda m: True, raising=False)

    ctrl._run_curriculum([object()])
    assert (
        captured["coding_env"] is None
    ), "controller must default coding_env=None (heuristic path)"

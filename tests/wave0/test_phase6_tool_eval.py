"""Phase 6 A-cycle: tool-use scoring must use the real SWEBenchToolEvaluator,
not substring 'tool mention' counting (which rewarded echoing tool names)."""

import torch
import torch.nn as nn


class _TinyNoGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(4, 4)

    def forward(self, x):
        return self.p(x)


class _StubTok:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, *args, **kwargs):
        return ""


def test_substring_tool_mention_scoring_is_gone():
    # The fake-signal path counted `tool in output_text.lower()`. It must not
    # exist any more, in either the optimizer or the persona B-cycle twin.
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2] / "src" / "phase6_baking" / "a_cycle_tool.py"
    )
    src = path.read_text(encoding="utf-8")
    assert "tool_mentions" not in src, "substring tool-mention scoring must be removed"
    assert "SWEBenchToolEvaluator(" in src, "default scoring must build the real evaluator"


def test_default_eval_fails_closed_without_real_data():
    # No external evaluator and no SWE-Bench data: must fail closed (honest),
    # NOT silently return a substring/synthetic score.
    from phase6_baking.a_cycle_tool import ACycleOptimizer
    from phase6_baking.swe_bench_eval import MissingSWEBenchDataError

    opt = ACycleOptimizer(tool_prompts=["use tools"], max_eval_tasks=2)
    try:
        opt._evaluate_tool_use(_TinyNoGen(), _StubTok(), evaluator=None)
        raised = False
    except MissingSWEBenchDataError:
        raised = True
    assert raised, "tool eval must fail closed without real SWE-Bench data"


def test_default_eval_uses_real_evaluator_when_synthetic_allowed():
    # With the explicit smoke-test escape hatch, the default path runs the real
    # SWEBenchToolEvaluator end-to-end (returns a [0,1] composite score) instead
    # of the substring proxy.
    from phase6_baking.a_cycle_tool import ACycleOptimizer, SWEBenchToolEvaluator

    opt = ACycleOptimizer(
        tool_prompts=["use tools"], max_eval_tasks=2, allow_synthetic_eval=True
    )
    score = opt._evaluate_tool_use(_TinyNoGen(), _StubTok(), evaluator=None)
    assert isinstance(score, float) and 0.0 <= score <= 1.0
    assert isinstance(opt._default_evaluator, SWEBenchToolEvaluator)


def test_file_load_tasks_is_idempotent(tmp_path):
    # Codex-caught: _load_from_file appended to self.tasks and reset() only
    # cleared results, so run_evaluation()'s re-load (and pre/post-bake reuse)
    # grew the task list every call. Loading must be idempotent.
    import json

    from phase6_baking.swe_bench_eval import SWEBenchEvaluator

    data = [
        {"instance_id": f"t{i}", "repo": "r", "base_commit": "c", "problem_statement": "p"}
        for i in range(3)
    ]
    path = tmp_path / "swe.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    ev = SWEBenchEvaluator(data_path=str(path))
    n1 = len(ev.load_tasks(max_tasks=3))
    n2 = len(ev.load_tasks(max_tasks=3))
    assert n1 == 3 and n2 == 3, "repeated load_tasks must not grow the task list"

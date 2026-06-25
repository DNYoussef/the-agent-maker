"""E7 gate - CRUCIBLE: Phase 6 baking honesty.

Synthesis: validate_output ignored result.success (a failed bake with total_iterations>=1
shipped green); persona scoring decoded the FULL generate() output (prompt + completion),
so trait words IN THE PROMPT fake-passed the persona test. E7 gates on success and scores
only the generated continuation.
"""

import torch
import torch.nn as nn

from cross_phase.orchestrator.base_controller import PhaseResult
from cross_phase.orchestrator.phase6_controller import Phase6Controller
from phase6_baking.b_cycle_persona import BCycleOptimizer, PersonaTask


def _result(success, iterations=5):
    return PhaseResult(success, "phase6", object(), {"total_iterations": iterations}, 0.0, {}, {})


def test_validate_output_rejects_failed_bake():
    c = Phase6Controller(config={}, session_id="t")
    assert c.validate_output(_result(success=True)) is True
    assert c.validate_output(_result(success=False)) is False, "failed bake must not ship green"


class _PromptHasTraitModel(nn.Module):
    """generate() returns prompt ids (1,2,3) + continuation ids (9,9). The trait word lives
    in the PROMPT, not the continuation."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def generate(self, input_ids=None, **kw):
        return torch.tensor([[1, 2, 3, 9, 9]])


class _Tok:
    def __call__(self, text, **kw):
        return {"input_ids": torch.tensor([[1, 2, 3]])}  # prompt = 3 tokens

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        # token 1 == the trait word, only present in the full sequence (the prompt)
        return "honest" if 1 in ids else "blah"


def test_persona_scores_continuation_not_prompt():
    opt = object.__new__(BCycleOptimizer)
    opt.persona_tasks = [PersonaTask(prompt="be honest", expected_traits=["honest"], difficulty=1)]
    score = opt._evaluate_persona(_PromptHasTraitModel(), _Tok())
    assert score == 0.0, "trait words in the prompt must not count; score only the continuation"

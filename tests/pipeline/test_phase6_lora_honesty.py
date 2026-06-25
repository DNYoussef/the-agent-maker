"""Tier2-P6 gate - CRUCIBLE: Phase 6 A-cycle is honest about LoRA.

Synthesis: lora_r/lora_alpha were stored but never used - the optimizer is full-parameter
AdamW, mislabeled as LoRA. P6 surfaces uses_lora=False and fails loud if a caller asks for
LoRA (use_lora=True) rather than silently doing full fine-tuning under a LoRA label.
"""

import pytest

from phase6_baking.a_cycle_tool import ACycleOptimizer


def test_reports_no_lora():
    opt = ACycleOptimizer(tool_prompts=["use the calculator tool"])
    assert opt.uses_lora is False, "A-cycle does full-parameter AdamW; must not claim LoRA"


def test_use_lora_fails_loud():
    with pytest.raises(NotImplementedError):
        ACycleOptimizer(tool_prompts=["x"], use_lora=True)

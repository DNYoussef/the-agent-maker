"""E2 gate - CRUCIBLE: phases 5-8 consume the threaded tokenizer, not a hardcoded gpt2.

The synthesis found phases 5-8 each fabricate get_tokenizer('gpt2'), so if Phase 1's
vocab != gpt2 every downstream tokenize/generate/CE-loss runs on wrong ids. E0 added the
carry slot (controller.input_tokenizer); E2 makes each phase prefer it, falling back to
gpt2 only when run standalone (no upstream tokenizer).
"""

import pytest

from cross_phase.orchestrator.phase5_controller import Phase5Controller
from cross_phase.orchestrator.phase6_controller import Phase6Controller
from cross_phase.orchestrator.phase7_controller import Phase7Controller
from cross_phase.orchestrator.phase8_controller import Phase8Controller

CONTROLLERS = [Phase5Controller, Phase6Controller, Phase7Controller, Phase8Controller]


@pytest.mark.parametrize("ctrl_cls", CONTROLLERS)
def test_phase_uses_threaded_tokenizer(ctrl_cls):
    c = ctrl_cls(config={}, session_id="t")
    sentinel = object()
    c.input_tokenizer = sentinel
    assert c._get_tokenizer() is sentinel, f"{ctrl_cls.__name__} ignored the threaded tokenizer"


@pytest.mark.parametrize("ctrl_cls", CONTROLLERS)
def test_phase_falls_back_to_gpt2_when_standalone(ctrl_cls):
    c = ctrl_cls(config={}, session_id="t")
    c.input_tokenizer = None
    tok = c._get_tokenizer()
    assert tok is not None, f"{ctrl_cls.__name__} returned no fallback tokenizer"

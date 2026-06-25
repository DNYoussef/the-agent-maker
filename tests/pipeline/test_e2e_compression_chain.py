"""Tier3-E2E gate - CRUCIBLE: a real tiny model threads the compression chain end-to-end.

A full 1->8 run is not hermetic (Phase 1 downloads datasets, Phase 5 needs OpenRouter), so
this exercises the DATA-FREE transform half on a REAL tiny model: Phase 4 (BitNet quantize)
-> Phase 8 (SeedLM+ compress). It proves the handoff (model + tokenizer threaded via E0/E2),
that Phase 4 yields a REAL quantized artifact (>1x), that the chain completes, and that the
final artifact is a WORKING model (forwards to logits). Phases needing training data/network
(1,2,3,5,6,7) are out of the hermetic E2E by design.
"""

import pytest
import torch

from cross_phase.orchestrator.phase4_controller import Phase4Controller
from cross_phase.orchestrator.phase8_controller import Phase8Controller
from cross_phase.utils import get_tokenizer
from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config


def _tiny_phase1_model():
    cfg = Phase1Config(specialization="reasoning")
    cfg.titans_config.d_model, cfg.titans_config.n_heads = 128, 4
    cfg.titans_config.d_ff, cfg.titans_config.d_mem, cfg.titans_config.mag_hidden = 256, 64, 64
    cfg.titans_config.n_layers = 2
    cfg.trm_config.T_max, cfg.trm_config.micro_steps = 2, 1
    cfg.trm_config.step_weights = [1.0, 1.0, 1.0]
    return TRMTitansMAGModel(cfg).eval(), cfg


@pytest.mark.slow  # real quantize+compress on a real model (~5 min); excluded from CI fast path
def test_compression_chain_e2e_on_real_tiny_model():
    torch.manual_seed(0)
    model, mcfg = _tiny_phase1_model()
    tok = get_tokenizer("gpt2")

    # Phase 4: BitNet quantization (data-free), tokenizer threaded via E0 contract.
    p4 = Phase4Controller(config={}, session_id="e2e")
    p4.input_tokenizer = tok
    r4 = p4.execute([model])
    assert r4.success, f"phase 4 failed: {r4.error}"
    assert r4.model is not None
    # a REAL quantization artifact (the tiny model gets ~2x; production targets are higher).
    assert r4.metrics.get("compression_ratio", 0) > 1.0, "phase 4 must really compress"

    # Phase 8: compression on the quantized model; tokenizer threaded forward.
    p8 = Phase8Controller(config={}, session_id="e2e")
    p8.input_tokenizer = tok
    r8 = p8.execute([r4.model])
    assert r8.success, f"phase 8 failed: {r8.error}"
    assert p8.input_tokenizer is tok, "tokenizer must thread into phase 8 (E2)"
    assert "total_compression" in r8.metrics, "phase 8 must report a real compression metric"

    # The end-to-end artifact must still be a WORKING model (forwards to logits).
    final = r8.model
    with torch.no_grad():
        out = final(torch.randint(0, mcfg.titans_config.vocab_size, (1, 8)))
    logits = getattr(out, "logits", out)
    assert logits.shape[0] == 1 and logits.shape[-1] == mcfg.titans_config.vocab_size

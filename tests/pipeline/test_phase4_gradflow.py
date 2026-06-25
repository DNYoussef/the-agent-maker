"""Tier2-P4 gate - CRUCIBLE: Phase 4 gradient-flow gate works for real LMs.

Synthesis: test_gradient_flow fed float torch.randn(1,512) to a token-embedding model
(needs long input_ids) and treated the output as a tensor (it's a ModelOutput). So it
falsely reported gradient-flow FAILURE for every real LM. P4 feeds long token ids and reads
.logits. (Note: _ste_finetune is an HONEST 'skipped MVP' stub needing training data, not
theater; the BitNet quantize/serialize correctness was fixed in E5.)
"""

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from phase4_bitnet.utils import test_gradient_flow as gradient_flow_check  # alias: not a test


def _small_model():
    cfg = Phase1Config(specialization="reasoning")
    cfg.titans_config.d_model, cfg.titans_config.n_heads = 128, 4
    cfg.titans_config.d_ff, cfg.titans_config.d_mem, cfg.titans_config.mag_hidden = 256, 64, 64
    cfg.titans_config.n_layers = 2
    cfg.trm_config.T_max, cfg.trm_config.micro_steps = 2, 1
    cfg.trm_config.step_weights = [1.0, 1.0, 1.0]
    return TRMTitansMAGModel(cfg).eval()


def test_gradient_flow_passes_for_token_embedding_model():
    model = _small_model()
    ok, err = gradient_flow_check(model, device="cpu")
    assert ok is True, f"gradient flow should pass for a token-embedding LM; got: {err}"


def test_gradient_flow_reports_no_grad_model():
    # A model whose params don't require grad must honestly report failure.
    model = _small_model()
    for p in model.parameters():
        p.requires_grad_(False)
    ok, _ = gradient_flow_check(model, device="cpu")
    assert ok is False

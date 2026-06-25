"""F0 gate - CRUCIBLE: Phase 1 model exposes a usable API for downstream phases.

The synthesis 1->2 break: full_model.forward returns a plain dict and has no .generate, so
Phase 2 fitness (model(input_ids)->logits tensor), Phase 3 anti-theater / Phase 5-7
generation all silently degrade. F0 makes the output expose .logits (HF-style) WITHOUT
breaking existing dict consumers, and adds a real autoregressive .generate.
"""

import torch

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config


def _small_model():
    cfg = Phase1Config(specialization="reasoning")
    cfg.titans_config.d_model, cfg.titans_config.n_heads = 128, 4
    cfg.titans_config.d_ff, cfg.titans_config.d_mem, cfg.titans_config.mag_hidden = 256, 64, 64
    cfg.titans_config.n_layers = 2
    cfg.trm_config.T_max, cfg.trm_config.micro_steps = 2, 1
    cfg.trm_config.step_weights = [1.0, 1.0, 1.0]
    return TRMTitansMAGModel(cfg).eval(), cfg


def test_forward_output_exposes_logits_attr_and_dict():
    model, cfg = _small_model()
    ids = torch.randint(0, cfg.titans_config.vocab_size, (2, 8))
    with torch.no_grad():
        out = model(ids)
    # attribute access (what HF-style consumers expect: outputs.logits)
    assert out.logits.shape == (2, 8, cfg.titans_config.vocab_size)
    # dict access still works (backwards compatible - existing callers use out["logits"])
    assert out["logits"] is out.logits
    assert "halting_steps" in out and out.halting_steps is out["halting_steps"]


def test_generate_produces_valid_tokens():
    model, cfg = _small_model()
    ids = torch.randint(0, cfg.titans_config.vocab_size, (1, 5))
    gen = model.generate(ids, max_new_tokens=6, do_sample=False)
    assert gen.shape == (1, 11), f"expected [1, 5+6], got {tuple(gen.shape)}"
    assert gen.dtype == torch.long
    assert int(gen.max()) < cfg.titans_config.vocab_size and int(gen.min()) >= 0
    assert torch.equal(gen[:, :5], ids), "generate must preserve the prompt prefix"


def test_generate_is_deterministic_greedy():
    model, cfg = _small_model()
    ids = torch.randint(0, cfg.titans_config.vocab_size, (1, 4))
    a = model.generate(ids, max_new_tokens=5, do_sample=False)
    b = model.generate(ids, max_new_tokens=5, do_sample=False)
    assert torch.equal(a, b), "greedy generate must be deterministic"


def test_generate_honors_max_length():
    # HF callers (e.g. benchmarks.py) pass max_length (TOTAL length), not max_new_tokens.
    model, cfg = _small_model()
    ids = torch.randint(0, cfg.titans_config.vocab_size, (1, 5))
    gen = model.generate(ids, max_length=8, do_sample=False)
    assert gen.shape[1] == 8, "max_length must cap the TOTAL sequence length"


def test_sample_rejects_nonpositive_temperature():
    import pytest

    model, cfg = _small_model()
    ids = torch.randint(0, cfg.titans_config.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        model.generate(ids, max_new_tokens=2, do_sample=True, temperature=0.0)


def test_generate_windows_context_past_max_seq_len():
    # generate must not crash when the running sequence exceeds the model's max_seq_len
    # (forward truncates internally; generate windows the most-recent tokens).
    model, cfg = _small_model()
    max_len = cfg.titans_config.max_seq_len
    ids = torch.randint(0, cfg.titans_config.vocab_size, (1, max_len - 1))
    gen = model.generate(ids, max_new_tokens=4, do_sample=False)
    assert gen.shape[1] == (max_len - 1) + 4

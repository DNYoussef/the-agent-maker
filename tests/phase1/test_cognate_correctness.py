"""Cognate Phase 3 correctness gate - CRUCIBLE, fail-first.

Locks the correctness bugs from the Codex audit that a causal LM must not have:
- the LM loss must be NEXT-token (shifted), not same-position (which trains the model
  to copy the token it already sees via causal self-attention),
- the LTM memory must not carry state across independent forwards (cross-batch leak /
  order dependence),
- a sequence longer than max_seq_len must not crash on a label/logits shape mismatch,
- the ACT EMA buffers must size to the configured recursion depth (no hardcoded 10).
"""

import torch
import torch.nn.functional as F

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import ACTConfig, Phase1Config, TRMConfig


def _cfg(**trm):
    cfg = Phase1Config(specialization="reasoning")
    if trm:
        cfg.trm_config = TRMConfig(**trm)
    return cfg


def test_lm_loss_is_next_token_shifted():
    # deep_supervision off -> a single clean CE we can check against the shifted target.
    cfg = _cfg(T_max=2, micro_steps=1, deep_supervision=False, step_weights=[1.0, 1.0, 1.0])
    torch.manual_seed(0)
    m = TRMTitansMAGModel(cfg).eval()
    vocab = cfg.titans_config.vocab_size
    ids = torch.randint(0, vocab, (2, 16))
    m.reset_memory()
    out = m(ids, labels=ids)  # labels == input_ids (the natural causal-LM convention)
    logits = out["logits"]
    shifted = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1))
    unshifted = F.cross_entropy(logits.reshape(-1, vocab), ids.reshape(-1))
    assert torch.allclose(out["loss_ce"], shifted, atol=1e-5), "loss_ce must be NEXT-token CE"
    assert not torch.allclose(shifted, unshifted), "sanity: shifted vs unshifted should differ"


def test_memory_is_stateless_across_forwards():
    # Two forwards on the SAME input (no manual reset) must give identical logits;
    # otherwise prior-batch memory leaks into this batch (order-dependent training).
    cfg = _cfg()
    torch.manual_seed(0)
    m = TRMTitansMAGModel(cfg).eval()
    vocab = cfg.titans_config.vocab_size
    ids = torch.randint(0, vocab, (2, 16))
    with torch.no_grad():
        a = m(ids)["logits"]
        b = m(ids)["logits"]
    assert torch.allclose(a, b, atol=1e-6), "forward is not stateless: LTM leaks across calls"


def test_long_sequence_truncates_and_loss_is_finite():
    cfg = _cfg(T_max=2, micro_steps=1, deep_supervision=False, step_weights=[1.0, 1.0, 1.0])
    max_len = cfg.titans_config.max_seq_len
    torch.manual_seed(0)
    m = TRMTitansMAGModel(cfg).eval()
    vocab = cfg.titans_config.vocab_size
    ids = torch.randint(0, vocab, (1, max_len + 8))  # longer than max_seq_len
    m.reset_memory()
    out = m(ids, labels=ids)  # must not raise a CE/ACT shape mismatch
    assert torch.isfinite(out["loss"]), "loss must be finite (not nan) after truncation"
    # loss_ce must equal the shifted CE over the TRUNCATED ids.
    trunc = ids[:, :max_len]
    shifted = F.cross_entropy(out["logits"][:, :-1].reshape(-1, vocab), trunc[:, 1:].reshape(-1))
    assert torch.allclose(out["loss_ce"], shifted, atol=1e-5)


def test_degenerate_supervision_does_not_nan():
    # seq_len==1 (no next token) and an all-padding label row must give a finite, zero
    # CE - not nan (which would poison training).
    cfg = _cfg(T_max=2, micro_steps=1, deep_supervision=False, step_weights=[1.0, 1.0, 1.0])
    torch.manual_seed(0)
    m = TRMTitansMAGModel(cfg).eval()
    vocab = cfg.titans_config.vocab_size
    m.reset_memory()
    out1 = m(torch.randint(0, vocab, (1, 1)), labels=torch.randint(0, vocab, (1, 1)))
    assert torch.isfinite(out1["loss"]) and out1["loss_ce"].item() == 0.0
    ids = torch.randint(0, vocab, (1, 8))
    all_pad = torch.full((1, 8), -100)
    m.reset_memory()
    out2 = m(ids, labels=all_pad)
    assert torch.isfinite(out2["loss"]) and out2["loss_ce"].item() == 0.0


def test_act_buffers_size_to_recursion_depth():
    # T_max=10 -> step index 10 exists; ACT EMA buffers must not be hardcoded to 10 slots.
    act = ACTConfig()
    if hasattr(act, "max_steps"):
        pass
    cfg = _cfg(T_max=10, micro_steps=1, deep_supervision=False, step_weights=[1.0] * 11)
    torch.manual_seed(0)
    m = TRMTitansMAGModel(cfg)
    vocab = cfg.titans_config.vocab_size
    ids = torch.randint(0, vocab, (1, 8))
    out = m(ids, labels=ids)  # must not IndexError in the ACT EMA buffers
    out["loss"].backward()

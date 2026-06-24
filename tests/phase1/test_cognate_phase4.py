"""Cognate Phase 4 gate - CRUCIBLE fail-first: the cheap correctness fixes.

The big efficiency items (O(seq^2) TRM attention, gradient checkpointing, a fused
sliding-window kernel) are genuine Wave-2 architecture work and stay documented, not
rewritten here. These are the cheap, real fixes:
- the "speed" specialization must halt EARLIEST (lowest threshold), not latest (#13),
- bf16 autocast must not crash on the ACT BCE loss (#16/bf16),
- an ambiguous [batch, seq] attention mask must be rejected with a clear error, not a
  confusing downstream shape crash (#1).
"""

import pytest
import torch

from phase1_cognate.model.components.attention import SlidingWindowAttention
from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config


def test_speed_halts_earliest_reasoning_latest():
    # halt fires when q > threshold, so a LOWER threshold halts sooner. "speed" must
    # have the lowest threshold and "reasoning" the highest.
    cfg = Phase1Config()
    th = cfg.act_thresholds
    assert th["speed"] < th["memory"] < th["reasoning"], f"halt thresholds mis-ordered: {th}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast needs a GPU")
def test_bf16_autocast_does_not_crash():
    cfg = Phase1Config()
    cfg.trm_config.T_max, cfg.trm_config.micro_steps = 2, 1
    cfg.trm_config.step_weights = [1.0, 1.0, 1.0]
    torch.manual_seed(0)
    m = TRMTitansMAGModel(cfg).to("cuda")
    vocab = cfg.titans_config.vocab_size
    ids = torch.randint(0, vocab, (1, 16), device="cuda")
    m.reset_memory()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(ids, labels=ids)
    out["loss"].backward()  # must not raise "binary_cross_entropy not implemented for BFloat16"
    assert torch.isfinite(out["loss"])


def test_batch_seq_mask_rejected_clearly():
    # A [batch, seq] key-padding mask is ambiguous with [seq, seq]; reject it with a
    # clear error instead of silently mis-slicing (Codex #1).
    attn = SlidingWindowAttention(d_model=64, n_heads=1, window=4, dropout=0.0)
    x = torch.randn(2, 5, 64)
    bad = torch.ones(2, 5)  # [batch, seq] - ambiguous, non-square
    with pytest.raises(ValueError, match="square"):
        attn(x, mask=bad)
    # A valid square [seq, seq] mask is accepted (the supported 2-D contract).
    good = torch.ones(5, 5)
    out = attn(x, mask=good)
    assert out.shape == (2, 5, 64)

"""Cognate Wave-2 efficiency gate - CRUCIBLE. Output-preserving optimisation.

For an optimisation the gate that FLIPS is memory (a previously-OOM config must now
fit); the correctness probes are EQUIVALENCE GUARDS that must stay green BEFORE and
AFTER, proving the speedup did not change outputs:

  GUARD  attention output == an independent banded-causal reference (the vectorised
         SWA must compute exactly the windowed-causal attention the loop did),
  GUARD  gradient_checkpointing ON == OFF for both loss AND gradients (checkpointing
         only trades compute for memory; it must not change the math),
  FAIL-FIRST  a train step at b4/s256 (which OOMs on the per-token-loop attention with
         no real checkpointing) fits under an 8 GB budget after the optimisation.
"""

import pytest
import torch

from phase1_cognate.model.components.attention import SlidingWindowAttention
from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config


def _loop_reference(attn: SlidingWindowAttention, x: torch.Tensor) -> torch.Tensor:
    """The OLD per-token-loop attention, kept here as the equivalence ORACLE: for each
    position, softmax over only its local window slice, @V. The vectorised impl must
    match this exactly (eval / dropout=0)."""
    b, s, _ = x.shape
    q = attn.w_q(x).view(b, s, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = attn.w_k(x).view(b, s, attn.n_heads, attn.head_dim).transpose(1, 2)
    v = attn.w_v(x).view(b, s, attn.n_heads, attn.head_dim).transpose(1, 2)
    wh = attn.window // 2
    out = torch.zeros_like(q)
    for pos in range(s):
        start = max(0, pos - wh)
        end = pos + 1 if attn.causal else min(s, pos + wh + 1)
        sc = (q[:, :, pos : pos + 1, :] @ k[:, :, start:end, :].transpose(-2, -1)) * attn.scale
        out[:, :, pos : pos + 1, :] = sc.softmax(dim=-1) @ v[:, :, start:end, :]
    return attn.w_o(out.transpose(1, 2).reshape(b, s, attn.d_model))


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("window", [5, 6])  # odd / even
@pytest.mark.parametrize("seq", [4, 13])  # shorter / longer than the window
def test_vectorized_attention_equals_old_loop(causal, window, seq):
    torch.manual_seed(0)
    attn = SlidingWindowAttention(d_model=64, n_heads=4, window=window, dropout=0.0, causal=causal)
    attn.eval()
    x = torch.randn(2, seq, 64)
    with torch.no_grad():
        got = attn(x)
        ref = _loop_reference(attn, x)
    assert torch.allclose(got, ref, atol=1e-5), f"max diff {(got - ref).abs().max().item()}"


def _small_cfg():
    cfg = Phase1Config(specialization="reasoning")
    cfg.titans_config.d_model, cfg.titans_config.n_heads = 256, 4
    cfg.titans_config.d_ff, cfg.titans_config.d_mem, cfg.titans_config.mag_hidden = 512, 128, 128
    cfg.titans_config.n_layers = 3
    cfg.trm_config.T_max, cfg.trm_config.micro_steps = 2, 1
    cfg.trm_config.step_weights = [1.0, 1.0, 1.0]
    return cfg


def test_gradient_checkpointing_preserves_loss_and_grads():
    vocab = 50257
    ids = torch.randint(0, vocab, (2, 24))

    def run(gc):
        cfg = _small_cfg()
        cfg.gradient_checkpointing = gc
        cfg.titans_config.gradient_checkpointing = gc
        torch.manual_seed(0)
        m = TRMTitansMAGModel(cfg).train()
        out = m(ids, labels=ids)
        out["loss"].backward()
        g = m.backbone.token_emb.weight.grad.clone()
        return out["loss"].detach(), g

    loss_on, g_on = run(True)
    loss_off, g_off = run(False)
    assert torch.allclose(loss_on, loss_off, atol=1e-4), "checkpointing changed the loss"
    assert torch.allclose(g_on, g_off, atol=1e-4), "checkpointing changed the gradients"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="memory gate needs a GPU")
def test_fits_b4_s256_on_8gb_after_optimisation():
    # b4/s256 OOMs on the per-token-loop attention with no real checkpointing (~17 GB).
    # Vectorised SWA + gradient checkpointing must bring a real train step under 8 GB.
    cfg = Phase1Config(specialization="reasoning")  # full 222M flagship
    cfg.gradient_checkpointing = True
    cfg.titans_config.gradient_checkpointing = True
    torch.manual_seed(0)
    m = TRMTitansMAGModel(cfg).to("cuda").train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, cfg.titans_config.vocab_size, (4, 256), device="cuda")
    m(ids, labels=ids)["loss"].backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated()
    assert peak < 7.5e9, f"b4/s256 train step peak {peak/1e9:.2f}GB exceeds 7.5GB budget"

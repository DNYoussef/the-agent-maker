"""Cognate resize gate - CRUCIBLE acceptance gate, built FIRST.

Verifies the phase1 Cognate (TRM x Titans-MAG) resize to a ~222M flagship and the
properties the resize must hold. Codex-audited: asserts EXACT geometry + a tight
param window (a loose band would pass wrong architectures), both TRM attention
modules, and a REAL on-GPU memory probe at the honestly-measured trainable config
(b2/s128 fp32). Causal (no-future-leak) is enforced across seeds/positions (Phase 2).

Hermetic: builds a fresh model from config, no checkpoints, no shared state.
"""

import pytest
import torch

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config

# Exact, measured geometry of the resized flagship (not a loose band).
EXPECT = dict(
    d_model=896, n_layers=12, n_heads=14, head_dim=64, d_ff=3584, d_mem=448, mag_hidden=448
)
EXPECT_PARAMS = 222_323_137
# Honestly measured: 222M trains on an 8 GB GPU only at b2/s128 fp32 (peak ~4.9 GB).
# Larger batch/seq and the bf16 path are blocked on efficiency work (gradient
# checkpointing, the O(seq^2) TRM attention, and a BCE-under-autocast bug) - a later phase.
TRAIN_B, TRAIN_S, TRAIN_VRAM_CAP = 2, 128, 6.0e9


def _model():
    cfg = Phase1Config(specialization="reasoning")
    torch.manual_seed(0)
    return cfg, TRMTitansMAGModel(cfg)


def test_exact_geometry():
    cfg, _ = _model()
    tc = cfg.titans_config
    got = dict(
        d_model=tc.d_model,
        n_layers=tc.n_layers,
        n_heads=tc.n_heads,
        head_dim=tc.head_dim,
        d_ff=tc.d_ff,
        d_mem=tc.d_mem,
        mag_hidden=tc.mag_hidden,
    )
    assert got == EXPECT, f"geometry drifted: {got} != {EXPECT}"
    assert tc.head_dim == tc.d_model // tc.n_heads == 64
    assert tc.vocab_size == 50257, "GPT-2 vocab must be kept for pipeline compatibility"


def test_param_count_exact():
    _, m = _model()
    n = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    assert n == EXPECT_PARAMS, f"{n:,} != expected {EXPECT_PARAMS:,}"


def test_forward_shape_and_backward_grads():
    cfg, m = _model()
    vocab = cfg.titans_config.vocab_size
    ids = torch.randint(0, vocab, (2, 16))
    labels = torch.randint(0, vocab, (2, 16))
    out = m(ids, labels=labels)
    assert out["logits"].shape == (2, 16, vocab)
    out["loss"].backward()
    assert m.act_head.w_halt.weight.grad.abs().sum() > 0, "ACT head got zero gradient"
    assert m.backbone.token_emb.weight.grad.abs().sum() > 0, "embedding got zero gradient"


def test_trm_both_attentions_track_d_model():
    # Scale-up hazard: TRM must not hardcode a head count; BOTH attention modules must
    # use head_dim 64 (feature_attn AND answer_attn - answer_attn must not drift).
    cfg, m = _model()
    d = cfg.titans_config.d_model
    for name in ("feature_attn", "answer_attn"):
        nh = getattr(m.trm.refiner, name).num_heads
        assert d % nh == 0 and d // nh == 64, f"TRM {name}: {nh} heads -> head_dim {d // nh} != 64"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="memory gate needs a GPU")
def test_trains_on_8gb_at_measured_config():
    # REAL memory gate (not a fictional batch16/seq512 claim): one fp32 train step at
    # the honestly-measured trainable config must fit well under 8 GB.
    cfg, m = _model()
    m = m.to("cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    vocab = cfg.titans_config.vocab_size
    ids = torch.randint(0, vocab, (TRAIN_B, TRAIN_S), device="cuda")
    labels = torch.randint(0, vocab, (TRAIN_B, TRAIN_S), device="cuda")
    m(ids, labels=labels)["loss"].backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated()
    assert (
        peak < TRAIN_VRAM_CAP
    ), f"train step peak {peak/1e9:.2f}GB exceeds cap {TRAIN_VRAM_CAP/1e9:.1f}GB"


def test_no_future_token_leakage():
    # Causality, proven across several seeds and perturbation positions (not one
    # black-box poke): perturbing token j must never change logits at any position < j.
    # Memory is reset per forward to isolate WITHIN-sequence causality; the cross-forward
    # memory-state leak is a separate Phase-3 bug (#10).
    cfg, m = _model()
    m.eval()
    vocab = cfg.titans_config.vocab_size
    for seed in (1, 7, 42):
        for j in (4, 8, 11):  # perturb an interior and the last position
            torch.manual_seed(seed)
            ids = torch.randint(0, vocab, (1, 12))
            with torch.no_grad():
                m.reset_memory()
                base = m(ids)["logits"]
                ids2 = ids.clone()
                ids2[0, j] = (ids2[0, j] + 7) % vocab
                m.reset_memory()
                pert = m(ids2)["logits"]
            delta = (base[0, :j] - pert[0, :j]).abs().max().item() if j > 0 else 0.0
            assert (
                delta < 1e-5
            ), f"seed{seed} perturb@{j}: earlier logits moved by {delta} (not causal)"

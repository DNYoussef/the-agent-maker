"""Wave-0 capstone canary: a REAL (non-mock) training step must actually reduce
loss. This wires the real Phase-1 model (Titans-MAG backbone + EMA memory + TRM +
real ACT loss + deep supervision), the real cross-entropy, and the real
MuGrokfast optimizer (Muon on 2-D params + real AdamW on 1-D) through real
autograd, and overfits a fixed tiny batch. If any of those links were stubbed or
detached, the loss would not move.

This is the gate the earlier crash-only canary deliberately did not cover: it
proves the pipeline LEARNS, not merely that it runs without crashing."""

import torch

from cross_phase.mugrokfast.optimizer import MuonGrokfast
from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import ACTConfig, Phase1Config, TitansMAGConfig, TRMConfig


def _tiny_config():
    titans = TitansMAGConfig(
        d_model=64,
        n_layers=1,
        n_heads=1,
        head_dim=64,
        d_ff=64,
        vocab_size=32,
        max_seq_len=32,
        sw_window=16,
        d_mem=32,
        mag_hidden=32,
    )
    trm = TRMConfig(T_max=1, micro_steps=1, step_weights=[0.5, 1.0])
    return Phase1Config(titans_config=titans, trm_config=trm, act_config=ACTConfig(), device="cpu")


def test_real_training_step_reduces_loss():
    torch.manual_seed(0)
    model = TRMTitansMAGModel(_tiny_config())
    model.train()

    optimizer = MuonGrokfast(
        model.parameters(), muon_lr=0.02, fallback_lr=0.02, grokfast_lambda=0.0
    )

    input_ids = torch.randint(0, 32, (2, 8))
    labels = torch.randint(0, 32, (2, 8))

    losses = []
    for _ in range(40):
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first, last = losses[0], losses[-1]
    assert all(l == l for l in losses), "loss must never be NaN"
    # Real learning on an overfit batch: loss must fall well below the start.
    assert (
        last < first * 0.5
    ), f"real training must reduce loss (first={first:.3f}, last={last:.3f})"

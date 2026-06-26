"""
Wave 4 deep-ML-correctness test: the ACT head must actually train.

Defect (pre-fix): full_model.forward built loss_act from `halting_steps`,
an integer step-count tensor produced by threshold comparisons in
_compute_halting_steps. Threshold comparison (q > t) is non-differentiable,
so loss_act.backward() delivered NO gradient to the halting Linear
(act_head.w_halt). The "learned halting" never learned.

Fix: build loss_act from the differentiable per-step halting PROBABILITIES
(sigmoid outputs of w_halt) via a PonderNet/ACT-style expected-steps
(ponder) cost, so gradient flows to w_halt.

This test does one forward + backward on a tiny model and asserts the
halting Linear weight received a real (non-None, non-zero) gradient.
"""

import torch

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import ACTConfig, Phase1Config, TitansMAGConfig, TRMConfig


def _tiny_config() -> Phase1Config:
    """Smallest config that satisfies all dataclass invariants."""
    titans = TitansMAGConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        sw_window=32,
        d_mem=32,
        mag_hidden=32,
    )
    trm = TRMConfig(T_max=2, micro_steps=1, step_weights=[0.5, 0.75, 1.0])
    act = ACTConfig()
    return Phase1Config(
        titans_config=titans,
        trm_config=trm,
        act_config=act,
        specialization="reasoning",
    )


def test_act_halting_weight_receives_gradient():
    torch.manual_seed(0)
    model = TRMTitansMAGModel(_tiny_config())
    model.train()

    batch, seq = 2, 16
    input_ids = torch.randint(0, 256, (batch, seq))
    labels = torch.randint(0, 256, (batch, seq))

    out = model(input_ids, labels=labels)
    assert "loss" in out
    model.zero_grad(set_to_none=True)
    out["loss"].backward()

    w = model.act_head.w_halt.weight
    assert w.grad is not None, "act_head.w_halt.weight.grad is None: halting param gets no gradient"
    gnorm = w.grad.abs().sum().item()
    assert (
        gnorm > 0.0
    ), f"act_head.w_halt.weight.grad is all-zero (sum={gnorm}): halting never learns"


def test_act_loss_is_differentiable_and_nonconstant():
    """loss_act must depend on w_halt (i.e. be part of the autograd graph),
    not be a detached constant built from integer step counts."""
    torch.manual_seed(0)
    model = TRMTitansMAGModel(_tiny_config())
    model.train()

    input_ids = torch.randint(0, 256, (2, 16))
    labels = torch.randint(0, 256, (2, 16))
    out = model(input_ids, labels=labels)

    loss_act = out["loss_act"]
    assert loss_act.requires_grad, "loss_act is detached (no grad_fn): not differentiable"

    # Gradient of loss_act alone must reach w_halt.
    model.zero_grad(set_to_none=True)
    loss_act.backward()
    g = model.act_head.w_halt.weight.grad
    assert (
        g is not None and g.abs().sum().item() > 0.0
    ), "loss_act does not propagate gradient to the halting Linear"

"""MuGrokfast: the 1-D-param fallback must be real AdamW (moment estimates +
bias correction), not plain SGD mislabeled as AdamW."""

import torch

from cross_phase.mugrokfast.optimizer import MuonGrokfast


def test_adamw_fallback_uses_moments_not_sgd():
    # A 1-D parameter routes to _adamw_update. With grokfast filtering disabled
    # (lambda=0) the filtered grad equals the raw grad, so we isolate the update
    # rule. AdamW normalizes by the second moment: the first-step magnitude is
    # ~lr regardless of the gradient magnitude. SGD would move lr*grad.
    p = torch.nn.Parameter(torch.zeros(4))
    opt = MuonGrokfast(
        [p],
        fallback_lr=0.1,
        grokfast_lambda=0.0,
        adamw_beta1=0.9,
        adamw_beta2=0.999,
        weight_decay=0.0,
    )
    p.grad = torch.full((4,), 10.0)
    opt.step()

    delta = p.data.abs()
    # AdamW: |delta| ~ lr = 0.1. SGD would give lr*grad = 1.0.
    assert torch.allclose(delta, torch.full((4,), 0.1), atol=1e-3), delta

    state = opt.state[p]
    assert "exp_avg" in state and "exp_avg_sq" in state, "AdamW must keep moment estimates"
    assert state["exp_avg"].abs().sum().item() > 0


def test_adamw_decoupled_weight_decay_applies():
    # Decoupled weight decay shrinks the parameter independent of the gradient.
    p = torch.nn.Parameter(torch.full((4,), 1.0))
    opt = MuonGrokfast(
        [p],
        fallback_lr=0.1,
        grokfast_lambda=0.0,
        weight_decay=0.5,
    )
    p.grad = torch.zeros(4)  # no gradient -> only weight decay moves the param
    before = p.data.clone()
    opt.step()
    # param *= (1 - lr*wd) = 1 - 0.05 = 0.95 (gradient term is zero).
    assert torch.allclose(p.data, before * 0.95, atol=1e-6), p.data


def test_adamw_step_survives_old_state_without_new_group_keys():
    # Codex-caught: a checkpoint saved before the AdamW keys existed restores
    # param_groups without adamw_beta1/beta2/eps; hard-indexing group[...] would
    # KeyError on the first step after resume. group.get(...) must tolerate it.
    p = torch.nn.Parameter(torch.zeros(4))
    opt = MuonGrokfast([p], fallback_lr=0.1, grokfast_lambda=0.0)
    # Simulate a legacy state dict: strip the new keys from the param group.
    for group in opt.param_groups:
        for key in ("adamw_beta1", "adamw_beta2", "adamw_eps", "weight_decay"):
            group.pop(key, None)

    p.grad = torch.full((4,), 10.0)
    opt.step()  # must not raise KeyError
    assert torch.isfinite(p.data).all()

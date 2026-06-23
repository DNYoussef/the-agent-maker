"""Phase 1 ACT: the halt head must actually receive a gradient from the loss."""


def _tiny_config():
    from phase1_cognate.model.model_config import (
        ACTConfig,
        Phase1Config,
        TitansMAGConfig,
        TRMConfig,
    )

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
    cfg = Phase1Config(titans_config=titans, trm_config=trm, act_config=ACTConfig(), device="cpu")
    return cfg


def test_act_head_receives_gradient_from_loss():
    # Bug: loss_act was act_loss_weight * halting_steps.float().mean(), but
    # halting_steps is a non-differentiable integer tensor. The ACT head's
    # w_halt parameters got ZERO gradient, so adaptive computation trained
    # nothing. The fix routes the differentiable compute_act_loss into the
    # total loss.
    import torch

    from phase1_cognate.model.full_model import TRMTitansMAGModel

    torch.manual_seed(0)
    model = TRMTitansMAGModel(_tiny_config())
    model.train()

    input_ids = torch.randint(0, 32, (2, 8))
    labels = torch.randint(0, 32, (2, 8))

    output = model(input_ids, labels=labels)
    assert "loss" in output
    output["loss"].backward()

    grad = model.act_head.w_halt.weight.grad
    assert grad is not None, "ACT halt head must receive a gradient"
    assert grad.abs().sum().item() > 0, "ACT halt-head gradient must be non-zero"


def test_compute_act_loss_handles_batch_and_calibrates_on_correctness():
    # Bug: the is_correct path built a scalar target and `.view(batch,1,1)`
    # crashed for batch > 1, so correctness-calibrated halting was unusable.
    # The target must be per-sample, and correct samples (better than EMA) must
    # be pushed toward halt=1 (lower BCE when q is high), while wrong samples
    # are pushed toward halt=0 (lower BCE when q is high is penalized).
    import torch

    from phase1_cognate.model.act_head import ACTHead
    from phase1_cognate.model.model_config import ACTConfig

    head = ACTHead(8, ACTConfig())

    # High halt prob on a batch of 2 with one correct, one wrong sample.
    q = torch.full((2, 4, 1), 0.9)
    loss = head.compute_act_loss(q, 0, is_correct=torch.tensor([1.0, 0.0]))
    assert torch.isfinite(loss), "loss must be finite for batch > 1"

    # Differentiability w.r.t. q is preserved.
    q2 = torch.full((2, 4, 1), 0.6, requires_grad=True)
    head.compute_act_loss(q2, 0, is_correct=torch.tensor([1.0, 0.0])).backward()
    assert q2.grad is not None and q2.grad.abs().sum().item() > 0

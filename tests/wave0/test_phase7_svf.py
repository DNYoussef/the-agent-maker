"""Phase 7 SVF: the REINFORCE baseline must not zero the logged loss on a scalar."""


def test_svf_reinforce_baseline_guarded_for_scalars():
    # Bug: `batch_loss - batch_loss.mean().detach()` on a scalar == `x - x.detach()`,
    # which zeroed the logged loss (~0) and reduced no variance. Now guarded to a
    # real batch (numel > 1) so scalar losses log their real value.
    import inspect

    import phase7_experts.svf_trainer as svf

    src = inspect.getsource(svf)
    assert "batch_loss.numel() > 1" in src, "scalar REINFORCE-baseline no-op must be guarded"


def test_svf_forward_step_propagates_gradient_to_singular_values():
    # Bug: `_svf_forward_step` set `module.weight.data = reconstructed`, which
    # detaches the autograd graph. The loss was then differentiable only w.r.t.
    # the untouched weight Parameter, never the trainable singular values, so
    # SVF trained nothing. The fix swaps the weight Parameter for the
    # graph-connected reconstruction so the gradient reaches sv_params.
    import torch
    import torch.nn as nn

    from phase7_experts.svf_trainer import SVFConfig, SVFTrainer

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8, bias=False)

        def forward(self, input_ids=None, **kwargs):
            x = nn.functional.one_hot(input_ids, num_classes=8).float()
            logits = self.proj(x)

            class Out:
                pass

            out = Out()
            out.logits = logits
            out.loss = None
            return out

    def fake_tokenizer(prompt, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    torch.manual_seed(0)
    model = TinyModel()
    trainer = SVFTrainer(SVFConfig(num_singular_values=4))
    trainer._extract_singular_values(model)
    assert trainer.sv_params, "expected at least one trainable singular-value parameter"

    loss = trainer._svf_forward_step(model, [{"prompt": "x"}], fake_tokenizer, torch.device("cpu"))
    assert loss is not None
    loss.backward()

    grads = [p.grad for p in trainer.sv_params.values()]
    assert all(g is not None for g in grads), "singular values must receive a gradient"
    assert any(g.abs().sum().item() > 0 for g in grads), "gradient must be non-zero"

    # The weight Parameter must be restored (a real nn.Parameter again).
    assert isinstance(model.proj.weight, nn.Parameter)


def test_svf_forward_step_restores_weight_on_baseexception():
    # Codex-caught: popping the weight Parameter without a finally leaves the
    # module with no `weight` if anything between pop and restore escapes. The
    # per-sample loop only catches Exception, so a BaseException (e.g.
    # KeyboardInterrupt) mid-forward must still trigger the restore.
    import torch
    import torch.nn as nn

    from phase7_experts.svf_trainer import SVFConfig, SVFTrainer

    class Boom(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8, bias=False)

        def forward(self, **kwargs):
            raise KeyboardInterrupt("simulated interrupt mid-forward")

    def fake_tokenizer(prompt, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    model = Boom()
    trainer = SVFTrainer(SVFConfig(num_singular_values=4))
    trainer._extract_singular_values(model)
    original = model.proj.weight

    raised = False
    try:
        trainer._svf_forward_step(model, [{"prompt": "x"}], fake_tokenizer, torch.device("cpu"))
    except KeyboardInterrupt:
        raised = True

    assert raised, "BaseException should propagate"
    assert isinstance(model.proj.weight, nn.Parameter), "weight must be restored on BaseException"
    assert model.proj.weight is original, "the original Parameter object must be reinstated"

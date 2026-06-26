"""Wave 4 deep-ML-correctness test for SVF (Singular Value Fine-tuning).

Finding: SVFTrainer._svf_forward_step rebuilt the layer weight from the
trainable singular values and then installed it with

    module.weight.data = reconstructed

Assigning through ``.data`` detaches the new weight from the autograd graph,
so the loss had NO dependency on ``self.sv_params``. After ``loss.backward()``
the singular-value parameters had ``.grad is None`` and the optimizer step was
a no-op: SVF "training" never moved a single singular value.

Two behavioral guarantees are pinned here:

  (a) ROUND-TRIP IDENTITY (math): SVD-decompose a known non-square weight and
      reconstruct it with the singular values unchanged. The reconstruction
      must equal the original weight (correct reduced-SVD shapes, no scaling
      bug), for both the truncated case (top-k SVs + remainder) and the
      full-rank case.

  (b) GRADIENT FLOW (the dead path): a forward pass through the SVF
      reconstruction must leave ``sv_params`` with a real, non-zero gradient,
      and an optimizer step must actually change the singular values.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.phase7_experts.svf_trainer import SVFTrainer, SVFConfig


class TinyLM(nn.Module):
    """Minimal model exposing a single SVF-able Linear and a ``.logits`` output,
    matching the consumer contract of ``_svf_forward_step``."""

    def __init__(self, vocab: int = 12, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, vocab)  # weight shape (vocab, dim) -> non-square

    def forward(self, input_ids=None, **kwargs):
        logits = self.proj(self.embed(input_ids))

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        out.loss = None
        return out


def _reconstruct(module, sv) -> torch.Tensor:
    """Rebuild the full weight the way the trainer does, given singular values
    ``sv`` for the trained top-k block."""
    rec = module._svf_U @ torch.diag(sv) @ module._svf_Vh
    if module._svf_S_original is not None:
        rec = rec + module._svf_U_rest @ torch.diag(module._svf_S_original) @ module._svf_Vh_rest
    return rec


def test_svd_roundtrip_identity_truncated_and_full():
    """Unchanged singular values must reconstruct the original weight exactly,
    for both a truncated decomposition and a full-rank one."""
    torch.manual_seed(0)

    # Truncated: min(shape)=16 > num_singular_values=8 -> remainder block exists.
    trunc_model = nn.Sequential(nn.Linear(16, 24))
    t1 = SVFTrainer(SVFConfig(num_singular_values=8))
    t1._extract_singular_values(trunc_model)
    assert t1.sv_params, "no Linear was decomposed (truncated case)"
    for name, module in trunc_model.named_modules():
        if hasattr(module, "_svf_S_param"):
            assert module._svf_S_original is not None, "expected a remainder block"
            rec = _reconstruct(module, t1.sv_params[name])
            assert rec.shape == module.weight.shape
            assert torch.allclose(rec, module.weight.data, atol=1e-5), (
                f"{name}: truncated round-trip is not identity"
            )

    # Full-rank: num_singular_values == min(shape) -> no remainder block.
    full_model = nn.Sequential(nn.Linear(16, 24))
    t2 = SVFTrainer(SVFConfig(num_singular_values=16))
    t2._extract_singular_values(full_model)
    assert t2.sv_params, "no Linear was decomposed (full-rank case)"
    for name, module in full_model.named_modules():
        if hasattr(module, "_svf_S_param"):
            assert module._svf_S_original is None, "did not expect a remainder block"
            rec = _reconstruct(module, t2.sv_params[name])
            assert rec.shape == module.weight.shape
            assert torch.allclose(rec, module.weight.data, atol=1e-5), (
                f"{name}: full-rank round-trip is not identity"
            )


def test_svf_forward_step_propagates_gradient_to_singular_values():
    """A forward pass through SVF reconstruction must give sv_params a real
    gradient, and an optimizer step must move the singular values."""
    torch.manual_seed(0)
    model = TinyLM()
    trainer = SVFTrainer(SVFConfig(num_singular_values=4))
    trainer._extract_singular_values(model)
    assert trainer.sv_params, "no Linear was decomposed"

    sv_params = list(trainer.sv_params.values())
    before = {name: p.detach().clone() for name, p in trainer.sv_params.items()}

    loss = trainer._svf_forward_step(model, [{"prompt": "hello"}], None, torch.device("cpu"))
    assert loss is not None and loss.requires_grad, "loss must depend on the trainable SVs"

    loss.backward()

    for name, p in trainer.sv_params.items():
        assert p.grad is not None, f"{name}: singular values received no gradient (dead path)"
        assert torch.any(p.grad != 0), f"{name}: gradient is all-zero (no learning signal)"

    opt = torch.optim.SGD(sv_params, lr=1.0)
    opt.step()
    moved = any(
        not torch.allclose(before[name], p.detach()) for name, p in trainer.sv_params.items()
    )
    assert moved, "optimizer step did not change any singular value"

    # The model's own weight Parameter must be intact and trainable afterwards
    # (the SVF path must not strand the layer without its weight).
    assert isinstance(model.proj.weight, nn.Parameter)
    assert model.proj.weight.shape == (12, 16)

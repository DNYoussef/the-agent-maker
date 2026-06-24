"""Prompt baking: the KL objective must distill the PROMPTED teacher (adapter
disabled, detached) into the UNPROMPTED student (adapter enabled). The old code
ran both models on the same input, so the prompt's effect was never isolated and
the teacher was not detached."""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _StubLoRAModel(nn.Module):
    """Minimal stand-in for a PEFT model: a trainable weight, a toggleable
    adapter, and a disable_adapter() context manager. Records each forward."""

    def __init__(self, vocab=6):
        super().__init__()
        self.w = nn.Parameter(torch.randn(vocab, vocab))
        self.adapter_enabled = True
        self.calls = []

    @contextlib.contextmanager
    def disable_adapter(self):
        prev = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = prev

    def forward(self, input_ids):
        self.calls.append(
            {
                "first_token": int(input_ids[0, 0]),
                "adapter_enabled": self.adapter_enabled,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        onehot = F.one_hot(input_ids, num_classes=self.w.shape[0]).float()
        logits = onehot @ self.w
        if self.adapter_enabled:
            logits = logits + 0.5  # adapter changes behavior
        return _Out(logits)


def test_compute_bake_loss_prompted_teacher_detached_unprompted_student():
    from cross_phase.prompt_baking.baker import PromptBaker, PromptBakingConfig

    baker = PromptBaker(PromptBakingConfig())
    model = _StubLoRAModel()
    prompted = torch.tensor([[1, 2, 3]])  # teacher sees the prompt
    unprompted = torch.tensor([[4, 5]])  # student does not

    loss = baker._compute_bake_loss(model, prompted, unprompted)
    assert torch.isfinite(loss) and loss.requires_grad

    loss.backward()
    assert (
        model.w.grad is not None and model.w.grad.abs().sum().item() > 0
    ), "student path must be differentiable"

    assert len(model.calls) == 2, "exactly one teacher and one student forward"
    teacher, student = model.calls
    # Teacher: prompted input, adapter disabled, under no_grad (detached).
    assert teacher["first_token"] == 1
    assert teacher["adapter_enabled"] is False
    assert teacher["grad_enabled"] is False
    # Student: unprompted input, adapter enabled, differentiable.
    assert student["first_token"] == 4
    assert student["adapter_enabled"] is True
    assert student["grad_enabled"] is True


def test_bake_prompt_loop_uses_the_kl_helper_on_distinct_inputs():
    # Lock the wiring: the training loop must call the helper with separate
    # prompted/unprompted inputs, not run both models on one shared sequence.
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2] / "src" / "cross_phase" / "prompt_baking" / "baker.py"
    )
    src = path.read_text(encoding="utf-8")
    body = src.split("def bake_prompt", 1)[1].split("\n    def ", 1)[0]
    assert "_compute_bake_loss" in body, "training loop must use the KL helper"
    assert "unprompted_input_ids" in body and "prompted_input_ids" in body


class _StubTok:
    """Deterministic char-level tokenizer: same text -> same ids regardless of
    the prompt prefix, so suffix alignment is checkable."""

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = [ord(c) % 97 + 1 for c in text]
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids])}


def test_prepare_batches_aligns_prompted_suffix_with_unprompted():
    # Codex-caught: independent truncation of (prompt+text) vs (text) misaligns
    # the final-position next-token target once the prompt eats the budget. The
    # prompted sequence must END with exactly the unprompted text tokens.
    from cross_phase.prompt_baking.baker import PromptBaker, PromptBakingConfig

    baker = PromptBaker(PromptBakingConfig())
    batches = baker._prepare_calibration_batches(
        "SYS PROMPT", _StubTok(), ["hello world"], device=torch.device("cpu")
    )
    assert len(batches) == 1
    prompted = batches[0]["prompted_input_ids"]
    unprompted = batches[0]["unprompted_input_ids"]
    n = unprompted.shape[1]
    assert prompted.shape[1] > n, "prompted must include the prompt prefix"
    assert torch.equal(
        prompted[:, -n:], unprompted
    ), "prompted must end with the identical text tokens (aligned final position)"

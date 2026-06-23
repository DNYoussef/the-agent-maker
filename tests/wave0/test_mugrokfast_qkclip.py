"""MuGrokfast QK-Clip honesty: the forward-hook QKClipHook MONITORS (counts)
threshold exceedances but does NOT clip - it runs after the module output, so it
cannot touch pre-softmax scores. Only apply_qk_clip (called inside attention)
actually clips. Locking this prevents the old docstring's false 'clips to
[-threshold, +threshold]' claim from creeping back."""

import torch
import torch.nn as nn

from cross_phase.mugrokfast.optimizer import QKClipHook, apply_qk_clip


class _AttnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Identity()  # name contains "attn" -> hook registers here

    def forward(self, x):
        return self.attn(x)


def test_qkclip_hook_monitors_but_does_not_clip():
    model = _AttnModel()
    monitor = QKClipHook(threshold=5.0)
    monitor.register(model)

    x = torch.full((2, 3), 100.0)
    out = model(x)

    # It does NOT clip: output passes through unchanged.
    assert torch.equal(out, x), "QKClipHook must not modify outputs (it is monitor-only)"
    # It DOES count the exceedance.
    assert monitor.get_clip_count() >= 1


def test_apply_qk_clip_actually_clips_pre_softmax_scores():
    scores = torch.tensor([[-100.0, 0.0, 100.0]])
    clipped, n = apply_qk_clip(scores, threshold=10.0)
    assert clipped.max().item() <= 10.0
    assert clipped.min().item() >= -10.0
    assert n == 2  # the -100 and +100 entries

"""Wave 0 crash gate: each test reproduces a confirmed crash bug in isolation.
Fail-first: these are RED before the fix and GREEN after. Hermetic, no heavy training.
Run: PYTHONPATH=<wt>;<wt>/src PYTHONIOENCODING=utf-8 pytest tests/wave0 -q
"""
import torch
import pytest


def test_muon_newton_schulz_non_square_no_crash():
    # Bug 0.8: _muon_update Newton-Schulz used wrong operand order (G_norm @ A
    # where A is [m,m]) -> shape mismatch on any non-square 2D param.
    # Agents reproduced 64x100 -> RuntimeError. Square/1D worked, non-square crashed.
    from cross_phase.mugrokfast.optimizer import MuonGrokfast

    for shape in [(100, 64), (64, 100), (256, 128), (128, 256), (64, 64)]:
        p = torch.nn.Parameter(torch.randn(*shape))
        opt = MuonGrokfast([p], muon_lr=0.01)
        p.grad = torch.randn(*shape)
        opt.step()  # must not raise
        assert torch.isfinite(p.data).all(), f"non-finite update for {shape}"


def test_muon_handles_1d_and_rank3_params_no_crash():
    # Routing sends only exactly-2D params to the matrix-only Muon path; 1D and
    # rank>2 must take the fallback and not crash (Codex-flagged latent crash).
    from cross_phase.mugrokfast.optimizer import MuonGrokfast

    for shape in [(128,), (8, 16, 32)]:
        p = torch.nn.Parameter(torch.randn(*shape))
        opt = MuonGrokfast([p], muon_lr=0.01)
        p.grad = torch.randn(*shape)
        opt.step()  # must not raise
        assert torch.isfinite(p.data).all(), f"non-finite update for {shape}"


def test_self_modeling_module_defines_logger():
    # Bug 0.3: self_modeling.py uses logger.warning/.info (lines 415,418,504,522)
    # but never imports logging / defines logger -> NameError the moment that path runs.
    import phase5_curriculum.self_modeling as sm

    assert hasattr(sm, "logger"), "module-level `logger` missing -> NameError at runtime"

"""E8 gate - CRUCIBLE: Phase 7 SVF covers fp16 layers and fails loud on total failure.

Synthesis: torch.linalg.svd raises on fp16/bf16 weights; the bare except:continue then
skipped EVERY layer, so SVF reported "success" with ~nothing parameterized. E8 runs SVD in
fp32 (covers fp16 models) and raises if 0 of N eligible layers got parameterized.
"""

import pytest
import torch
import torch.nn as nn

from phase7_experts.svf_trainer import SVFConfig, SVFTrainer


def _trainer(nsv=2):
    t = object.__new__(SVFTrainer)
    t.config = SVFConfig(num_singular_values=nsv)
    return t


def test_svf_covers_fp16_layers():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).half()
    t = _trainer()
    t._extract_singular_values(model)
    assert len(t.sv_params) == 2, "fp16 Linear layers must be covered (SVD upcast to fp32)"


def test_svf_covers_fp32_layers_unchanged():
    model = nn.Sequential(nn.Linear(8, 8))
    t = _trainer()
    t._extract_singular_values(model)
    assert len(t.sv_params) == 1


def test_svf_fails_loud_when_all_svd_fail(monkeypatch):
    model = nn.Sequential(nn.Linear(8, 8))
    t = _trainer()

    def boom(*a, **k):
        raise RuntimeError("svd backend failure")

    monkeypatch.setattr(torch.linalg, "svd", boom)
    with pytest.raises(RuntimeError):
        t._extract_singular_values(model)  # must NOT silently report 0 parameterized layers

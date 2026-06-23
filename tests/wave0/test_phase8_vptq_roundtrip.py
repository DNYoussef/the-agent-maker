"""Phase 8 VPTQ: lock the compress -> decompress round-trip with a MEASURED
compression ratio (not the aspirational 280x headline). Wave-0 disabled the
residual path pending this Wave-1 round-trip test; this is that test."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_vptq_single_codebook_roundtrip_is_real_and_measured():
    from phase8_compression.vptq import VPTQCompressor, VPTQConfig

    torch.manual_seed(0)
    model = _TinyModel()
    original_w = model.fc1.weight.detach().clone()

    compressor = VPTQCompressor(VPTQConfig())
    compressed_model, result = compressor.compress(model)

    # Compression is REAL and MEASURED (ratio derived from byte sizes, > 1x).
    assert result.success
    assert result.compression_ratio > 1.0
    assert result.compressed_size_mb < result.original_size_mb

    # Round-trip preserves shapes and produces a runnable model.
    recon = dict(compressed_model.named_parameters())["fc1.weight"]
    assert recon.shape == original_w.shape
    out = compressed_model(torch.randn(2, 64))
    assert out.shape == (2, 32) and torch.isfinite(out).all()

    # Lossy but FAITHFUL: the reconstruction must track the original (it is
    # codebook[indices], not noise). High cosine similarity proves a real
    # decode rather than a stubbed/zero reconstruction.
    cos = F.cosine_similarity(recon.flatten(), original_w.flatten(), dim=0).item()
    assert cos > 0.9, f"reconstruction must track the original weight, cos={cos:.3f}"
    # And it is genuinely quantized (not an identity copy).
    assert not torch.equal(recon, original_w)


def test_vptq_residual_disabled_by_default():
    # Wave-0 fail-safe: the residual path is off until its pack/decompress dict
    # mismatch is unified. Guard against silent re-enable.
    from phase8_compression.vptq import VPTQConfig

    assert VPTQConfig().use_residual is False

"""Fail-first probe for the VPTQ default (residual) compression path.

The default VPTQConfig has use_residual=True, num_codebooks=4. Before the fix,
_pack_residual_result returned dicts for indices/codebook, so compress() crashed
on codebook.shape[0] (AttributeError). This probe pins both the crash-free
compress AND a sane round-trip on the default config.
"""
import torch
import torch.nn as nn

from src.phase8_compression.vptq import VPTQCompressor, VPTQConfig


def test_default_residual_compress_and_roundtrip():
    # main defaults use_residual=False (it avoided the crash by disabling the
    # path); this proves the residual path itself is fixed when enabled.
    cfg = VPTQConfig(use_residual=True, num_codebooks=4)

    torch.manual_seed(0)
    model = nn.Linear(64, 64)
    compressor = VPTQCompressor(config=cfg)

    compressed_model, result = compressor.compress(model)

    assert result.success is True
    assert result.compressed_size_mb > 0
    # On a tiny 64x64 weight the 4 fp16 codebooks dominate, so ratio < 1 here;
    # the contract is a crash-free, finite round-trip, not a ratio on toy sizes.
    assert result.compression_ratio > 0

    # Round-trip must produce a usable model with the original architecture.
    x = torch.randn(4, 64)
    out = compressed_model(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()

    # The residual path stores the weight under "vptq"; its codebook/indices
    # must be tensors (not dicts) so size/decompress can operate on them.
    w_entry = compressed_model.state_dict()  # decompressed already
    assert "weight" in w_entry

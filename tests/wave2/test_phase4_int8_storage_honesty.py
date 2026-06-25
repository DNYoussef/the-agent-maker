"""
Behavioral proof that CompressedModel (Mode 1, BitLinear) reports the REAL
int8 on-disk footprint, not a theoretical packed 1.58-bit number.

Ternary weights carry 1.58 bits of information, but the implementation stores
them as int8 (1 byte/weight) with an fp16 scale - there is NO bit-packing.
A truthful compression ratio is therefore ~4x (fp32 -> int8), NOT the ~20x
that real 1.58-bit packing (32 / 1.58) would give.

These tests would FAIL if someone re-hardcoded a packed-1.58-bit ratio, or if
quantized_size_mb were a constant insensitive to model size.
"""

import torch
import torch.nn as nn

from src.phase4_bitnet.bitlinear import BitLinear
from src.phase4_bitnet.compressed_model import CompressedModel
from src.phase4_bitnet.config import Phase4Config
from src.phase4_bitnet.quantizer import BitNetQuantizer

# Theoretical ratio if ternary weights were truly bit-packed at 1.58 bits.
PACKED_1P58_RATIO = 32.0 / 1.58  # ~20.25x


def _make(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _compress(model):
    config = Phase4Config(wandb_enabled=False)
    quantizer = BitNetQuantizer(config)
    return CompressedModel(model, quantizer, config, use_bitlinear=True)


def _expected_int8_bytes(compressed):
    """Independently recompute int8 storage from the BitLinear modules."""
    total = 0
    for module in compressed.base_model.modules():
        if isinstance(module, BitLinear):
            weight_int8 = module.weight.nelement() * 1  # 1 byte/weight (int8)
            scale_fp16 = module.out_features * 2
            bias = module.bias.nelement() * 2 if module.bias is not None else 0
            total += weight_int8 + scale_fp16 + bias
    return total


def test_quantized_size_matches_int8_storage_not_packed():
    """Reported quantized_size_mb must equal the independently computed int8
    byte count - proving it is measured, not a fabricated packed number."""
    compressed = _compress(_make(64, 128, 64))
    stats = compressed.get_compression_stats()

    expected_mb = _expected_int8_bytes(compressed) / (1024 ** 2)
    assert expected_mb > 0
    assert abs(stats["quantized_size_mb"] - expected_mb) < 1e-6, (
        f"quantized_size_mb {stats['quantized_size_mb']} != int8 reality {expected_mb}"
    )
    # Honest labels present.
    assert stats["storage_dtype"] == "int8"
    assert stats["bit_packed"] is False


def test_compression_ratio_is_int8_not_1p58bit():
    """Ratio must be in the int8 regime (~4x), and must NOT be the ~20x that
    packed 1.58-bit storage would yield. Hardcoding the packed ratio fails."""
    compressed = _compress(_make(256, 512, 256))
    ratio = compressed.get_compression_stats()["compression_ratio"]

    assert 2.0 < ratio < 6.0, f"expected int8 ratio ~4x, got {ratio}"
    # Definitively not the theoretical packed-1.58-bit ratio.
    assert ratio < PACKED_1P58_RATIO - 5.0, (
        f"ratio {ratio} looks like packed 1.58-bit ({PACKED_1P58_RATIO:.1f}x); "
        "storage is int8, not bit-packed"
    )


def test_quantized_size_scales_with_model_size():
    """A bigger model must report a strictly bigger int8 footprint - a fake
    constant would be insensitive to input size."""
    small = _compress(_make(64, 128, 64)).get_compression_stats()
    large = _compress(_make(256, 512, 256)).get_compression_stats()

    assert large["quantized_size_mb"] > small["quantized_size_mb"] * 2, (
        "quantized_size_mb does not scale with model size - likely a constant"
    )

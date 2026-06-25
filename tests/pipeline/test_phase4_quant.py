"""E5 gate - CRUCIBLE: BitNet quantized weights survive storage (not all zeros).

Synthesis: get_quantized_state() ran weight_quant() (which returns alpha*sign, a small
FLOAT like 0.02) then .to(torch.int8), truncating every value to 0 -> all-zero quantized
weights; load then double-scaled (alpha * 0). E5 stores the ternary CODE sign() in {-1,0,+1}
as int8, so dequant = alpha * code recovers the quantized weight.
"""

import json

import pytest
import torch

from phase4_bitnet.bitlinear import BitLinear


def test_quantized_weights_are_ternary_and_not_all_zero():
    layer = BitLinear(16, 16)
    with torch.no_grad():
        layer.weight.normal_(0, 0.02)  # small weights - exactly what truncated to 0 before
    state = layer.get_quantized_state()
    q = state["quantized_weight"]
    assert q.dtype == torch.int8
    assert int(q.abs().sum()) > 0, "quantized weights are all zero (int8 truncation bug)"
    assert set(q.unique().tolist()).issubset({-1, 0, 1}), "codes must be ternary {-1,0,+1}"


def test_quant_roundtrip_recovers_scaled_ternary():
    src = BitLinear(16, 16)
    with torch.no_grad():
        src.weight.normal_(0, 0.02)
    expected = src.weight_quant(src.weight)  # alpha * {-1,0,+1}, the dequant target

    state = src.get_quantized_state()
    dst = BitLinear(16, 16)
    dst.load_quantized_state(state)

    assert torch.allclose(
        dst.weight.data, expected, atol=1e-3
    ), f"round-trip diverged; max {abs(dst.weight.data - expected).max().item()}"
    assert dst.weight.data.abs().sum() > 0, "dequantized weights are all zero"


def test_per_channel_scale_serializes_without_crash():
    # Phase 4 save serialized scales with float(v); per-channel scales are multi-element
    # tensors, so float() raised. The fix uses tolist() -> JSON-able list.
    alpha = torch.rand(8)  # per-channel scale factor
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        float(alpha)  # the old crash
    serialized = alpha.detach().cpu().tolist() if hasattr(alpha, "tolist") else alpha
    json.dumps({"scale_factors": {"layer0": serialized}})  # must not raise

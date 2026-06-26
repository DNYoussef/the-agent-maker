"""Branch-deletion audit capture: CompressionEngine._finalize hardcoded
success=True and returned model=current_model with no rollback, even when the
cumulative retention gate failed -> it would ship damaged weights reporting
success. This was still present on main (masked because lstsq SeedLM usually
passes its own gate). Fix: success reflects gate failure; on failure ship the
last good (undamaged) model.
"""
import torch
import torch.nn as nn

from src.phase8_compression.compression_engine import CompressionConfig, CompressionEngine


def test_final_gate_failure_is_not_reported_as_success_and_ships_original():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
    original = {k: v.clone() for k, v in model.state_dict().items()}

    # VPTQ passes its own per-stage gate, but force the FINAL cumulative gate to
    # fail with an unachievable threshold.
    cfg = CompressionConfig(
        seedlm_enabled=False,
        vptq_enabled=True,
        hyper_enabled=False,
        min_retention_vptq=0.0,
        min_retention_final=1.01,  # unachievable -> final gate always fails
        run_benchmarks=False,
    )
    result = CompressionEngine(cfg).run(model, tokenizer=None)

    assert result.success is False, "final-gate failure must NOT report success"
    out = result.model.state_dict()
    for k, v in original.items():
        assert torch.allclose(out[k], v), f"{k}: shipped damaged weights after final-gate failure"
    # Metrics must describe the model actually shipped (the original): no fake
    # compressed size / ratio / artifact for weights we are not delivering.
    assert (
        result.final_size_mb == result.original_size_mb
    ), "reported a compressed size for the original model"
    assert (
        abs(result.total_compression - 1.0) < 1e-6
    ), "reported compression ratio != 1x while shipping original"
    assert (
        getattr(result, "artifact_path", None) is None
    ), "wrote a compressed artifact for unshipped weights"

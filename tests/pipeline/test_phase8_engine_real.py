"""G2 gate - CRUCIBLE: Phase 8 engine reports the REAL encoded compression + a disk artifact.

Synthesis #4 / Codex-final: _finalize measured _get_model_size(current_model), but current_model
is the DEQUANTIZED reconstruction (~original size), so total_compression was always ~1.0x even
though SeedLM really compressed. G2 measures the smallest kept-stage ENCODED size and serializes
that compressed_state to disk.
"""

import os

import torch.nn as nn

from phase8_compression.compression_engine import CompressionConfig, CompressionEngine


def test_engine_reports_real_encoded_compression_and_writes_artifact(tmp_path):
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))  # real Linear weights
    cfg = CompressionConfig(
        seedlm_enabled=True,
        vptq_enabled=False,
        hyper_enabled=False,
        run_benchmarks=False,
        min_retention_seedlm=0.0,  # accept the lossy seedlm so the stage is KEPT
        artifacts_dir=str(tmp_path),
    )
    result = CompressionEngine(cfg).run(model, tokenizer=None)

    assert result.success
    # real encoded ratio (was ~1.0 measuring the dequantized module)
    assert (
        result.total_compression > 1.0
    ), f"expected real compression, got {result.total_compression}"
    assert result.final_size_mb < result.original_size_mb
    # a real, smaller artifact exists on disk
    assert result.artifact_path and os.path.exists(result.artifact_path)
    assert os.path.getsize(result.artifact_path) < result.original_size_mb * 1024 * 1024


def test_engine_no_op_reports_1x_when_all_stages_fail_quality():
    # If every stage fails its retention gate, NOTHING is kept -> honest 1.0x (no fake ratio).
    model = nn.Sequential(nn.Linear(64, 64))
    cfg = CompressionConfig(
        seedlm_enabled=True,
        vptq_enabled=False,
        hyper_enabled=False,
        run_benchmarks=False,
        min_retention_seedlm=1.01,  # impossible -> seedlm never kept
    )
    result = CompressionEngine(cfg).run(model, tokenizer=None)
    assert result.total_compression == 1.0, "no kept stage -> honest 1.0x, not a fabricated ratio"

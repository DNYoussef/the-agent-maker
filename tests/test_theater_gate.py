"""
Acceptance gate for the theater-remediation waves (crucible step 2).

Built FIRST, before any fix. Each probe encodes a load-bearing contract derived
from the two-model audit (2026-06-25). Behavioral probes use tiny CPU tensors;
honesty probes are source-derived (harvest the lie-literal straight from the
file). A probe is RED until its wave flips it green. Hermetic: no network, no
heavy training, fresh objects per test.

Run:
  PYTHONPATH="<repo>;<repo>/src" python -m pytest tests/test_theater_gate.py -q
"""
import ast
import os
import re

import pytest
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")


def _src(rel):
    """Read a source file under src/ as text (data-derived honesty probes)."""
    with open(os.path.join(SRC, rel), "r", encoding="utf-8") as fh:
        return fh.read()


def _tiny_linear(seed):
    torch.manual_seed(seed)
    return nn.Linear(8, 8)


# --------------------------------------------------------------------------
# WAVE 1 - Class A: crash / correctness bugs (behavioral, strong)
# --------------------------------------------------------------------------

def test_vptq_residual_path_compresses_without_crash():
    """Phase8 VPTQ: the residual path (use_residual=True, num_codebooks=4) must
    compress a real 2-D weight without the AttributeError on the dict codebook
    that crashed it. (main avoided the crash by defaulting use_residual=False;
    this proves the residual path itself is fixed, not merely disabled.)"""
    from src.phase8_compression.vptq import VPTQCompressor, VPTQConfig

    cfg = VPTQConfig(use_residual=True, num_codebooks=4)
    model = nn.Linear(64, 64)
    compressed_model, result = VPTQCompressor(config=cfg).compress(model)
    assert result.success is True
    assert result.compressed_size_mb > 0


def test_phase2_every_merger_accepts_a_model_list():
    """Phase2 EvoMerge: every configured merger must merge a list of models via
    one uniform signature merge(models: List[nn.Module]). Today ties/dare/
    frankenmerge/dfs require (target, refs) and raise TypeError on the pipeline's
    call path."""
    from src.phase2_evomerge.phase2_pipeline import Phase2Pipeline

    pipeline = Phase2Pipeline()  # default config = all 6 techniques
    models = [_tiny_linear(1), _tiny_linear(2)]
    for name, merger in pipeline._mergers.items():
        merged = merger.merge([_tiny_linear(1), _tiny_linear(2)])
        assert isinstance(merged, nn.Module), f"{name} did not return a model"
    assert set(pipeline._mergers) >= {"slerp", "ties", "dare", "linear", "frankenmerge", "dfs"}


# --------------------------------------------------------------------------
# WAVE 2 - Class B: fake evaluation (behavioral) + Class D: honesty (source)
# --------------------------------------------------------------------------

def test_edge_of_chaos_correctness_is_deterministic():
    """Phase5 assessment: _check_correctness must measure the response, not roll
    random.random(). Same (question, response) must give the same verdict."""
    from src.phase5_curriculum.assessment import EdgeOfChaosAssessment

    a = EdgeOfChaosAssessment.__new__(EdgeOfChaosAssessment)  # skip heavy __init__
    q = {"level": 30, "question": "2+2", "answer": "4", "expected": "4"}
    verdicts = {a._check_correctness(q, "4") for _ in range(25)}
    assert len(verdicts) == 1, "correctness verdict is non-deterministic (random)"
    # NOTE: a blunt "random.random() not in source" check is a false positive -
    # assessment legitimately uses random for question GENERATION, not grading.
    # Determinism of the grading verdict is the real contract.


def test_seedlm_does_not_report_success_on_destroyed_weights():
    """Phase8 SeedLM reconstructs Gaussian noise from a seed (corr ~0.2 on real
    weights). It must NOT return success=True while retention is low."""
    from src.phase8_compression.seedlm import SeedLMCompressor

    torch.manual_seed(0)
    model = nn.Linear(64, 64)
    compressed_model, result = SeedLMCompressor().compress(model)
    assert not (result.success and result.retention_score < 0.5), (
        f"claims success with retention={result.retention_score:.3f} "
        "(silent weight destruction)"
    )


def test_anti_theater_gate_is_not_hardcoded_true():
    """cross_phase Phase3Controller._validate_anti_theater hardcodes
    consistency_test and ablation_test to True. The gate must compute them."""
    text = _src("cross_phase/orchestrator/phase3_controller.py")
    assert 'results["consistency_test"] = True' not in text
    assert 'results["ablation_test"] = True' not in text


def test_phase1_final_metrics_are_not_hardcoded_constants():
    """Phase1 trainer streams hardcoded summary metrics to W&B and returns a
    canned validation loss. The placeholder literals must be gone."""
    text = _src("phase1_cognate/training/trainer.py")
    assert "dummy for now" not in text
    assert "Placeholder accuracies" not in text


# NOTE: a phase6 LoRA-honesty probe was intentionally dropped from this PR.
# main already makes Phase 6 honest about LoRA by failing loudly on use_lora
# (full-AdamW, no silent adapter) rather than removing the knobs - a different
# but legitimate fix. This PR does not touch phase6, so it does not re-litigate it.


def test_phase4_does_not_mislabel_int8_as_1p58bit():
    """Phase4 stores ternary weights as int8 (no bit-packing). No source may
    label a STORAGE quantity (a dict key / footprint field) as 1.58-bit, and any
    '1.58-bit' prose must sit next to an honest int8 note. (Scans the package so
    bitlinear.py is covered, not just compressed_model.py.)"""
    import glob

    # A storage-quantity labeled "1.58bit" (e.g. a dict key holding a byte count)
    # is the lie; the method name "1.58-bit quantization" (hyphenated prose) is
    # accurate and allowed. Ban the storage-key token across the package.
    # "1.58bit" as an identifier/key (e.g. quantized_1.58bit) is the lie; allow
    # the prose "1.58-bit" (hyphen) and "1.58 bits of information".
    bad_token = re.compile(r"1\.58 ?bit(?![s-])", re.IGNORECASE)
    for path in glob.glob(os.path.join(SRC, "phase4_bitnet", "*.py")):
        text = open(path, "r", encoding="utf-8").read()
        assert not bad_token.search(text), (
            f"{os.path.basename(path)}: a storage quantity is labeled '1.58bit'"
        )
    # In the stats file, any 1.58-bit prose must acknowledge int8 storage nearby.
    stats = _src("phase4_bitnet/compressed_model.py")
    for m in re.finditer(r"1\.58[ -]?bit", stats, re.IGNORECASE):
        window = stats[max(0, m.start() - 220): m.end() + 220].lower()
        assert "int8" in window, "1.58-bit claim without int8-storage honesty note"


# --------------------------------------------------------------------------
# WAVE 3 - Class C: wire the real twin / delete the broken
# --------------------------------------------------------------------------

def test_globalmoo_stub_not_on_production_path():
    """moo_bridge.GlobalMOOAdapter is a NotImplementedError stub; the real client
    lives in moo_utils. The stub's raises must be gone (deleted or replaced)."""
    text = _src("cross_phase/meta_calculus/moo_bridge.py")
    assert 'NotImplementedError("GlobalMOO API integration pending")' not in text


def test_phase2_quick_fitness_is_deterministic():
    """Phase2 _quick_fitness used random.uniform for 'speed' (and a formula for
    accuracy) and never ran the model. Fitness must be deterministic for a fixed
    model - a search that ranks population on random noise is theater."""
    from src.phase2_evomerge.phase2_pipeline import Phase2Pipeline

    pipeline = Phase2Pipeline()
    model = _tiny_linear(7)
    f1 = pipeline._quick_fitness(model)
    f2 = pipeline._quick_fitness(model)
    assert f1 == f2, f"non-deterministic fitness: {f1} != {f2}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

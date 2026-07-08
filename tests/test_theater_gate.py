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
    for name, merger in pipeline._mergers.items():
        merged = merger.merge([_tiny_linear(1), _tiny_linear(2)])
        assert isinstance(merged, nn.Module), f"{name} did not return a model"
    assert set(pipeline._mergers) >= {"slerp", "ties", "dare", "linear", "frankenmerge", "dfs"}


# --------------------------------------------------------------------------
# WAVE 2 - Class B: fake evaluation (behavioral) + Class D: honesty (source)
# --------------------------------------------------------------------------


def test_edge_of_chaos_grades_correctness_deterministically():
    """Phase5 assessment: _check_correctness must MEASURE the response against
    ground truth, not roll random.random(). Behavioral teeth: correct grades
    True, wrong grades False, missing ground truth fails closed, and the verdict
    is deterministic. A constant grader fails the True/False split; a random
    grader fails determinism (the old probe only checked determinism)."""
    from src.phase5_curriculum.assessment import EdgeOfChaosAssessment

    a = EdgeOfChaosAssessment.__new__(EdgeOfChaosAssessment)  # skip heavy __init__
    q = {"level": 30, "question": "2+2", "answer": "4"}
    assert a._check_correctness(q, "4") is True, "correct answer must grade True"
    assert a._check_correctness(q, "5") is False, "wrong answer must grade False"
    assert a._check_correctness({}, "4") is False, "no ground truth must fail closed"
    assert len({a._check_correctness(q, "4") for _ in range(25)}) == 1, "verdict not deterministic"


def test_seedlm_does_not_report_success_on_destroyed_weights():
    """Phase8 SeedLM reconstructs from seeds; retention on real weights is far
    below the 0.95 target. compress() must NOT hardcode success=True - success
    must reflect retention vs target. Teeth: the old probe used an arbitrary
    <0.5 cutoff, so a retention of 0.54 with success=True slipped through; this
    keys on the actual config target."""
    from src.phase8_compression.seedlm import SeedLMCompressor, SeedLMConfig

    cfg = SeedLMConfig()
    torch.manual_seed(0)
    model = nn.Linear(64, 64)
    compressed_model, result = SeedLMCompressor(config=cfg).compress(model)
    assert result.retention_score < cfg.target_retention, (
        f"premise broke: retention {result.retention_score:.3f} >= target "
        f"{cfg.target_retention} (pick weights that genuinely degrade)"
    )
    assert result.success is False, (
        f"SeedLM claims success at retention={result.retention_score:.3f} "
        f"< target {cfg.target_retention} (silent weight destruction)"
    )


def test_phase8_reports_original_size_when_final_gate_fails_after_partial_keep():
    """Phase8 CompressionEngine: when an earlier stage is KEPT (SeedLM passes) but
    a later stage rolls back (VPTQ fails) AND the cumulative final gate fails, the
    engine ships the pristine original - so it must report 1.0x / the original
    size, never the kept stage's compressed size. Teeth: the old guard keyed only
    on rollback_stage=='final', so this path shipped the original yet reported the
    SeedLM size (total_compression 2.0x with success False)."""
    import time as _time

    from src.phase8_compression.compression_engine import CompressionConfig, CompressionEngine

    cfg = CompressionConfig(run_benchmarks=False, artifacts_dir=None)
    engine = CompressionEngine(config=cfg)
    original_size = 0.02
    model = nn.Linear(8, 8)
    stage_results = {
        # SeedLM passed its gate (retention >= 0.95) and has a real smaller size...
        "seedlm": {
            "compression_ratio": 2.0,
            "retention": 0.96,
            "size_mb": original_size / 2,
            "compressed_state": None,
        },
        # ...VPTQ failed its gate, so rollback_stage is "vptq" and cumulative
        # retention (0.96 * 0.10 = 0.096) is below min_retention_final (0.84).
        "vptq": {
            "compression_ratio": 1.0,
            "retention": 0.10,
            "size_mb": original_size,
            "compressed_state": None,
        },
    }
    result = engine._finalize(
        model, model, original_size, stage_results, "vptq", None, None, _time.time()
    )
    assert result.success is False
    assert result.artifact_path is None
    assert abs(result.final_size_mb - original_size) < 1e-9, (
        f"shipped original but reported final_size={result.final_size_mb} "
        f"(should be original {original_size})"
    )
    assert (
        abs(result.total_compression - 1.0) < 1e-9
    ), f"shipped original but reported {result.total_compression:.2f}x compression"


def test_anti_theater_gate_distinguishes_real_from_constant_model():
    """cross_phase Phase3Controller._validate_anti_theater must COMPUTE divergence
    and consistency from real generate() output, not hardcode True. Behavioral
    teeth: a model that emits a constant for every input must FAIL divergence; a
    deterministic model that varies with input must PASS. A hardcoded-True gate
    would pass the constant model."""
    from src.cross_phase.orchestrator.phase3_controller import Phase3Controller

    class _Tok:
        def __call__(self, text, **kw):
            ids = [ord(c) % 50 for c in text][:8] or [0]
            return {"input_ids": torch.tensor([ids])}

    class _Varying(nn.Module):
        def eval(self):
            return self

        def generate(self, input_ids=None, **kw):
            return input_ids + 1  # output is a deterministic function of the input

    class _Constant(nn.Module):
        def eval(self):
            return self

        def generate(self, input_ids=None, **kw):
            return torch.zeros(1, 5, dtype=torch.long)  # same output for every input

    ctrl = Phase3Controller.__new__(Phase3Controller)  # skip heavy __init__
    good = ctrl._validate_anti_theater(_Varying(), _Tok())
    bad = ctrl._validate_anti_theater(_Constant(), _Tok())
    assert good["all_passed"] is True, "real varying model must pass anti-theater"
    assert bad["all_passed"] is False, "constant model must fail (gate is not hardcoded True)"
    # belt-and-suspenders: the original hardcoded literals stay gone
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
        assert not bad_token.search(
            text
        ), f"{os.path.basename(path)}: a storage quantity is labeled '1.58bit'"
    # In the stats file, any 1.58-bit prose must acknowledge int8 storage nearby.
    stats = _src("phase4_bitnet/compressed_model.py")
    for m in re.finditer(r"1\.58[ -]?bit", stats, re.IGNORECASE):
        window = stats[max(0, m.start() - 220) : m.end() + 220].lower()
        assert "int8" in window, "1.58-bit claim without int8-storage honesty note"


# --------------------------------------------------------------------------
# WAVE 3 - Class C: wire the real twin / delete the broken
# --------------------------------------------------------------------------


def test_globalmoo_cloud_stub_fails_loud_not_fake():
    """moo_bridge.GlobalMOOAdapter cloud path is NOT implemented. Behavioral teeth:
    it must fail loudly (raise) rather than return a fabricated job id, even with
    a key present - a silent fake job id would be the lie. The real backend is
    local pymoo (MOORunner)."""
    from src.cross_phase.meta_calculus.moo_bridge import GlobalMOOAdapter

    adapter = GlobalMOOAdapter(api_key="present")
    assert adapter.is_available() is True
    with pytest.raises((NotImplementedError, RuntimeError)):
        adapter.submit_problem(problem=None, config=None)
    # the specific old "pending" lie must stay gone
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


def test_phase2_quick_fitness_is_labeled_proxy():
    """The default _quick_fitness perplexity/accuracy are a parameter-variance
    PROXY, not measured task fitness. It must be labeled so a champion selected on
    it cannot be mistaken for one selected on a real benchmark (research-ready =
    every number measured OR labeled proxy)."""
    from src.phase2_evomerge.phase2_pipeline import Phase2Pipeline

    result = Phase2Pipeline()._quick_fitness(_tiny_linear(7))
    assert result.get("is_proxy") is True, "proxy fitness is not labeled is_proxy"
    assert result.get("fitness_method") == "proxy_param_variance"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

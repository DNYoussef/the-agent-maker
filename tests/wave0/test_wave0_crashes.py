"""Wave 0 crash gate: each test reproduces a confirmed crash bug in isolation.
Fail-first: these are RED before the fix and GREEN after. Hermetic, no heavy training.
Run: PYTHONPATH=<wt>;<wt>/src PYTHONIOENCODING=utf-8 pytest tests/wave0 -q
"""
import pytest
import torch


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


def test_self_modeling_quick_evaluate_deepcopies_model():
    # Bug 0.4: _quick_evaluate did `model_copy = model` then trained it, corrupting
    # the real model across every Pareto candidate. Must deep-copy.
    import inspect

    import phase5_curriculum.self_modeling as sm

    src = inspect.getsource(sm)
    assert (
        "model_copy = model\n" not in src and "model_copy = model " not in src
    ), "no-clone bug still present (`model_copy = model`)"
    assert "copy.deepcopy(model)" in src, "expected deepcopy of the model before training"


def test_meta_calculus_runner_has_no_undefined_name():
    # Bug 0.5: HybridMOORunner.run referenced free name `runner_config` -> NameError.
    import inspect

    from cross_phase.meta_calculus.moo_utils.globalmoo_adapter import HybridMOORunner

    run_src = inspect.getsource(HybridMOORunner.run)
    assert "runner_config" not in run_src, "undefined `runner_config` still referenced in run()"
    assert "self.config.objective_minimize" in run_src


def test_vptq_default_config_disables_crashing_residual_path():
    # Bug 0.7: default use_residual=True + num_codebooks=4 crashed on any real 2D
    # layer (residual packing returns dicts; size/decompress assume tensors).
    from phase8_compression.vptq import VPTQConfig

    assert VPTQConfig().use_residual is False, "crashing residual path must be off by default"


def test_prompt_baking_without_calibration_data_no_typeerror():
    # Bug 0.2: phase5 called bake_prompt without the required calibration_data ->
    # opaque TypeError. Now optional: honest logged no-op returning the model.
    from cross_phase.prompt_baking.baker import PromptBaker, PromptBakingConfig

    m = torch.nn.Linear(4, 4)
    baker = PromptBaker(PromptBakingConfig())
    out = baker.bake_prompt(m, "be a reasoning specialist", tokenizer=None)
    assert out is m


def test_phase2_merge_dispatch_handles_all_operator_signatures():
    # Bug 0.1: evolutionary path called every merge operator with a single list,
    # but DARE wants (finetuned, base) and DFS/TIES/Franken want (target, refs:List)
    # -> TypeError. _merge_models dispatches by signature; verify every shape works.
    import torch.nn as nn

    from phase2_evomerge.merge.dare_merge import DAREMerge  # merge(finetuned, base)
    from phase2_evomerge.merge.linear_merge import LinearMerge  # merge(models: List)
    from phase2_evomerge.merge.ties_merge import TIESMerge  # merge(target, refs: List)
    from phase2_evomerge.phase2_pipeline import Phase2Pipeline

    def tiny():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(8, 8))

    for Op in (LinearMerge, DAREMerge, TIESMerge):
        out = Phase2Pipeline._merge_models(Op(), [tiny(), tiny()])
        assert isinstance(out, nn.Module), f"{Op.__name__} dispatch returned non-module"


def test_phase7_adas_fail_closed_and_engine_skips_honestly():
    # Bug 0.6: engine called adas.optimize with no evaluator; ADAS correctly refuses
    # synthetic fitness (raises), but the engine's broad except swallowed it into a
    # silent success=False. Verify the fail-closed contract + the honest skip.
    import inspect

    from phase7_experts import experts_engine
    from phase7_experts.adas.evaluation import evaluate_individual

    with pytest.raises(RuntimeError):
        evaluate_individual(None, None, None, None, evaluator=None)

    src = inspect.getsource(experts_engine)
    assert (
        "except RuntimeError" in src and "adas_skipped" in src
    ), "engine must catch the no-evaluator RuntimeError and skip ADAS honestly"

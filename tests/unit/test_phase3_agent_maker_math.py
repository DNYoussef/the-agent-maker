"""Phase 3 regression tests for Agent Maker math and validation fixes."""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cross_phase.storage.model_registry import ModelRegistry  # noqa: E402
from phase1_cognate.model.components.attention import SlidingWindowAttention  # noqa: E402
from phase2_evomerge.fitness.composite import (  # noqa: E402
    DEFAULT_WEIGHTS,
    compute_composite_fitness,
)
from phase6_baking.validation import create_standard_benchmark_suite  # noqa: E402
from phase7_experts.svf_trainer import (  # noqa: E402
    REINFORCEConfig,
    REINFORCETrainer,
    SVFConfig,
    SVFPolicy,
    SVFTrainer,
)
from phase8_compression.hypercompression import HyperCompressor, HyperConfig  # noqa: E402
from phase8_compression.seedlm import SeedLMCompressor, SeedLMConfig  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"


def test_seedlm_compressed_size_charges_actual_seed_tensor_storage():
    compressor = SeedLMCompressor(SeedLMConfig(seed_bits=8, block_size=4, num_iterations=1))
    seeds = torch.zeros(10, dtype=torch.int64)
    compressed_state = {
        "linear.weight": {
            "type": "seedlm",
            "seeds": seeds,
            "scale": torch.tensor(1.0),
            "shape": torch.Size([40]),
        }
    }

    compressed_bytes = compressor._calculate_compressed_size(compressed_state) * 1024 * 1024
    assert compressed_bytes == pytest.approx(10 * 8 + 4 + 32)

    layer_ratio = compressor._calculate_compression(torch.zeros(40), seeds)
    assert layer_ratio == pytest.approx((40 * 4) / (10 * 8 + 36))


def test_seedlm_retention_is_reconstruction_fidelity_not_normalized_mae_proxy():
    # P8 contract: _compress_tensor returns (seeds, coeffs, scale, retention) and retention is
    # the real reconstruction fidelity (not a normalized-MAE proxy). Use a LOSSY config
    # (latent_dim < block_size) so fidelity is genuinely < 1.
    torch.manual_seed(0)
    compressor = SeedLMCompressor(
        SeedLMConfig(seed_bits=8, block_size=16, latent_dim=4, num_iterations=4)
    )
    tensor = torch.randn(4, 4)  # 16 elements -> one block of 16, k=4 < 16 (lossy)

    seeds, coeffs, scale, retention = compressor._compress_tensor(tensor)
    reconstructed = compressor._decompress_tensor(seeds, scale, tensor.shape, None, coeffs)
    expected = compressor._calculate_reconstruction_fidelity(tensor, reconstructed)

    assert retention == pytest.approx(expected)
    assert retention < 0.99  # lossy: k (4) < block (16)


def test_hypercompression_decode_uses_encoded_segment_lengths(monkeypatch):
    compressor = HyperCompressor(HyperConfig(num_params=2, curve_type="polynomial", num_segments=3))
    evaluate_lengths = []

    def fake_fit_segment(segment):
        return torch.tensor([float(len(segment)), 0.0])

    def fake_evaluate_curve(params, n):
        evaluate_lengths.append(n)
        return torch.full((n,), params[0].item())

    monkeypatch.setattr(compressor, "_fit_segment", fake_fit_segment)
    monkeypatch.setattr(compressor, "_evaluate_curve", fake_evaluate_curve)

    curve_params, segment_lengths, _retention, _metrics = compressor._fit_curves(
        torch.arange(100.0)
    )
    assert segment_lengths == [34, 33, 33]
    assert evaluate_lengths == segment_lengths

    evaluate_lengths.clear()
    reconstructed = compressor._decompress_tensor(
        curve_params,
        torch.Size([100]),
        "polynomial",
        segment_lengths,
    )

    assert evaluate_lengths == segment_lengths
    assert reconstructed.numel() == 100
    assert torch.equal(reconstructed.flatten()[:34], torch.full((34,), 34.0))
    assert torch.equal(reconstructed.flatten()[34:], torch.full((66,), 33.0))


def test_phase6_standard_benchmark_suite_fails_closed_until_real_evaluators_exist():
    suite = create_standard_benchmark_suite()

    assert set(suite) == {
        "swe_bench",
        "math",
        "commonsense_qa",
        "human_eval",
        "gsm8k",
    }

    with pytest.raises(RuntimeError, match="SWE-Bench evaluator is not configured"):
        suite["swe_bench"](nn.Linear(1, 1))


def test_model_registry_uses_thread_local_connections(tmp_path):
    registry = ModelRegistry(str(tmp_path / "registry.db"))
    main_connection_id = id(registry.conn)

    def register_from_worker():
        registry.create_session("worker-session", {"phase": "phase3"})
        return id(registry.conn)

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker_connection_id = pool.submit(register_from_worker).result()

    row = registry.conn.execute(
        "SELECT status FROM sessions WHERE session_id = ?",
        ("worker-session",),
    ).fetchone()

    assert worker_connection_id != main_connection_id
    assert row[0] == "running"
    registry.close()


def test_model_registry_register_model_accepts_future_artifact_path(tmp_path):
    registry = ModelRegistry(str(tmp_path / "registry.db"))
    future_model_path = tmp_path / "future-model.pt"

    model_id = registry.register_model(
        session_id="session-a",
        phase_name="phase2",
        model_name="candidate",
        model_path=str(future_model_path),
        metadata={"parameters": 123, "size_mb": 12.5},
    )

    row = registry.conn.execute(
        "SELECT model_path, size_mb, parameters FROM models WHERE model_id = ?",
        (model_id,),
    ).fetchone()

    assert row[0] == str(future_model_path)
    assert row[1] == pytest.approx(12.5)
    assert row[2] == 123
    registry.close()


def test_composite_fitness_caps_components_so_weights_bound_contribution():
    no_speed = compute_composite_fitness(
        perplexity=15.0,
        accuracy=0.5,
        speed=0.0,
        memory=500.0,
    )
    absurd_speed = compute_composite_fitness(
        perplexity=15.0,
        accuracy=0.5,
        speed=120_000.0,
        memory=500.0,
    )

    assert absurd_speed["components"]["perplexity_score"] == pytest.approx(1.0)
    assert absurd_speed["components"]["speed_score"] == pytest.approx(1.0)
    assert 0.0 <= absurd_speed["composite"] <= 1.0
    assert absurd_speed["composite"] - no_speed["composite"] == pytest.approx(
        DEFAULT_WEIGHTS["speed"]
    )


def test_composite_fitness_rejects_accuracy_outside_unit_interval():
    with pytest.raises(ValueError, match="Accuracy must be between"):
        compute_composite_fitness(
            perplexity=15.0,
            accuracy=45.2,
            speed=1200.0,
            memory=500.0,
        )


def test_reinforce_baseline_is_subtracted_before_reward_normalization():
    torch.manual_seed(0)
    policy = SVFPolicy(task_embed_dim=3, num_singular_values=4, hidden_dim=8)
    trainer = REINFORCETrainer(
        policy,
        REINFORCEConfig(
            baseline_decay=0.5,
            entropy_coeff=0.0,
            normalize_rewards=True,
            warmup_steps=0,
        ),
    )
    trainer.baseline = 100.0

    metrics = trainer.update(
        log_probs=torch.tensor([0.1, 0.2, 0.3], requires_grad=True),
        rewards=torch.tensor([100.0, 102.0, 104.0]),
        task_embeddings=torch.randn(3, 3),
    )

    assert abs(metrics["mean_advantage"]) < 1e-6
    assert metrics["baseline"] == pytest.approx(101.0)


def test_svf_training_fails_closed_when_all_samples_error():
    class FailingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, **_inputs):
            raise RuntimeError("sample failed")

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    trainer = SVFTrainer(SVFConfig(num_singular_values=2, num_epochs=1, batch_size=1))

    _model, result = trainer.train_expert(
        model=FailingModel(),
        expert_id=7,
        expert_capabilities=["reasoning"],
        tokenizer=Tokenizer(),
        training_data=[{"prompt": "bad sample"}],
    )

    assert result.success is False
    assert result.metrics["successful_batches"] == 0
    assert result.metrics["failed_batches"] == 1
    assert "No SVF training samples" in result.metrics["error"]


def test_sliding_window_attention_is_vectorized_and_matches_per_token_reference():
    # Wave-2 efficiency CONTRACT CHANGE (was: assert no full [b,h,s,s] matmul / per-token
    # loop). SWA was vectorized into a single masked softmax: the old per-token loop kept
    # score memory O(s*w) but built one autograd node per position - the real activation-
    # memory hog (measured: a 222M b4/s256 train step went 14.4 GB OOM -> 5.0 GB after
    # vectorizing). The vectorized path materializes a dense [b,h,s,s] score matrix
    # (O(s^2)); that is a documented ceiling for very long seq, and a banded/flash kernel
    # is deferred Wave-2 work. This test now guards the property that matters - CORRECTNESS
    # (vectorised == the per-token windowed-causal reference) - not the old implementation.
    seq_len = 16
    attention = SlidingWindowAttention(d_model=8, n_heads=2, window=4, dropout=0.0).eval()
    q = torch.randn(1, 2, seq_len, 4)
    k = torch.randn(1, 2, seq_len, 4)
    v = torch.randn(1, 2, seq_len, 4)

    output = attention._sliding_window_attn(q, k, v, mask=None)

    wh = attention.window // 2
    ref = torch.zeros_like(q)
    for pos in range(seq_len):
        start = max(0, pos - wh)
        end = pos + 1  # causal default
        sc = (q[:, :, pos : pos + 1, :] @ k[:, :, start:end, :].transpose(-2, -1)) * attention.scale
        ref[:, :, pos : pos + 1, :] = sc.softmax(dim=-1) @ v[:, :, start:end, :]

    assert output.shape == q.shape
    assert torch.allclose(output, ref, atol=1e-5), f"max diff {(output - ref).abs().max().item()}"


def test_thought_injector_is_reentrant_without_instance_last_injection_state():
    spec = importlib.util.spec_from_file_location(
        "legacy_quietstar_architecture",
        SRC_ROOT / "phase3_quietstar" / "architecture.py",
    )
    legacy_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(legacy_module)

    from phase3_quietstar.architecture.thought_injector import (
        ThoughtInjector as ModularThoughtInjector,
    )

    logits = torch.ones(1, 8)

    for injector_cls in (legacy_module.ThoughtInjector, ModularThoughtInjector):
        injector = injector_cls(threshold=0.1, min_interval=3)
        assert not hasattr(injector, "last_injection")
        assert injector(logits, None, None, position=0) is True
        assert injector(logits, None, None, position=0) is True
        assert injector(logits, None, None, position=1, last_injection=0) is False


def test_adas_optimizer_import_path_delegates_to_modular_owner():
    from phase7_experts.adas.optimizer import ADASOptimizer as ModularADASOptimizer
    from phase7_experts.adas_optimizer import ADASConfig, ADASOptimizer

    assert issubclass(ADASOptimizer, ModularADASOptimizer)

    optimizer = ADASOptimizer(ADASConfig(population_size=4, tournament_size=2))
    optimizer._initialize_population(num_experts=3)
    for rank, individual in enumerate(optimizer.population):
        individual.rank = rank
        individual.crowding_distance = float(rank)

    selected = optimizer._tournament_selection()
    assert len(selected) == 4


def test_phase1_training_smoke_script_is_not_importable_from_src_package():
    assert importlib.util.find_spec("phase1_cognate.test_training") is None
    assert not (SRC_ROOT / "phase1_cognate" / "test_training.py").exists()
    assert (REPO_ROOT / "tests" / "sandbox" / "phase1_training_pipeline_check.py").exists()


def test_phase5_curriculum_engine_has_single_canonical_owner():
    from phase5_curriculum.curriculum_engine import CurriculumEngine as TopLevelEngine
    from phase5_curriculum.engine.curriculum_engine import CurriculumEngine as EnginePackageEngine

    assert EnginePackageEngine is TopLevelEngine


def test_phase2_pipeline_fails_loudly_when_configured_merger_import_fails(monkeypatch):
    from phase2_evomerge import phase2_pipeline

    real_import_module = phase2_pipeline.importlib.import_module

    def fake_import_module(module_name):
        if module_name == "phase2_evomerge.merge.slerp_merge":
            raise ImportError("planted missing slerp")
        return real_import_module(module_name)

    monkeypatch.setattr(phase2_pipeline.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="slerp.*planted missing slerp"):
        phase2_pipeline.Phase2Pipeline(phase2_pipeline.EvolutionConfig(merge_techniques=["slerp"]))

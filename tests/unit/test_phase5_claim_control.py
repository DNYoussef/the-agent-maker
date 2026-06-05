"""Phase 5 claim-control regressions for Agent Maker."""

from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cross_phase.utils import validate_diversity
from phase4_bitnet.config import Phase4Config
from phase4_bitnet.phase_controller import Phase4Controller
from phase5_curriculum.curriculum_engine import CurriculumConfig, CurriculumEngine
from phase6_baking.swe_bench_eval import MissingSWEBenchDataError, SWEBenchEvaluator
from phase7_experts.adas.config import Individual
from phase7_experts.adas.evaluation import evaluate_individual
from phase8_compression.benchmarks import BenchmarkConfig, MMLUBenchmark


def test_mmlu_question_bank_does_not_fabricate_missing_subjects():
    benchmark = MMLUBenchmark(BenchmarkConfig(mmlu_subjects=99, mmlu_samples_per_subject=3, device="cpu"))

    questions = benchmark.get_questions(num_subjects=99, samples_per_subject=3)

    available_subjects = set(benchmark.SAMPLE_QUESTIONS)
    assert questions
    assert {q["subject"] for q in questions}.issubset(available_subjects)
    assert all(not q["question"].startswith("Sample ") for q in questions)
    assert all(q["answer"] in {"A", "B", "C", "D"} for q in questions)
    assert all(q["evidence_status"] == "local_sample_bank_not_official_mmlu" for q in questions)
    assert len(questions) == sum(min(3, len(items)) for items in benchmark.SAMPLE_QUESTIONS.values())


def test_phase4_fine_tuning_decision_uses_measured_quality_gate():
    config = Phase4Config(device="cpu", wandb_enabled=False, fine_tune_threshold=0.05)
    controller = Phase4Controller(config)

    assert controller._check_fine_tuning_needed({"perplexity": 10.0}, {"perplexity": 10.4}) is False
    assert controller.metrics["fine_tune_quality_gate"]["degradation"] == pytest.approx(0.04)

    assert controller._check_fine_tuning_needed({"perplexity": 10.0}, {"perplexity": 10.6}) is True
    assert controller.metrics["fine_tune_quality_gate"]["degradation"] == pytest.approx(0.06)

    disabled = Phase4Controller(Phase4Config(device="cpu", wandb_enabled=False, enable_fine_tuning=False))
    assert disabled._check_fine_tuning_needed({"perplexity": 10.0}, {"perplexity": 99.0}) is False


def test_adas_fitness_requires_real_evaluator_and_uses_supplied_one():
    individual = Individual(routing_weights=[0.5, 0.5], expert_configs={}, fitness_scores={})
    model = nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="fitness evaluator is not configured"):
        evaluate_individual(individual, model, experts=[], tokenizer=None, evaluator=None)

    result = evaluate_individual(
        individual,
        model,
        experts=[],
        tokenizer=None,
        evaluator=lambda *_args: {"accuracy": 0.91, "latency": 0.25, "diversity": 0.8},
    )

    assert result == {"accuracy": 0.91, "latency": 0.25, "diversity": 0.8}


def _linear_with_weight(value: float) -> nn.Linear:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def test_edge_of_chaos_diversity_validation_can_fail_and_pass():
    identical = [_linear_with_weight(1.0), _linear_with_weight(1.0), _linear_with_weight(1.0)]
    with pytest.raises(AssertionError, match="Model diversity below threshold"):
        validate_diversity(*identical, min_weight_diversity=0.01, min_parameter_delta=100)

    diverse = [_linear_with_weight(1.0), _linear_with_weight(-1.0), _linear_with_weight(0.5)]
    result = validate_diversity(*diverse, min_weight_diversity=0.01, min_parameter_delta=100)
    assert result["passed"] is True
    assert result["weight_diversity"] >= 0.01


def test_swe_bench_missing_dataset_fails_closed_unless_explicit_demo_mode(tmp_path):
    evaluator = SWEBenchEvaluator(data_path=str(tmp_path / "missing-swe-bench.json"))

    with pytest.raises(MissingSWEBenchDataError, match="not an official SWE-Bench score"):
        evaluator.load_tasks(max_tasks=3)

    demo = SWEBenchEvaluator(
        data_path=str(tmp_path / "missing-swe-bench.json"),
        allow_synthetic_tasks=True,
    )
    tasks = demo.load_tasks(max_tasks=3)
    assert len(tasks) == 3
    assert all(task.instance_id.startswith("synthetic-") for task in tasks)


def test_readme_does_not_claim_nonexistent_status_json():
    readme = (Path(__file__).parents[2] / "README.md").read_text(encoding="utf-8")
    status_block = readme.split("<!-- STATUS:START -->", 1)[1].split("<!-- STATUS:END -->", 1)[0]

    assert "2026-EXOSKELETON-STATUS.json" not in status_block
    assert "Status: 95%" not in status_block
    assert "does not ship a repository-local percentage-complete status file" in status_block


def test_phase5_curriculum_runs_offline_without_frontier_client():
    config = CurriculumConfig(
        num_levels=2,
        questions_per_level=2,
        bake_after_each_level=False,
        offline_mode=True,
    )
    engine = CurriculumEngine(config=config)
    model = nn.Linear(2, 2)

    result = engine.run(model, tokenizer=None, frontier_client=None, coding_env=None)

    assert result.success is True
    assert result.levels_completed == 2
    assert result.artifacts["assessment_results"]["evidence_status"] == "offline_deterministic_assessment"
    assert result.artifacts["curriculum_stats"]["total_questions"] == 4
    assert all(
        progress.mastered_questions == 2 and progress.current_questions == 0
        for progress in result.artifacts["level_progress"]
    )

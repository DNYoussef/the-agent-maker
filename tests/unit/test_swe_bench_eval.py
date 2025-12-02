"""
Unit tests for Phase 6: SWE-Bench Evaluation

Tests:
- SWEBenchTask dataclass
- EvaluationResult dataclass
- SWEBenchEvaluator class
- EvaluationMode enum

Target: >=90% coverage for SWE-Bench evaluation
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from phase6_baking.swe_bench_eval import (
    SWEBenchEvaluator,
    SWEBenchTask,
    EvaluationResult,
    EvaluationMode,
)


class TestSWEBenchTask:
    """Test SWEBenchTask dataclass."""

    def test_basic_creation(self):
        """Test basic task creation."""
        task = SWEBenchTask(
            instance_id="test-1",
            repo="test/repo",
            base_commit="abc123",
            problem_statement="Fix the bug"
        )

        assert task.instance_id == "test-1"
        assert task.repo == "test/repo"
        assert task.base_commit == "abc123"
        assert task.problem_statement == "Fix the bug"

    def test_optional_fields(self):
        """Test optional fields have defaults."""
        task = SWEBenchTask(
            instance_id="test-1",
            repo="test/repo",
            base_commit="abc123",
            problem_statement="Fix it"
        )

        assert task.hints_text == ""
        assert task.patch == ""
        assert task.test_patch == ""

    def test_all_fields(self):
        """Test task with all fields."""
        task = SWEBenchTask(
            instance_id="django-123",
            repo="django/django",
            base_commit="abc123def",
            problem_statement="QuerySet filter fails on empty list",
            hints_text="Check the filter method",
            patch="diff content here",
            test_patch="test diff content"
        )

        assert task.instance_id == "django-123"
        assert task.hints_text == "Check the filter method"
        assert task.patch == "diff content here"

    def test_from_dict(self):
        """Test creating task from dictionary."""
        data = {
            "instance_id": "test-from-dict",
            "repo": "test/repo",
            "base_commit": "commit123",
            "problem_statement": "Test problem",
            "patch": "test patch"
        }

        task = SWEBenchTask.from_dict(data)

        assert task.instance_id == "test-from-dict"
        assert task.repo == "test/repo"
        assert task.patch == "test patch"


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = EvaluationResult(
            instance_id="test-1",
            exact_match=0.0,
            partial_match=0.5,
            syntax_valid=1.0,
            semantic_similar=0.7,
            prediction="def fix(): pass"
        )

        assert result.instance_id == "test-1"
        assert result.exact_match == 0.0
        assert result.partial_match == 0.5
        assert result.syntax_valid == 1.0

    def test_default_values(self):
        """Test default values."""
        result = EvaluationResult(
            instance_id="test-1",
            exact_match=0.0,
            partial_match=0.0,
            syntax_valid=0.0,
            semantic_similar=0.0
        )

        assert result.prediction == ""
        assert result.ground_truth == ""
        assert result.error is None

    def test_optional_error(self):
        """Test optional error field."""
        result = EvaluationResult(
            instance_id="test-1",
            exact_match=0.0,
            partial_match=0.0,
            syntax_valid=0.0,
            semantic_similar=0.0,
            error="Generation failed"
        )

        assert result.error == "Generation failed"

    def test_composite_score(self):
        """Test composite score calculation."""
        result = EvaluationResult(
            instance_id="test-1",
            exact_match=1.0,
            partial_match=1.0,
            syntax_valid=1.0,
            semantic_similar=1.0
        )

        # 0.4*1 + 0.3*1 + 0.2*1 + 0.1*1 = 1.0
        assert result.composite_score == pytest.approx(1.0)

    def test_composite_score_partial(self):
        """Test composite score with partial values."""
        result = EvaluationResult(
            instance_id="test-1",
            exact_match=0.0,
            partial_match=0.5,
            syntax_valid=1.0,
            semantic_similar=0.0
        )

        # 0.4*0 + 0.3*0.5 + 0.2*1 + 0.1*0 = 0.35
        assert result.composite_score == pytest.approx(0.35)


class TestEvaluationMode:
    """Test EvaluationMode enum."""

    def test_modes_exist(self):
        """Test all modes exist."""
        assert hasattr(EvaluationMode, 'EXACT')
        assert hasattr(EvaluationMode, 'SEMANTIC')
        assert hasattr(EvaluationMode, 'PARTIAL')
        assert hasattr(EvaluationMode, 'SYNTAX')

    def test_mode_values(self):
        """Test mode values."""
        assert EvaluationMode.EXACT.value == "exact"
        assert EvaluationMode.SEMANTIC.value == "semantic"
        assert EvaluationMode.PARTIAL.value == "partial"
        assert EvaluationMode.SYNTAX.value == "syntax"


class TestSWEBenchEvaluator:
    """Test SWEBenchEvaluator class."""

    def test_initialization_no_data(self):
        """Test initialization without data path."""
        evaluator = SWEBenchEvaluator(data_path=None)

        assert evaluator.data_path is None
        assert evaluator.tasks == []
        assert evaluator.results == []

    def test_initialization_with_path(self):
        """Test initialization with data path."""
        evaluator = SWEBenchEvaluator(data_path="test/path.json")

        assert evaluator.data_path == Path("test/path.json")

    def test_load_tasks_synthetic(self):
        """Test loading synthetic tasks (no data file)."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.load_tasks(max_tasks=10)

        assert len(evaluator.tasks) == 10
        assert all(isinstance(t, SWEBenchTask) for t in evaluator.tasks)

    def test_load_tasks_max_limit(self):
        """Test max_tasks limits loaded tasks."""
        evaluator = SWEBenchEvaluator(data_path=None)

        evaluator.load_tasks(max_tasks=5)
        assert len(evaluator.tasks) == 5

        evaluator.load_tasks(max_tasks=20)
        assert len(evaluator.tasks) == 20

    def test_synthetic_task_structure(self):
        """Test synthetic tasks have valid structure."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.load_tasks(max_tasks=5)

        for task in evaluator.tasks:
            assert task.instance_id is not None
            assert task.problem_statement is not None
            assert len(task.problem_statement) > 0

    def test_evaluate_single_basic(self):
        """Test evaluating a single task."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.load_tasks(max_tasks=1)

        task = evaluator.tasks[0]
        generated = "def fix(): pass"

        result = evaluator.evaluate_single(task, generated)

        assert isinstance(result, EvaluationResult)
        assert result.instance_id == task.instance_id

    def test_evaluate_single_syntax_check(self):
        """Test syntax validation in evaluation."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.load_tasks(max_tasks=1)

        task = evaluator.tasks[0]

        # Valid Python - syntax_valid should be 1.0
        result_valid = evaluator.evaluate_single(task, "def fix():\n    return 1")
        assert result_valid.syntax_valid == 1.0

        # Invalid Python - syntax_valid should be 0.0
        result_invalid = evaluator.evaluate_single(task, "def fix( return")
        assert result_invalid.syntax_valid == 0.0

    def test_evaluate_single_empty_generation(self):
        """Test evaluation with empty generation."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.load_tasks(max_tasks=1)

        task = evaluator.tasks[0]
        result = evaluator.evaluate_single(task, "")

        assert result.exact_match == 0.0

    def test_run_evaluation_basic(self):
        """Test running full evaluation."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.load_tasks(max_tasks=5)

        def dummy_generate(problem: str) -> str:
            return "def fix(): pass"

        metrics = evaluator.run_evaluation(
            generate_fn=dummy_generate,
            max_tasks=5,
            verbose=False
        )

        assert 'composite_score' in metrics
        assert metrics['composite_score'] >= 0.0
        assert metrics['composite_score'] <= 1.0

    def test_run_evaluation_stores_results(self):
        """Test evaluation stores results."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.load_tasks(max_tasks=5)

        def dummy_generate(problem: str) -> str:
            return "def fix(): pass"

        evaluator.run_evaluation(
            generate_fn=dummy_generate,
            max_tasks=5,
            verbose=False
        )

        assert len(evaluator.results) == 5

    def test_aggregate_results(self):
        """Test aggregating results into metrics."""
        evaluator = SWEBenchEvaluator(data_path=None)

        # Manually add results
        evaluator.results = [
            EvaluationResult("t1", exact_match=1.0, partial_match=1.0, syntax_valid=1.0, semantic_similar=1.0),
            EvaluationResult("t2", exact_match=0.0, partial_match=0.5, syntax_valid=1.0, semantic_similar=0.5),
            EvaluationResult("t3", exact_match=0.0, partial_match=0.0, syntax_valid=0.0, semantic_similar=0.0),
        ]

        metrics = evaluator.aggregate_results()

        assert 'composite_score' in metrics
        assert 'exact_match' in metrics

    def test_reset(self):
        """Test reset clears results."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.results = [Mock(), Mock()]

        evaluator.reset()

        assert evaluator.results == []

    def test_get_failed_tasks(self):
        """Test getting failed tasks."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.load_tasks(max_tasks=3)

        # Manually set results with varying scores
        evaluator.results = [
            EvaluationResult("t1", exact_match=0.0, partial_match=0.8, syntax_valid=1.0, semantic_similar=0.9),
            EvaluationResult("t2", exact_match=0.0, partial_match=0.2, syntax_valid=0.0, semantic_similar=0.1),
            EvaluationResult("t3", exact_match=1.0, partial_match=1.0, syntax_valid=1.0, semantic_similar=1.0),
        ]

        failed = evaluator.get_failed_tasks(threshold=0.5)

        # Should return tasks where score < threshold
        assert len(failed) >= 1


class TestSWEBenchWithMockModel:
    """Test SWE-Bench evaluation with mock model."""

    def test_evaluate_with_mock_model(self):
        """Test evaluation with mocked model."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.load_tasks(max_tasks=3)

        call_count = [0]

        def generate_fn(problem: str) -> str:
            call_count[0] += 1
            return f"def solution_{call_count[0]}(): pass"

        metrics = evaluator.run_evaluation(
            generate_fn=generate_fn,
            max_tasks=3,
            verbose=False
        )

        assert call_count[0] == 3, "Should call generate for each task"
        assert 'composite_score' in metrics


class TestLoadFromFile:
    """Test loading tasks from file."""

    def test_load_from_json_file(self):
        """Test loading tasks from JSON file."""
        # Create temporary JSON file
        tasks_data = [
            {
                "instance_id": "test-1",
                "repo": "test/repo1",
                "problem_statement": "Fix bug 1",
                "base_commit": "abc123",
                "patch": "diff 1",
                "test_patch": "test 1"
            },
            {
                "instance_id": "test-2",
                "repo": "test/repo2",
                "problem_statement": "Fix bug 2",
                "base_commit": "def456",
                "patch": "diff 2",
                "test_patch": "test 2"
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(tasks_data, f)
            temp_path = f.name

        try:
            evaluator = SWEBenchEvaluator(data_path=temp_path)
            evaluator.load_tasks(max_tasks=10)

            assert len(evaluator.tasks) == 2
            assert evaluator.tasks[0].instance_id == "test-1"
            assert evaluator.tasks[1].instance_id == "test-2"
        finally:
            Path(temp_path).unlink()

    def test_load_from_nonexistent_file(self):
        """Test loading from nonexistent file falls back to synthetic."""
        evaluator = SWEBenchEvaluator(data_path="nonexistent/path.json")
        evaluator.load_tasks(max_tasks=5)

        # Should fall back to synthetic tasks
        assert len(evaluator.tasks) == 5


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_results_aggregate(self):
        """Test aggregating empty results."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.results = []

        metrics = evaluator.aggregate_results()

        assert metrics['composite_score'] == 0.0

    def test_very_long_problem_statement(self):
        """Test with very long problem statement."""
        evaluator = SWEBenchEvaluator(data_path=None)

        long_problem = "Fix the bug. " * 1000
        task = SWEBenchTask(
            instance_id="long-1",
            repo="test/repo",
            base_commit="abc123",
            problem_statement=long_problem
        )

        result = evaluator.evaluate_single(task, "def fix(): pass")

        assert result.instance_id == "long-1"

    def test_unicode_in_generation(self):
        """Test handling unicode in generated code."""
        evaluator = SWEBenchEvaluator(data_path=None)
        evaluator.load_tasks(max_tasks=1)

        task = evaluator.tasks[0]
        unicode_code = 'def fix():\n    return "Hello"'

        result = evaluator.evaluate_single(task, unicode_code)

        assert result is not None
        # syntax_valid is 1.0 for valid syntax (float, not bool)
        assert result.syntax_valid == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

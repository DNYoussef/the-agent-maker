"""
Phase 6: Cross-Task Catastrophic Forgetting Validation

Tests that baking on task A doesn't degrade performance on tasks B, C, D.
Paper Figure 6: Cross-task performance heatmap shows <3.4% max decrease.

Research: "Prompt Baking" (arXiv:2409.13697v1)
Key Finding: Prompt baking is remarkably robust to catastrophic forgetting.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ValidationConfig:
    """Configuration for cross-task validation."""

    max_acceptable_degradation: float = 0.034  # Paper: 3.4% max decrease
    min_task_score: float = 0.5  # Minimum acceptable absolute score
    num_validation_samples: int = 100  # Samples per task
    temperature: float = 0.7  # Generation temperature


@dataclass
class TaskResult:
    """Result for a single task."""

    task_name: str
    base_score: float  # Score before baking
    baked_score: float  # Score after baking
    degradation: float  # Negative = improvement, positive = degradation
    degradation_percent: float  # As percentage
    passed: bool  # Whether within acceptable bounds
    samples_evaluated: int


@dataclass
class ValidationResult:
    """Result from cross-task validation."""

    success: bool
    baked_task: str  # Task that was baked
    task_results: List[TaskResult]
    max_degradation: float  # Worst degradation across all tasks
    avg_degradation: float  # Average degradation
    tasks_passed: int  # Number of tasks within bounds
    tasks_failed: int  # Number of tasks that degraded too much
    error: Optional[str] = None


class CrossTaskValidator:
    """
    Cross-Task Catastrophic Forgetting Validator.

    Tests whether baking a prompt for task A causes the model to forget
    how to do tasks B, C, D, etc.

    Paper Finding:
        - Baking on one task causes <3.4% degradation on other tasks
        - This is MUCH better than standard fine-tuning (often 20-40% degradation)
        - Baking is naturally resistant to catastrophic forgetting

    Process:
        1. Evaluate base model on all tasks -> baseline scores
        2. Bake prompt for task A
        3. Evaluate baked model on all tasks -> post-bake scores
        4. Compare: degradation[task] = (baseline - post_bake) / baseline
        5. Assert: max(degradation) < 3.4%

    Example:
        >>> validator = CrossTaskValidator()
        >>> tasks = {
        ...     "math": math_evaluator,
        ...     "coding": coding_evaluator,
        ...     "reasoning": reasoning_evaluator,
        ... }
        >>>
        >>> # Bake for "coding"
        >>> result = validator.validate_cross_task_forgetting(
        ...     base_model=model,
        ...     baked_model=baked_coding_model,
        ...     baked_task="coding",
        ...     all_tasks=tasks
        ... )
        >>>
        >>> print(f"Max degradation: {result.max_degradation:.3f}")
        >>> print(f"Tasks passed: {result.tasks_passed}/{len(tasks)}")
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize cross-task validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.metrics = {
            "total_validations": 0,
            "validations_passed": 0,
            "avg_max_degradation": 0.0,
            "worst_degradation_seen": 0.0,
        }

    def validate_cross_task_forgetting(
        self,
        base_model: nn.Module,
        baked_model: nn.Module,
        baked_task: str,
        all_tasks: Dict[str, Callable[[nn.Module], float]],
        tokenizer: Any = None,
    ) -> ValidationResult:
        """
        Validate that baking didn't cause catastrophic forgetting.

        Args:
            base_model: Model before baking
            baked_model: Model after baking on baked_task
            baked_task: Name of task that was baked
            all_tasks: Dict of {task_name: evaluator_function}
                      Evaluator returns score 0.0-1.0
            tokenizer: Optional tokenizer (if evaluators need it)

        Returns:
            ValidationResult with per-task analysis

        Example:
            >>> tasks = {
            ...     "swe_bench": lambda m: evaluate_swe_bench(m),
            ...     "math_qa": lambda m: evaluate_math(m),
            ...     "commonsense": lambda m: evaluate_commonsense(m),
            ... }
            >>> result = validator.validate_cross_task_forgetting(
            ...     base_model, baked_model, "swe_bench", tasks
            ... )
        """
        print(f"Validating cross-task forgetting after baking '{baked_task}'")
        print(f"Testing on {len(all_tasks)} tasks...")

        task_results = []
        all_passed = True

        for task_name, evaluator in all_tasks.items():
            print(f"\n  Evaluating task: {task_name}")

            try:
                # Evaluate base model
                print(f"    Base model...", end=" ")
                base_score = evaluator(base_model)
                print(f"{base_score:.4f}")

                # Evaluate baked model
                print(f"    Baked model...", end=" ")
                baked_score = evaluator(baked_model)
                print(f"{baked_score:.4f}")

                # Calculate degradation
                if base_score > 0:
                    degradation = (base_score - baked_score) / base_score
                    degradation_percent = degradation * 100
                else:
                    degradation = 0.0
                    degradation_percent = 0.0

                # Check if passed
                passed = (
                    degradation <= self.config.max_acceptable_degradation
                    and baked_score >= self.config.min_task_score
                )

                if not passed:
                    all_passed = False

                # Create result
                task_result = TaskResult(
                    task_name=task_name,
                    base_score=base_score,
                    baked_score=baked_score,
                    degradation=degradation,
                    degradation_percent=degradation_percent,
                    passed=passed,
                    samples_evaluated=self.config.num_validation_samples,
                )

                task_results.append(task_result)

                # Log status
                status = "PASS" if passed else "FAIL"
                print(
                    f"    Degradation: {degradation_percent:+.2f}% [{status}]"
                )

            except Exception as e:
                print(f"    ERROR: {e}")
                task_results.append(
                    TaskResult(
                        task_name=task_name,
                        base_score=0.0,
                        baked_score=0.0,
                        degradation=0.0,
                        degradation_percent=0.0,
                        passed=False,
                        samples_evaluated=0,
                    )
                )
                all_passed = False

        # Calculate aggregate statistics
        if task_results:
            max_degradation = max(r.degradation for r in task_results)
            avg_degradation = sum(r.degradation for r in task_results) / len(task_results)
            tasks_passed = sum(1 for r in task_results if r.passed)
            tasks_failed = len(task_results) - tasks_passed
        else:
            max_degradation = 0.0
            avg_degradation = 0.0
            tasks_passed = 0
            tasks_failed = 0

        # Print summary
        print(f"\n=== Validation Summary ===")
        print(f"Baked task: {baked_task}")
        print(f"Tasks evaluated: {len(task_results)}")
        print(f"Tasks passed: {tasks_passed}/{len(task_results)}")
        print(f"Max degradation: {max_degradation*100:.2f}% (threshold: {self.config.max_acceptable_degradation*100:.1f}%)")
        print(f"Avg degradation: {avg_degradation*100:.2f}%")

        if all_passed:
            print(f"VALIDATION PASSED: No catastrophic forgetting detected")
        else:
            print(f"VALIDATION FAILED: {tasks_failed} task(s) degraded beyond threshold")

        # Update metrics
        self.metrics["total_validations"] += 1
        if all_passed:
            self.metrics["validations_passed"] += 1

        self.metrics["avg_max_degradation"] = (
            self.metrics["avg_max_degradation"] * (self.metrics["total_validations"] - 1)
            + max_degradation
        ) / self.metrics["total_validations"]

        self.metrics["worst_degradation_seen"] = max(
            self.metrics["worst_degradation_seen"], max_degradation
        )

        return ValidationResult(
            success=all_passed,
            baked_task=baked_task,
            task_results=task_results,
            max_degradation=max_degradation,
            avg_degradation=avg_degradation,
            tasks_passed=tasks_passed,
            tasks_failed=tasks_failed,
        )

    def generate_forgetting_heatmap_data(
        self,
        base_model: nn.Module,
        tasks: Dict[str, Callable[[nn.Module], float]],
        baking_fn: Callable[[nn.Module, str], nn.Module],
        tokenizer: Any = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate data for Paper Figure 6 style heatmap.

        For each task:
            1. Bake model on that task
            2. Evaluate on all tasks
            3. Record degradation matrix

        Args:
            base_model: Base model
            tasks: Dict of {task_name: evaluator}
            baking_fn: Function to bake model for a task
            tokenizer: Optional tokenizer

        Returns:
            Dict[baked_task][eval_task] = degradation_percent

        Example:
            >>> heatmap = validator.generate_forgetting_heatmap_data(
            ...     model, tasks, lambda m, t: bake_for_task(m, t)
            ... )
            >>> # Visualize with matplotlib/seaborn
            >>> import seaborn as sns
            >>> sns.heatmap(pd.DataFrame(heatmap))
        """
        print("Generating cross-task forgetting heatmap...")
        heatmap = {}

        for baked_task in tasks.keys():
            print(f"\nBaking for task: {baked_task}")

            # Bake model
            baked_model = baking_fn(base_model, baked_task)

            # Validate
            result = self.validate_cross_task_forgetting(
                base_model, baked_model, baked_task, tasks, tokenizer
            )

            # Store degradation matrix
            heatmap[baked_task] = {
                r.task_name: r.degradation_percent for r in result.task_results
            }

        print(f"\nHeatmap generation complete")
        print(f"Matrix size: {len(tasks)} x {len(tasks)}")

        return heatmap

    def get_metrics(self) -> Dict:
        """Get validation metrics."""
        return self.metrics.copy()


def create_standard_benchmark_suite() -> Dict[str, Callable[[nn.Module], float]]:
    """
    Create standard benchmark suite for cross-task validation.

    Returns dict of evaluators for common tasks:
        - SWE-Bench (code generation)
        - MATH (mathematical reasoning)
        - CommonsenseQA (commonsense reasoning)
        - HumanEval (code correctness)
        - GSM8K (grade school math)

    Returns:
        Dict[task_name, evaluator_function]
    """
    # This would integrate with actual benchmark implementations
    # For now, return placeholder structure

    def swe_bench_evaluator(model: nn.Module) -> float:
        """SWE-Bench code generation score."""
        # Would call actual SWE-Bench evaluation
        return 0.65  # Placeholder

    def math_evaluator(model: nn.Module) -> float:
        """MATH dataset score."""
        return 0.58  # Placeholder

    def commonsense_evaluator(model: nn.Module) -> float:
        """CommonsenseQA score."""
        return 0.72  # Placeholder

    def humaneval_evaluator(model: nn.Module) -> float:
        """HumanEval pass@1 score."""
        return 0.45  # Placeholder

    def gsm8k_evaluator(model: nn.Module) -> float:
        """GSM8K math reasoning score."""
        return 0.62  # Placeholder

    return {
        "swe_bench": swe_bench_evaluator,
        "math": math_evaluator,
        "commonsense_qa": commonsense_evaluator,
        "human_eval": humaneval_evaluator,
        "gsm8k": gsm8k_evaluator,
    }


__all__ = [
    "CrossTaskValidator",
    "ValidationConfig",
    "ValidationResult",
    "TaskResult",
    "create_standard_benchmark_suite",
]

"""
Phase 5: Edge-of-Chaos Assessment

Finds the optimal learning threshold where model achieves ~75% accuracy.
Based on research: "Intelligence at the Edge of Chaos"

Key insight: Maximum learning occurs at the edge of chaos where
systems exhibit maximum information processing capacity.
"""

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.cross_phase.evaluation.evaluator import matches

import torch
import torch.nn as nn

# Import meta-calculus for MOO edge-of-chaos finding
try:
    from src.cross_phase.meta_calculus.moo_utils import HybridMOORunner

    MOO_AVAILABLE = True
except ImportError:
    MOO_AVAILABLE = False

# OpenRouter client for real question generation
try:
    from .openrouter_client import OpenRouterClient, ModelProvider

    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False


@dataclass
class AssessmentResult:
    """Result of a single assessment question."""

    level: int
    question: str
    correct: bool
    confidence: float
    response: str


class EdgeOfChaosAssessment:
    """
    Finds the edge-of-chaos level for a model.

    The edge of chaos is where accuracy ~= 75%, representing
    the optimal learning zone (Goldilocks zone).

    Process:
    1. Generate questions across difficulty scale 1-100
    2. Test model on each level
    3. Find level where accuracy ~= threshold (75%)
    """

    def __init__(self, threshold: float = 0.75, num_questions: int = 2000, tolerance: float = 0.05):
        """
        Initialize assessment.

        Args:
            threshold: Target accuracy (default 75%)
            num_questions: Total questions for assessment
            tolerance: Acceptable deviation from threshold
        """
        self.threshold = threshold
        self.num_questions = num_questions
        self.tolerance = tolerance

    def find_baseline(
        self, model: nn.Module, tokenizer: Any, frontier_client: Optional[Any] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Find the baseline level where model achieves ~75% accuracy.

        Args:
            model: Model to assess
            tokenizer: Tokenizer for encoding
            frontier_client: Optional client for generating questions

        Returns:
            Tuple of (baseline_level, assessment_results)
        """
        print(f"  Running edge-of-chaos assessment...")
        print(f"  Target threshold: {self.threshold:.0%}")

        # Generate or load assessment questions
        questions = self._generate_assessment_questions(frontier_client)

        # Test model on each difficulty level
        level_accuracies = {}
        level_results = {}

        model.eval()
        with torch.no_grad():
            for level in range(1, 101, 5):  # Sample every 5 levels for speed
                level_questions = [q for q in questions if q["level"] == level]

                if not level_questions:
                    # Generate placeholder questions for this level
                    level_questions = self._generate_level_questions(level, 20)

                correct = 0
                total = len(level_questions)

                for q in level_questions:
                    result = self._evaluate_question(model, tokenizer, q)
                    if result.correct:
                        correct += 1

                accuracy = correct / total if total > 0 else 0
                level_accuracies[level] = accuracy
                level_results[level] = {
                    "accuracy": accuracy,
                    "n_questions": total,
                    "correct": correct,
                }

                print(f"    Level {level}: {accuracy:.1%} ({correct}/{total})")

        # Find edge of chaos (closest to threshold)
        baseline_level = self._find_threshold_level(level_accuracies)

        print(f"  Edge-of-chaos level: {baseline_level}")
        print(f"  Accuracy at baseline: {level_accuracies.get(baseline_level, 0):.1%}")

        return baseline_level, {
            "level_accuracies": level_accuracies,
            "level_results": level_results,
            "baseline_level": baseline_level,
            "threshold": self.threshold,
        }

    def _generate_assessment_questions(self, frontier_client: Optional[Any]) -> List[Dict]:
        """Generate assessment questions across difficulty scale."""
        questions = []

        if frontier_client:
            # Use frontier models to generate questions
            for level in range(1, 101, 5):
                level_qs = self._request_questions_from_frontier(frontier_client, level, count=20)
                questions.extend(level_qs)
        else:
            # Generate placeholder questions
            for level in range(1, 101, 5):
                level_qs = self._generate_level_questions(level, 20)
                questions.extend(level_qs)

        return questions

    def _request_questions_from_frontier(self, client: Any, level: int, count: int) -> List[Dict]:
        """Request questions from frontier model API."""
        if not OPENROUTER_AVAILABLE:
            raise RuntimeError("OpenRouter client not available.")

        api_key = client.api_key if hasattr(client, "api_key") else None
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required for frontier question generation.")

        prompt = (
            "Generate a JSON array of coding questions with test cases. "
            f"Difficulty level: {level}/100. Count: {count}. "
            "Each item must include fields: question, expected_type, test_cases."
        )
        system_prompt = "You are a precise curriculum generator. Output JSON only."

        try:
            response = asyncio.run(
                self._async_complete(api_key, prompt, system_prompt)
            )
        except Exception as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        if not response.success:
            raise RuntimeError(f"OpenRouter error: {response.error}")

        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenRouter response was not valid JSON.") from exc

        questions: List[Dict] = []
        for idx, item in enumerate(payload):
            questions.append(
                {
                    "id": f"assess_{level}_{idx}",
                    "level": level,
                    "question": item.get("question", ""),
                    "expected_type": item.get("expected_type", "code"),
                    "test_cases": item.get("test_cases", []),
                }
            )

        return questions

    async def _async_complete(
        self, api_key: str, prompt: str, system_prompt: str
    ):
        async with OpenRouterClient(api_key=api_key, default_model=ModelProvider.QWEN_FREE) as client:
            return await client.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
            )

    def _generate_level_questions(self, level: int, count: int) -> List[Dict]:
        """Generate placeholder questions for a difficulty level."""
        questions = []

        # Coding question templates by difficulty
        easy_templates = [
            "Write a function to print 'Hello, World!'",
            "Create a variable that stores the number {n}",
            "Write a loop that prints numbers 1 to {n}",
        ]

        medium_templates = [
            "Implement a function to find the maximum in a list",
            "Write a function to check if a number is prime",
            "Create a class representing a {entity}",
        ]

        hard_templates = [
            "Implement a binary search tree with insert and search",
            "Write a function to solve the N-queens problem",
            "Implement a LRU cache with O(1) operations",
        ]

        expert_templates = [
            "Implement a distributed consensus algorithm",
            "Write a neural network training loop from scratch",
            "Design a lock-free concurrent data structure",
        ]

        # Select template based on level
        if level <= 25:
            templates = easy_templates
        elif level <= 50:
            templates = medium_templates
        elif level <= 75:
            templates = hard_templates
        else:
            templates = expert_templates

        for i in range(count):
            template = random.choice(templates)
            question = template.format(n=random.randint(1, 100), entity="Car")

            questions.append(
                {
                    "id": f"assess_{level}_{i}",
                    "level": level,
                    "question": question,
                    "expected_type": "code",
                    "test_cases": self._generate_test_cases(level),
                }
            )

        return questions

    def _generate_test_cases(self, level: int) -> List[Dict]:
        """Generate test cases based on difficulty level."""
        # Placeholder test cases
        return [{"input": "test_input", "expected": "test_output"}]

    def _evaluate_question(
        self, model: nn.Module, tokenizer: Any, question: Dict
    ) -> AssessmentResult:
        """Evaluate model's answer to a question."""
        prompt = question["question"]

        # Encode prompt
        if hasattr(tokenizer, "__call__"):
            inputs = tokenizer(
                prompt, return_tensors="pt", max_length=512, truncation=True, padding=True
            )
        else:
            # Mock tokenizer fallback
            inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

        # Generate response
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        try:
            if hasattr(model, "generate"):
                outputs = model.generate(
                    **inputs, max_new_tokens=256, temperature=0.7, do_sample=True
                )
                response = (
                    tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if hasattr(tokenizer, "decode")
                    else str(outputs[0].tolist())
                )
            else:
                # Fallback for models without generate
                with torch.no_grad():
                    logits = model(**inputs)
                response = "Model output (no generate method)"
        except Exception as e:
            response = f"Error: {e}"

        # Evaluate correctness (simplified)
        # In full implementation, would execute code and check test cases
        correct = self._check_correctness(question, response)

        return AssessmentResult(
            level=question["level"],
            question=question["question"],
            correct=correct,
            confidence=0.5,  # Placeholder
            response=response,
        )

    def _check_correctness(self, question: Dict, response: str) -> bool:
        """Check whether the response matches the question's expected answer(s).

        Replaces the previous `random.random()` placeholder with a real, deterministic
        match (shared evaluator logic) against the question's expected outputs. The
        curriculum's test-case generation is itself still placeholder (Wave 1), so
        this yields honest-but-low accuracy until real expected answers are wired -
        which is correct behavior, NOT random noise. With no ground truth it returns
        False (fail-closed) rather than inventing a success probability.
        """
        expecteds = []
        for tc in question.get("test_cases", []) or []:
            if isinstance(tc, dict) and tc.get("expected") is not None:
                expecteds.append(str(tc["expected"]))
        if question.get("answer") is not None:
            expecteds.append(str(question["answer"]))
        if not expecteds:
            return False
        return any(matches(response, e) for e in expecteds)

    def _find_threshold_level(self, level_accuracies: Dict[int, float]) -> int:
        """Find level closest to threshold accuracy."""
        if not level_accuracies:
            return 40  # Default baseline

        # Find level with accuracy closest to threshold
        best_level = 40
        best_diff = float("inf")

        for level, accuracy in level_accuracies.items():
            diff = abs(accuracy - self.threshold)
            if diff < best_diff:
                best_diff = diff
                best_level = level

        return best_level

    def find_baseline_moo(
        self, model: nn.Module, tokenizer: Any, frontier_client: Optional[Any] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Find the baseline level using multi-objective optimization.

        Instead of single binary search point, finds Pareto-optimal "zone"
        balancing multiple objectives:
        1. Accuracy close to threshold (75%)
        2. Confidence stability (low variance)
        3. Response efficiency (low latency)

        Args:
            model: Model to assess
            tokenizer: Tokenizer for encoding
            frontier_client: Optional client for generating questions

        Returns:
            Tuple of (baseline_level, moo_results)
        """
        if not MOO_AVAILABLE:
            print("  [Warning] MOO not available, falling back to single-objective")
            return self.find_baseline(model, tokenizer, frontier_client)

        print("  Running MOO edge-of-chaos assessment...")
        print(f"  Target threshold: {self.threshold:.0%}")
        print("  Objectives: accuracy, confidence_stability, efficiency")

        # Cache for evaluated levels
        evaluation_cache: Dict[int, Dict] = {}

        def evaluate_level(level_array) -> List[float]:
            """Evaluate a difficulty level on 3 objectives."""
            level = int(round(level_array[0]))
            level = max(1, min(100, level))

            if level in evaluation_cache:
                cached = evaluation_cache[level]
            else:
                # Generate and evaluate questions at this level
                questions = self._generate_level_questions(level, 20)
                correct = 0
                total = len(questions)
                confidences = []
                latencies = []

                model.eval()
                with torch.no_grad():
                    for q in questions:
                        import time

                        start = time.time()
                        result = self._evaluate_question(model, tokenizer, q)
                        latency = time.time() - start

                        if result.correct:
                            correct += 1
                        confidences.append(result.confidence)
                        latencies.append(latency)

                accuracy = correct / total if total > 0 else 0
                conf_variance = (
                    sum((c - sum(confidences) / len(confidences)) ** 2 for c in confidences)
                    / len(confidences)
                    if confidences
                    else 1.0
                )
                avg_latency = sum(latencies) / len(latencies) if latencies else 1.0

                evaluation_cache[level] = {
                    "accuracy": accuracy,
                    "conf_variance": conf_variance,
                    "avg_latency": avg_latency,
                }
                cached = evaluation_cache[level]

            # Objectives (all minimized):
            # 1. Distance from threshold (want accuracy ~= 0.75)
            accuracy_distance = abs(cached["accuracy"] - self.threshold)
            # 2. Confidence variance (want low variance = stable)
            conf_variance = cached["conf_variance"]
            # 3. Average latency (want fast)
            avg_latency = cached["avg_latency"]

            return [accuracy_distance, conf_variance, avg_latency]

        # Run MOO
        runner = HybridMOORunner.from_evaluator(
            evaluator=evaluate_level,
            n_vars=1,  # difficulty level
            n_objs=3,  # accuracy_dist, conf_var, latency
            xl=[1],  # min level
            xu=[100],  # max level
        )

        result = runner.run(n_generations=15)

        # Select balanced solution from Pareto front
        baseline_level = self._select_balanced_solution(result.X, result.F)

        print(f"  MOO Pareto front size: {len(result.X)}")
        print(f"  Selected baseline level: {baseline_level}")
        print(f"  Cached evaluations: {len(evaluation_cache)}")

        return baseline_level, {
            "pareto_X": result.X.tolist() if hasattr(result.X, "tolist") else result.X,
            "pareto_F": result.F.tolist() if hasattr(result.F, "tolist") else result.F,
            "baseline_level": baseline_level,
            "evaluation_cache": evaluation_cache,
            "backend_used": result.backend_used,
            "n_evaluations": result.n_evaluations,
        }

    def _select_balanced_solution(self, X, F) -> int:
        """Select balanced solution from Pareto front using weighted sum."""
        if len(X) == 0:
            return 40  # Default

        # Normalize objectives
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1  # Avoid division by zero

        F_norm = (F - F_min) / F_range

        # Weighted sum: accuracy (50%), stability (30%), efficiency (20%)
        weights = [0.5, 0.3, 0.2]
        scores = (F_norm * weights).sum(axis=1)

        best_idx = scores.argmin()
        return int(round(X[best_idx][0]))


__all__ = ["EdgeOfChaosAssessment", "AssessmentResult"]

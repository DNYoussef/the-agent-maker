"""
Phase 5: Edge-of-Chaos Assessment

Finds the optimal learning threshold where model achieves ~75% accuracy.
Based on research: "Intelligence at the Edge of Chaos"

Key insight: Maximum learning occurs at the edge of chaos where
systems exhibit maximum information processing capacity.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import random

import torch
import torch.nn as nn


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

    def __init__(
        self,
        threshold: float = 0.75,
        num_questions: int = 2000,
        tolerance: float = 0.05
    ):
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
        self,
        model: nn.Module,
        tokenizer: Any,
        frontier_client: Optional[Any] = None
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
                level_questions = [q for q in questions if q['level'] == level]

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
                    'accuracy': accuracy,
                    'n_questions': total,
                    'correct': correct
                }

                print(f"    Level {level}: {accuracy:.1%} ({correct}/{total})")

        # Find edge of chaos (closest to threshold)
        baseline_level = self._find_threshold_level(level_accuracies)

        print(f"  Edge-of-chaos level: {baseline_level}")
        print(f"  Accuracy at baseline: {level_accuracies.get(baseline_level, 0):.1%}")

        return baseline_level, {
            'level_accuracies': level_accuracies,
            'level_results': level_results,
            'baseline_level': baseline_level,
            'threshold': self.threshold
        }

    def _generate_assessment_questions(
        self,
        frontier_client: Optional[Any]
    ) -> List[Dict]:
        """Generate assessment questions across difficulty scale."""
        questions = []

        if frontier_client:
            # Use frontier models to generate questions
            for level in range(1, 101, 5):
                level_qs = self._request_questions_from_frontier(
                    frontier_client, level, count=20
                )
                questions.extend(level_qs)
        else:
            # Generate placeholder questions
            for level in range(1, 101, 5):
                level_qs = self._generate_level_questions(level, 20)
                questions.extend(level_qs)

        return questions

    def _request_questions_from_frontier(
        self,
        client: Any,
        level: int,
        count: int
    ) -> List[Dict]:
        """Request questions from frontier model API."""
        # Placeholder - would call OpenRouter API
        return self._generate_level_questions(level, count)

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

            questions.append({
                'id': f"assess_{level}_{i}",
                'level': level,
                'question': question,
                'expected_type': 'code',
                'test_cases': self._generate_test_cases(level)
            })

        return questions

    def _generate_test_cases(self, level: int) -> List[Dict]:
        """Generate test cases based on difficulty level."""
        # Placeholder test cases
        return [
            {'input': 'test_input', 'expected': 'test_output'}
        ]

    def _evaluate_question(
        self,
        model: nn.Module,
        tokenizer: Any,
        question: Dict
    ) -> AssessmentResult:
        """Evaluate model's answer to a question."""
        prompt = question['question']

        # Encode prompt
        if hasattr(tokenizer, '__call__'):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
        else:
            # Mock tokenizer fallback
            inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}

        # Generate response
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        try:
            if hasattr(model, 'generate'):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True) \
                    if hasattr(tokenizer, 'decode') else str(outputs[0].tolist())
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
            level=question['level'],
            question=question['question'],
            correct=correct,
            confidence=0.5,  # Placeholder
            response=response
        )

    def _check_correctness(self, question: Dict, response: str) -> bool:
        """Check if response is correct."""
        # Simplified correctness check
        # Full implementation would:
        # 1. Extract code from response
        # 2. Execute in sandbox
        # 3. Run test cases
        # 4. Check outputs

        # Placeholder: random based on difficulty
        level = question['level']
        base_success_rate = max(0.1, 1.0 - (level / 100))
        return random.random() < base_success_rate

    def _find_threshold_level(self, level_accuracies: Dict[int, float]) -> int:
        """Find level closest to threshold accuracy."""
        if not level_accuracies:
            return 40  # Default baseline

        # Find level with accuracy closest to threshold
        best_level = 40
        best_diff = float('inf')

        for level, accuracy in level_accuracies.items():
            diff = abs(accuracy - self.threshold)
            if diff < best_diff:
                best_diff = diff
                best_level = level

        return best_level


__all__ = ['EdgeOfChaosAssessment', 'AssessmentResult']

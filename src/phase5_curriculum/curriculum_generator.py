"""
Phase 5: Adaptive Curriculum Generator

Generates 20,000 questions across 10 difficulty levels using frontier models.
Rescales difficulty from baseline to create personalized curriculum.

Process:
1. Map baseline level -> Level 1, original 100 -> Level 10
2. Request questions from multiple frontier models
3. Shuffle questions within each level
4. Return structured curriculum

M5 TIER 1: Integrated with OpenRouter FREE models (Qwen, Gemma, Mistral, Llama).
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .curriculum_engine import SpecializationType

# Import OpenRouter client
try:
    from .openrouter_client import (
        CompletionResponse,
        ModelProvider,
        OpenRouterClient,
        get_free_models,
    )

    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """A curriculum question."""

    id: str
    level: int
    original_difficulty: int
    question: str
    source: str  # Which frontier model generated it
    test_cases: List[Dict]
    hints: List[str]
    success_count: int = 0
    attempt_count: int = 0


class AdaptiveCurriculumGenerator:
    """
    Generates adaptive curriculum based on model's baseline level.

    Rescales difficulty so that:
    - Baseline level -> New Level 1 (starting point)
    - Original Level 100 -> New Level 10 (maximum challenge)
    """

    def __init__(
        self,
        baseline_level: int,
        num_levels: int = 10,
        questions_per_level: int = 2000,
        frontier_models: Optional[List[str]] = None,
        specialization: SpecializationType = SpecializationType.CODING,
    ):
        """
        Initialize curriculum generator.

        Args:
            baseline_level: Model's baseline (edge-of-chaos) level
            num_levels: Number of curriculum levels (default 10)
            questions_per_level: Questions per level (default 2000)
            frontier_models: List of frontier model names
            specialization: Type of agent specialization
        """
        self.baseline_level = baseline_level
        self.num_levels = num_levels
        self.questions_per_level = questions_per_level
        # Use FREE OpenRouter models by default
        self.frontier_models = frontier_models or [
            "qwen/qwen-2-7b-instruct:free",
            "google/gemma-7b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3-8b-instruct:free",
        ]
        self.specialization = specialization

        # Model provider mapping for OpenRouter
        self._model_map = {
            "qwen/qwen-2-7b-instruct:free": ModelProvider.QWEN_FREE
            if OPENROUTER_AVAILABLE
            else None,
            "google/gemma-7b-it:free": ModelProvider.GEMMA_FREE if OPENROUTER_AVAILABLE else None,
            "mistralai/mistral-7b-instruct:free": ModelProvider.MISTRAL_FREE
            if OPENROUTER_AVAILABLE
            else None,
            "meta-llama/llama-3-8b-instruct:free": ModelProvider.LLAMA_FREE
            if OPENROUTER_AVAILABLE
            else None,
        }

    def generate(self, frontier_client: Optional[Any] = None) -> Dict[int, List[Question]]:
        """
        Generate full curriculum for all levels.

        Args:
            frontier_client: Client for frontier model API

        Returns:
            Dict mapping level -> list of questions
        """
        curriculum = {}

        for level in range(1, self.num_levels + 1):
            # Map new level to original difficulty
            original_difficulty = self._map_to_original_difficulty(level)

            print(f"  Generating level {level} (original difficulty: {original_difficulty})...")

            # Generate questions from each frontier model
            level_questions = []
            questions_per_model = self.questions_per_level // len(self.frontier_models)

            for model_name in self.frontier_models:
                model_questions = self._generate_from_frontier(
                    frontier_client, model_name, original_difficulty, level, questions_per_model
                )
                level_questions.extend(model_questions)

            # Shuffle questions within level
            random.shuffle(level_questions)

            curriculum[level] = level_questions
            print(f"    Generated {len(level_questions)} questions")

        return curriculum

    def _map_to_original_difficulty(self, new_level: int) -> int:
        """
        Map new curriculum level to original difficulty scale.

        Formula: original = baseline + (new_level - 1) * (100 - baseline) / (num_levels - 1)
        """
        if self.num_levels <= 1:
            return self.baseline_level

        original = self.baseline_level + (new_level - 1) * (100 - self.baseline_level) / (
            self.num_levels - 1
        )

        return int(round(original))

    def _generate_from_frontier(
        self,
        client: Optional[Any],
        model_name: str,
        original_difficulty: int,
        level: int,
        count: int,
    ) -> List[Question]:
        """Generate questions from a frontier model."""
        if client:
            return self._request_from_api(client, model_name, original_difficulty, level, count)
        else:
            return self._generate_placeholder(model_name, original_difficulty, level, count)

    def _request_from_api(
        self, client: Any, model_name: str, difficulty: int, level: int, count: int
    ) -> List[Question]:
        """
        Request questions from frontier model API (OpenRouter).

        M5 TIER 1: Real API integration with FREE models.

        Args:
            client: OpenRouterClient instance
            model_name: Model identifier
            difficulty: Original difficulty level (1-100)
            level: Curriculum level (1-10)
            count: Number of questions to generate

        Returns:
            List of Question objects from API
        """
        if not OPENROUTER_AVAILABLE or client is None:
            logger.info(f"  Using placeholder (OpenRouter unavailable)")
            return self._generate_placeholder(model_name, difficulty, level, count)

        # Get model provider
        model_provider = self._model_map.get(model_name)
        if model_provider is None:
            logger.warning(f"  Unknown model {model_name}, using placeholder")
            return self._generate_placeholder(model_name, difficulty, level, count)

        # Build generation prompt
        prompt = self._build_generation_prompt(difficulty, count)
        system_prompt = self._get_system_prompt()

        try:
            # Run async completion synchronously
            response = asyncio.run(
                self._async_generate(client, model_provider, prompt, system_prompt)
            )

            if response.success:
                questions = self._parse_questions(response.content, level, model_name, difficulty)
                if questions:
                    logger.info(f"  Generated {len(questions)} questions from {model_name}")
                    return questions

            logger.warning(f"  API call failed: {response.error}, using placeholder")
        except Exception as e:
            logger.warning(f"  API error: {e}, using placeholder")

        return self._generate_placeholder(model_name, difficulty, level, count)

    async def _async_generate(
        self, client: "OpenRouterClient", model: "ModelProvider", prompt: str, system_prompt: str
    ) -> "CompletionResponse":
        """Async wrapper for OpenRouter completion."""
        async with OpenRouterClient(
            api_key=client.api_key if hasattr(client, "api_key") else None, default_model=model
        ) as async_client:
            return await async_client.complete(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.8,
            )

    def _build_generation_prompt(self, difficulty: int, count: int) -> str:
        """Build prompt for question generation."""
        spec_name = self.specialization.value
        difficulty_desc = self._get_difficulty_description(difficulty)

        return f"""Generate {count} {spec_name} questions at {difficulty_desc} difficulty level.

Requirements:
1. Each question should be clear and unambiguous
2. Include test cases or expected outputs where applicable
3. Difficulty should match level {difficulty}/100

Format your response as JSON array:
[
  {{
    "question": "The question text",
    "test_cases": [{{"input": "...", "expected": "..."}}],
    "hints": ["hint1", "hint2"]
  }}
]

Generate exactly {count} questions."""

    def _get_system_prompt(self) -> str:
        """Get system prompt for question generation."""
        return """You are an expert curriculum designer specializing in adaptive learning.
Your task is to generate high-quality educational questions that:
1. Are precisely calibrated to the specified difficulty level
2. Test genuine understanding, not just memorization
3. Include appropriate hints for scaffolded learning
4. Have clear, testable success criteria

Always respond with valid JSON."""

    def _get_difficulty_description(self, difficulty: int) -> str:
        """Convert numeric difficulty to description."""
        if difficulty <= 20:
            return "beginner"
        elif difficulty <= 40:
            return "easy"
        elif difficulty <= 60:
            return "intermediate"
        elif difficulty <= 80:
            return "advanced"
        else:
            return "expert"

    def _parse_questions(
        self, response_text: str, level: int, model_name: str, difficulty: int
    ) -> List[Question]:
        """Parse API response into Question objects."""
        questions = []

        try:
            # Try to extract JSON from response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)

                for i, item in enumerate(parsed):
                    if isinstance(item, dict) and "question" in item:
                        question = Question(
                            id=f"q_{level}_{model_name}_{i}",
                            level=level,
                            original_difficulty=difficulty,
                            question=item.get("question", ""),
                            source=model_name,
                            test_cases=item.get("test_cases", []),
                            hints=item.get("hints", []),
                            success_count=0,
                            attempt_count=0,
                        )
                        questions.append(question)
        except json.JSONDecodeError as e:
            logger.warning(f"  JSON parse error: {e}")
        except Exception as e:
            logger.warning(f"  Parse error: {e}")

        return questions

    def _generate_placeholder(
        self, model_name: str, difficulty: int, level: int, count: int
    ) -> List[Question]:
        """Generate placeholder questions."""
        questions = []

        templates = self._get_templates_for_difficulty(difficulty)

        for i in range(count):
            template = random.choice(templates)
            question_text = self._fill_template(template, difficulty)

            question = Question(
                id=f"q_{level}_{model_name}_{i}",
                level=level,
                original_difficulty=difficulty,
                question=question_text,
                source=model_name,
                test_cases=self._generate_test_cases(difficulty),
                hints=[],
                success_count=0,
                attempt_count=0,
            )
            questions.append(question)

        return questions

    def _get_templates_for_difficulty(self, difficulty: int) -> List[str]:
        """Get question templates appropriate for difficulty level."""
        if self.specialization == SpecializationType.CODING:
            return self._get_coding_templates(difficulty)
        elif self.specialization == SpecializationType.RESEARCH:
            return self._get_research_templates(difficulty)
        elif self.specialization == SpecializationType.WRITING:
            return self._get_writing_templates(difficulty)
        else:
            return self._get_coding_templates(difficulty)  # Default

    def _get_coding_templates(self, difficulty: int) -> List[str]:
        """Get coding question templates."""
        if difficulty <= 30:
            return [
                "Write a function that returns the sum of two numbers",
                "Create a function to check if a string is empty",
                "Write code to print numbers from 1 to {n}",
                "Implement a function to find the length of a list",
                "Write a function that reverses a string",
            ]
        elif difficulty <= 50:
            return [
                "Implement binary search on a sorted array",
                "Write a function to check if a number is prime",
                "Create a function to merge two sorted lists",
                "Implement a stack using a list",
                "Write a function to find all duplicates in an array",
            ]
        elif difficulty <= 70:
            return [
                "Implement a binary search tree with insert and search operations",
                "Write a function to solve the subset sum problem",
                "Implement Dijkstra's shortest path algorithm",
                "Create a function to validate a binary search tree",
                "Implement a LRU cache with O(1) get and put operations",
            ]
        else:
            return [
                "Implement a red-black tree with balancing",
                "Write a function to solve the traveling salesman problem",
                "Implement a concurrent hash map with fine-grained locking",
                "Design and implement a B+ tree for database indexing",
                "Implement the Raft consensus algorithm",
            ]

    def _get_research_templates(self, difficulty: int) -> List[str]:
        """Get research question templates."""
        if difficulty <= 40:
            return [
                "Summarize the main findings of this abstract: {abstract}",
                "What is the definition of {term} in machine learning?",
                "List three applications of {technology}",
            ]
        else:
            return [
                "Compare and contrast {method1} and {method2} approaches",
                "Identify potential limitations in this methodology: {description}",
                "Synthesize findings from multiple sources on {topic}",
            ]

    def _get_writing_templates(self, difficulty: int) -> List[str]:
        """Get writing question templates."""
        if difficulty <= 40:
            return [
                "Write a clear introduction paragraph about {topic}",
                "Summarize this text in 3 sentences: {text}",
                "Rewrite this sentence to improve clarity: {sentence}",
            ]
        else:
            return [
                "Write a persuasive argument for {position}",
                "Create a detailed outline for an essay on {topic}",
                "Edit this paragraph for style and coherence: {paragraph}",
            ]

    def _fill_template(self, template: str, difficulty: int) -> str:
        """Fill template placeholders with appropriate values."""
        replacements = {
            "{n}": str(random.randint(5, 100)),
            "{term}": random.choice(["gradient descent", "backpropagation", "attention"]),
            "{technology}": random.choice(["transformers", "CNNs", "RNNs"]),
            "{topic}": random.choice(["machine learning", "data structures", "algorithms"]),
            "{method1}": "supervised learning",
            "{method2}": "unsupervised learning",
            "{abstract}": "[Sample abstract text]",
            "{description}": "[Methodology description]",
            "{text}": "[Text to summarize]",
            "{sentence}": "[Sentence to rewrite]",
            "{position}": "open source software",
            "{paragraph}": "[Paragraph to edit]",
        }

        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)

        return result

    def _generate_test_cases(self, difficulty: int) -> List[Dict]:
        """Generate test cases for validation."""
        # Placeholder test cases
        num_cases = min(5, 2 + difficulty // 20)

        return [
            {
                "input": f"test_input_{i}",
                "expected": f"test_output_{i}",
                "description": f"Test case {i}",
            }
            for i in range(num_cases)
        ]


__all__ = ["AdaptiveCurriculumGenerator", "Question"]

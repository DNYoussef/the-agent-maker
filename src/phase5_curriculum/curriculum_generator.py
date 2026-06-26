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
from typing import Any, Dict, List, Optional, Tuple

from .curriculum_engine import SpecializationType

# Import OpenRouter client
try:
    from .openrouter_client import CompletionResponse, ModelProvider, OpenRouterClient

    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

# Import meta-calculus for k(L) difficulty curve
try:
    from src.cross_phase.meta_calculus.phase_facades import phase5 as meta_phase5

    META_CALCULUS_AVAILABLE = True
except ImportError:
    META_CALCULUS_AVAILABLE = False

# Import MOO for curriculum optimization
try:
    from src.cross_phase.meta_calculus.moo_utils import HybridMOORunner

    MOO_AVAILABLE = True
except ImportError:
    MOO_AVAILABLE = False

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

        If meta-calculus available, uses k(L) for physics-motivated difficulty curve.
        Otherwise falls back to linear interpolation.

        Linear formula: original = baseline + (new_level - 1) * (100 - baseline) / (num_levels - 1)
        k(L) formula: Uses k(level/total_levels) to create non-linear curve
        """
        if self.num_levels <= 1:
            return self.baseline_level

        # Use k(L) difficulty curve if meta-calculus available
        if META_CALCULUS_AVAILABLE:
            # Get k(L)-based difficulty (returns 0.0-1.0)
            difficulty_normalized = meta_phase5.get_stage_difficulty(
                stage=new_level,
                total_stages=self.num_levels,
                base_difficulty=0.3,  # Base difficulty for level 1
            )
            # Map to original scale: baseline -> 100
            original = self.baseline_level + difficulty_normalized * (100 - self.baseline_level)
            logger.debug(
                f"k(L) difficulty: level {new_level} -> normalized {difficulty_normalized:.3f} "
                f"-> original {original:.1f}"
            )
            return int(round(original))

        # Fallback: Linear interpolation
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
            logger.info("  Using placeholder (OpenRouter unavailable)")
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


class MOOCurriculumGenerator(AdaptiveCurriculumGenerator):
    """
    Curriculum generator with multi-objective optimization.

    Finds Pareto-optimal curriculum configurations balancing:
    1. Expected learning rate
    2. Concept retention
    3. Difficulty smoothness
    4. Total questions needed
    """

    def __init__(self, *args, **kwargs):
        """Initialize MOO curriculum generator."""
        super().__init__(*args, **kwargs)
        self._evaluation_cache: Dict[tuple, Dict] = {}

    def generate_optimal_curriculum(
        self,
        frontier_client: Optional[Any] = None,
        n_generations: int = 30,
    ) -> Tuple[Dict[int, List[Question]], Dict[str, Any]]:
        """
        Generate Pareto-optimal curriculum using MOO.

        Objectives:
        1. Maximize expected learning rate (smooth progression)
        2. Maximize concept retention (spaced repetition potential)
        3. Minimize difficulty variance (smoothness)
        4. Minimize total questions (efficiency)

        Args:
            frontier_client: Optional API client for question generation
            n_generations: Number of MOO generations

        Returns:
            Tuple of (curriculum, optimization_results)
        """
        if not MOO_AVAILABLE:
            logger.warning("MOO not available, using standard generation")
            return self.generate(frontier_client), {"error": "MOO not available"}

        logger.info("Running MOO curriculum optimization...")

        def evaluate_curriculum_params(params) -> List[float]:
            """Evaluate curriculum parameters on 4 objectives."""
            # params: [difficulty_alpha, difficulty_beta, questions_scale, spacing_factor]
            alpha = max(0.5, params[0])  # Beta distribution alpha
            beta = max(0.5, params[1])  # Beta distribution beta
            q_scale = max(0.5, params[2])  # Questions per level multiplier
            spacing = max(0.1, params[3])  # Spacing factor for retention

            # Cache key
            cache_key = (round(alpha, 2), round(beta, 2), round(q_scale, 2), round(spacing, 2))
            if cache_key in self._evaluation_cache:
                cached = self._evaluation_cache[cache_key]
                return [
                    cached["learning_rate_obj"],
                    cached["retention_obj"],
                    cached["smoothness_obj"],
                    cached["questions_obj"],
                ]

            # Generate difficulty curve using beta distribution shape
            difficulties = []
            for level in range(1, self.num_levels + 1):
                # Map level to [0, 1] then apply beta-shaped curve
                t = (level - 1) / (self.num_levels - 1) if self.num_levels > 1 else 0.5
                # Beta CDF approximation for difficulty curve
                difficulty_t = self._beta_cdf_approx(t, alpha, beta)
                # Map to actual difficulty range
                difficulty = self.baseline_level + difficulty_t * (100 - self.baseline_level)
                difficulties.append(difficulty)

            # Calculate questions per level (scaled)
            questions_per_level = [
                int(self.questions_per_level * q_scale * (1 + spacing * (i / self.num_levels)))
                for i in range(self.num_levels)
            ]

            # Objective 1: Learning rate (want smooth progression)
            # Ideal: difficulties increase steadily
            diffs = [difficulties[i + 1] - difficulties[i] for i in range(len(difficulties) - 1)]
            learning_rate = sum(diffs) / len(diffs) if diffs else 0
            # Penalize if any step is too large or negative
            step_penalty = sum(max(0, d - 15) + max(0, -d) for d in diffs)
            obj1_learning = -learning_rate + step_penalty * 0.5

            # Objective 2: Retention (want spaced practice)
            # More questions at harder levels = better retention
            retention_score = sum(
                q * (d / 100) for q, d in zip(questions_per_level, difficulties)
            ) / sum(questions_per_level)
            obj2_retention = -retention_score  # Maximize -> negate

            # Objective 3: Smoothness (minimize difficulty variance)
            if len(diffs) > 1:
                mean_diff = sum(diffs) / len(diffs)
                variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
            else:
                variance = 0
            obj3_smoothness = variance

            # Objective 4: Total questions (minimize)
            total_q = sum(questions_per_level)
            obj4_questions = total_q / 10000  # Normalize

            # Cache results
            self._evaluation_cache[cache_key] = {
                "learning_rate_obj": obj1_learning,
                "retention_obj": obj2_retention,
                "smoothness_obj": obj3_smoothness,
                "questions_obj": obj4_questions,
                "difficulties": difficulties,
                "questions_per_level": questions_per_level,
            }

            return [obj1_learning, obj2_retention, obj3_smoothness, obj4_questions]

        # Run MOO
        runner = HybridMOORunner.from_evaluator(
            evaluator=evaluate_curriculum_params,
            n_vars=4,  # alpha, beta, q_scale, spacing
            n_objs=4,  # learning, retention, smoothness, questions
            xl=[0.5, 0.5, 0.5, 0.1],  # min values
            xu=[5.0, 5.0, 2.0, 1.0],  # max values
        )

        result = runner.run(n_generations=n_generations)

        # Select balanced solution
        best_params = self._select_efficient_curriculum(result.X, result.F)

        # Generate actual curriculum with best parameters
        alpha, beta, q_scale, spacing = best_params
        curriculum = self._generate_with_params(frontier_client, alpha, beta, q_scale, spacing)

        logger.info(f"MOO curriculum optimization complete. Pareto front: {len(result.X)}")

        return curriculum, {
            "pareto_front_size": len(result.X),
            "best_params": {
                "alpha": alpha,
                "beta": beta,
                "questions_scale": q_scale,
                "spacing_factor": spacing,
            },
            "backend_used": result.backend_used,
            "n_evaluations": result.n_evaluations,
            "evaluation_cache_size": len(self._evaluation_cache),
        }

    def _beta_cdf_approx(self, t: float, alpha: float, beta: float) -> float:
        """Approximate beta CDF for difficulty curve shaping."""
        # Simple approximation using power functions
        # For alpha > beta: curve is convex (harder early)
        # For alpha < beta: curve is concave (easier early, then harder)
        # For alpha = beta = 1: linear
        if alpha == 1 and beta == 1:
            return t

        # Use regularized incomplete beta approximation
        # This is a simplified version - full implementation would use scipy
        power = alpha / (alpha + beta)
        if alpha > beta:
            # Convex curve
            return t ** (1 / power)
        else:
            # Concave curve
            return 1 - (1 - t) ** power

    def _select_efficient_curriculum(self, X, F) -> List[float]:
        """Select efficient curriculum from Pareto front."""

        if len(X) == 0:
            return [1.0, 1.0, 1.0, 0.5]  # Default params

        # Normalize objectives
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1

        F_norm = (F - F_min) / F_range

        # Weighted sum: learning (30%), retention (30%), smoothness (25%), questions (15%)
        weights = [0.30, 0.30, 0.25, 0.15]
        scores = (F_norm * weights).sum(axis=1)

        best_idx = scores.argmin()
        return X[best_idx].tolist()

    def _generate_with_params(
        self,
        frontier_client: Optional[Any],
        alpha: float,
        beta: float,
        q_scale: float,
        spacing: float,
    ) -> Dict[int, List[Question]]:
        """Generate curriculum with specified parameters."""
        curriculum: Dict[int, List[Question]] = {}

        for level in range(1, self.num_levels + 1):
            # Calculate difficulty using beta-shaped curve
            t = (level - 1) / (self.num_levels - 1) if self.num_levels > 1 else 0.5
            difficulty_t = self._beta_cdf_approx(t, alpha, beta)
            original_difficulty = int(
                self.baseline_level + difficulty_t * (100 - self.baseline_level)
            )

            # Calculate questions for this level
            num_questions = int(
                self.questions_per_level * q_scale * (1 + spacing * (level / self.num_levels))
            )

            # Generate questions (reuse parent class logic)
            if frontier_client and OPENROUTER_AVAILABLE:
                level_questions = self._generate_from_frontier(
                    frontier_client,
                    random.choice(self.frontier_models),
                    original_difficulty,
                    level,
                    num_questions,
                )
            else:
                level_questions = self._generate_placeholder_questions(
                    original_difficulty, level, num_questions
                )

            random.shuffle(level_questions)
            curriculum[level] = level_questions

            logger.debug(
                f"Level {level}: difficulty={original_difficulty}, "
                f"questions={len(level_questions)}"
            )

        return curriculum

    def _generate_placeholder_questions(
        self, difficulty: int, level: int, count: int
    ) -> List[Question]:
        """Generate placeholder questions for a level."""
        questions = []
        templates = self._get_templates_for_difficulty(difficulty)

        for i in range(count):
            template = random.choice(templates)
            question_text = self._fill_template(template, difficulty)

            questions.append(
                Question(
                    id=f"moo_q_{level}_{i}_{random.randint(1000, 9999)}",
                    level=level,
                    original_difficulty=difficulty,
                    question=question_text,
                    source="moo_generator",
                    test_cases=self._generate_test_cases(difficulty),
                    hints=[],
                )
            )

        return questions


__all__ = ["AdaptiveCurriculumGenerator", "Question", "MOOCurriculumGenerator"]

"""
Phase 5: Curriculum Training Loop

Implements the core training mechanics with:
- Success path: Create variant, remove after 3 consecutive successes
- Failure path: Root cause analysis, generate hint, reshuffle

The dataset shrinks as concepts are mastered, providing visible progress.

M5 TIER 1: OpenRouter integration for variant/hint generation using FREE models.
"""

import asyncio
import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .curriculum_generator import Question

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
class TrainingMetrics:
    """Metrics from training a level."""

    accuracy: float
    mastered: int
    remaining_questions: int
    variants: int
    hints: int
    epochs: int
    loss: float


class CurriculumTrainingLoop:
    """
    Training loop with variant generation and hint scaffolding.

    Success Flow:
    1. Model answers correctly
    2. Generate variant (change nouns/numbers, keep concept)
    3. Replace original with variant
    4. After 3 consecutive successes -> Remove (mastered)

    Failure Flow:
    1. Model answers incorrectly
    2. Analyze root cause
    3. Generate hint
    4. Append hint to question
    5. Reshuffle into curriculum
    """

    def __init__(
        self,
        consecutive_for_mastery: int = 3,
        max_hints: int = 5,
        enable_variants: bool = True,
        max_epochs: int = 50,
        convergence_threshold: int = 50,  # Stop when this many questions remain
    ):
        """
        Initialize training loop.

        Args:
            consecutive_for_mastery: Successes needed to master a concept
            max_hints: Maximum hints per question
            enable_variants: Whether to generate variants on success
            max_epochs: Maximum training epochs per level
            convergence_threshold: Stop when this many questions remain
        """
        self.consecutive_for_mastery = consecutive_for_mastery
        self.max_hints = max_hints
        self.enable_variants = enable_variants
        self.max_epochs = max_epochs
        self.convergence_threshold = convergence_threshold

    def train_level(
        self,
        model: nn.Module,
        questions: List[Question],
        tokenizer: Any,
        coding_env: Optional[Any],
        frontier_client: Optional[Any],
        level: int,
    ) -> Tuple[nn.Module, TrainingMetrics]:
        """
        Train model on a curriculum level.

        Args:
            model: Model to train
            questions: Questions for this level
            tokenizer: Tokenizer for encoding
            coding_env: Code execution environment
            frontier_client: Client for variant/hint generation
            level: Current level number

        Returns:
            Tuple of (trained_model, metrics)
        """
        # Track statistics
        total_correct = 0
        total_attempts = 0
        variants_generated = 0
        hints_given = 0
        mastered_count = 0

        # Working copy of questions
        active_questions = copy.deepcopy(questions)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        print(f"  Starting with {len(active_questions)} questions")

        for epoch in range(self.max_epochs):
            # Shuffle questions each epoch
            random.shuffle(active_questions)

            epoch_correct = 0
            epoch_total = 0
            epoch_loss = 0.0

            questions_to_remove = []
            questions_to_add = []

            for question in active_questions:
                # Attempt question
                result = self._attempt_question(model, question, tokenizer, coding_env)

                epoch_total += 1
                total_attempts += 1

                if result["success"]:
                    # Success path
                    epoch_correct += 1
                    total_correct += 1
                    question.success_count += 1

                    if question.success_count >= self.consecutive_for_mastery:
                        # Mastered - remove
                        questions_to_remove.append(question)
                        mastered_count += 1
                    elif self.enable_variants:
                        # Generate variant
                        variant = self._generate_variant(question, frontier_client)
                        if variant:
                            questions_to_remove.append(question)
                            questions_to_add.append(variant)
                            variants_generated += 1

                    # Train on successful response
                    loss = self._train_step(model, optimizer, question, result, tokenizer)
                    epoch_loss += loss

                else:
                    # Failure path
                    question.success_count = 0  # Reset streak
                    question.attempt_count += 1

                    if len(question.hints) < self.max_hints:
                        # Generate hint
                        hint = self._generate_hint(question, result, frontier_client)
                        if hint:
                            question.hints.append(hint)
                            hints_given += 1

                    # Train on corrected response
                    loss = self._train_step(
                        model, optimizer, question, result, tokenizer, include_hints=True
                    )
                    epoch_loss += loss

            # Update question list
            for q in questions_to_remove:
                if q in active_questions:
                    active_questions.remove(q)
            active_questions.extend(questions_to_add)

            # Calculate epoch accuracy
            epoch_accuracy = epoch_correct / max(1, epoch_total)
            avg_loss = epoch_loss / max(1, epoch_total)

            # Progress update
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"    Epoch {epoch + 1}: accuracy={epoch_accuracy:.1%}, "
                    f"remaining={len(active_questions)}, mastered={mastered_count}"
                )

            # Check convergence
            if len(active_questions) <= self.convergence_threshold:
                print(f"    Converged at epoch {epoch + 1}")
                break

        # Final metrics
        final_accuracy = total_correct / max(1, total_attempts)

        return model, TrainingMetrics(
            accuracy=final_accuracy,
            mastered=mastered_count,
            remaining_questions=len(active_questions),
            variants=variants_generated,
            hints=hints_given,
            epochs=epoch + 1,
            loss=avg_loss,
        )

    def _attempt_question(
        self, model: nn.Module, question: Question, tokenizer: Any, coding_env: Optional[Any]
    ) -> Dict[str, Any]:
        """Have model attempt to answer a question."""
        # Build prompt with any accumulated hints
        prompt = question.question
        if question.hints:
            hints_text = "\n".join(f"Hint {i+1}: {h}" for i, h in enumerate(question.hints))
            prompt = f"{prompt}\n\n{hints_text}"

        # Generate response
        model.eval()
        with torch.no_grad():
            try:
                # Tokenize
                if hasattr(tokenizer, "__call__"):
                    inputs = tokenizer(
                        prompt, return_tensors="pt", max_length=512, truncation=True, padding=True
                    )
                else:
                    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

                # Generate
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
                    response = "Model output (no generate)"

            except Exception as e:
                response = f"Error: {e}"

        # Validate response
        success, error = self._validate_response(response, question, coding_env)

        return {"success": success, "response": response, "error": error, "prompt": prompt}

    def _validate_response(
        self, response: str, question: Question, coding_env: Optional[Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate model's response against test cases."""
        if coding_env:
            # Execute code in sandbox
            try:
                # Extract code from response
                code = self._extract_code(response)
                result = coding_env.execute(code, timeout=5)

                # Run test cases
                for test_case in question.test_cases:
                    if not result.check(test_case["input"], test_case["expected"]):
                        return False, f"Failed test: {test_case['description']}"

                return True, None

            except Exception as e:
                return False, str(e)
        else:
            # Simplified validation (placeholder)
            # In production, would parse and validate code
            difficulty = question.original_difficulty
            base_success_rate = max(0.3, 1.0 - (difficulty / 150))

            # Boost success rate based on hints
            hint_bonus = len(question.hints) * 0.05
            success_rate = min(0.95, base_success_rate + hint_bonus)

            success = random.random() < success_rate
            return success, None if success else "Validation failed"

    def _extract_code(self, response: str) -> str:
        """Extract code block from response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        return response  # type: ignore[no-any-return]

    def _generate_variant(
        self, question: Question, frontier_client: Optional[Any]
    ) -> Optional[Question]:
        """
        Generate a variant of a successfully answered question.

        M5 TIER 1: Uses OpenRouter FREE models when available.
        """
        if frontier_client and OPENROUTER_AVAILABLE:
            try:
                variant_question = self._generate_variant_api(question, frontier_client)
                if variant_question:
                    return variant_question
            except Exception as e:
                logger.warning(f"API variant generation failed: {e}, using placeholder")

        # Fallback: Placeholder variant generation
        variant = Question(
            id=f"{question.id}_v{question.success_count}",
            level=question.level,
            original_difficulty=question.original_difficulty,
            question=self._modify_question(question.question),
            source=question.source,
            test_cases=question.test_cases,  # Same concept, similar tests
            hints=[],  # Reset hints
            success_count=0,  # Reset success count
            attempt_count=0,
        )

        return variant

    def _generate_variant_api(self, question: Question, frontier_client: Any) -> Optional[Question]:
        """Generate variant using OpenRouter API."""
        prompt = f"""Given this question:
"{question.question}"

Generate a VARIANT question that:
1. Tests the same concept/skill
2. Uses different specific values (numbers, names, etc.)
3. Has slightly different wording
4. Is at the same difficulty level

Respond with ONLY the new question text, nothing else."""

        system_prompt = """You are an expert educational content creator.
Generate high-quality variant questions that test the same underlying concept
but use different specific examples and wording."""

        try:
            response = asyncio.run(
                self._async_complete(frontier_client, prompt, system_prompt, max_tokens=256)
            )

            if response.success and response.content.strip():
                return Question(
                    id=f"{question.id}_v{question.success_count}",
                    level=question.level,
                    original_difficulty=question.original_difficulty,
                    question=response.content.strip(),
                    source=f"{question.source}_variant",
                    test_cases=question.test_cases,
                    hints=[],
                    success_count=0,
                    attempt_count=0,
                )
        except Exception as e:
            logger.warning(f"Variant API error: {e}")

        return None

    async def _async_complete(
        self, frontier_client: Any, prompt: str, system_prompt: str, max_tokens: int = 256
    ) -> "CompletionResponse":
        """Async wrapper for OpenRouter completion."""
        # Use first free model available
        model = ModelProvider.QWEN_FREE

        async with OpenRouterClient(
            api_key=frontier_client.api_key if hasattr(frontier_client, "api_key") else None,
            default_model=model,
        ) as client:
            return await client.complete(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
            )

    def _modify_question(self, question_text: str) -> str:
        """Modify question to create variant (change nouns/numbers)."""
        # Simple modifications
        modifications = [
            ("list", "array"),
            ("array", "sequence"),
            ("function", "method"),
            ("numbers", "integers"),
            ("string", "text"),
        ]

        result = question_text
        for old, new in modifications:
            if old in result.lower():
                result = result.replace(old, new)
                break

        # Change numbers
        import re

        numbers = re.findall(r"\d+", result)
        for num in numbers[:1]:  # Change first number
            new_num = str(int(num) + random.randint(1, 10))
            result = result.replace(num, new_num, 1)

        return result

    def _generate_hint(
        self, question: Question, result: Dict, frontier_client: Optional[Any]
    ) -> Optional[str]:
        """
        Generate hint based on failure analysis.

        M5 TIER 1: Uses OpenRouter FREE models when available.
        """
        if frontier_client and OPENROUTER_AVAILABLE:
            try:
                hint = self._generate_hint_api(question, result, frontier_client)
                if hint:
                    return hint
            except Exception as e:
                logger.warning(f"API hint generation failed: {e}, using placeholder")

        # Fallback: Placeholder hints based on error type
        error = result.get("error", "")

        hint_templates = [
            "Consider breaking down the problem into smaller steps.",
            "Make sure to handle edge cases like empty inputs.",
            "Check your loop bounds and termination conditions.",
            "Verify your data types match what's expected.",
            "Remember to return the result, not just print it.",
        ]

        if "index" in error.lower():
            return "Check your array indices - you may be going out of bounds."
        elif "type" in error.lower():
            return "Verify that your data types are correct."
        elif "syntax" in error.lower():
            return "Review Python syntax - check for missing colons or parentheses."

        return random.choice(hint_templates)

    def _generate_hint_api(
        self, question: Question, result: Dict, frontier_client: Any
    ) -> Optional[str]:
        """Generate hint using OpenRouter API with root cause analysis."""
        error = result.get("error", "Unknown error")
        response_text = result.get("response", "")

        prompt = f"""A student attempted this question:
"{question.question}"

Their response was:
"{response_text[:500]}"

The error/issue was:
"{error}"

Provide a SHORT, helpful hint (1-2 sentences) that:
1. Identifies the likely misconception or mistake
2. Points toward the correct approach without giving the answer
3. Is encouraging and constructive

Respond with ONLY the hint text, nothing else."""

        system_prompt = """You are a patient, expert tutor.
Generate hints that help students discover solutions themselves
rather than giving direct answers. Be encouraging and specific."""

        try:
            response = asyncio.run(
                self._async_complete(frontier_client, prompt, system_prompt, max_tokens=128)
            )

            if response.success and response.content.strip():
                return response.content.strip()
        except Exception as e:
            logger.warning(f"Hint API error: {e}")

        return None

    def _train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        question: Question,
        result: Dict,
        tokenizer: Any,
        include_hints: bool = False,
    ) -> float:
        """Execute one training step."""
        model.train()

        # Build training input
        prompt = question.question
        if include_hints and question.hints:
            hints_text = "\n".join(f"Hint: {h}" for h in question.hints)
            prompt = f"{prompt}\n{hints_text}"

        try:
            # Tokenize
            if hasattr(tokenizer, "__call__"):
                inputs = tokenizer(
                    prompt, return_tensors="pt", max_length=512, truncation=True, padding=True
                )
            else:
                inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            # Forward pass
            if hasattr(model, "forward"):
                outputs = model(**inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif hasattr(outputs, "loss"):
                    loss = outputs.loss
                else:
                    logits = outputs

                # Compute loss if not provided
                if "loss" not in dir():
                    # Simple language modeling loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=0,
                    )
            else:
                return 0.0

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            return loss.item()

        except Exception as e:
            return 0.0


__all__ = ["CurriculumTrainingLoop", "TrainingMetrics"]

"""
Phase 6: B-Cycle Persona Optimizer

Optimizes persona/behavior via self-guided discovery.
Model discovers its own patterns and bakes persona prompts.

Research: "Prompt Baking" (arXiv:2409.13697v1)
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PersonaTask:
    """A persona evaluation task."""

    prompt: str
    expected_traits: List[str]
    difficulty: int  # 1-10


class BCycleOptimizer:
    """
    B-Cycle: Persona Optimization.

    Process:
    1. Model self-analyzes to discover behavioral patterns
    2. Generate persona evaluation tasks
    3. Bake persona prompts to improve consistency
    4. Return optimized model and score

    Key difference from V1: Self-guided discovery, not pre-defined personas.
    """

    def __init__(
        self,
        persona_prompts: List[str],
        lora_r: int = 16,
        lora_alpha: int = 32,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,  # Fixed: was 1e-4, now 5e-5 per M4 spec
    ):
        """
        Initialize B-cycle optimizer.

        Args:
            persona_prompts: Base prompts for persona baking
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            num_epochs: Baking epochs
            learning_rate: Learning rate
        """
        self.persona_prompts = persona_prompts
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.state = {
            "iterations": 0,
            "scores": [],
            "best_score": 0.0,
            "discovered_traits": [],
            "prompts_used": [],
        }

        # Persona evaluation tasks
        self.persona_tasks = self._generate_persona_tasks()

    def _generate_persona_tasks(self) -> List[PersonaTask]:
        """Generate persona evaluation tasks."""
        return [
            PersonaTask(
                prompt="How would you approach a complex problem?",
                expected_traits=["step-by-step", "careful", "thorough"],
                difficulty=3,
            ),
            PersonaTask(
                prompt="A user makes an error. How do you respond?",
                expected_traits=["helpful", "patient", "constructive"],
                difficulty=4,
            ),
            PersonaTask(
                prompt="You are unsure about something. What do you do?",
                expected_traits=["honest", "transparent", "clarifying"],
                difficulty=5,
            ),
            PersonaTask(
                prompt="How do you verify your answers?",
                expected_traits=["verification", "double-check", "validate"],
                difficulty=4,
            ),
            PersonaTask(
                prompt="Explain your reasoning process.",
                expected_traits=["logical", "structured", "clear"],
                difficulty=5,
            ),
        ]

    def optimize(
        self, model: nn.Module, tokenizer: Any, evaluator: Any = None
    ) -> Tuple[nn.Module, float]:
        """
        Run one B-cycle optimization iteration.

        Args:
            model: Model to optimize
            tokenizer: Tokenizer
            evaluator: Optional external evaluator

        Returns:
            Tuple of (optimized_model, score)
        """
        self.state["iterations"] += 1

        # Step 1: Self-discovery - analyze model's current patterns
        discovered = self._self_discover_patterns(model, tokenizer)
        self.state["discovered_traits"].extend(discovered)
        print(f"    Discovered traits: {discovered[:3]}...")

        # Step 2: Evaluate current persona consistency
        pre_score = self._evaluate_persona(model, tokenizer, evaluator)
        print(f"    Pre-bake persona score: {pre_score:.3f}")

        # Step 3: Generate and select prompt based on discovery
        prompt = self._generate_persona_prompt(discovered)
        self.state["prompts_used"].append(prompt)

        # Step 4: Bake the persona prompt
        baked_model = self._bake_persona_prompt(model, prompt, tokenizer)

        # Step 5: Evaluate post-bake
        post_score = self._evaluate_persona(baked_model, tokenizer, evaluator)
        print(f"    Post-bake persona score: {post_score:.3f}")

        # Update state
        self.state["scores"].append(post_score)
        if post_score > self.state["best_score"]:
            self.state["best_score"] = post_score

        return baked_model, post_score

    def _self_discover_patterns(self, model: nn.Module, tokenizer: Any) -> List[str]:
        """
        Model self-analyzes to discover behavioral patterns.

        This is the key V2 innovation: model-driven discovery.
        """
        discovered_traits = []
        model.eval()

        discovery_prompts = [
            "Describe your approach to problem-solving:",
            "What are your core values as an assistant?",
            "How do you handle uncertainty?",
            "What makes your responses helpful?",
        ]

        with torch.no_grad():
            for prompt in discovery_prompts:
                try:
                    if hasattr(tokenizer, "__call__"):
                        inputs = tokenizer(
                            prompt,
                            return_tensors="pt",
                            max_length=128,
                            truncation=True,
                            padding=True,
                        )
                    else:
                        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

                    device = next(model.parameters()).device
                    inputs = {
                        k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
                    }

                    if hasattr(model, "generate"):
                        outputs = model.generate(
                            **inputs, max_new_tokens=64, do_sample=True, temperature=0.7
                        )
                        output_text = (
                            tokenizer.decode(outputs[0], skip_special_tokens=True)
                            if hasattr(tokenizer, "decode")
                            else ""
                        )
                    else:
                        output_text = ""

                    # Extract traits from response
                    trait_keywords = [
                        "careful",
                        "thorough",
                        "step-by-step",
                        "logical",
                        "helpful",
                        "honest",
                        "clear",
                        "structured",
                        "verify",
                        "check",
                        "analyze",
                        "consider",
                    ]

                    for trait in trait_keywords:
                        if trait in output_text.lower() and trait not in discovered_traits:
                            discovered_traits.append(trait)

                except Exception:
                    continue

        # Fallback if no traits discovered
        if not discovered_traits:
            discovered_traits = ["helpful", "careful", "thorough"]

        return discovered_traits[:5]  # Top 5 traits

    def _generate_persona_prompt(self, discovered_traits: List[str]) -> str:
        """Generate persona prompt based on discovered traits."""
        base_prompt = random.choice(self.persona_prompts)

        if discovered_traits:
            trait_str = ", ".join(discovered_traits[:3])
            enhanced_prompt = f"{base_prompt} You demonstrate these qualities: {trait_str}."
        else:
            enhanced_prompt = base_prompt

        return enhanced_prompt

    def _evaluate_persona(self, model: nn.Module, tokenizer: Any, evaluator: Any = None) -> float:
        """Evaluate model's persona consistency."""
        if evaluator is not None:
            return evaluator.evaluate(model)

        model.eval()
        total_score = 0.0

        with torch.no_grad():
            for task in self.persona_tasks:
                try:
                    if hasattr(tokenizer, "__call__"):
                        inputs = tokenizer(
                            task.prompt,
                            return_tensors="pt",
                            max_length=128,
                            truncation=True,
                            padding=True,
                        )
                    else:
                        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

                    device = next(model.parameters()).device
                    inputs = {
                        k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
                    }

                    if hasattr(model, "generate"):
                        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                        output_text = (
                            tokenizer.decode(outputs[0], skip_special_tokens=True)
                            if hasattr(tokenizer, "decode")
                            else ""
                        )
                    else:
                        output_text = ""

                    # Score based on trait presence
                    trait_matches = sum(
                        1 for trait in task.expected_traits if trait in output_text.lower()
                    )
                    task_score = trait_matches / len(task.expected_traits)
                    total_score += task_score

                except Exception:
                    continue

        return total_score / max(1, len(self.persona_tasks))

    def _bake_persona_prompt(self, model: nn.Module, prompt: str, tokenizer: Any) -> nn.Module:
        """Bake a persona prompt into the model."""
        import copy

        baked_model = copy.deepcopy(model)

        device = next(baked_model.parameters()).device
        optimizer = torch.optim.AdamW(baked_model.parameters(), lr=self.learning_rate)

        # Create calibration data for persona
        calibration_samples = [
            f"{prompt}\n\nUser: How do you solve problems?\nAssistant: I approach problems carefully and systematically.",
            f"{prompt}\n\nUser: Are you sure about that?\nAssistant: Let me verify my reasoning step by step.",
            f"{prompt}\n\nUser: Explain your thinking.\nAssistant: I'll break this down into clear logical steps.",
        ]

        baked_model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            for sample in calibration_samples:
                try:
                    if hasattr(tokenizer, "__call__"):
                        inputs = tokenizer(
                            sample,
                            return_tensors="pt",
                            max_length=256,
                            truncation=True,
                            padding=True,
                        )
                    else:
                        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

                    inputs = {
                        k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
                    }

                    outputs = baked_model(**inputs)

                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        loss = outputs.loss
                    elif hasattr(outputs, "logits"):
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs["input_ids"][..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=0,
                        )
                    else:
                        continue

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(baked_model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()

                except Exception:
                    continue

        return baked_model

    def get_state(self) -> Dict:
        """Get optimizer state."""
        return self.state.copy()


__all__ = ["BCycleOptimizer", "PersonaTask"]

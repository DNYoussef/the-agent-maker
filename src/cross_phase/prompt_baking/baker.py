"""
Prompt Baking System
Converts prompts into weight updates via KL divergence minimization

Based on: Prompt Baking paper (arXiv:2409.13697v1)
Core Algorithm: θ_u = argmin D_KL(P_θ(·|u) || P_θu(·))
"""

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class PromptBakingConfig:
    """Configuration for prompt baking"""

    lora_r: int = 16  # LoRA rank (16 or 32)
    lora_alpha: int = 32
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-4
    half_baking: bool = False  # Stop early for partial strength
    half_baking_factor: float = 0.5  # 50% strength


class PromptBaker:
    """
    Prompt Baking System

    Features:
    - Fast: 5 minutes per prompt (LoRA-based)
    - Half-Baking: Stop early for partial prompt strength
    - Prompt Pursuit: Iterative re-baking for amplification
    - Sequential Baking: Compose multiple prompts
    """

    def __init__(self, config: PromptBakingConfig):
        self.config = config

    def bake_prompt(
        self,
        model: nn.Module,
        prompt: str,
        tokenizer,
        calibration_data: Optional[list] = None,
        half_bake: bool = False,
    ) -> nn.Module:
        """
        Bake a prompt into model weights

        Args:
            model: Base model
            prompt: Prompt to bake (e.g., "You are a reasoning specialist...")
            tokenizer: Tokenizer
            calibration_data: Calibration dataset
            half_bake: Use half-baking (50% strength)

        Returns:
            Baked model
        """
        # Wave-0: callers (phase5 curriculum_engine) invoked this without
        # calibration_data, raising an opaque TypeError. Fail honestly instead:
        # without calibration data there is nothing to bake, so warn loudly and
        # return the model unchanged. (The KL objective itself is fixed in Wave 1.)
        if not calibration_data:
            logger.warning(
                "bake_prompt called without calibration_data; skipping bake and "
                "returning model unchanged (Wave-0 honest no-op)."
            )
            return model

        # Add LoRA adapters
        model_with_lora = self._add_lora_adapters(model)

        # Each calibration text tokenized twice: with the prompt (teacher
        # context) and without it (student context).
        device = next(model_with_lora.parameters()).device
        batches = self._prepare_calibration_batches(prompt, tokenizer, calibration_data, device)

        # Train LoRA to match prompted behavior
        optimizer = torch.optim.AdamW(model_with_lora.parameters(), lr=self.config.learning_rate)

        num_epochs = self.config.num_epochs
        if half_bake:
            num_epochs = int(num_epochs * self.config.half_baking_factor)

        for epoch in range(num_epochs):
            for batch in batches:
                kl_loss = self._compute_bake_loss(
                    model_with_lora,
                    batch["prompted_input_ids"],
                    batch["unprompted_input_ids"],
                )
                kl_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Merge LoRA into base model
        baked_model = self._merge_lora(model_with_lora)

        return baked_model

    def sequential_baking(
        self, model: nn.Module, prompts: list, tokenizer, calibration_data: list
    ) -> nn.Module:
        """
        Sequential baking: θ_u1u2 = B(B(θ, u1), u2)

        Args:
            model: Base model
            prompts: List of prompts to bake sequentially
            tokenizer: Tokenizer
            calibration_data: Calibration data

        Returns:
            Sequentially baked model
        """
        baked_model = model

        for i, prompt in enumerate(prompts):
            print(f"Baking prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            baked_model = self.bake_prompt(baked_model, prompt, tokenizer, calibration_data)

        return baked_model

    def prompt_pursuit(
        self,
        model: nn.Module,
        prompt: str,
        tokenizer,
        calibration_data: list,
        num_iterations: int = 3,
    ) -> nn.Module:
        """
        Prompt Pursuit: Iterative re-baking for amplification

        Amplifies prompt strength by 15-40%

        Args:
            model: Base model
            prompt: Prompt to amplify
            tokenizer: Tokenizer
            calibration_data: Calibration data
            num_iterations: Number of pursuit iterations

        Returns:
            Amplified baked model
        """
        baked_model = model

        for i in range(num_iterations):
            print(f"Prompt pursuit iteration {i+1}/{num_iterations}...")
            baked_model = self.bake_prompt(baked_model, prompt, tokenizer, calibration_data)

        return baked_model

    def _add_lora_adapters(self, model: nn.Module) -> nn.Module:
        """Add LoRA adapters to model for efficient fine-tuning.

        Uses PEFT library to inject LoRA adapters into attention layers.
        Target modules: q_proj, v_proj, k_proj, o_proj (standard transformer attention)
        """
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            print("Warning: peft not installed. Run: pip install peft")
            return model

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        return get_peft_model(model, lora_config)

    def _prepare_calibration_batches(
        self, prompt: str, tokenizer, calibration_data: list, device
    ) -> list:
        """Tokenize each calibration text twice: with the prompt prepended (the
        teacher's context, P_theta(.|u)) and without it (the student's context,
        P_theta_u(.)). The KL is taken over the next-token distribution at the
        final position, so no generation is needed - one example per batch keeps
        the two contexts' differing lengths trivially aligned.
        """
        # Tokenize the prompt prefix once, then reuse the SAME text tokens in
        # both contexts: unprompted = text_ids, prompted = [prompt_ids, text_ids].
        # Both therefore end with identical text tokens, so the final-position
        # next-token distributions are aligned. (Tokenizing prompt+text and text
        # independently would truncate text differently once the prompt eats the
        # length budget, misaligning teacher and student.)
        prompt_ids = tokenizer(
            f"{prompt}\n\n", return_tensors="pt", truncation=True, max_length=256
        )["input_ids"]
        text_budget = max(1, 512 - prompt_ids.shape[1])

        batches = []
        for text in calibration_data:
            text_ids = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=text_budget
            )["input_ids"]
            prompted_input_ids = torch.cat([prompt_ids, text_ids], dim=1)
            batches.append(
                {
                    "prompted_input_ids": prompted_input_ids.to(device),
                    "unprompted_input_ids": text_ids.to(device),
                }
            )
        return batches

    @staticmethod
    def _extract_logits(output):
        """Models may return a tensor or a ModelOutput with .logits."""
        return output.logits if hasattr(output, "logits") else output

    def _compute_bake_loss(
        self,
        model_with_lora: nn.Module,
        prompted_input_ids: torch.Tensor,
        unprompted_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prompt-baking KL: D_KL(P_theta(.|u) || P_theta_u(.)).

        Teacher = the ORIGINAL model with the prompt in context (LoRA adapter
        disabled), detached so no gradient leaks into the shared base weights.
        Student = the LoRA weights WITHOUT the prompt. Minimizing this KL pushes
        the unprompted student to behave as if the prompt were present, i.e.
        bakes the prompt into the weights. KL is over the next-token distribution
        at the final position.
        """
        disable = getattr(model_with_lora, "disable_adapter", None)
        teacher_ctx = disable() if callable(disable) else contextlib.nullcontext()
        with torch.no_grad(), teacher_ctx:
            teacher_logits = self._extract_logits(model_with_lora(prompted_input_ids))[:, -1, :]

        student_logits = self._extract_logits(model_with_lora(unprompted_input_ids))[:, -1, :]

        return F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )

    def _merge_lora(self, model_with_lora: nn.Module) -> nn.Module:
        """Merge LoRA adapters into base model weights.

        Uses PEFT's merge_and_unload() to fold LoRA weights into base model.
        After merging, the model no longer needs LoRA overhead at inference.

        Args:
            model_with_lora: PEFT model with trained LoRA adapters

        Returns:
            Merged base model with LoRA weights incorporated
        """
        try:
            from peft import PeftModel
        except ImportError:
            print("Warning: peft not installed. Returning model as-is.")
            return model_with_lora

        # Check if model is a PEFT model
        if isinstance(model_with_lora, PeftModel):
            # merge_and_unload folds LoRA weights into base model
            merged_model = model_with_lora.merge_and_unload()
            return merged_model
        else:
            # Model doesn't have LoRA adapters, return as-is
            return model_with_lora


def bake_prompt(
    model: nn.Module,
    prompt: str,
    tokenizer,
    calibration_data: list,
    config: Optional[PromptBakingConfig] = None,
) -> nn.Module:
    """
    Convenience function for prompt baking

    Args:
        model: Base model
        prompt: Prompt to bake
        tokenizer: Tokenizer
        calibration_data: Calibration dataset
        config: Optional config (uses defaults if None)

    Returns:
        Baked model
    """
    if config is None:
        config = PromptBakingConfig()

    baker = PromptBaker(config)
    return baker.bake_prompt(model, prompt, tokenizer, calibration_data)

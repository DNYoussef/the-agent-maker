"""
Prompt Baking System
Converts prompts into weight updates via KL divergence minimization

Based on: Prompt Baking paper (arXiv:2409.13697v1)
Core Algorithm: θ_u = argmin D_KL(P_θ(·|u) || P_θu(·))
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


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
        calibration_data: list,
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
        # Add LoRA adapters
        model_with_lora = self._add_lora_adapters(model)

        # Generate prompted responses for KL target
        prompted_responses = self._generate_prompted_responses(
            model, prompt, tokenizer, calibration_data
        )

        # Train LoRA to match prompted behavior
        optimizer = torch.optim.AdamW(model_with_lora.parameters(), lr=self.config.learning_rate)

        num_epochs = self.config.num_epochs
        if half_bake:
            num_epochs = int(num_epochs * self.config.half_baking_factor)

        for epoch in range(num_epochs):
            for batch in prompted_responses:
                # KL divergence loss
                base_logits = model(batch["input_ids"])
                lora_logits = model_with_lora(batch["input_ids"])

                # D_KL(P_θ(·|u) || P_θu(·))
                kl_loss = F.kl_div(
                    F.log_softmax(lora_logits, dim=-1),
                    F.softmax(base_logits, dim=-1),
                    reduction="batchmean",
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

    def _generate_prompted_responses(
        self, model: nn.Module, prompt: str, tokenizer, calibration_data: list
    ) -> list:
        """Generate responses with prompt prepended for KL divergence targets.

        Args:
            model: Base model to generate with
            prompt: System prompt to prepend
            tokenizer: Tokenizer for encoding/decoding
            calibration_data: List of input texts for calibration

        Returns:
            List of batched response dictionaries with input_ids
        """
        responses = []
        model.eval()

        with torch.no_grad():
            for text in calibration_data:
                # Prepend prompt to each calibration example
                prompted_text = f"{prompt}\n\n{text}"

                inputs = tokenizer(
                    prompted_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                # Move to model device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

                responses.append({"input_ids": outputs, "attention_mask": torch.ones_like(outputs)})

        # Batch responses
        batched = []
        for i in range(0, len(responses), self.config.batch_size):
            batch = responses[i : i + self.config.batch_size]
            if batch:
                batched.append({"input_ids": torch.cat([r["input_ids"] for r in batch], dim=0)})

        return batched

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

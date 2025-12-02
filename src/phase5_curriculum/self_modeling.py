"""
Phase 5: Self-Modeling System

Implements temperature-range self-prediction training.
Model learns to predict its own outputs at different temperatures,
developing meta-cognitive awareness.

Based on: "Unexpected Benefits of Self-Modeling in Neural Systems"
Key insight: Models that predict their own outputs develop better
internal representations and confidence calibration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemperatureRange:
    """A temperature range for self-modeling."""
    start: float
    end: float
    midpoint: float
    index: int


class SelfModelingTrainer:
    """
    Trains model to predict its own outputs across temperature ranges.

    Process:
    1. Generate outputs at various temperatures
    2. Mask portions of generated text
    3. Train model to predict masked tokens (knowing it generated them)
    4. Repeat until model "groks" itself (>95% self-prediction accuracy)
    """

    def __init__(
        self,
        temperature_ranges: List[Dict],
        mask_rate: float = 0.2,
        target_accuracy: float = 0.95,
        max_epochs: int = 5,
        samples_per_range: int = 100
    ):
        """
        Initialize self-modeling trainer.

        Args:
            temperature_ranges: List of temperature range dicts
            mask_rate: Fraction of tokens to mask
            target_accuracy: Target self-prediction accuracy
            max_epochs: Maximum training epochs
            samples_per_range: Samples to generate per temperature range
        """
        self.temperature_ranges = [
            TemperatureRange(**r) if isinstance(r, dict) else r
            for r in temperature_ranges
        ]
        self.mask_rate = mask_rate
        self.target_accuracy = target_accuracy
        self.max_epochs = max_epochs
        self.samples_per_range = samples_per_range

    def train(
        self,
        model: nn.Module,
        tokenizer: Any
    ) -> nn.Module:
        """
        Train model on self-prediction task.

        Args:
            model: Model to train
            tokenizer: Tokenizer for encoding

        Returns:
            Model with improved self-modeling capability
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        device = next(model.parameters()).device

        print(f"  Training self-modeling across {len(self.temperature_ranges)} temperature ranges")

        for epoch in range(self.max_epochs):
            total_correct = 0
            total_predictions = 0
            epoch_loss = 0.0

            for temp_range in self.temperature_ranges:
                # Step 1: Generate outputs at this temperature
                generated_samples = self._generate_at_temperature(
                    model, tokenizer, temp_range.midpoint, device
                )

                # Step 2: Train on masked prediction
                for sample in generated_samples:
                    # Mask tokens
                    masked_ids, target_ids, mask_positions = self._mask_tokens(
                        sample['token_ids'], tokenizer
                    )

                    # Step 3: Self-prediction with context
                    loss, correct, total = self._self_prediction_step(
                        model, optimizer, masked_ids, target_ids,
                        mask_positions, temp_range.midpoint, device
                    )

                    total_correct += correct
                    total_predictions += total
                    epoch_loss += loss

            # Calculate epoch metrics
            accuracy = total_correct / max(1, total_predictions)
            avg_loss = epoch_loss / max(1, len(self.temperature_ranges) * self.samples_per_range)

            print(f"    Epoch {epoch + 1}: self-prediction accuracy={accuracy:.1%}, loss={avg_loss:.4f}")

            # Check convergence
            if accuracy >= self.target_accuracy:
                print(f"    Reached target accuracy at epoch {epoch + 1}")
                break

        return model

    def _generate_at_temperature(
        self,
        model: nn.Module,
        tokenizer: Any,
        temperature: float,
        device: torch.device
    ) -> List[Dict]:
        """Generate samples at a specific temperature."""
        samples = []
        model.eval()

        # Sample prompts for generation
        prompts = [
            "Write a function to",
            "Explain how to",
            "The algorithm works by",
            "To solve this problem,",
            "Consider the following approach:",
        ]

        with torch.no_grad():
            for i in range(self.samples_per_range):
                prompt = random.choice(prompts)

                try:
                    # Tokenize
                    if hasattr(tokenizer, '__call__'):
                        inputs = tokenizer(
                            prompt,
                            return_tensors="pt",
                            max_length=64,
                            truncation=True,
                            padding=True
                        )
                    else:
                        inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}

                    inputs = {k: v.to(device) for k, v in inputs.items()
                              if isinstance(v, torch.Tensor)}

                    # Generate with specific temperature
                    if hasattr(model, 'generate'):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=64,
                            temperature=max(0.1, temperature),  # Avoid division by zero
                            do_sample=True,
                            top_p=0.9
                        )
                        token_ids = outputs[0].cpu().tolist()
                    else:
                        token_ids = inputs['input_ids'][0].cpu().tolist()

                    samples.append({
                        'prompt': prompt,
                        'token_ids': token_ids,
                        'temperature': temperature
                    })

                except Exception:
                    # Fallback sample
                    samples.append({
                        'prompt': prompt,
                        'token_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'temperature': temperature
                    })

        return samples

    def _mask_tokens(
        self,
        token_ids: List[int],
        tokenizer: Any
    ) -> tuple:
        """Mask a portion of tokens for prediction."""
        token_ids = list(token_ids)
        n_tokens = len(token_ids)
        n_mask = max(1, int(n_tokens * self.mask_rate))

        # Select random positions to mask
        mask_positions = random.sample(range(n_tokens), min(n_mask, n_tokens))

        # Store targets
        targets = [token_ids[pos] for pos in mask_positions]

        # Create masked version
        masked_ids = token_ids.copy()
        mask_token_id = getattr(tokenizer, 'mask_token_id', 0) or 0

        for pos in mask_positions:
            masked_ids[pos] = mask_token_id

        return masked_ids, targets, mask_positions

    def _self_prediction_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        masked_ids: List[int],
        target_ids: List[int],
        mask_positions: List[int],
        temperature: float,
        device: torch.device
    ) -> tuple:
        """Execute one self-prediction training step."""
        model.train()

        try:
            # Convert to tensors
            input_tensor = torch.tensor([masked_ids], device=device)

            # Forward pass
            outputs = model(input_ids=input_tensor)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            # Extract predictions at mask positions
            loss = torch.tensor(0.0, device=device)
            correct = 0
            total = 0

            for i, pos in enumerate(mask_positions):
                if pos < logits.size(1):
                    # Get prediction at masked position
                    position_logits = logits[0, pos, :]
                    target = torch.tensor([target_ids[i]], device=device)

                    # Compute loss
                    pos_loss = F.cross_entropy(position_logits.unsqueeze(0), target)
                    loss = loss + pos_loss

                    # Check if prediction is correct
                    predicted = position_logits.argmax().item()
                    if predicted == target_ids[i]:
                        correct += 1
                    total += 1

            if total > 0:
                loss = loss / total

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            return loss.item(), correct, total

        except Exception:
            return 0.0, 0, 1


class SelfModelingMetrics:
    """Track self-modeling performance across temperature ranges."""

    def __init__(self):
        self.range_accuracies = {}
        self.range_losses = {}

    def update(self, temp_range: TemperatureRange, accuracy: float, loss: float):
        """Update metrics for a temperature range."""
        key = f"{temp_range.start:.1f}-{temp_range.end:.1f}"
        self.range_accuracies[key] = accuracy
        self.range_losses[key] = loss

    def get_summary(self) -> Dict:
        """Get summary of self-modeling metrics."""
        return {
            'range_accuracies': self.range_accuracies,
            'range_losses': self.range_losses,
            'avg_accuracy': sum(self.range_accuracies.values()) / max(1, len(self.range_accuracies)),
            'avg_loss': sum(self.range_losses.values()) / max(1, len(self.range_losses))
        }


__all__ = ['SelfModelingTrainer', 'TemperatureRange', 'SelfModelingMetrics']

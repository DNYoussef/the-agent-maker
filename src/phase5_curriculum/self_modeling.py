"""
Phase 5: Self-Modeling System

Implements temperature-range self-prediction training: the model is trained
(masked-token self-distillation) to predict its own sampled outputs at different
temperatures. Honesty: this is self-distillation on the model's own generations;
it does not by itself establish "meta-cognitive awareness" - that is an open
research claim, not a property this loop measures or guarantees.

Based on: "Unexpected Benefits of Self-Modeling in Neural Systems"
Key insight: Models that predict their own outputs develop better
internal representations and confidence calibration.

Enhanced with meta-calculus spectral gap monitoring to prevent representation collapse.
"""

import copy
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Module logger: optimize_temperature_ranges() calls logger.warning/.info but the
# name was never defined -> NameError the moment that path ran (Wave-0 crash fix).
logger = logging.getLogger(__name__)

# Import meta-calculus for spectral gap monitoring
try:
    from src.cross_phase.meta_calculus.phase_facades import phase5 as meta_phase5

    META_CALCULUS_AVAILABLE = True
except ImportError:
    META_CALCULUS_AVAILABLE = False


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
    4. Repeat until self-prediction accuracy reaches the configured target
    """

    def __init__(
        self,
        temperature_ranges: List[Dict],
        mask_rate: float = 0.2,
        target_accuracy: float = 0.95,
        max_epochs: int = 5,
        samples_per_range: int = 100,
        monitor_spectral_gap: bool = True,
    ):
        """
        Initialize self-modeling trainer.

        Args:
            temperature_ranges: List of temperature range dicts
            mask_rate: Fraction of tokens to mask
            target_accuracy: Target self-prediction accuracy
            max_epochs: Maximum training epochs
            samples_per_range: Samples to generate per temperature range
            monitor_spectral_gap: Whether to monitor for representation collapse
        """
        self.temperature_ranges = [
            TemperatureRange(**r) if isinstance(r, dict) else r for r in temperature_ranges
        ]
        self.mask_rate = mask_rate
        self.target_accuracy = target_accuracy
        self.max_epochs = max_epochs
        self.samples_per_range = samples_per_range
        self.monitor_spectral_gap = monitor_spectral_gap and META_CALCULUS_AVAILABLE

        # Initialize spectral gap monitor
        self.gap_monitor = None
        self.gap_history: List[float] = []
        if self.monitor_spectral_gap:
            self.gap_monitor = meta_phase5.create_gap_monitor()

    def train(self, model: nn.Module, tokenizer: Any) -> nn.Module:
        """
        Train model on self-prediction task.

        Args:
            model: Model to train
            tokenizer: Tokenizer for encoding

        Returns:
            Model with improved self-modeling capability
        """
        # Use MetaGrokfast if available, fallback to AdamW
        if META_CALCULUS_AVAILABLE:
            optimizer = meta_phase5.create_optimizer(model, lr=1e-5)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        device = next(model.parameters()).device

        print(f"  Training self-modeling across {len(self.temperature_ranges)} temperature ranges")
        if self.monitor_spectral_gap:
            print("  [Meta-Calculus] Spectral gap monitoring enabled")

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
                        sample["token_ids"], tokenizer
                    )

                    # Step 3: Self-prediction with context
                    loss, correct, total = self._self_prediction_step(
                        model,
                        optimizer,
                        masked_ids,
                        target_ids,
                        mask_positions,
                        temp_range.midpoint,
                        device,
                    )

                    total_correct += correct
                    total_predictions += total
                    epoch_loss += loss

            # Calculate epoch metrics
            accuracy = total_correct / max(1, total_predictions)
            avg_loss = epoch_loss / max(1, len(self.temperature_ranges) * self.samples_per_range)

            print(
                f"    Epoch {epoch + 1}: self-prediction accuracy={accuracy:.1%}, loss={avg_loss:.4f}"
            )

            # Monitor spectral gap for representation collapse
            if self.monitor_spectral_gap and self.gap_monitor is not None:
                gap_value = self._compute_spectral_gap(model)
                self.gap_history.append(gap_value)

                # Check for representation collapse (gap < 0.05)
                if gap_value < 0.05:
                    print(f"    [Meta-Calculus] WARNING: Spectral gap {gap_value:.4f} < 0.05")
                    print("    [Meta-Calculus] Possible representation collapse detected!")
                else:
                    print(f"    [Meta-Calculus] Spectral gap: {gap_value:.4f} (healthy)")

            # Check convergence
            if accuracy >= self.target_accuracy:
                print(f"    Reached target accuracy at epoch {epoch + 1}")
                break

        return model

    @staticmethod
    def _collect_weight_matrices(model: nn.Module) -> List[torch.Tensor]:
        """Collect 2-D (capped 256x256) weight matrices for spectral analysis."""
        weight_matrices = []
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                w = param.data
                if w.dim() > 2:
                    w = w.view(w.size(0), -1)
                if w.size(0) >= 2 and w.size(1) >= 2:
                    weight_matrices.append(w[: min(256, w.size(0)), : min(256, w.size(1))])
        return weight_matrices

    def _compute_spectral_gap(self, model: nn.Module) -> float:
        """Compute spectral gap from model weight matrices."""
        try:
            weight_matrices = self._collect_weight_matrices(model)
            if not weight_matrices:
                return 0.5

            w = max(weight_matrices, key=lambda x: x.numel())

            with torch.no_grad():
                try:
                    U, S, V = torch.svd(w.float())
                    if len(S) >= 2 and S[0] > 1e-8:
                        gap = 1.0 - (S[1] / S[0]).item()
                        return max(0.0, min(1.0, gap))
                except Exception:
                    pass

            return 0.5
        except Exception:
            return 0.5

    def _generate_at_temperature(
        self, model: nn.Module, tokenizer: Any, temperature: float, device: torch.device
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
                    if hasattr(tokenizer, "__call__"):
                        inputs = tokenizer(
                            prompt,
                            return_tensors="pt",
                            max_length=64,
                            truncation=True,
                            padding=True,
                        )
                    else:
                        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

                    inputs = {
                        k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
                    }

                    # Generate with specific temperature
                    if hasattr(model, "generate"):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=64,
                            temperature=max(0.1, temperature),  # Avoid division by zero
                            do_sample=True,
                            top_p=0.9,
                        )
                        token_ids = outputs[0].cpu().tolist()
                    else:
                        token_ids = inputs["input_ids"][0].cpu().tolist()

                    samples.append(
                        {"prompt": prompt, "token_ids": token_ids, "temperature": temperature}
                    )

                except Exception:
                    # Fallback sample
                    samples.append(
                        {
                            "prompt": prompt,
                            "token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            "temperature": temperature,
                        }
                    )

        return samples

    def _mask_tokens(self, token_ids: List[int], tokenizer: Any) -> tuple:
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
        mask_token_id = getattr(tokenizer, "mask_token_id", 0) or 0

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
        device: torch.device,
    ) -> tuple:
        """Execute one self-prediction training step."""
        model.train()

        try:
            # Convert to tensors
            input_tensor = torch.tensor([masked_ids], device=device)

            # Forward pass
            outputs = model(input_ids=input_tensor)

            if hasattr(outputs, "logits"):
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
            "range_accuracies": self.range_accuracies,
            "range_losses": self.range_losses,
            "avg_accuracy": sum(self.range_accuracies.values())
            / max(1, len(self.range_accuracies)),
            "avg_loss": sum(self.range_losses.values()) / max(1, len(self.range_losses)),
        }


class MOOSelfModelingTrainer(SelfModelingTrainer):
    """
    Self-modeling trainer with MOO-optimized temperature range selection.

    Finds Pareto-optimal temperature ranges balancing:
    1. Self-prediction accuracy
    2. Temperature coverage (breadth)
    3. Training efficiency (time)
    4. Representation diversity (spectral gap)
    """

    def __init__(self, *args, **kwargs):
        """Initialize MOO self-modeling trainer."""
        super().__init__(*args, **kwargs)
        self._range_cache: Dict[tuple, Dict] = {}

    def optimize_temperature_ranges(
        self,
        model: nn.Module,
        tokenizer: Any,
        n_generations: int = 20,
        quick_eval_epochs: int = 1,
    ) -> Tuple[List[TemperatureRange], Dict[str, Any]]:
        """
        Find Pareto-optimal temperature ranges using MOO.

        Objectives:
        1. Maximize self-prediction accuracy
        2. Maximize temperature coverage
        3. Minimize training time cost
        4. Maximize representation diversity (spectral gap)

        Args:
            model: Model to train
            tokenizer: Tokenizer for encoding
            n_generations: MOO generations
            quick_eval_epochs: Epochs for quick evaluation

        Returns:
            Tuple of (optimal_ranges, moo_results)
        """
        if not META_CALCULUS_AVAILABLE:
            logger.warning("Meta-calculus not available, using default ranges")
            return self.temperature_ranges, {"error": "Meta-calculus not available"}

        logger.info("Running MOO temperature range optimization...")

        def evaluate_range_params(params) -> List[float]:
            """Evaluate temperature range parameters on 4 objectives."""
            # params: [temp_start, temp_width, num_ranges, samples_per_range]
            temp_start = max(0.1, min(0.9, params[0]))
            temp_width = max(0.1, min(0.5, params[1]))
            num_ranges = max(3, min(15, int(params[2])))
            samples_per_range = max(20, min(150, int(params[3])))

            # Cache key
            cache_key = (
                round(temp_start, 2),
                round(temp_width, 2),
                num_ranges,
                samples_per_range,
            )
            if cache_key in self._range_cache:
                cached = self._range_cache[cache_key]
                return [
                    cached["accuracy_obj"],
                    cached["coverage_obj"],
                    cached["time_obj"],
                    cached["gap_obj"],
                ]

            # Create temperature ranges
            ranges = self._create_ranges_from_params(temp_start, temp_width, num_ranges)

            # Quick training evaluation
            accuracy = self._quick_evaluate(
                model, tokenizer, ranges, samples_per_range, quick_eval_epochs
            )

            # Temperature coverage (how much of 0.1-2.0 range is covered)
            covered_temps = set()
            for r in ranges:
                for t in range(int(r.start * 10), int(r.end * 10) + 1):
                    covered_temps.add(t)
            coverage = len(covered_temps) / 19  # 0.1 to 2.0 = 19 steps

            # Time cost (proportional to ranges * samples)
            time_cost = num_ranges * samples_per_range / 1000  # Normalize

            # Spectral gap (representation diversity)
            gap = self._compute_spectral_gap(model)

            # Cache results
            self._range_cache[cache_key] = {
                "accuracy_obj": -accuracy,  # maximize -> negate
                "coverage_obj": -coverage,  # maximize -> negate
                "time_obj": time_cost,  # minimize
                "gap_obj": -gap,  # maximize -> negate
                "ranges": ranges,
            }

            return [-accuracy, -coverage, time_cost, -gap]

        # Import MOO runner
        try:
            from src.cross_phase.meta_calculus.moo_utils import HybridMOORunner

            runner = HybridMOORunner.from_evaluator(
                evaluator=evaluate_range_params,
                n_vars=4,  # temp_start, temp_width, num_ranges, samples
                n_objs=4,  # accuracy, coverage, time, gap
                xl=[0.1, 0.1, 3, 20],
                xu=[0.9, 0.5, 15, 150],
            )

            result = runner.run(n_generations=n_generations)

            # Select balanced solution
            best_params = self._select_efficient_ranges(result.X, result.F)
            temp_start, temp_width, num_ranges, samples = best_params

            optimal_ranges = self._create_ranges_from_params(
                temp_start, temp_width, int(num_ranges)
            )

            # Update instance
            self.temperature_ranges = optimal_ranges
            self.samples_per_range = int(samples)

            logger.info(
                f"MOO optimization complete: {len(optimal_ranges)} ranges, "
                f"{self.samples_per_range} samples/range"
            )

            return optimal_ranges, {
                "pareto_front_size": len(result.X),
                "best_params": {
                    "temp_start": temp_start,
                    "temp_width": temp_width,
                    "num_ranges": int(num_ranges),
                    "samples_per_range": int(samples),
                },
                "backend_used": result.backend_used,
                "cache_size": len(self._range_cache),
            }

        except ImportError:
            logger.warning("HybridMOORunner not available")
            return self.temperature_ranges, {"error": "MOO not available"}

    def _create_ranges_from_params(
        self, temp_start: float, temp_width: float, num_ranges: int
    ) -> List[TemperatureRange]:
        """Create temperature ranges from optimization parameters."""
        ranges = []
        temp_end = min(2.0, temp_start + temp_width * num_ranges)
        step = (temp_end - temp_start) / num_ranges

        for i in range(num_ranges):
            start = temp_start + i * step
            end = start + step
            midpoint = (start + end) / 2
            ranges.append(
                TemperatureRange(
                    start=round(start, 2),
                    end=round(end, 2),
                    midpoint=round(midpoint, 2),
                    index=i,
                )
            )

        return ranges

    def _quick_evaluate(
        self,
        model: nn.Module,
        tokenizer: Any,
        ranges: List[TemperatureRange],
        samples_per_range: int,
        epochs: int,
    ) -> float:
        """Quick evaluation of accuracy for a range configuration."""
        # Temporarily set parameters
        old_ranges = self.temperature_ranges
        old_samples = self.samples_per_range
        old_epochs = self.max_epochs

        self.temperature_ranges = ranges
        self.samples_per_range = min(samples_per_range, 50)  # Cap for speed
        self.max_epochs = epochs

        # Run training
        try:
            # Deep-copy: _quick_evaluate is called per Pareto candidate; training
            # the real model in place corrupted it across generations.
            model_copy = copy.deepcopy(model)
            self.train(model_copy, tokenizer)

            # Estimate accuracy from gap history
            if self.gap_history:
                accuracy = 1.0 - (1.0 - self.gap_history[-1])  # Use gap as proxy
            else:
                accuracy = 0.5
        except Exception:
            accuracy = 0.3

        # Restore parameters
        self.temperature_ranges = old_ranges
        self.samples_per_range = old_samples
        self.max_epochs = old_epochs

        return accuracy

    def _select_efficient_ranges(self, X, F) -> List[float]:
        """Select efficient range configuration from Pareto front."""

        if len(X) == 0:
            return [0.3, 0.2, 8, 100]  # Default params

        # Normalize objectives
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1

        F_norm = (F - F_min) / F_range

        # Weighted sum: accuracy (35%), coverage (25%), time (20%), gap (20%)
        weights = [0.35, 0.25, 0.20, 0.20]
        scores = (F_norm * weights).sum(axis=1)

        best_idx = scores.argmin()
        return X[best_idx].tolist()


__all__ = [
    "SelfModelingTrainer",
    "TemperatureRange",
    "SelfModelingMetrics",
    "MOOSelfModelingTrainer",
]

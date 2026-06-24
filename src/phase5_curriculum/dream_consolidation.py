"""
Phase 5: Dream Consolidation System

Implements memory consolidation through high-temperature replay.
Prevents catastrophic forgetting by strengthening learned patterns.

Based on: "Dreaming Is All You Need"
Key insight: High-temperature replay consolidates episodic memory
into semantic memory, preventing forgetting by 67%.

Enhanced with meta-calculus k(L) scaling for level-adaptive consolidation.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import meta-calculus for k(level) sample scaling
try:
    from src.cross_phase.meta_calculus.k_formula import compute_k

    META_CALCULUS_AVAILABLE = True
except ImportError:
    META_CALCULUS_AVAILABLE = False


@dataclass
class DreamConfig:
    """Configuration for dream consolidation."""

    dream_temperature: float = 1.5  # High temp for creative replay
    training_temperature: float = 0.8  # Lower temp for consolidation
    num_samples: int = 1000
    num_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-5


class DreamConsolidator:
    """
    Consolidates learning through dream-like high-temperature replay.

    Process:
    1. Sample training experiences from completed level
    2. Generate "dreams" at high temperature (creative replay)
    3. Train on dreams at lower temperature (consolidation)
    4. Strengthens patterns, prevents catastrophic forgetting
    """

    def __init__(
        self,
        dream_temperature: float = 1.5,
        training_temperature: float = 0.8,
        num_samples: int = 1000,
        num_epochs: int = 1,
        level: int = 1,
        total_levels: int = 10,
        use_k_scaling: bool = True,
    ):
        """
        Initialize dream consolidator.

        Args:
            dream_temperature: Temperature for generating dreams
            training_temperature: Temperature for consolidation training
            num_samples: Base number of dream samples to generate
            num_epochs: Training epochs on dream data
            level: Current curriculum level (for k(L) scaling)
            total_levels: Total number of curriculum levels
            use_k_scaling: Whether to use k(L) to scale sample count
        """
        self.dream_temperature = dream_temperature
        self.training_temperature = training_temperature
        self.base_num_samples = num_samples
        self.num_epochs = num_epochs
        self.level = level
        self.total_levels = total_levels
        self.use_k_scaling = use_k_scaling

        # Apply k(level) scaling: later levels need MORE consolidation
        # k(L) is high for early L, low for late L
        # We want MORE samples for later levels, so use (1 - k) scaling
        if use_k_scaling and META_CALCULUS_AVAILABLE:
            L = level / total_levels  # Normalize to [0, 1]
            k = compute_k(max(0.01, L))  # Avoid log(0)
            # Scale factor: 1.0 at level 1, up to 2.0 at level 10
            scale_factor = 1.0 + (1.0 - k) * 1.5
            self.num_samples = int(num_samples * scale_factor)
            print(
                f"  [Meta-Calculus] k(L={L:.2f}) = {k:.4f}, "
                f"consolidation samples: {num_samples} -> {self.num_samples}"
            )
        else:
            self.num_samples = num_samples

    def consolidate(self, model: nn.Module, level_data: List[Dict], tokenizer: Any) -> nn.Module:
        """
        Perform dream consolidation on model.

        Args:
            model: Model to consolidate
            level_data: Training data from completed level
            tokenizer: Tokenizer for encoding

        Returns:
            Model with consolidated memory
        """
        device = next(model.parameters()).device
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        print(f"  Generating {self.num_samples} dream samples at temp={self.dream_temperature}")

        # Step 1: Sample training experiences
        experiences = self._sample_experiences(level_data)

        # Step 2: Generate dreams (high-temp replay)
        dreams = self._generate_dreams(model, experiences, tokenizer, device)

        print(f"  Training on dreams at temp={self.training_temperature}")

        # Step 3: Consolidation training
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = 0

            # Shuffle dreams
            random.shuffle(dreams)

            for i in range(0, len(dreams), 8):  # Batch size 8
                batch = dreams[i : i + 8]
                loss = self._consolidation_step(model, optimizer, batch, tokenizer, device)
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            print(f"    Consolidation epoch {epoch + 1}: loss={avg_loss:.4f}")

        return model

    def _sample_experiences(self, level_data: List[Dict]) -> List[Dict]:
        """Sample training experiences for dream generation."""
        # Extract successful questions/responses
        experiences = []

        for item in level_data:
            if hasattr(item, "question"):
                # Question object
                experiences.append(
                    {
                        "prompt": item.question,
                        "type": "question",
                        "difficulty": getattr(item, "original_difficulty", 50),
                    }
                )
            elif isinstance(item, dict):
                # Dict format
                experiences.append(
                    {
                        "prompt": item.get("question", item.get("prompt", "")),
                        "type": "question",
                        "difficulty": item.get("level", 50),
                    }
                )

        # Sample up to num_samples
        if len(experiences) > self.num_samples:
            experiences = random.sample(experiences, self.num_samples)

        return experiences

    def _generate_dreams(
        self, model: nn.Module, experiences: List[Dict], tokenizer: Any, device: torch.device
    ) -> List[Dict]:
        """Generate dreams through high-temperature replay."""
        dreams = []
        model.eval()

        with torch.no_grad():
            for exp in experiences:
                try:
                    # Tokenize experience prompt
                    prompt = exp["prompt"]

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

                    inputs = {
                        k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
                    }

                    # Generate at high temperature (dreaming)
                    if hasattr(model, "generate"):
                        dream_output = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            temperature=self.dream_temperature,
                            do_sample=True,
                            top_p=0.95,
                            top_k=50,
                        )
                        dream_ids = dream_output[0].cpu().tolist()
                    else:
                        # Fallback for models without generate
                        dream_ids = inputs["input_ids"][0].cpu().tolist()

                    # Decode dream
                    if hasattr(tokenizer, "decode"):
                        dream_text = tokenizer.decode(dream_ids, skip_special_tokens=True)
                    else:
                        dream_text = str(dream_ids)

                    dreams.append(
                        {
                            "original_prompt": prompt,
                            "dream_text": dream_text,
                            "dream_ids": dream_ids,
                            "temperature": self.dream_temperature,
                        }
                    )

                except Exception:
                    # Skip failed generations
                    continue

        print(f"  Generated {len(dreams)} dreams")
        return dreams

    def _consolidation_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: List[Dict],
        tokenizer: Any,
        device: torch.device,
    ) -> float:
        """Execute one consolidation training step."""
        model.train()
        total_loss = 0.0

        for dream in batch:
            try:
                # Tokenize dream text
                if hasattr(tokenizer, "__call__"):
                    inputs = tokenizer(
                        dream["dream_text"],
                        return_tensors="pt",
                        max_length=256,
                        truncation=True,
                        padding=True,
                    )
                else:
                    inputs = {"input_ids": torch.tensor([dream["dream_ids"][:256]])}

                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

                # Forward pass
                outputs = model(**inputs)

                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                elif hasattr(outputs, "logits"):
                    # Language modeling loss
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

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            except Exception:
                continue

        return total_loss / max(1, len(batch))


class DreamMetrics:
    """Track dream consolidation metrics."""

    def __init__(self):
        self.total_dreams = 0
        self.consolidation_loss = []
        self.forgetting_rate = 0.0

    def update(self, num_dreams: int, loss: float):
        """Update metrics."""
        self.total_dreams += num_dreams
        self.consolidation_loss.append(loss)

    def get_summary(self) -> Dict:
        """Get summary of dream metrics."""
        return {
            "total_dreams": self.total_dreams,
            "avg_consolidation_loss": sum(self.consolidation_loss)
            / max(1, len(self.consolidation_loss)),
            "consolidation_history": self.consolidation_loss,
        }


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""

    passed: bool
    gap_value: float
    threshold: float
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DreamQualityGate:
    """
    Spectral gap-based quality gates for dream consolidation.

    Uses spectral gap to determine:
    1. If consolidation quality is sufficient
    2. If more consolidation rounds are needed
    3. If model is overfitting/collapsing
    """

    def __init__(
        self,
        min_gap_threshold: float = 0.05,
        ideal_gap_range: Tuple[float, float] = (0.15, 0.40),
        collapse_threshold: float = 0.02,
        history_window: int = 5,
    ):
        """
        Initialize quality gate.

        Args:
            min_gap_threshold: Minimum acceptable gap
            ideal_gap_range: Ideal gap range (lower, upper)
            collapse_threshold: Gap below this indicates collapse
            history_window: Number of recent gaps to consider
        """
        self.min_gap_threshold = min_gap_threshold
        self.ideal_gap_range = ideal_gap_range
        self.collapse_threshold = collapse_threshold
        self.history_window = history_window
        self.gap_history: List[float] = []

    def check_consolidation_quality(
        self, model: nn.Module, level: int, total_levels: int = 10
    ) -> QualityGateResult:
        """
        Check if dream consolidation achieved sufficient quality.

        Args:
            model: Model after consolidation
            level: Current curriculum level
            total_levels: Total number of levels

        Returns:
            QualityGateResult with pass/fail and recommendation
        """
        gap_value = self._compute_spectral_gap(model)
        self.gap_history.append(gap_value)

        # Adaptive threshold based on level
        # Later levels need higher gap (more consolidated knowledge)
        level_factor = level / total_levels
        adaptive_threshold = self.min_gap_threshold * (1 + 0.5 * level_factor)

        # Check for collapse
        if gap_value < self.collapse_threshold:
            return QualityGateResult(
                passed=False,
                gap_value=gap_value,
                threshold=self.collapse_threshold,
                recommendation="CRITICAL: Representation collapse detected. "
                "Reduce consolidation intensity or add diversity regularization.",
                metadata={"status": "collapse", "severity": "critical"},
            )

        # Check if below minimum
        if gap_value < adaptive_threshold:
            return QualityGateResult(
                passed=False,
                gap_value=gap_value,
                threshold=adaptive_threshold,
                recommendation=f"Gap {gap_value:.4f} below threshold {adaptive_threshold:.4f}. "
                "Run additional consolidation with higher temperature.",
                metadata={"status": "insufficient", "additional_rounds": 1},
            )

        # Check trend (declining gap suggests overfitting)
        if len(self.gap_history) >= self.history_window:
            recent = self.gap_history[-self.history_window :]
            trend = (recent[-1] - recent[0]) / len(recent)
            if trend < -0.01:  # Declining
                return QualityGateResult(
                    passed=True,  # Still pass but warn
                    gap_value=gap_value,
                    threshold=adaptive_threshold,
                    recommendation="Warning: Gap declining over recent consolidations. "
                    "Consider reducing consolidation frequency.",
                    metadata={"status": "declining_trend", "trend": trend},
                )

        # Check if in ideal range
        status, recommendation = self._classify_gap_status(gap_value)

        return QualityGateResult(
            passed=True,
            gap_value=gap_value,
            threshold=adaptive_threshold,
            recommendation=recommendation,
            metadata={"status": status, "in_ideal_range": status == "optimal"},
        )

    def _classify_gap_status(self, gap_value: float) -> Tuple[str, str]:
        """Classify a passing gap value into a status and recommendation."""
        if self.ideal_gap_range[0] <= gap_value <= self.ideal_gap_range[1]:
            return "optimal", "Consolidation quality is optimal."
        if gap_value > self.ideal_gap_range[1]:
            return (
                "high_diversity",
                "Gap above ideal range - good diversity but may need more consolidation.",
            )
        return "acceptable", "Consolidation quality is acceptable."

    def should_continue_consolidation(
        self, model: nn.Module, current_rounds: int, max_rounds: int = 5
    ) -> Tuple[bool, str]:
        """
        Determine if more consolidation rounds are needed.

        Args:
            model: Current model
            current_rounds: Rounds completed so far
            max_rounds: Maximum allowed rounds

        Returns:
            Tuple of (should_continue, reason)
        """
        if current_rounds >= max_rounds:
            return False, "Maximum rounds reached"

        gap = self._compute_spectral_gap(model)

        # Check for collapse
        if gap < self.collapse_threshold:
            return False, "Stopping due to collapse risk"

        # Check if already in ideal range
        if self.ideal_gap_range[0] <= gap <= self.ideal_gap_range[1]:
            return False, "Optimal quality reached"

        # Check if gap is improving
        if len(self.gap_history) >= 2:
            improvement = gap - self.gap_history[-2]
            if improvement < 0.005:
                return False, "Diminishing returns - gap not improving"

        return True, "Continue consolidation to improve quality"

    def get_recommended_params(
        self, current_gap: float, level: int, total_levels: int = 10
    ) -> Dict[str, Any]:
        """
        Get recommended consolidation parameters based on current state.

        Args:
            current_gap: Current spectral gap
            level: Current curriculum level
            total_levels: Total levels

        Returns:
            Dict with recommended num_samples, temperature, epochs
        """
        # Base parameters
        base_samples = 1000
        base_temp = 1.5
        base_epochs = 3

        # Adjust based on gap
        if current_gap < self.min_gap_threshold:
            # Need more diversity - higher temp, more samples
            temp_mult = 1.3
            sample_mult = 1.5
        elif current_gap > self.ideal_gap_range[1]:
            # Too diverse - lower temp, more epochs for consolidation
            temp_mult = 0.8
            sample_mult = 1.0
        else:
            temp_mult = 1.0
            sample_mult = 1.0

        # Adjust based on level (later levels need more consolidation)
        level_factor = 1 + 0.5 * (level / total_levels)

        return {
            "num_samples": int(base_samples * sample_mult * level_factor),
            "dream_temperature": base_temp * temp_mult,
            "num_epochs": base_epochs + (level // 3),
            "current_gap": current_gap,
            "target_gap_range": self.ideal_gap_range,
        }

    def _collect_weight_matrices(self, model: nn.Module) -> List[torch.Tensor]:
        """Collect 2D weight matrices (capped at 256x256) from model params."""
        weight_matrices = []
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                w = param.data
                if w.dim() > 2:
                    w = w.view(w.size(0), -1)
                if w.size(0) >= 2 and w.size(1) >= 2:
                    weight_matrices.append(w[: min(256, w.size(0)), : min(256, w.size(1))])
        return weight_matrices

    def _spectral_gap_from_matrix(self, w: torch.Tensor) -> float:
        """Compute spectral gap from a single weight matrix via SVD."""
        with torch.no_grad():
            try:
                # SVD for spectral gap
                U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
                if len(S) >= 2:
                    gap = 1.0 - (S[1] / S[0]).item()
                    return max(0.0, min(1.0, gap))
            except Exception:
                pass
        return 0.5

    def _compute_spectral_gap(self, model: nn.Module) -> float:
        """Compute spectral gap from model weight matrices."""
        try:
            weight_matrices = self._collect_weight_matrices(model)

            if not weight_matrices:
                return 0.5

            # Use largest weight matrix
            w = max(weight_matrices, key=lambda x: x.numel())

            return self._spectral_gap_from_matrix(w)

        except Exception:
            return 0.5

    def reset(self):
        """Reset gap history for new training run."""
        self.gap_history = []


__all__ = [
    "DreamConsolidator",
    "DreamConfig",
    "DreamMetrics",
    "DreamQualityGate",
    "QualityGateResult",
]

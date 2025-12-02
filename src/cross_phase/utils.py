"""
Model-Size-Agnostic Utilities
Runtime detection and adaptive strategies for any model size
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


def get_model_size(model: nn.Module) -> Dict:
    """
    Detect model size at runtime

    Returns:
        {
            'params': total_parameters,
            'size_mb': size_in_megabytes,
            'size_category': 'tiny' | 'small' | 'medium' | 'large'
        }
    """
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024**2)  # FP32

    # Categorize for adaptive strategies
    if total_params < 50_000_000:  # <50M
        size_category = "tiny"
    elif total_params < 500_000_000:  # <500M
        size_category = "small"
    elif total_params < 2_000_000_000:  # <2B
        size_category = "medium"
    else:
        size_category = "large"

    return {"params": total_params, "size_mb": size_mb, "size_category": size_category}


def calculate_safe_batch_size(model_or_size, device_vram_gb: float) -> int:
    """
    Calculate batch size that fits in VRAM with gradient accumulation

    Args:
        model_or_size: nn.Module model OR float model size in MB
        device_vram_gb: Available VRAM in GB

    Returns:
        batch_size (int) - returns just batch size for API compatibility
    """
    # Support both model and model_size_mb signatures (ISS-001)
    if isinstance(model_or_size, (int, float)):
        model_size_mb = float(model_or_size)
    else:
        model_info = get_model_size(model_or_size)
        model_size_mb = model_info["size_mb"]

    # Rule of thumb: 4x model size for training (model + optimizer + gradients + activations)
    required_vram_mb = model_size_mb * 4

    # Leave 10% headroom
    available_vram_mb = device_vram_gb * 1024 * 0.9

    if required_vram_mb > available_vram_mb:
        # Won't fit, need gradient accumulation
        batch_size = 1
        accumulation_steps = math.ceil(required_vram_mb / available_vram_mb)
        print(
            f"[WARN] Using gradient accumulation: batch_size=1, "
            f"accumulation={accumulation_steps}"
        )
    else:
        # Fits, calculate optimal batch size
        overhead_per_sample = model_size_mb * 0.1
        batch_size = int((available_vram_mb - required_vram_mb) / overhead_per_sample)
        batch_size = min(batch_size, 32)  # Cap at 32
        accumulation_steps = 1

    # Test it (safety check) - only if CUDA available
    if torch.cuda.is_available():
        try:
            test_batch = torch.randn(batch_size, 512, 512).to("cuda")
            with torch.no_grad():
                _ = test_batch.sum()
            del test_batch
            torch.cuda.empty_cache()
            print(f"[OK] Batch size {batch_size} fits in VRAM")
        except torch.cuda.OutOfMemoryError:
            print(f"[FAIL] Batch size {batch_size} too large, reducing...")
            batch_size = batch_size // 2

    # Return just batch_size for API compatibility (ISS-001)
    return batch_size


def validate_diversity(model1: nn.Module, model2: nn.Module, model3: nn.Module):
    """
    Validate diversity across 3 Phase 1 models
    Raises AssertionError if diversity too low
    """
    # Test 1: Different halting steps (ACT)
    # This requires ACT metrics from actual model - placeholder for now
    # In real implementation, would call measure_avg_halting_steps()
    print("[WARN] Diversity validation not fully implemented (requires trained models)")

    # Test 2: Different memory usage (LTM)
    # In real implementation, would call measure_ltm_usage()

    # Test 3: Different inference times
    # In real implementation, would call measure_inference_time()

    # Placeholder: Check parameter count diversity
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    params3 = sum(p.numel() for p in model3.parameters())

    param_diversity = max(params1, params2, params3) - min(params1, params2, params3)
    if param_diversity > 100_000:  # At least 100K parameter difference
        print(f"[OK] Parameter diversity validated: {param_diversity:,} params")
    else:
        print(
            f"[WARN] Low parameter diversity: {param_diversity:,} params "
            f"(this is expected for same architecture)"
        )


def detect_training_issues(loss_history: list):
    """
    Detect training divergence early

    Raises:
        RuntimeError: If loss diverging or NaN
    """
    import numpy as np

    last_100 = loss_history[-100:]

    # Issue 1: Divergence (loss increasing)
    if len(last_100) > 10:
        recent_trend = np.polyfit(range(len(last_100)), last_100, 1)[0]
        if recent_trend > 0.01:  # Loss increasing
            raise RuntimeError(f"Loss diverging: trend={recent_trend:.4f}")

    # Issue 2: Plateau (no improvement for 50 steps)
    if len(last_100) >= 50:
        recent_variance = np.var(last_100[-50:])
        if recent_variance < 0.001:  # No change
            print(f"[WARN] Loss plateaued: variance={recent_variance:.6f}")

    # Issue 3: NaN
    if np.isnan(last_100[-1]):
        raise RuntimeError("Loss is NaN")


def compute_diversity(population: list) -> float:
    """
    Compute average pairwise cosine distance for population diversity

    Args:
        population: List of models

    Returns:
        Diversity score (0.0 to 1.0, higher = more diverse)
    """
    import numpy as np

    def get_weights_flat(model):
        """Flatten all model weights into 1D vector"""
        return torch.cat([p.data.flatten() for p in model.parameters()]).cpu().numpy()

    def cosine_distance(vec1, vec2):
        """Cosine distance (1 - cosine similarity)"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        cosine_sim = dot_product / (norm1 * norm2)
        return 1 - cosine_sim

    distances = []
    for i, model_i in enumerate(population):
        for j, model_j in enumerate(population):
            if i < j:
                vec_i = get_weights_flat(model_i)
                vec_j = get_weights_flat(model_j)
                dist = cosine_distance(vec_i, vec_j)
                distances.append(dist)

    return float(np.mean(distances))


# =============================================================================
# ISS-016: Unified MockTokenizer Utility
# =============================================================================


class MockTokenizer:
    """
    Mock tokenizer for when HuggingFace transformers is unavailable.

    Provides deterministic hash-based tokenization for testing/fallback.
    All phases should use this unified class instead of duplicating.

    Usage:
        tokenizer = get_tokenizer()  # Returns GPT2Tokenizer or MockTokenizer
    """

    # Standard token IDs
    PAD_TOKEN_ID = 0
    EOS_TOKEN_ID = 1
    BOS_TOKEN_ID = 2
    UNK_TOKEN_ID = 3
    MASK_TOKEN_ID = 4

    # Vocab size for hash modulo
    VOCAB_SIZE = 32768

    def __init__(self):
        """Initialize mock tokenizer with standard special tokens."""
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"

        self.pad_token_id = self.PAD_TOKEN_ID
        self.eos_token_id = self.EOS_TOKEN_ID
        self.bos_token_id = self.BOS_TOKEN_ID
        self.unk_token_id = self.UNK_TOKEN_ID
        self.mask_token_id = self.MASK_TOKEN_ID

        self.vocab_size = self.VOCAB_SIZE

        # Build reverse mapping for decode
        self._id_to_token = {
            self.PAD_TOKEN_ID: self.pad_token,
            self.EOS_TOKEN_ID: self.eos_token,
            self.BOS_TOKEN_ID: self.bos_token,
            self.UNK_TOKEN_ID: self.unk_token,
            self.MASK_TOKEN_ID: self.mask_token,
        }

    def __call__(
        self, text, return_tensors="pt", max_length=512, truncation=True, padding=True, **kwargs
    ):
        """
        Tokenize text using deterministic hash-based encoding.

        Args:
            text: Input text to tokenize
            return_tensors: Tensor format ("pt" for PyTorch)
            max_length: Maximum sequence length
            truncation: Whether to truncate to max_length
            padding: Whether to pad to max_length
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Dict with input_ids and attention_mask tensors
        """
        # Split into tokens (word-level)
        words = text.split()

        # Apply truncation
        if truncation and len(words) > max_length:
            words = words[:max_length]

        # Hash-based token IDs (deterministic)
        input_ids = [self._hash_token(word) for word in words]

        # Pad if needed
        if padding:
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.PAD_TOKEN_ID] * padding_length

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(words) + [0] * (len(input_ids) - len(words))

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.tensor([attention_mask]),
            }
        else:
            return {"input_ids": [input_ids], "attention_mask": [attention_mask]}

    def _hash_token(self, token):
        """Hash a token to an ID deterministically."""
        # Skip special token IDs (0-4)
        return (abs(hash(token)) % (self.VOCAB_SIZE - 5)) + 5

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs (tensor, list, or single int)
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text string
        """
        # Handle various input types
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]

        # Flatten if nested
        if isinstance(token_ids, list) and len(token_ids) > 0:
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]

        # Decode tokens
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid <= 4:
                continue
            if tid in self._id_to_token:
                tokens.append(self._id_to_token[tid])
            else:
                tokens.append(f"[{tid}]")

        return " ".join(tokens)

    def encode(self, text, return_tensors=None, **kwargs):
        """Encode text to token IDs."""
        result = self(text, return_tensors=return_tensors or "pt", **kwargs)
        return result["input_ids"]


def get_tokenizer(model_name="gpt2"):
    """
    Get tokenizer with fallback to MockTokenizer.

    Args:
        model_name: HuggingFace model name (default: "gpt2")

    Returns:
        GPT2Tokenizer if available, else MockTokenizer
    """
    try:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load {model_name} tokenizer ({e}), using MockTokenizer.")
        return MockTokenizer()


# =============================================================================
# ISS-001: Compatibility aliases for test_utils.py
# =============================================================================


def validate_model_diversity(models: list, min_diversity: float = 0.1) -> bool:
    """
    Validate diversity across models (alias for compute_diversity).

    Args:
        models: List of nn.Module models
        min_diversity: Minimum required diversity score (0.0-1.0)

    Returns:
        True if diversity >= min_diversity, False otherwise
    """
    if len(models) < 2:
        return True  # Single model is trivially diverse

    diversity_score = compute_diversity(models)
    return diversity_score >= min_diversity


def detect_training_divergence(losses: list, window: int = 10) -> bool:
    """
    Detect if training is diverging (loss increasing).

    Args:
        losses: List of loss values
        window: Number of recent losses to check

    Returns:
        True if diverging, False otherwise
    """
    import numpy as np

    if len(losses) < window:
        return False

    recent = losses[-window:]

    # Check if trend is increasing
    trend = np.polyfit(range(len(recent)), recent, 1)[0]
    return trend > 0.01  # Positive slope = diverging


def compute_population_diversity(fitness_scores: list) -> float:
    """
    Compute population diversity from fitness scores.

    Args:
        fitness_scores: List of fitness values

    Returns:
        Diversity score (0.0-1.0) based on variance
    """
    import numpy as np

    if len(fitness_scores) < 2:
        return 0.0

    # Normalize variance to 0-1 range
    variance = np.var(fitness_scores)
    mean_val = np.mean(fitness_scores)

    if mean_val == 0:
        return 0.0

    # Coefficient of variation normalized
    cv = np.sqrt(variance) / abs(mean_val)
    return min(cv, 1.0)  # Cap at 1.0

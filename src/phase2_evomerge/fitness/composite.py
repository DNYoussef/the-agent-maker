"""
Composite fitness scoring for evolutionary optimization.

This module combines four component metrics (perplexity, accuracy, speed, memory)
into a single composite fitness score using weighted averaging.
"""

from typing import Any, Dict, Optional

# Default fitness weights (from V1 Phase 2)
DEFAULT_WEIGHTS = {
    "perplexity": 0.4,  # 40% - Language modeling quality
    "accuracy": 0.3,  # 30% - Task performance
    "speed": 0.2,  # 20% - Inference efficiency
    "memory": 0.1,  # 10% - Resource usage
}

# Default expected values for normalization
DEFAULT_EXPECTED = {
    "perplexity": 15.0,  # Typical for 25M param model
    "speed": 1200.0,  # tokens/sec on GTX 1660
    "memory": 500.0,  # MB (25M params × 4 bytes × 2)
}


def compute_composite_fitness(
    perplexity: float,
    accuracy: float,
    speed: float,
    memory: float,
    weights: Optional[Dict[str, float]] = None,
    expected_values: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute composite fitness score from component metrics.

    Fitness formula:
        composite = w_ppl * (1/ppl) + w_acc * acc +
                   w_spd * (spd/expected_spd) + w_mem * (expected_mem/mem)

    Where:
    - Lower perplexity = better (inverted)
    - Higher accuracy = better (direct)
    - Higher speed = better (normalized)
    - Lower memory = better (inverted)

    Args:
        perplexity: Perplexity value (lower is better)
        accuracy: Accuracy value (0.0-1.0, higher is better)
        speed: Tokens/second (higher is better)
        memory: Peak memory MB (lower is better)
        weights: Fitness weights dict (default: 0.4/0.3/0.2/0.1)
        expected_values: Normalization baselines (default: 15/1200/500)

    Returns:
        Dictionary with composite score and components:
        {
            'composite': 0.185,  # Weighted average
            'components': {
                'perplexity': 15.2,
                'perplexity_score': 0.0658,  # 1/15.2
                'accuracy': 0.48,
                'speed': 1250.0,
                'speed_score': 1.042,  # 1250/1200
                'memory': 520.0,
                'memory_score': 0.962  # 500/520
            }
        }

    Raises:
        ValueError: If weights don't sum to 1.0
        ValueError: If any value is negative
        ValueError: If perplexity or memory is zero
    """
    # Use default weights/expected if not provided
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if expected_values is None:
        expected_values = DEFAULT_EXPECTED

    # Validate weights sum to 1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.6f}")

    # Validate no negative values
    if perplexity < 0 or accuracy < 0 or speed < 0 or memory < 0:
        raise ValueError(
            f"All values must be non-negative: "
            f"ppl={perplexity}, acc={accuracy}, "
            f"speed={speed}, memory={memory}"
        )

    # Validate no zero division
    if perplexity == 0:
        raise ValueError("Perplexity cannot be zero")
    if memory == 0:
        raise ValueError("Memory cannot be zero")

    # Compute component scores
    perplexity_score = 1.0 / perplexity
    accuracy_score = accuracy
    speed_score = speed / expected_values["speed"]
    memory_score = expected_values["memory"] / memory

    # Compute composite fitness
    composite_fitness = (
        weights["perplexity"] * perplexity_score
        + weights["accuracy"] * accuracy_score
        + weights["speed"] * speed_score
        + weights["memory"] * memory_score
    )

    # Return results
    return {
        "composite": composite_fitness,
        "components": {
            "perplexity": perplexity,
            "perplexity_score": perplexity_score,
            "accuracy": accuracy,
            "speed": speed,
            "speed_score": speed_score,
            "memory": memory,
            "memory_score": memory_score,
        },
    }

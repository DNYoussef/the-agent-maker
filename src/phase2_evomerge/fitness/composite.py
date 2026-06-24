"""
Composite fitness scoring for evolutionary optimization.

This module combines four normalized component metrics (perplexity, accuracy,
speed, memory) into a single composite fitness score using weighted averaging.
"""

from typing import Any, Dict, Optional

# Default fitness weights (from V1 Phase 2)
DEFAULT_WEIGHTS = {
    "perplexity": 0.4,  # 40% - Language modeling quality
    "accuracy": 0.3,  # 30% - Task performance
    "speed": 0.2,  # 20% - Inference efficiency
    "memory": 0.1,  # 10% - Resource usage
}

REQUIRED_METRICS = {"perplexity", "accuracy", "speed", "memory"}
REQUIRED_EXPECTED = {"perplexity", "speed", "memory"}

# Default expected values for normalization
DEFAULT_EXPECTED = {
    "perplexity": 15.0,  # Typical for 25M param model
    "speed": 1200.0,  # tokens/sec on GTX 1660
    "memory": 500.0,  # MB (25M params × 4 bytes × 2)
}


def _validate_fitness_weights(weights: Dict[str, float]) -> None:
    """Validate fitness weights contain exactly the required metrics and sum to 1.0."""
    missing_weights = REQUIRED_METRICS - set(weights)
    extra_weights = set(weights) - REQUIRED_METRICS
    if missing_weights or extra_weights:
        raise ValueError(
            f"Weights must contain exactly {sorted(REQUIRED_METRICS)}; "
            f"missing={sorted(missing_weights)}, extra={sorted(extra_weights)}"
        )

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.6f}")


def _validate_expected_values(expected_values: Dict[str, float]) -> None:
    """Validate expected normalization baselines are present and positive."""
    missing_expected = REQUIRED_EXPECTED - set(expected_values)
    if missing_expected:
        raise ValueError(f"Missing expected value(s): {sorted(missing_expected)}")

    for key in REQUIRED_EXPECTED:
        if expected_values[key] <= 0:
            raise ValueError(f"Expected value for {key} must be positive")


def _validate_fitness_inputs(
    perplexity: float,
    accuracy: float,
    speed: float,
    memory: float,
) -> None:
    """Validate raw metric inputs are non-negative, non-zero where required, bounded."""
    if perplexity < 0 or accuracy < 0 or speed < 0 or memory < 0:
        raise ValueError(
            f"All values must be non-negative: "
            f"ppl={perplexity}, acc={accuracy}, "
            f"speed={speed}, memory={memory}"
        )

    if perplexity == 0:
        raise ValueError("Perplexity cannot be zero")
    if memory == 0:
        raise ValueError("Memory cannot be zero")
    if accuracy > 1:
        raise ValueError("Accuracy must be between 0.0 and 1.0")


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
        composite = w_ppl * norm_ppl + w_acc * acc +
                   w_spd * norm_spd + w_mem * norm_mem

    Where:
    - Lower perplexity = better, normalized as min(expected_ppl / ppl, 1.0)
    - Higher accuracy = better, already bounded to [0, 1]
    - Higher speed = better, normalized as min(speed / expected_speed, 1.0)
    - Lower memory = better, normalized as min(expected_memory / memory, 1.0)

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
                'perplexity_score': 0.987,  # expected_ppl / 15.2
                'accuracy': 0.48,
                'accuracy_score': 0.48,
                'speed': 1250.0,
                'speed_score': 1.0,  # capped speed/expected_speed
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

    # Validate inputs
    _validate_fitness_weights(weights)
    _validate_expected_values(expected_values)
    _validate_fitness_inputs(perplexity, accuracy, speed, memory)

    # Compute bounded component scores so nominal weights control contribution.
    perplexity_score = min(expected_values["perplexity"] / perplexity, 1.0)
    accuracy_score = accuracy
    speed_score = min(speed / expected_values["speed"], 1.0)
    memory_score = min(expected_values["memory"] / memory, 1.0)

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
            "accuracy_score": accuracy_score,
            "speed": speed,
            "speed_score": speed_score,
            "memory": memory,
            "memory_score": memory_score,
        },
    }

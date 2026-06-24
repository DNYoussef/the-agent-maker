"""
k-Formula Expert Routing Utilities

Expert routing temperature and top-k selection based on
k(L) = -0.0137*log10(L) + 0.1593 where L = input complexity.

Key Insight:
    - Complex inputs (low k) -> softer routing, use multiple experts
    - Simple inputs (high k) -> sharper routing, single expert sufficient

Phase Applications:
    - Phase 7 (Experts): Adaptive routing temperature and expert selection
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# Layer 1 import only
from ..k_formula import compute_k, normalize_k_value


@dataclass
class RoutingConfig:
    """Configuration for expert routing."""

    # Temperature range
    base_temperature: float = 1.0
    min_temperature: float = 0.1
    max_temperature: float = 10.0

    # Top-k range
    min_k: int = 1
    max_k: int = 4

    # Noise scale range
    min_noise: float = 0.0
    max_noise: float = 0.1

    # k-formula parameters
    epsilon: float = 0.1  # Added to k to avoid division issues


# =============================================================================
# ROUTING TEMPERATURE (Phase 7 Experts)
# =============================================================================


def get_routing_temperature(
    input_complexity: float,
    config: Optional[RoutingConfig] = None,
) -> float:
    """
    Get routing temperature based on input complexity.

    Complex inputs (low k) -> high temperature (soft routing, multiple experts)
    Simple inputs (high k) -> low temperature (sharp routing, single expert)

    Formula: temperature = base / (k + epsilon)

    Args:
        input_complexity: Input complexity measure (e.g., entropy, perplexity)
                         Higher values = more complex
        config: Optional configuration

    Returns:
        Routing temperature

    Example:
        >>> # Complex input (high entropy)
        >>> temp_complex = get_routing_temperature(5.0)
        >>> print(f"Complex: temp = {temp_complex:.2f}")
        Complex: temp = 7.14

        >>> # Simple input (low entropy)
        >>> temp_simple = get_routing_temperature(0.1)
        >>> print(f"Simple: temp = {temp_simple:.2f}")
        Simple: temp = 1.23
    """
    config = config or RoutingConfig()

    # Compute k from complexity (higher complexity -> lower k)
    k = compute_k(max(input_complexity, 1e-10))

    # Temperature inversely proportional to k
    # Low k (complex) -> high temperature (soft routing)
    # High k (simple) -> low temperature (sharp routing)
    temperature = config.base_temperature / (k + config.epsilon)

    return max(config.min_temperature, min(config.max_temperature, temperature))


def get_routing_temperatures_batch(
    complexities: list,
    config: Optional[RoutingConfig] = None,
) -> list:
    """
    Get routing temperatures for a batch of inputs.

    Args:
        complexities: List of complexity values
        config: Optional configuration

    Returns:
        List of temperatures
    """
    return [get_routing_temperature(c, config) for c in complexities]


# =============================================================================
# ADAPTIVE TOP-K (Phase 7 Experts)
# =============================================================================


def get_adaptive_top_k(
    input_complexity: float,
    config: Optional[RoutingConfig] = None,
) -> int:
    """
    Get adaptive top-k value based on input complexity.

    Complex inputs -> use more experts (higher k)
    Simple inputs -> use fewer experts (lower k)

    Args:
        input_complexity: Input complexity measure
        config: Optional configuration

    Returns:
        Number of experts to use (top-k)

    Example:
        >>> # Complex input needs multiple experts
        >>> k_complex = get_adaptive_top_k(5.0)
        >>> print(f"Complex: top-{k_complex}")
        Complex: top-4

        >>> # Simple input needs single expert
        >>> k_simple = get_adaptive_top_k(0.1)
        >>> print(f"Simple: top-{k_simple}")
        Simple: top-1
    """
    config = config or RoutingConfig()

    # Compute k from complexity
    k_formula = compute_k(max(input_complexity, 1e-10))

    # Normalize k to [0, 1]
    k_normalized = normalize_k_value(k_formula)

    # Inverse: low k (complex) -> use more experts
    # High k (simple) -> use fewer experts
    k_range = config.max_k - config.min_k
    top_k = config.min_k + int(k_range * (1 - k_normalized))

    return max(config.min_k, min(config.max_k, top_k))


def get_adaptive_top_k_batch(
    complexities: list,
    config: Optional[RoutingConfig] = None,
) -> list:
    """
    Get adaptive top-k values for a batch of inputs.

    Args:
        complexities: List of complexity values
        config: Optional configuration

    Returns:
        List of top-k values
    """
    return [get_adaptive_top_k(c, config) for c in complexities]


# =============================================================================
# ROUTING NOISE (Phase 7 Experts)
# =============================================================================


def get_routing_noise_scale(
    input_complexity: float,
    config: Optional[RoutingConfig] = None,
) -> float:
    """
    Get noise scale for load balancing in expert routing.

    Complex inputs -> more noise (encourage exploration)
    Simple inputs -> less noise (stick with best expert)

    Args:
        input_complexity: Input complexity measure
        config: Optional configuration

    Returns:
        Noise scale for gating logits

    Example:
        >>> noise = get_routing_noise_scale(5.0)
        >>> # Add to gating logits: logits + noise * torch.randn_like(logits)
    """
    config = config or RoutingConfig()

    # Compute k from complexity
    k = compute_k(max(input_complexity, 1e-10))

    # Normalize k to [0, 1]
    k_normalized = normalize_k_value(k)

    # Inverse: low k (complex) -> more noise
    noise_range = config.max_noise - config.min_noise
    noise_scale = config.min_noise + noise_range * (1 - k_normalized)

    return max(config.min_noise, min(config.max_noise, noise_scale))


# =============================================================================
# COMBINED ROUTING PARAMETERS
# =============================================================================


def get_routing_params(
    input_complexity: float,
    config: Optional[RoutingConfig] = None,
) -> dict:
    """
    Get all routing parameters for an input.

    Args:
        input_complexity: Input complexity measure
        config: Optional configuration

    Returns:
        Dictionary with temperature, top_k, noise_scale

    Example:
        >>> params = get_routing_params(3.0)
        >>> print(params)
        {'temperature': 3.45, 'top_k': 3, 'noise_scale': 0.05, 'k_value': 0.137}
    """
    config = config or RoutingConfig()

    k = compute_k(max(input_complexity, 1e-10))

    return {
        "temperature": get_routing_temperature(input_complexity, config),
        "top_k": get_adaptive_top_k(input_complexity, config),
        "noise_scale": get_routing_noise_scale(input_complexity, config),
        "k_value": k,
        "complexity": input_complexity,
    }


def get_routing_params_batch(
    complexities: list,
    config: Optional[RoutingConfig] = None,
) -> list:
    """
    Get routing parameters for a batch of inputs.

    Args:
        complexities: List of complexity values
        config: Optional configuration

    Returns:
        List of parameter dictionaries
    """
    return [get_routing_params(c, config) for c in complexities]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def estimate_complexity_from_entropy(entropy: float) -> float:
    """
    Estimate input complexity from entropy.

    Args:
        entropy: Input entropy (e.g., from softmax distribution)

    Returns:
        Complexity estimate suitable for routing functions
    """
    # Entropy is already a good complexity measure
    # Scale to reasonable range
    return max(entropy, 1e-10)


def estimate_complexity_from_perplexity(perplexity: float) -> float:
    """
    Estimate input complexity from perplexity.

    Args:
        perplexity: Model perplexity on input

    Returns:
        Complexity estimate suitable for routing functions
    """
    # Perplexity = exp(entropy), so log(perplexity) = entropy
    return max(math.log(perplexity + 1), 1e-10)


def estimate_complexity_from_loss(loss: float) -> float:
    """
    Estimate input complexity from loss value.

    Args:
        loss: Loss value on input

    Returns:
        Complexity estimate suitable for routing functions
    """
    # Higher loss = more complex
    return max(loss, 1e-10)


def print_routing_table(
    complexity_range: Tuple[float, float] = (0.1, 10.0),
    num_samples: int = 10,
    config: Optional[RoutingConfig] = None,
) -> None:
    """
    Print a table of routing parameters across complexity range.

    Args:
        complexity_range: (min, max) complexity values
        num_samples: Number of samples to show
        config: Optional configuration
    """
    config = config or RoutingConfig()

    print(f"\nRouting Parameter Table")
    print("-" * 65)
    print(f"{'Complexity':>10} | {'k value':>8} | {'Temp':>8} | {'Top-k':>6} | {'Noise':>8}")
    print("-" * 65)

    min_c, max_c = complexity_range
    for i in range(num_samples):
        complexity = min_c + (max_c - min_c) * i / (num_samples - 1)
        params = get_routing_params(complexity, config)

        print(
            f"{complexity:>10.2f} | {params['k_value']:>8.4f} | "
            f"{params['temperature']:>8.2f} | {params['top_k']:>6} | "
            f"{params['noise_scale']:>8.4f}"
        )

    print("-" * 65)

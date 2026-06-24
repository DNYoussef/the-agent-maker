"""
Log-Space Operation Utilities

Generic log-space operations for numerical stability.

Phase Applications:
    - All phases: Stable computations for log-normal distributions
    - Phase 3: Log-space RL updates (natural gradient)
    - Phase 8: Log-space compression operations
"""

import math
from dataclasses import dataclass
from typing import Optional

# Layer 1 imports only


@dataclass
class LogSpaceConfig:
    """Configuration for log-space operations."""

    # Numerical stability
    epsilon: float = 1e-10

    # Clamping
    min_value: float = -100.0
    max_value: float = 100.0


# =============================================================================
# SAFE LOG/EXP OPERATIONS
# =============================================================================


def safe_log(x, config: Optional[LogSpaceConfig] = None):
    """
    Numerically stable logarithm.

    Handles zeros and negative values safely.

    Args:
        x: Input tensor or value
        config: Optional configuration

    Returns:
        log(|x| + epsilon)
    """
    config = config or LogSpaceConfig()

    try:
        import torch

        if isinstance(x, torch.Tensor):
            result = torch.log(x.abs() + config.epsilon)
            return torch.clamp(result, config.min_value, config.max_value)
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            result = np.log(np.abs(x) + config.epsilon)
            return np.clip(result, config.min_value, config.max_value)
    except ImportError:
        pass

    # Scalar fallback
    result = math.log(abs(x) + config.epsilon)
    return max(config.min_value, min(config.max_value, result))


def safe_exp(x, config: Optional[LogSpaceConfig] = None):
    """
    Numerically stable exponential.

    Clamps input to prevent overflow.

    Args:
        x: Input tensor or value
        config: Optional configuration

    Returns:
        exp(clamp(x))
    """
    config = config or LogSpaceConfig()

    try:
        import torch

        if isinstance(x, torch.Tensor):
            clamped = torch.clamp(x, config.min_value, config.max_value)
            return torch.exp(clamped)
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            clamped = np.clip(x, config.min_value, config.max_value)
            return np.exp(clamped)
    except ImportError:
        pass

    # Scalar fallback
    clamped = max(config.min_value, min(config.max_value, x))
    return math.exp(clamped)


# =============================================================================
# LOG-SPACE STATISTICS
# =============================================================================


def log_space_mean(x, config: Optional[LogSpaceConfig] = None):
    """
    Compute geometric mean via log-space.

    For positive values: exp(mean(log(x)))
    Handles signs by computing mean of absolute values.

    Args:
        x: Input tensor
        config: Optional configuration

    Returns:
        Geometric mean
    """
    config = config or LogSpaceConfig()

    try:
        import torch

        if isinstance(x, torch.Tensor):
            log_abs = torch.log(x.abs() + config.epsilon)
            return torch.exp(log_abs.mean())
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            log_abs = np.log(np.abs(x) + config.epsilon)
            return np.exp(np.mean(log_abs))
    except ImportError:
        pass

    # List fallback
    log_vals = [math.log(abs(v) + config.epsilon) for v in x]
    return math.exp(sum(log_vals) / len(log_vals))


def log_space_std(x, config: Optional[LogSpaceConfig] = None):
    """
    Compute standard deviation in log-space.

    Useful for analyzing distributions of values across orders of magnitude.

    Args:
        x: Input tensor
        config: Optional configuration

    Returns:
        Standard deviation of log values
    """
    config = config or LogSpaceConfig()

    try:
        import torch

        if isinstance(x, torch.Tensor):
            log_abs = torch.log(x.abs() + config.epsilon)
            return log_abs.std()
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            log_abs = np.log(np.abs(x) + config.epsilon)
            return np.std(log_abs)
    except ImportError:
        pass

    # List fallback
    log_vals = [math.log(abs(v) + config.epsilon) for v in x]
    mean = sum(log_vals) / len(log_vals)
    variance = sum((v - mean) ** 2 for v in log_vals) / len(log_vals)
    return math.sqrt(variance)


def log_space_normalize(x, config: Optional[LogSpaceConfig] = None):
    """
    Normalize tensor in log-space.

    Centers log values to have mean 0, then converts back.
    Result has geometric mean = 1.

    Args:
        x: Input tensor
        config: Optional configuration

    Returns:
        Normalized tensor (geometric mean = 1)
    """
    config = config or LogSpaceConfig()

    try:
        import torch

        if isinstance(x, torch.Tensor):
            signs = torch.sign(x)
            log_abs = torch.log(x.abs() + config.epsilon)
            log_centered = log_abs - log_abs.mean()
            return torch.exp(log_centered) * signs
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            signs = np.sign(x)
            log_abs = np.log(np.abs(x) + config.epsilon)
            log_centered = log_abs - np.mean(log_abs)
            return np.exp(log_centered) * signs
    except ImportError:
        pass

    raise TypeError("Input must be torch.Tensor or numpy.ndarray")


# =============================================================================
# LOG-SPACE INTERPOLATION
# =============================================================================


def log_space_interpolate(
    x1,
    x2,
    alpha: float = 0.5,
    config: Optional[LogSpaceConfig] = None,
):
    """
    Interpolate between two tensors in log-space.

    Equivalent to weighted geometric mean:
    result = x1^alpha * x2^(1-alpha)

    Args:
        x1: First tensor
        x2: Second tensor
        alpha: Interpolation factor (0=x2, 1=x1)
        config: Optional configuration

    Returns:
        Interpolated tensor

    Example:
        >>> # Geometric mean (alpha=0.5)
        >>> geom_mean = log_space_interpolate(a, b, 0.5)
        >>> # Closer to a (alpha=0.8)
        >>> closer_to_a = log_space_interpolate(a, b, 0.8)
    """
    config = config or LogSpaceConfig()

    try:
        import torch

        if isinstance(x1, torch.Tensor):
            # Handle signs
            sign1 = torch.sign(x1)
            sign2 = torch.sign(x2)

            log1 = torch.log(x1.abs() + config.epsilon)
            log2 = torch.log(x2.abs() + config.epsilon)

            log_interp = alpha * log1 + (1 - alpha) * log2
            abs_interp = torch.exp(log_interp)

            # Merge signs
            sign_interp = torch.sign(alpha * sign1 + (1 - alpha) * sign2 + config.epsilon)

            return abs_interp * sign_interp
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(x1, np.ndarray):
            sign1 = np.sign(x1)
            sign2 = np.sign(x2)

            log1 = np.log(np.abs(x1) + config.epsilon)
            log2 = np.log(np.abs(x2) + config.epsilon)

            log_interp = alpha * log1 + (1 - alpha) * log2
            abs_interp = np.exp(log_interp)

            sign_interp = np.sign(alpha * sign1 + (1 - alpha) * sign2 + config.epsilon)

            return abs_interp * sign_interp
    except ImportError:
        pass

    raise TypeError("Inputs must be torch.Tensor or numpy.ndarray")


# =============================================================================
# LOG-SPACE POLICY GRADIENT (Phase 3)
# =============================================================================


def log_space_policy_gradient(
    log_probs,
    advantages,
    config: Optional[LogSpaceConfig] = None,
):
    """
    Compute policy gradient in log-probability space.

    This is related to natural gradient methods and provides
    more stable updates for RL.

    Args:
        log_probs: Log probabilities of actions
        advantages: Advantage estimates

    Returns:
        Policy gradient loss

    Example:
        >>> loss = log_space_policy_gradient(log_probs, advantages)
        >>> loss.backward()
    """
    config = config or LogSpaceConfig()

    try:
        import torch

        if isinstance(log_probs, torch.Tensor):
            # Standard policy gradient: -log_prob * advantage
            # But we compute in a numerically stable way
            clamped_log = torch.clamp(log_probs, config.min_value, 0)
            return -(clamped_log * advantages).mean()
    except ImportError:
        pass

    raise TypeError("log_probs must be torch.Tensor")


def log_space_kl_divergence(
    log_p,
    log_q,
    config: Optional[LogSpaceConfig] = None,
):
    """
    Compute KL divergence in log-space.

    KL(P||Q) = sum(P * (log_P - log_Q))

    Args:
        log_p: Log probabilities of P
        log_q: Log probabilities of Q

    Returns:
        KL divergence
    """
    config = config or LogSpaceConfig()

    try:
        import torch

        if isinstance(log_p, torch.Tensor):
            # P = exp(log_p)
            p = torch.exp(log_p)
            kl = (p * (log_p - log_q)).sum()
            return kl
    except ImportError:
        pass

    raise TypeError("Inputs must be torch.Tensor")

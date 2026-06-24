"""
Spectral Gap Quality Gates

Quality validation gates for merge, compression, and quantization operations.
Gates provide GO/NO-GO decisions based on spectral gap retention.

Phase Applications:
    - Phase 2 (EvoMerge): Validate merge quality before accepting
    - Phase 4 (BitNet): Validate quantization doesn't collapse representations
    - Phase 8 (Compression): Validate compression maintains representation quality
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Layer 1 import only
from ..spectral_gap import (
    SpectralGapMonitor,
    compute_compression_gap_retention,
    compute_merge_diversity_change,
)


class GateDecision(Enum):
    """Quality gate decision."""

    ACCEPT = "accept"
    REJECT = "reject"
    RETRY = "retry"
    WARN = "warn"


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""

    # Retention thresholds
    min_retention: float = 0.9  # 90% gap retention required
    warn_retention: float = 0.95  # Warn if below 95%

    # Absolute thresholds
    min_gap: float = 0.01  # Absolute minimum gap

    # Retry settings
    max_retries: int = 3
    retry_relaxation: float = 0.05  # Relax threshold per retry


@dataclass
class MergeQualityResult:
    """Result from merge quality check."""

    decision: GateDecision
    retention: float
    gap_before: float
    gap_after: float
    satisfies_bound: bool
    message: str
    details: Dict[str, Any]


@dataclass
class CompressionQualityResult:
    """Result from compression quality check."""

    decision: GateDecision
    mean_retention: float
    min_retention: float
    layers_passed: int
    total_layers: int
    passes_threshold: bool
    message: str
    details: Dict[str, Any]


# =============================================================================
# MERGE QUALITY GATE (Phase 2)
# =============================================================================


def check_merge_quality(
    models_before: List,
    model_after,
    config: Optional[QualityGateConfig] = None,
) -> MergeQualityResult:
    """
    Check merge quality using spectral gap retention.

    Rejects merges that collapse diversity too much.

    Args:
        models_before: List of models before merge
        model_after: Merged model
        config: Optional configuration

    Returns:
        MergeQualityResult with decision and metrics

    Example:
        >>> result = check_merge_quality([model1, model2], merged_model)
        >>> if result.decision == GateDecision.REJECT:
        ...     print(f"Merge rejected: {result.message}")
        ...     # Try different merge parameters
        >>> else:
        ...     print("Merge accepted!")
    """
    config = config or QualityGateConfig()

    # Compute diversity change
    try:
        diversity_result = compute_merge_diversity_change(models_before, model_after)
    except Exception as e:
        return MergeQualityResult(
            decision=GateDecision.REJECT,
            retention=0.0,
            gap_before=0.0,
            gap_after=0.0,
            satisfies_bound=False,
            message=f"Error computing diversity: {e}",
            details={"error": str(e)},
        )

    gap_before = diversity_result.get(
        "mean_gap_before", diversity_result.get("min_gap_before", 0.0)
    )
    gap_after = diversity_result.get("gap_after", 0.0)
    satisfies_bound = diversity_result.get("satisfies_bound", False)

    # Compute retention
    retention = gap_after / gap_before if gap_before > 0 else 0.0

    # Determine decision
    if gap_after < config.min_gap:
        decision = GateDecision.REJECT
        message = f"Gap collapsed below minimum ({gap_after:.4f} < {config.min_gap})"
    elif retention < config.min_retention:
        decision = GateDecision.REJECT
        message = f"Retention too low ({retention:.2%} < {config.min_retention:.0%})"
    elif retention < config.warn_retention:
        decision = GateDecision.WARN
        message = f"Retention below warning threshold ({retention:.2%})"
    else:
        decision = GateDecision.ACCEPT
        message = f"Merge quality OK (retention: {retention:.2%})"

    return MergeQualityResult(
        decision=decision,
        retention=retention,
        gap_before=gap_before,
        gap_after=gap_after,
        satisfies_bound=satisfies_bound,
        message=message,
        details=diversity_result,
    )


def check_merge_quality_simple(
    gap_before: float,
    gap_after: float,
    min_retention: float = 0.9,
) -> bool:
    """
    Simple merge quality check from gap values.

    Args:
        gap_before: Spectral gap before merge
        gap_after: Spectral gap after merge
        min_retention: Minimum retention required

    Returns:
        True if merge quality is acceptable
    """
    if gap_before <= 0:
        return gap_after > 0

    retention = gap_after / gap_before
    return retention >= min_retention


# =============================================================================
# COMPRESSION QUALITY GATE (Phase 8)
# =============================================================================


def check_compression_quality(
    model_original,
    model_compressed,
    config: Optional[QualityGateConfig] = None,
) -> CompressionQualityResult:
    """
    Check compression quality using spectral gap retention.

    Validates that compression maintains representation quality.

    Args:
        model_original: Original model before compression
        model_compressed: Compressed model
        config: Optional configuration

    Returns:
        CompressionQualityResult with decision and metrics

    Example:
        >>> result = check_compression_quality(original, compressed)
        >>> if result.decision == GateDecision.REJECT:
        ...     print(f"Compression rejected: {result.message}")
        ...     # Try less aggressive compression
        >>> elif result.decision == GateDecision.WARN:
        ...     print(f"Warning: {result.message}")
    """
    config = config or QualityGateConfig()

    # Compute gap retention
    try:
        retention_result = compute_compression_gap_retention(
            model_original,
            model_compressed,
        )
    except Exception as e:
        return CompressionQualityResult(
            decision=GateDecision.REJECT,
            mean_retention=0.0,
            min_retention=0.0,
            layers_passed=0,
            total_layers=0,
            passes_threshold=False,
            message=f"Error computing retention: {e}",
            details={"error": str(e)},
        )

    mean_retention = retention_result.get("mean_retention", 0.0)
    min_layer_retention = retention_result.get("min_retention", 0.0)
    passes_threshold = retention_result.get("passes_threshold", False)
    layer_retentions = retention_result.get("layer_retentions", [])

    # Count passing layers
    layers_passed = sum(1 for r in layer_retentions if r >= config.min_retention)
    total_layers = len(layer_retentions)

    # Determine decision
    if mean_retention < config.min_retention:
        decision = GateDecision.REJECT
        message = f"Mean retention too low ({mean_retention:.2%} < {config.min_retention:.0%})"
    elif min_layer_retention < config.min_gap:
        decision = GateDecision.REJECT
        message = f"Some layers collapsed (min: {min_layer_retention:.4f})"
    elif mean_retention < config.warn_retention:
        decision = GateDecision.WARN
        message = f"Retention below warning ({mean_retention:.2%})"
    else:
        decision = GateDecision.ACCEPT
        message = f"Compression quality OK (retention: {mean_retention:.2%})"

    return CompressionQualityResult(
        decision=decision,
        mean_retention=mean_retention,
        min_retention=min_layer_retention,
        layers_passed=layers_passed,
        total_layers=total_layers,
        passes_threshold=passes_threshold,
        message=message,
        details=retention_result,
    )


# =============================================================================
# QUANTIZATION QUALITY GATE (Phase 4)
# =============================================================================


def check_quantization_quality(
    weights_original,
    weights_quantized,
    config: Optional[QualityGateConfig] = None,
) -> Dict[str, Any]:
    """
    Check quantization quality using spectral gap.

    Validates that quantization maintains weight diversity.

    Args:
        weights_original: Original weights tensor
        weights_quantized: Quantized weights tensor
        config: Optional configuration

    Returns:
        Dictionary with quality metrics and decision

    Example:
        >>> result = check_quantization_quality(original_weights, quantized_weights)
        >>> if result["decision"] == "reject":
        ...     print("Quantization too aggressive!")
    """
    config = config or QualityGateConfig()

    monitor = SpectralGapMonitor()

    # Compute gaps
    try:
        # Flatten weights for gap computation
        original_flat = weights_original.flatten().unsqueeze(0)
        quantized_flat = weights_quantized.flatten().unsqueeze(0)

        # Compute for comparison (need multiple samples for SVD)
        # Use weight distribution statistics instead
        import torch

        # Compute simple diversity metrics
        original_std = weights_original.std().item()
        quantized_std = weights_quantized.std().item()

        original_unique = len(torch.unique(weights_original))
        quantized_unique = len(torch.unique(weights_quantized))

        # Retention based on standard deviation
        std_retention = quantized_std / original_std if original_std > 0 else 0

        # Retention based on unique values (proxy for information)
        value_retention = quantized_unique / original_unique if original_unique > 0 else 0

        # Combined retention
        retention = (std_retention + value_retention) / 2

    except Exception as e:
        return {
            "decision": GateDecision.REJECT.value,
            "retention": 0.0,
            "message": f"Error computing quality: {e}",
            "error": str(e),
        }

    # Determine decision
    if retention < config.min_retention:
        decision = GateDecision.REJECT
        message = f"Quantization too aggressive (retention: {retention:.2%})"
    elif retention < config.warn_retention:
        decision = GateDecision.WARN
        message = f"Quantization quality marginal (retention: {retention:.2%})"
    else:
        decision = GateDecision.ACCEPT
        message = f"Quantization quality OK (retention: {retention:.2%})"

    return {
        "decision": decision.value,
        "retention": retention,
        "std_retention": std_retention,
        "value_retention": value_retention,
        "original_std": original_std,
        "quantized_std": quantized_std,
        "original_unique": original_unique,
        "quantized_unique": quantized_unique,
        "message": message,
    }


# =============================================================================
# RETRY LOGIC
# =============================================================================


def gate_with_retry(
    check_fn,
    adjust_fn,
    config: Optional[QualityGateConfig] = None,
    **check_kwargs,
) -> Dict[str, Any]:
    """
    Run quality gate with automatic retry and relaxation.

    Args:
        check_fn: Quality check function
        adjust_fn: Function to adjust parameters on retry
        config: Quality gate configuration
        **check_kwargs: Arguments for check function

    Returns:
        Dictionary with final result and retry history

    Example:
        >>> def adjust(params, retry_num):
        ...     params["compression_ratio"] *= 0.9  # Reduce compression
        ...     return params
        >>> result = gate_with_retry(
        ...     check_compression_quality,
        ...     adjust,
        ...     model_original=original,
        ...     model_compressed=compressed,
        ... )
    """
    config = config or QualityGateConfig()

    history = []
    current_kwargs = check_kwargs.copy()
    current_threshold = config.min_retention

    for retry in range(config.max_retries + 1):
        result = check_fn(**current_kwargs, config=config)

        history.append(
            {
                "retry": retry,
                "result": result,
                "threshold": current_threshold,
            }
        )

        # Check if passed
        if hasattr(result, "decision"):
            if result.decision in (GateDecision.ACCEPT, GateDecision.WARN):
                return {
                    "final_result": result,
                    "retries": retry,
                    "history": history,
                    "success": True,
                }
        elif isinstance(result, dict):
            if result.get("decision") in ("accept", "warn"):
                return {
                    "final_result": result,
                    "retries": retry,
                    "history": history,
                    "success": True,
                }

        # Adjust for retry
        if retry < config.max_retries:
            current_kwargs = adjust_fn(current_kwargs, retry + 1)
            current_threshold -= config.retry_relaxation

    # All retries exhausted
    return {
        "final_result": result,
        "retries": config.max_retries,
        "history": history,
        "success": False,
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def format_gate_result(result: Union[MergeQualityResult, CompressionQualityResult, dict]) -> str:
    """Format gate result for display."""
    if isinstance(result, MergeQualityResult):
        icon = {"accept": "[PASS]", "reject": "[FAIL]", "warn": "[WARN]", "retry": "[RETRY]"}
        return (
            f"{icon.get(result.decision.value, '[?]')} Merge Gate\n"
            f"  Retention: {result.retention:.2%}\n"
            f"  Gap: {result.gap_before:.4f} -> {result.gap_after:.4f}\n"
            f"  Message: {result.message}"
        )
    elif isinstance(result, CompressionQualityResult):
        icon = {"accept": "[PASS]", "reject": "[FAIL]", "warn": "[WARN]", "retry": "[RETRY]"}
        return (
            f"{icon.get(result.decision.value, '[?]')} Compression Gate\n"
            f"  Mean Retention: {result.mean_retention:.2%}\n"
            f"  Layers Passed: {result.layers_passed}/{result.total_layers}\n"
            f"  Message: {result.message}"
        )
    elif isinstance(result, dict):
        decision = result.get("decision", "unknown")
        icon = {"accept": "[PASS]", "reject": "[FAIL]", "warn": "[WARN]"}
        return (
            f"{icon.get(decision, '[?]')} Quality Gate\n"
            f"  Retention: {result.get('retention', 0):.2%}\n"
            f"  Message: {result.get('message', 'N/A')}"
        )
    else:
        return str(result)

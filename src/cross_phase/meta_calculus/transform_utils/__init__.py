"""
Transform Utilities (Layer 2)

Bigeometric and log-space transformation utilities that DO NOT require MOO.
These provide gradient/weight transformations based on meta-calculus principles.

Submodules:
    gradients: Gradient transformation utilities
    weights: Weight merging and fitting utilities
    log_space: Generic log-space operations
    quantization: Bigeometric quantization utilities

Usage:
    from src.cross_phase.meta_calculus.transform_utils import (
        apply_bigeometric_to_gradients,
        bigeometric_merge,
        to_log_space,
        bigeometric_threshold,
    )

Design:
    - All functions operate on tensors directly
    - No dependencies on other Layer 2 modules
    - Only imports from Layer 1 (bigeometric.py, k_formula.py)
"""

from .gradients import (
    GradientTransformConfig,
    LogSpaceGradientAccumulator,
    apply_bigeometric_to_gradients,
    apply_bigeometric_to_model,
)
from .log_space import (
    LogSpaceConfig,
    log_space_interpolate,
    log_space_mean,
    log_space_normalize,
    log_space_std,
    safe_exp,
    safe_log,
)
from .quantization import (
    LogSpaceSTE,
    QuantizationConfig,
    apply_bigeometric_quantization,
    bigeometric_threshold,
    get_ternary_mapping,
)
from .weights import (
    WeightMergeConfig,
    bigeometric_merge,
    bigeometric_merge_models,
    bigeometric_merge_tensors,
    fit_weights_log_space,
)

__all__ = [
    # gradients
    "apply_bigeometric_to_gradients",
    "apply_bigeometric_to_model",
    "LogSpaceGradientAccumulator",
    "GradientTransformConfig",
    # weights
    "bigeometric_merge",
    "bigeometric_merge_tensors",
    "bigeometric_merge_models",
    "fit_weights_log_space",
    "WeightMergeConfig",
    # log_space
    "safe_log",
    "safe_exp",
    "log_space_mean",
    "log_space_std",
    "log_space_normalize",
    "log_space_interpolate",
    "LogSpaceConfig",
    # quantization
    "bigeometric_threshold",
    "apply_bigeometric_quantization",
    "get_ternary_mapping",
    "LogSpaceSTE",
    "QuantizationConfig",
]

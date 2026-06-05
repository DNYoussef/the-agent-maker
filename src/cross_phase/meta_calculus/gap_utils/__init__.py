"""
Spectral Gap Utilities (Layer 2)

Spectral gap monitoring, regularization, and quality gates that DO NOT require MOO.
These provide diversity tracking and quality control across all phases.

Submodules:
    monitoring: Phase-specific spectral gap monitoring with thresholds
    regularization: Diversity regularizers and loss functions
    gates: Quality gates for merge/compression validation

Usage:
    from src.cross_phase.meta_calculus.gap_utils import (
        PhaseGapMonitor,
        create_thought_regularizer,
        check_merge_quality,
    )

Design:
    - All functions operate on embeddings/weights directly
    - No dependencies on other Layer 2 modules
    - Only imports from Layer 1 (spectral_gap.py)
"""

from .monitoring import (
    PhaseGapMonitor,
    PhaseGapConfig,
    should_advance_stage,
    get_gap_health_status,
    GapHealthStatus,
)

from .regularization import (
    create_thought_regularizer,
    create_expert_regularizer,
    create_generic_regularizer,
    diversity_loss,
    anti_collapse_loss,
    DiversityRegularizerConfig,
)

from .gates import (
    check_merge_quality,
    check_compression_quality,
    check_quantization_quality,
    MergeQualityResult,
    CompressionQualityResult,
    QualityGateConfig,
)

__all__ = [
    # monitoring
    "PhaseGapMonitor",
    "PhaseGapConfig",
    "should_advance_stage",
    "get_gap_health_status",
    "GapHealthStatus",
    # regularization
    "create_thought_regularizer",
    "create_expert_regularizer",
    "create_generic_regularizer",
    "diversity_loss",
    "anti_collapse_loss",
    "DiversityRegularizerConfig",
    # gates
    "check_merge_quality",
    "check_compression_quality",
    "check_quantization_quality",
    "MergeQualityResult",
    "CompressionQualityResult",
    "QualityGateConfig",
]

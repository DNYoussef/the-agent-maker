"""
Quality Validation Module

Library components deployed from: ~/.claude/library/components/validation/
- quality_validator: Evidence-based quality gate system with threshold-based scoring
- spec_validation: Generalized validation framework for specification documents
"""

from .quality_validator import (
    AnalysisResult,
    EvidenceQuality,
    QualityClaim,
    QualityValidationResult,
    QualityValidator,
    RiskLevel,
    Severity,
    ValidationResult,
    Violation,
)
from .spec_validation import (  # Default schemas
    DEFAULT_CONTEXT_SCHEMA,
    DEFAULT_IMPLEMENTATION_PLAN_SCHEMA,
    DEFAULT_PHASE_SCHEMA,
    DEFAULT_REQUIREMENTS_SCHEMA,
    DEFAULT_SPEC_RECOMMENDED_SECTIONS,
    DEFAULT_SPEC_REQUIRED_SECTIONS,
    DEFAULT_SUBTASK_SCHEMA,
    DEFAULT_VERIFICATION_SCHEMA,
    BaseValidator,
    ContextValidator,
    ImplementationPlanValidator,
    JSONFileValidator,
    MarkdownDocumentValidator,
    PrereqsValidator,
    SpecDocumentValidator,
    SpecValidationResult,
    SpecValidator,
    ValidationSchema,
    create_validator_from_config,
    validate_spec_directory,
)

__all__ = [
    # Quality Validator exports
    "AnalysisResult",
    "EvidenceQuality",
    "QualityClaim",
    "QualityValidationResult",
    "QualityValidator",
    "RiskLevel",
    "Severity",
    "ValidationResult",
    "Violation",
    # Spec Validation exports
    "BaseValidator",
    "ContextValidator",
    "ImplementationPlanValidator",
    "JSONFileValidator",
    "MarkdownDocumentValidator",
    "PrereqsValidator",
    "SpecDocumentValidator",
    "SpecValidator",
    "SpecValidationResult",
    "ValidationSchema",
    "create_validator_from_config",
    "validate_spec_directory",
    # Default schemas
    "DEFAULT_CONTEXT_SCHEMA",
    "DEFAULT_IMPLEMENTATION_PLAN_SCHEMA",
    "DEFAULT_PHASE_SCHEMA",
    "DEFAULT_REQUIREMENTS_SCHEMA",
    "DEFAULT_SPEC_RECOMMENDED_SECTIONS",
    "DEFAULT_SPEC_REQUIRED_SECTIONS",
    "DEFAULT_SUBTASK_SCHEMA",
    "DEFAULT_VERIFICATION_SCHEMA",
]

"""
Phase 8: Compression Target Validation

Validates the compression pipeline meets its targets:
- SeedLM: 2x compression with >95% retention
- VPTQ: 20x compression with >95% retention
- Hypercompression: 6.25x compression with >90% retention
- Cumulative: 250-280x compression with >84% retention

Usage:
    from validation import CompressionValidator, ValidationResult

    validator = CompressionValidator()
    result = validator.validate_full_pipeline(
        original_model, compressed_model, tokenizer
    )
    print(f"Pipeline valid: {result.all_passed}")
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class StageValidationResult:
    """Result from validating a single compression stage."""

    stage_name: str
    target_compression: float
    achieved_compression: float
    compression_passed: bool
    target_retention: float
    achieved_retention: float
    retention_passed: bool
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineValidationResult:
    """Result from validating the full compression pipeline."""

    stages: List[StageValidationResult]
    cumulative_compression: float
    cumulative_retention: float
    target_cumulative_compression: Tuple[float, float]  # (min, max)
    target_cumulative_retention: float
    cumulative_compression_passed: bool
    cumulative_retention_passed: bool
    all_passed: bool
    summary: str


@dataclass
class CompressionTargets:
    """Compression targets for validation."""

    # SeedLM
    seedlm_compression: float = 2.0
    seedlm_retention: float = 0.95

    # VPTQ
    vptq_compression: float = 20.0
    vptq_retention: float = 0.95

    # Hypercompression
    hyper_compression: float = 6.25
    hyper_retention: float = 0.90

    # Cumulative
    cumulative_compression_min: float = 250.0
    cumulative_compression_max: float = 280.0
    cumulative_retention: float = 0.84

    # Tolerance for validation
    compression_tolerance: float = 0.1  # 10% tolerance
    retention_tolerance: float = 0.02  # 2% tolerance


class CompressionValidator:
    """
    Validates compression pipeline meets its targets.

    Targets:
    - SeedLM: 2x compression, >95% retention
    - VPTQ: 20x compression, >95% retention
    - Hypercompression: 6.25x compression, >90% retention
    - Cumulative: 250-280x compression, >84% retention
    """

    def __init__(self, targets: CompressionTargets = None):
        """
        Initialize validator.

        Args:
            targets: Compression targets (uses defaults if None)
        """
        self.targets = targets or CompressionTargets()

    def validate_seedlm(
        self, original_size_mb: float, compressed_size_mb: float, retention: float
    ) -> StageValidationResult:
        """
        Validate SeedLM stage.

        Target: 2x compression, >95% retention
        """
        compression = original_size_mb / max(compressed_size_mb, 0.001)

        # Check compression (with tolerance)
        compression_passed = compression >= (
            self.targets.seedlm_compression * (1 - self.targets.compression_tolerance)
        )

        # Check retention (with tolerance)
        retention_passed = retention >= (
            self.targets.seedlm_retention - self.targets.retention_tolerance
        )

        return StageValidationResult(
            stage_name="SeedLM",
            target_compression=self.targets.seedlm_compression,
            achieved_compression=compression,
            compression_passed=compression_passed,
            target_retention=self.targets.seedlm_retention,
            achieved_retention=retention,
            retention_passed=retention_passed,
            passed=compression_passed and retention_passed,
            details={"original_mb": original_size_mb, "compressed_mb": compressed_size_mb},
        )

    def validate_vptq(
        self, original_size_mb: float, compressed_size_mb: float, retention: float
    ) -> StageValidationResult:
        """
        Validate VPTQ stage.

        Target: 20x compression, >95% retention
        """
        compression = original_size_mb / max(compressed_size_mb, 0.001)

        compression_passed = compression >= (
            self.targets.vptq_compression * (1 - self.targets.compression_tolerance)
        )

        retention_passed = retention >= (
            self.targets.vptq_retention - self.targets.retention_tolerance
        )

        return StageValidationResult(
            stage_name="VPTQ",
            target_compression=self.targets.vptq_compression,
            achieved_compression=compression,
            compression_passed=compression_passed,
            target_retention=self.targets.vptq_retention,
            achieved_retention=retention,
            retention_passed=retention_passed,
            passed=compression_passed and retention_passed,
            details={"original_mb": original_size_mb, "compressed_mb": compressed_size_mb},
        )

    def validate_hypercompression(
        self,
        original_size_mb: float,
        compressed_size_mb: float,
        retention: float,
        r_squared: float = None,
    ) -> StageValidationResult:
        """
        Validate Hypercompression stage.

        Target: 6.25x compression, >90% retention, R^2 > 0.95
        """
        compression = original_size_mb / max(compressed_size_mb, 0.001)

        compression_passed = compression >= (
            self.targets.hyper_compression * (1 - self.targets.compression_tolerance)
        )

        retention_passed = retention >= (
            self.targets.hyper_retention - self.targets.retention_tolerance
        )

        details = {"original_mb": original_size_mb, "compressed_mb": compressed_size_mb}

        # Optional R^2 check
        r_squared_passed = True
        if r_squared is not None:
            r_squared_passed = r_squared >= 0.95
            details["r_squared"] = r_squared
            details["r_squared_passed"] = r_squared_passed

        return StageValidationResult(
            stage_name="Hypercompression",
            target_compression=self.targets.hyper_compression,
            achieved_compression=compression,
            compression_passed=compression_passed,
            target_retention=self.targets.hyper_retention,
            achieved_retention=retention,
            retention_passed=retention_passed,
            passed=compression_passed and retention_passed and r_squared_passed,
            details=details,
        )

    def validate_cumulative(
        self, stages: List[StageValidationResult]
    ) -> Tuple[float, float, bool, bool]:
        """
        Validate cumulative compression and retention.

        Target: 250-280x compression, >84% retention
        """
        # Calculate cumulative compression (product of all stages)
        cumulative_compression = 1.0
        for stage in stages:
            cumulative_compression *= stage.achieved_compression

        # Calculate cumulative retention (product of all stages)
        cumulative_retention = 1.0
        for stage in stages:
            cumulative_retention *= stage.achieved_retention

        # Check cumulative targets
        compression_passed = (
            self.targets.cumulative_compression_min * (1 - self.targets.compression_tolerance)
            <= cumulative_compression
            <= self.targets.cumulative_compression_max * (1 + self.targets.compression_tolerance)
        )

        retention_passed = cumulative_retention >= (
            self.targets.cumulative_retention - self.targets.retention_tolerance
        )

        return (cumulative_compression, cumulative_retention, compression_passed, retention_passed)

    def validate_full_pipeline(
        self,
        seedlm_result: Optional[Dict] = None,
        vptq_result: Optional[Dict] = None,
        hyper_result: Optional[Dict] = None,
    ) -> PipelineValidationResult:
        """
        Validate the full compression pipeline.

        Args:
            seedlm_result: Dict with keys: original_size, compressed_size, retention
            vptq_result: Dict with keys: original_size, compressed_size, retention
            hyper_result: Dict with keys: original_size, compressed_size, retention, r_squared

        Returns:
            PipelineValidationResult with comprehensive validation
        """
        stages = []

        # Validate each stage if provided
        if seedlm_result:
            stages.append(
                self.validate_seedlm(
                    seedlm_result["original_size"],
                    seedlm_result["compressed_size"],
                    seedlm_result["retention"],
                )
            )

        if vptq_result:
            stages.append(
                self.validate_vptq(
                    vptq_result["original_size"],
                    vptq_result["compressed_size"],
                    vptq_result["retention"],
                )
            )

        if hyper_result:
            stages.append(
                self.validate_hypercompression(
                    hyper_result["original_size"],
                    hyper_result["compressed_size"],
                    hyper_result["retention"],
                    hyper_result.get("r_squared"),
                )
            )

        # Validate cumulative
        if stages:
            (
                cumulative_compression,
                cumulative_retention,
                compression_passed,
                retention_passed,
            ) = self.validate_cumulative(stages)
        else:
            cumulative_compression = 1.0
            cumulative_retention = 1.0
            compression_passed = False
            retention_passed = False

        # Check if all stages passed
        all_stages_passed = all(s.passed for s in stages)
        all_passed = all_stages_passed and compression_passed and retention_passed

        # Generate summary
        summary = self._generate_summary(
            stages,
            cumulative_compression,
            cumulative_retention,
            compression_passed,
            retention_passed,
            all_passed,
        )

        return PipelineValidationResult(
            stages=stages,
            cumulative_compression=cumulative_compression,
            cumulative_retention=cumulative_retention,
            target_cumulative_compression=(
                self.targets.cumulative_compression_min,
                self.targets.cumulative_compression_max,
            ),
            target_cumulative_retention=self.targets.cumulative_retention,
            cumulative_compression_passed=compression_passed,
            cumulative_retention_passed=retention_passed,
            all_passed=all_passed,
            summary=summary,
        )

    def _generate_summary(
        self,
        stages: List[StageValidationResult],
        cumulative_compression: float,
        cumulative_retention: float,
        compression_passed: bool,
        retention_passed: bool,
        all_passed: bool,
    ) -> str:
        """Generate human-readable validation summary."""
        lines = ["=" * 60, "COMPRESSION PIPELINE VALIDATION REPORT", "=" * 60, ""]

        # Stage-by-stage results
        for stage in stages:
            status = "PASS" if stage.passed else "FAIL"
            lines.append(f"{stage.stage_name}: [{status}]")
            lines.append(
                f"  Compression: {stage.achieved_compression:.2f}x (target: {stage.target_compression}x) - {'PASS' if stage.compression_passed else 'FAIL'}"
            )
            lines.append(
                f"  Retention: {stage.achieved_retention:.2%} (target: {stage.target_retention:.0%}) - {'PASS' if stage.retention_passed else 'FAIL'}"
            )
            lines.append("")

        # Cumulative results
        lines.append("-" * 60)
        lines.append("CUMULATIVE RESULTS:")
        lines.append(
            f"  Total Compression: {cumulative_compression:.1f}x (target: {self.targets.cumulative_compression_min}-{self.targets.cumulative_compression_max}x) - {'PASS' if compression_passed else 'FAIL'}"
        )
        lines.append(
            f"  Total Retention: {cumulative_retention:.2%} (target: {self.targets.cumulative_retention:.0%}) - {'PASS' if retention_passed else 'FAIL'}"
        )
        lines.append("")

        # Overall status
        lines.append("=" * 60)
        overall = "ALL TARGETS MET" if all_passed else "TARGETS NOT MET"
        lines.append(f"OVERALL STATUS: {overall}")
        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def calculate_model_size(model: nn.Module) -> float:
        """Calculate model size in MB."""
        total_bytes = 0
        for param in model.parameters():
            if param.dtype == torch.float32:
                total_bytes += param.numel() * 4
            elif param.dtype == torch.float16:
                total_bytes += param.numel() * 2
            elif param.dtype in [torch.int8, torch.uint8]:
                total_bytes += param.numel() * 1
            else:
                total_bytes += param.numel() * 4
        return total_bytes / (1024 * 1024)


def validate_compression_targets(
    original_size_mb: float = 100.0,
    seedlm_size_mb: float = 50.0,
    vptq_size_mb: float = 2.5,
    hyper_size_mb: float = 0.4,
    seedlm_retention: float = 0.97,
    vptq_retention: float = 0.96,
    hyper_retention: float = 0.92,
    hyper_r_squared: float = 0.96,
) -> PipelineValidationResult:
    """
    Quick validation function with example values.

    Default values demonstrate the target compression pipeline:
    - 100MB original
    - 50MB after SeedLM (2x)
    - 2.5MB after VPTQ (20x)
    - 0.4MB after Hypercompression (6.25x)
    - Total: 250x compression, ~85% retention

    Args:
        All size and retention parameters

    Returns:
        PipelineValidationResult
    """
    validator = CompressionValidator()

    result = validator.validate_full_pipeline(
        seedlm_result={
            "original_size": original_size_mb,
            "compressed_size": seedlm_size_mb,
            "retention": seedlm_retention,
        },
        vptq_result={
            "original_size": seedlm_size_mb,
            "compressed_size": vptq_size_mb,
            "retention": vptq_retention,
        },
        hyper_result={
            "original_size": vptq_size_mb,
            "compressed_size": hyper_size_mb,
            "retention": hyper_retention,
            "r_squared": hyper_r_squared,
        },
    )

    print(result.summary)
    return result


__all__ = [
    "CompressionValidator",
    "CompressionTargets",
    "StageValidationResult",
    "PipelineValidationResult",
    "validate_compression_targets",
]

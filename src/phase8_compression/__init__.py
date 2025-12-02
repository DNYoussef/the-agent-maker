# Phase 8: Final Compression
"""
Phase 8: Final Compression

Triple Compression Pipeline:
1. SeedLM - Seed-based weight projection (2x compression)
2. VPTQ - Vector post-training quantization (20x compression)
3. Hypercompression - Parametric curve fitting (6.25x compression)

Total: ~280x compression (100MB -> 0.4MB)

Research:
- "SeedLM: Compressing LLM Weights into Seeds"
- "VPTQ: Extreme Compression for LLMs"
- "Hyper-Compression of LLM Weights"

Quality Gates:
- Each stage: >95% retention
- Final cumulative: >84% retention
- Automatic rollback if quality fails
"""

from .benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    CompressionBenchmarkResult,
    GSM8KBenchmark,
    MMLUBenchmark,
)
from .compression_engine import CompressionConfig, CompressionEngine, Phase8Result
from .grokfast_optimizer import GrokfastConfig, GrokfastOptimizer, create_grokfast_optimizer
from .hypercompression import CurveFitMetrics, HyperCompressor, HyperConfig, HyperResult
from .seedlm import SeedLMCompressor, SeedLMConfig, SeedLMResult
from .validation import (
    CompressionTargets,
    CompressionValidator,
    PipelineValidationResult,
    StageValidationResult,
    validate_compression_targets,
)
from .vptq import ResidualQuantizationResult, VPTQCompressor, VPTQConfig, VPTQResult

__all__ = [
    # Main engine
    "CompressionEngine",
    "CompressionConfig",
    "Phase8Result",
    # SeedLM
    "SeedLMCompressor",
    "SeedLMConfig",
    "SeedLMResult",
    # VPTQ
    "VPTQCompressor",
    "VPTQConfig",
    "VPTQResult",
    "ResidualQuantizationResult",
    # Hypercompression
    "HyperCompressor",
    "HyperConfig",
    "HyperResult",
    "CurveFitMetrics",
    # Grokfast
    "GrokfastOptimizer",
    "GrokfastConfig",
    "create_grokfast_optimizer",
    # Benchmarks
    "BenchmarkConfig",
    "BenchmarkResult",
    "CompressionBenchmarkResult",
    "MMLUBenchmark",
    "GSM8KBenchmark",
    "BenchmarkSuite",
    # Validation
    "CompressionValidator",
    "CompressionTargets",
    "StageValidationResult",
    "PipelineValidationResult",
    "validate_compression_targets",
]

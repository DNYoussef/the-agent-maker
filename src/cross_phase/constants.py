"""
Cross-Phase Constants (ISS-015)

Centralized location for all magic numbers and thresholds.
Import from here instead of hardcoding values.

Usage:
    from cross_phase.constants import (
        DEFAULT_MAX_LENGTH,
        VOCAB_SIZE,
        ValidationThresholds
    )
"""

# =============================================================================
# Tokenizer Constants
# =============================================================================
VOCAB_SIZE = 32768
DEFAULT_MAX_LENGTH = 512
DEFAULT_PAD_TOKEN_ID = 0
DEFAULT_EOS_TOKEN_ID = 1
DEFAULT_BOS_TOKEN_ID = 2
DEFAULT_UNK_TOKEN_ID = 3
DEFAULT_MASK_TOKEN_ID = 4

# =============================================================================
# Memory & Size Constants
# =============================================================================
BYTES_PER_MB = 1024 ** 2
BYTES_PER_GB = 1024 ** 3
FP32_BYTES = 4  # 4 bytes per FP32 parameter

# VRAM headroom (10% reserved for system)
VRAM_HEADROOM_RATIO = 0.9

# Max batch size cap
MAX_BATCH_SIZE = 32

# Test tensor dimensions for VRAM testing
TEST_TENSOR_DIM = 512

# =============================================================================
# Training Constants
# =============================================================================
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_WARMUP_STEPS = 500
DEFAULT_LEARNING_RATE = 1e-4

# Loss history window for detecting issues
LOSS_HISTORY_WINDOW = 100
PLATEAU_DETECTION_STEPS = 50
PLATEAU_VARIANCE_THRESHOLD = 0.001
DIVERGENCE_TREND_THRESHOLD = 0.01

# =============================================================================
# Phase-Specific Constants
# =============================================================================

# Phase 2: EvoMerge
EVOMERGE_GENERATIONS = 50
EVOMERGE_POPULATION_SIZE = 10

# Phase 5: Curriculum Learning
CURRICULUM_LEVELS = 10
EDGE_OF_CHAOS_ACCURACY = 0.75  # Target accuracy for difficulty tuning
CONVERGENCE_THRESHOLD = 50  # Questions remaining to consider converged
CONSECUTIVE_FOR_MASTERY = 3
MAX_HINTS_PER_QUESTION = 5

# Phase 7: Self-Guided Experts
MIN_EXPERTS = 3
MAX_EXPERTS = 10
ADAS_POPULATION = 50
ADAS_GENERATIONS = 100
SVF_SINGULAR_VALUES = 32

# Phase 8: Compression
SEEDLM_COMPRESSION_RATIO = 2.0
VPTQ_COMPRESSION_RATIO = 20.0
HYPERCOMPRESSION_RATIO = 6.25
TOTAL_COMPRESSION_TARGET = 280.0  # 2 x 20 x 6.25 = 280

# =============================================================================
# Quality & Validation Thresholds
# =============================================================================

class ValidationThresholds:
    """Thresholds for phase validation (ISS-022)."""

    # Accuracy thresholds
    MIN_BAKING_ACCURACY = 0.85
    MIN_COMPRESSION_RETENTION = 0.84
    MIN_STAGE_RETENTION = 0.95

    # Diversity thresholds for Phase 1
    MIN_PARAMETER_DIVERSITY = 100_000  # 100K params difference

    # Quality gate thresholds
    PHASE1_MIN_VAL_ACCURACY = 0.6
    PHASE2_MIN_FITNESS_GAIN = 0.1
    PHASE3_MIN_COHERENCE = 0.7
    PHASE4_MIN_COMPRESSION = 4.0
    PHASE5_MIN_LEVELS_COMPLETED = 1
    PHASE6_MIN_CYCLES_COMPLETED = 1
    PHASE7_MIN_EXPERTS_DISCOVERED = 2
    PHASE8_MIN_COMPRESSION = 1.0

    # Model size thresholds (bytes)
    TINY_MODEL_MAX = 50_000_000  # <50M params
    SMALL_MODEL_MAX = 500_000_000  # <500M params
    MEDIUM_MODEL_MAX = 2_000_000_000  # <2B params


# =============================================================================
# W&B Metrics Count (for validation)
# =============================================================================

class WandBMetricsCounts:
    """Expected metric counts per phase."""
    PHASE1 = 37
    PHASE2 = 370
    PHASE3 = 17
    PHASE4 = 19
    PHASE5 = 78
    PHASE6 = 32
    PHASE7 = 28
    PHASE8 = 95


# =============================================================================
# QK-Clip Constants
# =============================================================================
DEFAULT_QK_CLIP_THRESHOLD = 30.0
RL_QK_CLIP_THRESHOLD = 25.0  # Tighter for RL training

# =============================================================================
# Sliding Window Attention
# =============================================================================
DEFAULT_WINDOW_SIZE = 256


__all__ = [
    # Tokenizer
    'VOCAB_SIZE',
    'DEFAULT_MAX_LENGTH',
    'DEFAULT_PAD_TOKEN_ID',
    'DEFAULT_EOS_TOKEN_ID',
    'DEFAULT_BOS_TOKEN_ID',
    'DEFAULT_UNK_TOKEN_ID',
    'DEFAULT_MASK_TOKEN_ID',
    # Memory
    'BYTES_PER_MB',
    'BYTES_PER_GB',
    'FP32_BYTES',
    'VRAM_HEADROOM_RATIO',
    'MAX_BATCH_SIZE',
    'TEST_TENSOR_DIM',
    # Training
    'DEFAULT_GRADIENT_CLIP',
    'DEFAULT_WARMUP_STEPS',
    'DEFAULT_LEARNING_RATE',
    'LOSS_HISTORY_WINDOW',
    'PLATEAU_DETECTION_STEPS',
    'PLATEAU_VARIANCE_THRESHOLD',
    'DIVERGENCE_TREND_THRESHOLD',
    # Phases
    'EVOMERGE_GENERATIONS',
    'EVOMERGE_POPULATION_SIZE',
    'CURRICULUM_LEVELS',
    'EDGE_OF_CHAOS_ACCURACY',
    'CONVERGENCE_THRESHOLD',
    'CONSECUTIVE_FOR_MASTERY',
    'MAX_HINTS_PER_QUESTION',
    'MIN_EXPERTS',
    'MAX_EXPERTS',
    'ADAS_POPULATION',
    'ADAS_GENERATIONS',
    'SVF_SINGULAR_VALUES',
    'SEEDLM_COMPRESSION_RATIO',
    'VPTQ_COMPRESSION_RATIO',
    'HYPERCOMPRESSION_RATIO',
    'TOTAL_COMPRESSION_TARGET',
    # Validation
    'ValidationThresholds',
    'WandBMetricsCounts',
    # QK-Clip
    'DEFAULT_QK_CLIP_THRESHOLD',
    'RL_QK_CLIP_THRESHOLD',
    # Attention
    'DEFAULT_WINDOW_SIZE',
]

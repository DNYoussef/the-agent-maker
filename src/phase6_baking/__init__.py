# Phase 6: Tool and Persona Baking
"""
Phase 6: Tool & Persona Baking

Implements iterative A/B optimization loops:
- A-Cycle: Tool use optimization (SWE-Bench style)
- B-Cycle: Persona optimization (self-guided discovery)
- Half-baking for gradual integration
- Plateau detection for automatic cycle switching

Research: "Prompt Baking" (arXiv:2409.13697v1)
Key insight: Model-driven persona discovery, not pre-defined templates.
"""

from .baking_engine import (
    BakingEngine,
    BakingConfig,
    BakingResult,
    BakingCycleType
)
from .a_cycle_tool import ACycleOptimizer, ToolTask, SWEBenchToolEvaluator
from .b_cycle_persona import BCycleOptimizer, PersonaTask
from .half_baking import HalfBaker, HalfBakeConfig, StrengthScheduler
from .plateau_detector import PlateauDetector, PlateauConfig, AdaptivePlateauDetector
from .loss_functions import (
    kl_divergence_loss,
    reverse_kl_divergence_loss,
    jensen_shannon_divergence,
    distillation_loss,
    KLDivergenceLoss,
)
from .swe_bench_eval import (
    SWEBenchEvaluator,
    SWEBenchTask,
    EvaluationResult,
    EvaluationMode,
)

__all__ = [
    # Main engine
    'BakingEngine',
    'BakingConfig',
    'BakingResult',
    'BakingCycleType',
    # A-Cycle
    'ACycleOptimizer',
    'ToolTask',
    'SWEBenchToolEvaluator',  # M4 TIER 1: SWE-Bench adapter for A-cycle
    # B-Cycle
    'BCycleOptimizer',
    'PersonaTask',
    # Half-baking
    'HalfBaker',
    'HalfBakeConfig',
    'StrengthScheduler',
    # Plateau detection
    'PlateauDetector',
    'PlateauConfig',
    'AdaptivePlateauDetector',
    # Loss functions (M4 TIER 1)
    'kl_divergence_loss',
    'reverse_kl_divergence_loss',
    'jensen_shannon_divergence',
    'distillation_loss',
    'KLDivergenceLoss',
    # SWE-Bench evaluation (M4 TIER 1)
    'SWEBenchEvaluator',
    'SWEBenchTask',
    'EvaluationResult',
    'EvaluationMode',
]

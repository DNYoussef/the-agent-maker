"""
Meta-Calculus Integration for Agent Forge V2

This module provides meta-calculus enhancements adapted from the
meta-calculus-toolkit project for use across all 8 phases of Agent Forge.

Core Components:
    - k_formula: Scale-dependent adaptation via k(L) = -0.0137*log10(L) + 0.1593
    - bigeometric: Bigeometric gradient transforms (D_BG[x^n] = e^n)
    - moo_bridge: Multi-objective optimization infrastructure
    - meta_grokfast: Enhanced Grokfast optimizer with meta-calculus
    - spectral_gap: Representation diversity monitoring

Quick Start:
    # Create optimizer for a specific phase
    from src.cross_phase.meta_calculus import MetaGrokfast
    optimizer = MetaGrokfast.for_phase("phase1_cognate", model.parameters())

    # Monitor spectral gaps during training
    from src.cross_phase.meta_calculus import SpectralGapMonitor
    monitor = SpectralGapMonitor()
    gaps = monitor.compute_model_gaps(model)

    # Run multi-objective optimization for Phase 7
    from src.cross_phase.meta_calculus import ExpertDiscoveryProblem, MOORunner
    problem = ExpertDiscoveryProblem(expert_evaluator=my_evaluator)
    result = MOORunner().optimize(problem)

Phase Integration Summary:
    - Phase 1 (Cognate): MetaGrokfast with gentle filtering
    - Phase 2 (EvoMerge): MOO for multi-objective merging
    - Phase 3 (Quiet-STaR): Spectral gap for thought diversity
    - Phase 4 (BitNet): k(L) for layer-wise quantization
    - Phase 5 (Curriculum): MetaGrokfast with STE mode
    - Phase 6 (Baking): Adaptive half-baking strength
    - Phase 7 (Experts): MOO for expert discovery
    - Phase 8 (Compression): Log-space transforms, gap retention
"""

__version__ = "2.0.0"
__author__ = "Agent Forge V2 Team"

# =============================================================================
# k(L) Formula
# =============================================================================
from .k_formula import (
    # Core formula
    compute_k,
    KFormulaConfig,

    # Domain-specific k functions
    k_from_gradient,
    k_from_layer_index,
    k_from_entropy,
    k_from_parameter_variance,

    # Utilities
    k_schedule,
    print_k_table,
    normalize_k_value,

    # Constants
    K_SLOPE,
    K_INTERCEPT,
    K_MIN,
    K_MAX,
)

# =============================================================================
# Bigeometric Transforms
# =============================================================================
from .bigeometric import (
    # Core classes
    BigeometricTransform,
    BigeometricDerivative,
    BigeometricConfig,

    # Main functions
    bigeometric_gradient_transform,
    bigeometric_gradient_with_stats,

    # Log-space operations
    to_log_space,
    from_log_space,
    log_space_interpolation,

    # Verification
    run_verification as verify_bigeometric,
)

# =============================================================================
# MOO Bridge
# =============================================================================
from .moo_bridge import (
    # Configuration
    MOOConfig,
    ObjectiveDefinition,

    # Pre-defined objectives
    EVOMERGE_OBJECTIVES,
    EXPERT_DISCOVERY_OBJECTIVES,
    COMPRESSION_OBJECTIVES,

    # Problem classes
    AgentForgeMOOProblem,
    EvoMergeProblem,
    ExpertDiscoveryProblem,

    # Runner and utilities
    MOORunner,
    select_from_pareto,
    analyze_pareto_front,

    # Cloud adapter
    GlobalMOOAdapter,
)

# =============================================================================
# MetaGrokfast Optimizer
# =============================================================================
from .meta_grokfast import (
    # Main optimizer
    MetaGrokfast,
    MetaGrokfastConfig,

    # Phase configurations
    PHASE_CONFIGS,
    GrokfastFilterType,

    # Utilities
    create_optimizer_for_model,
)

# =============================================================================
# Spectral Gap Monitoring
# =============================================================================
from .spectral_gap import (
    # Core monitoring
    SpectralGapMonitor,
    SpectralGapConfig,

    # Phase-specific functions
    compute_expert_diversity,
    compute_thought_diversity,
    compute_merge_diversity_change,
    compute_compression_gap_retention,

    # Loss/regularization
    SpectralGapRegularizer,
    thought_diversity_loss,
)

# =============================================================================
# Phase Integration (Auto-Triggering)
# =============================================================================
from .phase_integration import (
    # Main integration class
    PhaseIntegration,
    PhaseIntegrationConfig,
    PHASE_INTEGRATION_DEFAULTS,

    # Convenience functions
    get_phase_optimizer,
    get_phase_monitors,
    auto_integrate_phase,

    # MOO functions (Phase 2, 7)
    create_moo_problem,
    run_moo_optimization,

    # Status
    print_integration_status,
)

# =============================================================================
# Convenience Aliases
# =============================================================================

# Quick access to phase-specific optimizer creation
def create_phase_optimizer(phase: str, model, **kwargs):
    """
    Create optimizer configured for a specific Agent Forge phase.

    Args:
        phase: One of 'phase1_cognate', 'phase3_quietstar', 'phase5_curriculum',
               'phase6_baking', 'phase7_experts', 'phase8_compression'
        model: PyTorch model
        **kwargs: Override config values

    Returns:
        MetaGrokfast optimizer
    """
    return MetaGrokfast.for_phase(phase, model.parameters(), **kwargs)


def quick_gap_check(model) -> dict:
    """
    Quick spectral gap check for a model.

    Args:
        model: PyTorch model

    Returns:
        Summary of gap health across layers
    """
    monitor = SpectralGapMonitor()
    gaps = monitor.compute_model_gaps(model)

    healthy = sum(1 for g in gaps.values() if g["is_healthy"])
    collapsed = sum(1 for g in gaps.values() if g["is_collapsed"])

    return {
        "total_layers": len(gaps),
        "healthy_layers": healthy,
        "collapsed_layers": collapsed,
        "health_ratio": healthy / len(gaps) if gaps else 0,
        "needs_attention": collapsed > 0,
    }


# =============================================================================
# Module Info
# =============================================================================

__all__ = [
    # k_formula
    "compute_k",
    "KFormulaConfig",
    "k_from_gradient",
    "k_from_layer_index",
    "k_from_entropy",
    "k_from_parameter_variance",
    "k_schedule",
    "print_k_table",
    "normalize_k_value",
    "K_SLOPE",
    "K_INTERCEPT",
    "K_MIN",
    "K_MAX",

    # bigeometric
    "BigeometricTransform",
    "BigeometricDerivative",
    "BigeometricConfig",
    "bigeometric_gradient_transform",
    "bigeometric_gradient_with_stats",
    "to_log_space",
    "from_log_space",
    "log_space_interpolation",
    "verify_bigeometric",

    # moo_bridge
    "MOOConfig",
    "ObjectiveDefinition",
    "EVOMERGE_OBJECTIVES",
    "EXPERT_DISCOVERY_OBJECTIVES",
    "COMPRESSION_OBJECTIVES",
    "AgentForgeMOOProblem",
    "EvoMergeProblem",
    "ExpertDiscoveryProblem",
    "MOORunner",
    "select_from_pareto",
    "analyze_pareto_front",
    "GlobalMOOAdapter",

    # meta_grokfast
    "MetaGrokfast",
    "MetaGrokfastConfig",
    "PHASE_CONFIGS",
    "GrokfastFilterType",
    "create_optimizer_for_model",

    # spectral_gap
    "SpectralGapMonitor",
    "SpectralGapConfig",
    "compute_expert_diversity",
    "compute_thought_diversity",
    "compute_merge_diversity_change",
    "compute_compression_gap_retention",
    "SpectralGapRegularizer",
    "thought_diversity_loss",

    # phase_integration
    "PhaseIntegration",
    "PhaseIntegrationConfig",
    "PHASE_INTEGRATION_DEFAULTS",
    "get_phase_optimizer",
    "get_phase_monitors",
    "auto_integrate_phase",
    "create_moo_problem",
    "run_moo_optimization",
    "print_integration_status",

    # convenience
    "create_phase_optimizer",
    "quick_gap_check",

    # Layer 2 utility modules (subpackages)
    "k_utils",
    "gap_utils",
    "transform_utils",
    "moo_utils",

    # Layer 3 phase facades
    "phase_facades",
]

# =============================================================================
# Layer 2: Domain Utility Modules
# =============================================================================
from . import k_utils
from . import gap_utils
from . import transform_utils
from . import moo_utils

# =============================================================================
# Layer 3: Phase Facades (Simple Per-Phase API)
# =============================================================================
from . import phase_facades


def print_module_info():
    """Print module information and available components."""
    print(f"""
================================================================================
META-CALCULUS INTEGRATION FOR AGENT FORGE V2
================================================================================
Version: {__version__}

ARCHITECTURE (3-Layer):
-----------------------
Layer 1 (Core):     k_formula, bigeometric, moo_bridge, meta_grokfast, spectral_gap
Layer 2 (Utils):    k_utils/, gap_utils/, transform_utils/, moo_utils/
Layer 3 (Facades):  phase_facades/ (phase1-phase8 simple imports)

LAYER 1 - CORE:
---------------
k_formula, bigeometric, moo_bridge, meta_grokfast, spectral_gap

LAYER 2 - UTILITIES:
--------------------
k_utils/         - Scheduling, layer ratios, routing, adaptive params
gap_utils/       - Monitoring, regularization, quality gates
transform_utils/ - Gradients, weights, log-space, quantization
moo_utils/       - Architecture search, hyperparams, selection, constraints

LAYER 3 - PHASE FACADES:
------------------------
phase_facades/phase1 through phase8 - Simple per-phase imports

QUICK START:
------------
>>> from src.cross_phase.meta_calculus.phase_facades import phase2
>>> ratios = phase2.get_merge_ratios(num_layers=8)

>>> from src.cross_phase.meta_calculus import quick_gap_check
>>> health = quick_gap_check(model)
================================================================================
""")


if __name__ == "__main__":
    print_module_info()

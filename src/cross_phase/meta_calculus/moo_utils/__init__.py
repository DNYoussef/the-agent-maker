"""
MOO Utilities (Layer 2)

Multi-objective optimization utilities using pymoo/GlobalMOO.
These provide MOO-only functionality that does NOT depend on meta-calculus formulas.

Submodules:
    architecture: Architecture and configuration search
    hyperparams: Hyperparameter optimization
    selection: Pareto selection strategies
    constraints: Constraint handling utilities

Usage:
    from src.cross_phase.meta_calculus.moo_utils import (
        search_architecture,
        optimize_training_hyperparams,
        select_knee_point,
        make_param_constraint,
    )

Design:
    - All functions use pymoo/GlobalMOO for optimization
    - No dependencies on k_formula or bigeometric (Layer 2 isolation)
    - Only imports from Layer 1 (moo_bridge.py)
"""

from .architecture import (
    ArchitectureSearchProblem,
    MixedPrecisionProblem,
    ExpertCountProblem,
    search_architecture,
    search_precision_assignment,
    search_expert_count,
    ArchitectureSearchConfig,
)

from .hyperparams import (
    TrainingHyperparamProblem,
    ThoughtHyperparamProblem,
    CurriculumScheduleProblem,
    ABCycleInterleavingProblem,
    optimize_training_hyperparams,
    optimize_thought_hyperparams,
    optimize_curriculum_schedule,
    optimize_ab_interleaving,
    HyperparamSearchConfig,
)

from .selection import (
    select_balanced,
    select_knee_point,
    select_by_constraint,
    select_by_preference,
    analyze_tradeoffs,
    compute_hypervolume,
    SelectionConfig,
)

from .constraints import (
    make_param_constraint,
    make_latency_constraint,
    make_memory_constraint,
    make_accuracy_constraint,
    combine_constraints,
    ConstraintConfig,
)

from .globalmoo_adapter import (
    HybridMOOConfig,
    HybridMOOResult,
    HybridMOORunner,
    create_hybrid_runner,
    check_moo_backends,
    GLOBALMOO_AVAILABLE,
    PYMOO_AVAILABLE,
)

__all__ = [
    # architecture
    "ArchitectureSearchProblem",
    "MixedPrecisionProblem",
    "ExpertCountProblem",
    "search_architecture",
    "search_precision_assignment",
    "search_expert_count",
    "ArchitectureSearchConfig",
    # hyperparams
    "TrainingHyperparamProblem",
    "ThoughtHyperparamProblem",
    "CurriculumScheduleProblem",
    "ABCycleInterleavingProblem",
    "optimize_training_hyperparams",
    "optimize_thought_hyperparams",
    "optimize_curriculum_schedule",
    "optimize_ab_interleaving",
    "HyperparamSearchConfig",
    # selection
    "select_balanced",
    "select_knee_point",
    "select_by_constraint",
    "select_by_preference",
    "analyze_tradeoffs",
    "compute_hypervolume",
    "SelectionConfig",
    # constraints
    "make_param_constraint",
    "make_latency_constraint",
    "make_memory_constraint",
    "make_accuracy_constraint",
    "combine_constraints",
    "ConstraintConfig",
    # globalmoo_adapter
    "HybridMOOConfig",
    "HybridMOOResult",
    "HybridMOORunner",
    "create_hybrid_runner",
    "check_moo_backends",
    "GLOBALMOO_AVAILABLE",
    "PYMOO_AVAILABLE",
]

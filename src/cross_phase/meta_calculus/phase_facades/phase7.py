"""
Phase 7: Self-Guided Experts Facade

Model-driven expert discovery with Transformer2 and ADAS optimizer.
This facade provides:
- Expert count optimization via MOO
- k(L) adaptive routing temperature
- Expert diversity monitoring (spectral gap)
- Expert discovery via NSGA-II

Usage:
    from src.cross_phase.meta_calculus.phase_facades import phase7

    # Get routing temperature based on input complexity
    temp = phase7.get_routing_temperature(complexity=0.5)

    # Run expert discovery MOO
    result = phase7.run_expert_moo(evaluator)

    # Monitor expert diversity
    diversity = phase7.compute_expert_diversity(expert_weights)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

# Layer 1 imports (core)
from ..moo_bridge import ExpertDiscoveryProblem, MOORunner, MOOConfig
from ..spectral_gap import compute_expert_diversity as _compute_expert_diversity
from ..meta_grokfast import MetaGrokfast, GrokfastConfig

# Layer 2 imports (utilities)
from ..k_utils.routing import (
    get_routing_temperature,
    get_adaptive_top_k,
    get_routing_params,
    get_routing_params_batch,
    RoutingConfig,
    estimate_complexity_from_entropy,
)
from ..gap_utils.regularization import (
    create_expert_regularizer,
    expert_diversity_loss,
    DiversityRegularizerConfig,
)
from ..moo_utils.architecture import (
    ExpertCountProblem,
    search_expert_count,
)
from ..moo_utils.selection import (
    select_balanced,
    select_knee_point,
    SelectionResult,
)

# Phase 7 specific defaults
PHASE7_CONFIG = GrokfastConfig(
    alpha=0.96,
    lamb=1.5,
    filter_type="bigeometric",
    warmup_steps=200,
)


def create_optimizer(
    model: Any,
    lr: float = 1e-4,
    config: Optional[GrokfastConfig] = None,
    **adamw_kwargs,
) -> MetaGrokfast:
    """
    Create Phase 7 optimizer for expert training.

    Args:
        model: PyTorch model (MoE)
        lr: Learning rate
        config: Optional custom config
        **adamw_kwargs: Additional AdamW kwargs

    Returns:
        MetaGrokfast configured for Phase 7
    """
    if config is None:
        return MetaGrokfast.for_phase("phase7_experts", model, lr, **adamw_kwargs)

    return MetaGrokfast(model, config=config, lr=lr, **adamw_kwargs)


def get_temperature(
    complexity: float,
    base_temp: float = 1.0,
) -> float:
    """
    Get k(L)-adaptive routing temperature.

    Low complexity: higher temperature (soft routing, explore)
    High complexity: lower temperature (sharp routing, exploit)

    Args:
        complexity: Input complexity [0-1]
        base_temp: Base temperature

    Returns:
        Routing temperature
    """
    config = RoutingConfig(base_temperature=base_temp)
    return get_routing_temperature(complexity, config=config)


def get_top_k(
    complexity: float,
    total_experts: int,
    min_k: int = 1,
    max_k: int = 4,
) -> int:
    """
    Get adaptive top-k for expert selection.

    High complexity: more experts (need specialization)
    Low complexity: fewer experts (efficiency)

    Args:
        complexity: Input complexity [0-1]
        total_experts: Total number of experts
        min_k: Minimum experts to select
        max_k: Maximum experts to select

    Returns:
        Number of experts to route to
    """
    config = RoutingConfig(min_k=min_k, max_k=min(max_k, total_experts))
    return get_adaptive_top_k(complexity, config=config)


def get_all_routing_params(
    complexity: float,
    total_experts: int = 8,
) -> Dict[str, Any]:
    """
    Get all routing parameters for given complexity.

    Args:
        complexity: Input complexity
        total_experts: Number of experts

    Returns:
        Dict with temperature, top_k, noise_scale, etc.
    """
    config = RoutingConfig(max_k=min(4, total_experts))
    return get_routing_params(complexity, config=config)


def estimate_complexity(
    hidden_states: Any,
    method: str = "entropy",
) -> float:
    """
    Estimate input complexity from hidden states.

    Args:
        hidden_states: Hidden state tensor
        method: "entropy", "variance", or "norm"

    Returns:
        Complexity estimate [0-1]
    """
    import torch

    if method == "entropy":
        # Approximate entropy from activation distribution
        if isinstance(hidden_states, torch.Tensor):
            probs = torch.softmax(hidden_states.flatten(), dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            max_entropy = torch.log(torch.tensor(len(probs.flatten())))
            return float((entropy / max_entropy).clamp(0, 1))
        else:
            return estimate_complexity_from_entropy(float(np.std(hidden_states)))
    elif method == "variance":
        if isinstance(hidden_states, torch.Tensor):
            var = hidden_states.var().item()
        else:
            var = np.var(hidden_states)
        return min(1.0, var / 10.0)  # Normalize
    else:
        if isinstance(hidden_states, torch.Tensor):
            norm = hidden_states.norm().item()
        else:
            norm = np.linalg.norm(hidden_states)
        return min(1.0, norm / 100.0)  # Normalize


def create_diversity_regularizer(
    target_gap: float = 0.1,
    weight: float = 0.01,
) -> Callable:
    """
    Create expert diversity regularizer.

    Penalizes expert collapse (when experts become too similar).

    Args:
        target_gap: Target spectral gap between experts
        weight: Regularization weight

    Returns:
        Regularizer function
    """
    config = DiversityRegularizerConfig(
        target_diversity=target_gap,
        weight=weight,
    )
    return create_expert_regularizer(config=config)


def compute_expert_diversity(
    expert_weights: Any,
) -> float:
    """
    Compute diversity (spectral gap) of expert weight matrices.

    Higher gap = more diverse experts.

    Args:
        expert_weights: Tensor of expert weights (n_experts, ...)

    Returns:
        Spectral gap value
    """
    return _compute_expert_diversity(list(expert_weights)).get("gap", 0.0)


def run_expert_moo(
    evaluator: Callable,
    n_generations: int = 50,
    pop_size: int = 30,
) -> Any:
    """
    Run multi-objective expert discovery.

    Objectives: task_loss, routing_entropy, compute_cost, expert_diversity, load_balance

    Args:
        evaluator: Function (expert_config) -> objective_values
        n_generations: Evolution generations
        pop_size: Population size

    Returns:
        pymoo Result with Pareto-optimal expert configurations
    """
    problem = ExpertDiscoveryProblem(expert_evaluator=evaluator)
    runner = MOORunner(MOOConfig(n_generations=n_generations, population_size=pop_size))
    return runner.optimize(problem)


def select_best_config(
    result: Any,
    method: str = "knee",
) -> SelectionResult:
    """
    Select best expert configuration from Pareto front.

    Args:
        result: MOO result
        method: "knee" or "balanced"

    Returns:
        SelectionResult with best configuration
    """
    if method == "knee":
        return select_knee_point(result)
    else:
        return select_balanced(result)


def search_optimal_count(
    evaluator: Callable,
    min_experts: int = 2,
    max_experts: int = 16,
    n_generations: int = 30,
) -> Any:
    """
    Search for optimal number of experts.

    Objectives: capacity, efficiency, diversity

    Args:
        evaluator: Function (n_experts) -> objectives
        min_experts: Minimum experts
        max_experts: Maximum experts
        n_generations: Evolution generations

    Returns:
        Pareto-optimal expert counts
    """
    from ..moo_utils.architecture import ArchitectureSearchConfig

    config = ArchitectureSearchConfig(
        n_generations=n_generations,
        min_layers=min_experts,
        max_layers=max_experts,
    )
    return search_expert_count(evaluator, config=config)


__all__ = [
    "create_optimizer",
    "get_temperature",
    "get_top_k",
    "get_all_routing_params",
    "estimate_complexity",
    "create_diversity_regularizer",
    "compute_expert_diversity",
    "run_expert_moo",
    "select_best_config",
    "search_optimal_count",
    "PHASE7_CONFIG",
]

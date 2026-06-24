"""
Multi-Objective Optimization Bridge for Agent Forge V2

This module provides MOO infrastructure adapted from meta-calculus-toolkit
for use in Agent Forge phases:
    - Phase 2 (EvoMerge): Multi-objective model merging
    - Phase 7 (Experts): Expert discovery and routing optimization
    - General hyperparameter optimization

Key Features:
    - NSGA-II integration via pymoo
    - GlobalMOO API adapter (optional cloud optimization)
    - Pre-defined objective sets for common use cases
    - Pareto front analysis utilities
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.indicators.hv import HV
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    Problem = object  # Placeholder

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MOOConfig:
    """Configuration for multi-objective optimization."""

    # Algorithm settings
    population_size: int = 50
    n_generations: int = 100
    seed: int = 42

    # NSGA-II parameters
    crossover_prob: float = 0.9
    crossover_eta: float = 15
    mutation_eta: float = 20

    # Termination
    max_evaluations: Optional[int] = None
    convergence_threshold: float = 1e-6

    # Output
    verbose: bool = True
    save_history: bool = True


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Definition of a single optimization objective.

    Frozen: instances are shared across objective tables (MDL_OBJECTIVE
    appears in both EVOMERGE and EXPERT_DISCOVERY lists), so mutation would
    alias across problems. Use dataclasses.replace() to derive variants.
    """

    name: str
    minimize: bool = True  # False for maximize
    weight: float = 1.0
    bounds: Tuple[float, float] = (-np.inf, np.inf)
    description: str = ""


def description_length_bits(
    *,
    active_components: int,
    total_components: int,
    param_count: float,
    residual_nll_bits: Optional[float] = None,
) -> float:
    """Two-part MDL score in bits: L(model) + L(data | model).

    L(model): log2 C(total, active) for the support choice, plus
              active * log2(param_count) as the pointer/precision proxy.
    L(data):  residual_nll_bits when the caller supplies it. Defaults to
              0.0 because perplexity (negative log-likelihood) is already a
              SEPARATE objective in every predefined set - passing it here
              too would double count the data term.

    Refs: Rissanen 1978; Grunwald, The MDL Principle, ch. 5 (two-part codes).

    Args:
        active_components: components actually used (weight above threshold,
            experts selected, ...). Clamped to total_components.
        total_components: size of the component pool. Must be >= 1.
        param_count: parameter-count proxy for pointer precision. Values
            below 1 are treated as 1 (zero pointer bits).
        residual_nll_bits: optional explicit data term in bits.

    Returns:
        Description length in bits (always finite, >= 0).
    """
    if total_components < 1:
        raise ValueError(f"total_components must be >= 1, got {total_components}")
    if active_components < 0:
        raise ValueError(f"active_components must be >= 0, got {active_components}")

    active = min(active_components, total_components)
    log_binom = (
        math.lgamma(total_components + 1)
        - math.lgamma(active + 1)
        - math.lgamma(total_components - active + 1)
    ) / math.log(2)
    pointer_bits = active * math.log2(max(param_count, 1.0))
    data_bits = residual_nll_bits if residual_nll_bits is not None else 0.0
    return log_binom + pointer_bits + data_bits


# =============================================================================
# Pre-defined Objective Sets for Agent Forge Phases
# =============================================================================

MDL_OBJECTIVE = ObjectiveDefinition(
    name="description_length",
    minimize=True,
    description=(
        "Two-part MDL model bits (lower is better). The data half of MDL is "
        "the perplexity/task-loss objective; do not double count."
    ),
)

EVOMERGE_OBJECTIVES = [
    ObjectiveDefinition(
        name="perplexity", minimize=True, description="Task performance (lower is better)"
    ),
    ObjectiveDefinition(
        name="spectral_gap",
        minimize=False,
        description="Representation diversity (higher is better)",
    ),
    ObjectiveDefinition(
        name="weight_norm", minimize=True, description="Compression readiness (lower is better)"
    ),
    ObjectiveDefinition(
        name="invariance_score",
        minimize=False,
        description="Cross-seed agreement (higher is better)",
    ),
    ObjectiveDefinition(
        name="fragility", minimize=True, description="Perturbation sensitivity (lower is better)"
    ),
    MDL_OBJECTIVE,
]

EXPERT_DISCOVERY_OBJECTIVES = [
    ObjectiveDefinition(name="task_loss", minimize=True, description="Primary task performance"),
    ObjectiveDefinition(
        name="expert_diversity", minimize=False, description="Spectral gap of expert weights"
    ),
    ObjectiveDefinition(
        name="routing_entropy", minimize=False, description="Balanced expert utilization"
    ),
    ObjectiveDefinition(name="compute_cost", minimize=True, description="Inference FLOPs"),
    ObjectiveDefinition(
        name="robustness", minimize=False, description="Cross-seed variance (inverse)"
    ),
    MDL_OBJECTIVE,
]

COMPRESSION_OBJECTIVES = [
    ObjectiveDefinition(
        name="accuracy_loss", minimize=True, description="Accuracy drop from compression"
    ),
    ObjectiveDefinition(
        name="compression_ratio", minimize=False, description="Size reduction factor"
    ),
    ObjectiveDefinition(
        name="spectral_gap_retention", minimize=False, description="Representation preservation"
    ),
    ObjectiveDefinition(
        name="inference_speedup", minimize=False, description="Runtime improvement"
    ),
]


ALL_OBJECTIVE_DEFINITIONS = (
    EVOMERGE_OBJECTIVES + EXPERT_DISCOVERY_OBJECTIVES + COMPRESSION_OBJECTIVES
)
OBJECTIVE_DEFINITIONS_BY_NAME = {obj.name: obj for obj in ALL_OBJECTIVE_DEFINITIONS}


def _minimize_flags_from_names(names: List[str], fallback: bool = True) -> np.ndarray:
    """Return objective directions for raw, user-facing objective values."""
    return np.array(
        [
            OBJECTIVE_DEFINITIONS_BY_NAME.get(
                name, ObjectiveDefinition(name, minimize=fallback)
            ).minimize
            for name in names
        ],
        dtype=bool,
    )


def _normalized_costs(F: np.ndarray, minimize_flags: np.ndarray) -> np.ndarray:
    """Normalize objectives so lower is always better."""
    F_min = F.min(axis=0)
    F_max = F.max(axis=0)
    denom = np.where(F_max > F_min, F_max - F_min, 1.0)
    F_norm = (F - F_min) / denom
    costs = F_norm.copy()
    costs[:, ~minimize_flags] = 1.0 - costs[:, ~minimize_flags]
    return costs


# =============================================================================
# Base Problem Classes
# =============================================================================


class AgentForgeMOOProblem(Problem if PYMOO_AVAILABLE else ABC):
    """
    Base class for Agent Forge multi-objective optimization problems.

    Subclasses must implement:
        - _evaluate(): Compute objectives for given decision variables
    """

    def __init__(
        self,
        n_var: int,
        objectives: Optional[List[ObjectiveDefinition]] = None,
        xl: Optional[np.ndarray] = None,
        xu: Optional[np.ndarray] = None,
        evaluate_fn: Optional[Callable] = None,
        n_obj: Optional[int] = None,
        n_constr: int = 0,
    ):
        """
        Initialize MOO problem.

        Args:
            n_var: Number of decision variables
            objectives: List of objective definitions
            xl: Lower bounds for variables
            xu: Upper bounds for variables
            evaluate_fn: Optional evaluation function
            n_obj: Backward-compatible objective count when definitions are omitted
            n_constr: Number of inequality constraints for pymoo
        """
        if objectives is None:
            if n_obj is None:
                raise ValueError("objectives or n_obj must be provided")
            objectives = [ObjectiveDefinition(name=f"objective_{i}") for i in range(n_obj)]

        if xl is None or xu is None:
            raise ValueError("xl and xu bounds are required")

        self.objectives = objectives
        self.objective_names = [obj.name for obj in objectives]
        self._evaluate_fn = evaluate_fn

        n_obj = len(objectives)

        if PYMOO_AVAILABLE:
            super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        else:
            self.n_var = n_var
            self.n_obj = n_obj
            self.n_constr = n_constr
            self.xl = xl
            self.xu = xu

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """
        Evaluate objectives for decision variables.

        Args:
            x: Decision variable array (n_solutions x n_var)
            out: Output dictionary for objectives

        Note: Pymoo minimizes all objectives. For maximize objectives,
              we negate them internally.
        """
        if self._evaluate_fn is not None:
            raw_objectives = self._evaluate_fn(x)
        else:
            raw_objectives = self.compute_objectives(x)

        # Convert to minimization (negate maximize objectives)
        F = np.zeros((len(x), len(self.objectives)))
        for i, obj in enumerate(self.objectives):
            if obj.minimize:
                F[:, i] = raw_objectives[:, i]
            else:
                F[:, i] = -raw_objectives[:, i]  # Negate for minimization

        out["F"] = F

    def compute_objectives(self, x: np.ndarray) -> np.ndarray:
        """
        Compute raw objective values. Override in subclasses.

        Args:
            x: Decision variables (n_solutions x n_var)

        Returns:
            Objective values (n_solutions x n_obj)
        """
        raise NotImplementedError("Subclasses must implement compute_objectives()")


class EvoMergeProblem(AgentForgeMOOProblem):
    """
    MOO problem for Phase 2 EvoMerge optimization.

    Decision Variables:
        - merge_ratios: Layer-wise merge ratios [0, 1]
        - technique_weights: Weights for each merge technique

    Objectives:
        - Perplexity, spectral gap, weight norm, invariance, fragility
    """

    def __init__(
        self,
        n_layers: int,
        n_techniques: int = 6,
        model_evaluator: Optional[Callable] = None,
        include_mdl: bool = True,
    ):
        """
        Initialize EvoMerge problem.

        Args:
            n_layers: Number of model layers
            n_techniques: Number of merge techniques (default 6)
            model_evaluator: Function to evaluate merged model
            include_mdl: Include the description_length objective. Disable
                for fixed-architecture sweeps where model bits are constant
                and the objective would waste an NSGA-II dimension.
        """
        self.n_layers = n_layers
        self.n_techniques = n_techniques
        self.model_evaluator = model_evaluator
        self.include_mdl = include_mdl

        # Decision variables: layer ratios + technique weights
        n_var = n_layers + n_techniques

        # Bounds
        xl = np.zeros(n_var)
        xu = np.ones(n_var)

        objectives = (
            EVOMERGE_OBJECTIVES
            if include_mdl
            else [obj for obj in EVOMERGE_OBJECTIVES if obj.name != "description_length"]
        )

        super().__init__(n_var=n_var, objectives=objectives, xl=xl, xu=xu)

    def compute_objectives(self, x: np.ndarray) -> np.ndarray:
        """Compute objectives for merge configurations."""
        n_solutions = len(x)
        F = np.zeros((n_solutions, len(self.objectives)))

        for i, solution in enumerate(x):
            layer_ratios = solution[: self.n_layers]
            technique_weights = solution[self.n_layers :]
            weight_sum = technique_weights.sum()
            if weight_sum > 0:
                technique_weights = technique_weights / weight_sum

            if self.model_evaluator is not None:
                results = self.model_evaluator(layer_ratios, technique_weights)
                F[i, 0] = results.get("perplexity", 0)
                F[i, 1] = results.get("spectral_gap", 0)
                F[i, 2] = results.get("weight_norm", 0)
                F[i, 3] = results.get("invariance_score", 0)
                F[i, 4] = results.get("fragility", 0)
                if self.include_mdl:
                    # Evaluators may supply their own description_length;
                    # otherwise compute model bits from the merge recipe.
                    # The 1e-3 activity threshold applies to NORMALIZED
                    # weights deliberately: that is the recipe actually
                    # applied, and a component contributing <0.1% of the
                    # merge carries no description-worthy information.
                    # (With include_mdl=False, an evaluator-supplied
                    # description_length key is ignored - you opted out.)
                    F[i, 5] = results.get(
                        "description_length",
                        description_length_bits(
                            active_components=int((technique_weights > 1e-3).sum()),
                            total_components=self.n_techniques,
                            param_count=float(results.get("param_count", 1e8)),
                        ),
                    )
            else:
                raise RuntimeError(
                    "EvoMergeProblem requires a model_evaluator; synthetic random objectives are disabled"
                )

        return F


class ExpertDiscoveryProblem(AgentForgeMOOProblem):
    """
    MOO problem for Phase 7 expert discovery.

    Decision Variables:
        - n_experts: Number of experts [3, 10]
        - sparsity: Router sparsity [0, 1]
        - router_temp: Router temperature [0.1, 10]
        - svf_rank: SVF rank [4, 64]
        - z_dim: Z-vector dimension [8, 128]
    """

    def __init__(self, expert_evaluator: Optional[Callable] = None, include_mdl: bool = True):
        """
        Initialize expert discovery problem.

        Args:
            expert_evaluator: Function to evaluate expert configuration
            include_mdl: Include the description_length objective. Disable
                for fixed-architecture sweeps where model bits are constant.
        """
        self.expert_evaluator = expert_evaluator
        self.include_mdl = include_mdl

        # Decision variables
        n_var = 5

        # Bounds [n_experts, sparsity, router_temp, svf_rank, z_dim]
        xl = np.array([3, 0.0, 0.1, 4, 8])
        xu = np.array([10, 1.0, 10.0, 64, 128])
        self._max_experts = int(xu[0])

        objectives = (
            EXPERT_DISCOVERY_OBJECTIVES
            if include_mdl
            else [obj for obj in EXPERT_DISCOVERY_OBJECTIVES if obj.name != "description_length"]
        )

        super().__init__(n_var=n_var, objectives=objectives, xl=xl, xu=xu)

    def compute_objectives(self, x: np.ndarray) -> np.ndarray:
        """Compute objectives for expert configurations."""
        n_solutions = len(x)
        F = np.zeros((n_solutions, len(self.objectives)))

        for i, solution in enumerate(x):
            n_experts = int(solution[0])
            sparsity = solution[1]
            router_temp = solution[2]
            svf_rank = int(solution[3])
            z_dim = int(solution[4])

            if self.expert_evaluator is not None:
                results = self.expert_evaluator(n_experts, sparsity, router_temp, svf_rank, z_dim)
                F[i, 0] = results.get("task_loss", 0)
                F[i, 1] = results.get("expert_diversity", 0)
                F[i, 2] = results.get("routing_entropy", 0)
                F[i, 3] = results.get("compute_cost", 0)
                F[i, 4] = results.get("robustness", 0)
                if self.include_mdl:
                    F[i, 5] = results.get(
                        "description_length",
                        description_length_bits(
                            active_components=n_experts,
                            total_components=self._max_experts,
                            param_count=float(max(1, n_experts * svf_rank * z_dim)),
                        ),
                    )
            else:
                raise RuntimeError(
                    "ExpertDiscoveryProblem requires an expert_evaluator; synthetic random objectives are disabled"
                )

        return F


# =============================================================================
# MOO Runner and Utilities
# =============================================================================


class MOORunner:
    """
    Runner for multi-objective optimization.

    Wraps pymoo for easy use with Agent Forge problems.
    """

    def __init__(self, config: Optional[MOOConfig] = None):
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo is required for MOO. Install with: pip install pymoo")

        self.config = config or MOOConfig()
        self.history = []

    def optimize(self, problem: AgentForgeMOOProblem) -> dict:
        """
        Run multi-objective optimization.

        Args:
            problem: MOO problem to optimize

        Returns:
            Dictionary with Pareto front and metadata
        """
        # Configure algorithm
        algorithm = NSGA2(
            pop_size=self.config.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.config.crossover_prob, eta=self.config.crossover_eta),
            mutation=PM(eta=self.config.mutation_eta),
            eliminate_duplicates=True,
        )

        # Run optimization
        result = minimize(
            problem,
            algorithm,
            ("n_gen", self.config.n_generations),
            seed=self.config.seed,
            verbose=self.config.verbose,
            save_history=self.config.save_history,
        )

        # Extract Pareto front
        pareto_F = result.F.copy()

        # Convert back from minimization (un-negate maximize objectives)
        for i, obj in enumerate(problem.objectives):
            if not obj.minimize:
                pareto_F[:, i] = -pareto_F[:, i]

        return {
            "pareto_front": pareto_F,
            "pareto_solutions": result.X,
            "objective_names": problem.objective_names,
            "objective_minimize": [obj.minimize for obj in problem.objectives],
            "n_solutions": len(result.X),
            "n_generations": self.config.n_generations,
            "hypervolume": self._compute_hypervolume(result.F) if len(result.F) > 0 else 0,
        }

    def _compute_hypervolume(self, F: np.ndarray) -> float:
        """Compute hypervolume indicator for Pareto front."""
        try:
            # Reference point (worst case for each objective)
            ref_point = F.max(axis=0) * 1.1
            hv = HV(ref_point=ref_point)
            return hv(F)
        except Exception:
            return 0.0


def select_from_pareto(
    pareto_result: dict, preference: Dict[str, float], method: str = "weighted_sum"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a single solution from Pareto front based on preferences.

    Args:
        pareto_result: Result from MOORunner.optimize()
        preference: Dict mapping objective name to weight
        method: Selection method ("weighted_sum", "closest_to_ideal")

    Returns:
        Tuple of (selected_solution, selected_objectives)
    """
    F = pareto_result["pareto_front"]
    X = pareto_result["pareto_solutions"]
    names = pareto_result["objective_names"]

    minimize_flags = np.array(
        pareto_result.get("objective_minimize", _minimize_flags_from_names(names)),
        dtype=bool,
    )

    if method == "weighted_sum":
        # Normalize objectives to a cost space where lower is always better.
        costs = _normalized_costs(F, minimize_flags)

        weights = np.array([preference.get(name, 1.0) for name in names], dtype=float)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        scores = costs @ weights

        best_idx = int(np.argmin(scores))

    elif method == "closest_to_ideal":
        costs = _normalized_costs(F, minimize_flags)
        distances = np.linalg.norm(costs, axis=1)
        best_idx = int(np.argmin(distances))

    else:
        raise ValueError(f"Unknown selection method: {method}")

    # Governance: which point the frontier collapsed to, and under which
    # weights, is a regime choice that must be auditable after the fact.
    logger.info(
        "pareto-select: method=%s idx=%d front_size=%d preference=%s",
        method,
        best_idx,
        len(F),
        preference,
    )

    return X[best_idx], F[best_idx]


def analyze_pareto_front(pareto_result: dict) -> dict:
    """
    Analyze Pareto front for insights.

    Args:
        pareto_result: Result from MOORunner.optimize()

    Returns:
        Analysis dictionary
    """
    F = pareto_result["pareto_front"]
    names = pareto_result["objective_names"]
    minimize_flags = np.array(
        pareto_result.get("objective_minimize", _minimize_flags_from_names(names)),
        dtype=bool,
    )

    analysis = {
        "n_solutions": len(F),
        "objective_ranges": {},
        "correlations": {},
        "extreme_solutions": {},
    }

    # Objective ranges
    for i, name in enumerate(names):
        analysis["objective_ranges"][name] = {
            "min": float(F[:, i].min()),
            "max": float(F[:, i].max()),
            "mean": float(F[:, i].mean()),
            "std": float(F[:, i].std()),
        }

    # Pairwise correlations
    if len(F) > 2:
        corr_matrix = np.corrcoef(F.T)
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i < j:
                    key = f"{name_i}_vs_{name_j}"
                    analysis["correlations"][key] = float(corr_matrix[i, j])

    # Extreme solutions (best for each objective)
    for i, name in enumerate(names):
        best_idx = F[:, i].argmin() if minimize_flags[i] else F[:, i].argmax()
        analysis["extreme_solutions"][name] = {
            "index": int(best_idx),
            "objectives": {n: float(F[best_idx, j]) for j, n in enumerate(names)},
        }

    return analysis


# =============================================================================
# GlobalMOO API Adapter (Optional Cloud Integration)
# =============================================================================


class GlobalMOOAdapter:
    """
    Adapter for GlobalMOO cloud API.

    Provides cloud-based hyperparameter optimization when local
    compute is insufficient.
    """

    def __init__(self, api_key: Optional[str] = None, api_url: str = "https://api.globalmoo.com"):
        self.api_key = api_key
        self.api_url = api_url
        self._session = None

    def is_available(self) -> bool:
        """Check if GlobalMOO API is available."""
        return self.api_key is not None

    def submit_problem(self, problem: AgentForgeMOOProblem, config: MOOConfig) -> str:
        """
        Submit problem to GlobalMOO cloud.

        Args:
            problem: MOO problem
            config: Optimization config

        Returns:
            Job ID for tracking
        """
        if not self.is_available():
            raise RuntimeError("GlobalMOO API key not configured")

        # Placeholder for actual API implementation
        raise NotImplementedError("GlobalMOO API integration pending")

    def get_results(self, job_id: str) -> dict:
        """Get results from GlobalMOO cloud job."""
        raise NotImplementedError("GlobalMOO API integration pending")


# =============================================================================
# Quick Test
# =============================================================================


def run_demo():
    """Run a demo optimization."""
    if not PYMOO_AVAILABLE:
        print("pymoo not available. Install with: pip install pymoo")
        return

    print("\n" + "=" * 60)
    print("MOO BRIDGE DEMO")
    print("=" * 60)

    # Create test problem
    print("\n1. Creating Expert Discovery Problem...")

    def toy_expert_evaluator(n_experts, sparsity, router_temp, svf_rank, z_dim):
        return {
            "task_loss": 1.0 / n_experts + 0.01 * router_temp,
            "expert_diversity": min(1.0, n_experts / 10.0) * (1.0 - 0.5 * sparsity),
            "routing_entropy": min(1.0, router_temp / 10.0),
            "compute_cost": n_experts * svf_rank * z_dim,
            "robustness": 1.0 / (1.0 + sparsity),
        }

    problem = ExpertDiscoveryProblem(expert_evaluator=toy_expert_evaluator)
    print(f"   Variables: {problem.n_var}")
    print(f"   Objectives: {[obj.name for obj in problem.objectives]}")

    # Run optimization
    print("\n2. Running NSGA-II (20 generations, 30 population)...")
    config = MOOConfig(n_generations=20, population_size=30, verbose=False)
    runner = MOORunner(config)
    result = runner.optimize(problem)

    print(f"   Found {result['n_solutions']} Pareto-optimal solutions")
    print(f"   Hypervolume: {result['hypervolume']:.4f}")

    # Analyze
    print("\n3. Analyzing Pareto Front...")
    analysis = analyze_pareto_front(result)
    for name, stats in analysis["objective_ranges"].items():
        print(f"   {name}: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # Select best
    print("\n4. Selecting solution with preference...")
    preference = {"task_loss": 2.0, "expert_diversity": 1.5}
    best_x, best_f = select_from_pareto(result, preference)
    print(f"   Best solution: {best_x}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()

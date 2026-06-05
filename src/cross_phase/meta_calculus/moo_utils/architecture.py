"""
Architecture Search MOO Utilities

Multi-objective architecture and configuration search.

Phase Applications:
    - Phase 1 (Cognate): Neural Architecture Search for TRM x Titans-MAG
    - Phase 4 (BitNet): Mixed-precision layer assignment
    - Phase 7 (Experts): Expert count and configuration optimization
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import numpy as np

# Layer 1 import only
from ..moo_bridge import (
    AgentForgeMOOProblem,
    MOORunner,
    MOOConfig,
    ObjectiveDefinition,
)


@dataclass
class ArchitectureSearchConfig:
    """Configuration for architecture search."""

    # MOO settings
    n_generations: int = 50
    population_size: int = 100

    # Search space defaults
    min_layers: int = 4
    max_layers: int = 24
    min_hidden: int = 128
    max_hidden: int = 1024
    min_heads: int = 2
    max_heads: int = 16

    # Constraints
    max_params: Optional[int] = 25_000_000  # 25M default
    max_latency_ms: Optional[float] = None
    max_memory_gb: Optional[float] = None


# =============================================================================
# ARCHITECTURE SEARCH (Phase 1)
# =============================================================================

class ArchitectureSearchProblem(AgentForgeMOOProblem):
    """
    Multi-objective architecture search for neural networks.

    Objectives:
        - param_count: Minimize parameter count
        - latency: Minimize inference latency
        - perplexity: Minimize perplexity (maximize quality)
        - memory: Minimize memory footprint

    Decision Variables:
        - n_layers: Number of transformer layers
        - hidden_dim: Hidden dimension
        - n_heads: Number of attention heads
        - ff_ratio: Feed-forward ratio
    """

    OBJECTIVES = [
        ObjectiveDefinition("param_count", minimize=True, weight=1.0),
        ObjectiveDefinition("latency", minimize=True, weight=1.0),
        ObjectiveDefinition("perplexity", minimize=True, weight=2.0),
        ObjectiveDefinition("memory", minimize=True, weight=0.5),
    ]

    def __init__(
        self,
        model_factory: Callable,
        evaluator: Callable,
        config: Optional[ArchitectureSearchConfig] = None,
    ):
        """
        Initialize architecture search problem.

        Args:
            model_factory: Function(n_layers, hidden_dim, n_heads, ff_ratio) -> model
            evaluator: Function(model) -> {"param_count", "latency", "perplexity", "memory"}
            config: Search configuration
        """
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.config = config or ArchitectureSearchConfig()

        # Define search space bounds
        # [n_layers, hidden_dim, n_heads, ff_ratio]
        xl = np.array([
            self.config.min_layers,
            self.config.min_hidden,
            self.config.min_heads,
            1.0,  # ff_ratio min
        ])
        xr = np.array([
            self.config.max_layers,
            self.config.max_hidden,
            self.config.max_heads,
            4.0,  # ff_ratio max
        ])

        super().__init__(
            n_var=4,
            objectives=self.OBJECTIVES,
            n_constr=1 if self.config.max_params else 0,
            xl=xl,
            xu=xr,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate a batch of architectures."""
        f = np.zeros((len(x), self.n_obj))
        g = np.zeros((len(x), self.n_constr)) if self.n_constr > 0 else None

        for i, xi in enumerate(x):
            n_layers = int(xi[0])
            hidden_dim = int(xi[1])
            n_heads = int(xi[2])
            ff_ratio = xi[3]

            # Build and evaluate model
            try:
                model = self.model_factory(n_layers, hidden_dim, n_heads, ff_ratio)
                metrics = self.evaluator(model)

                f[i, 0] = metrics.get("param_count", 1e9)
                f[i, 1] = metrics.get("latency", 1e6)
                f[i, 2] = metrics.get("perplexity", 1e6)
                f[i, 3] = metrics.get("memory", 1e6)

                # Constraint: param_count <= max_params
                if self.n_constr > 0 and self.config.max_params:
                    g[i, 0] = metrics.get("param_count", 0) - self.config.max_params

            except Exception as e:
                # Invalid architecture - assign bad fitness
                f[i, :] = 1e9
                if g is not None:
                    g[i, 0] = 1e9

        out["F"] = f
        if g is not None:
            out["G"] = g


def search_architecture(
    model_factory: Callable,
    evaluator: Callable,
    config: Optional[ArchitectureSearchConfig] = None,
) -> Dict[str, Any]:
    """
    Search for optimal architecture using NSGA-II.

    Args:
        model_factory: Function to create model from architecture params
        evaluator: Function to evaluate model performance
        config: Search configuration

    Returns:
        Dictionary with Pareto front and best solutions

    Example:
        >>> def factory(n_layers, hidden, heads, ff):
        ...     return TransformerModel(n_layers, hidden, heads, ff)
        >>> def evaluate(model):
        ...     return {"param_count": count_params(model), ...}
        >>> result = search_architecture(factory, evaluate)
        >>> best_arch = result["best_balanced"]
    """
    config = config or ArchitectureSearchConfig()

    problem = ArchitectureSearchProblem(model_factory, evaluator, config)

    moo_config = MOOConfig(
        n_generations=config.n_generations,
        population_size=config.population_size,
    )

    runner = MOORunner(moo_config)
    result = runner.optimize(problem)

    return result


# =============================================================================
# MIXED PRECISION SEARCH (Phase 4)
# =============================================================================

class MixedPrecisionProblem(AgentForgeMOOProblem):
    """
    Multi-objective mixed-precision assignment search.

    Objectives:
        - model_size: Minimize compressed model size
        - accuracy_loss: Minimize accuracy degradation
        - inference_speed: Maximize inference speed

    Decision Variables:
        - Per-layer precision: 0=FP16, 1=INT8, 2=INT4, 3=TERNARY
    """

    OBJECTIVES = [
        ObjectiveDefinition("model_size", minimize=True, weight=1.0),
        ObjectiveDefinition("accuracy_loss", minimize=True, weight=2.0),
        ObjectiveDefinition("inference_speed", minimize=True, weight=1.0),  # Minimize latency
    ]

    def __init__(
        self,
        n_layers: int,
        quantize_fn: Callable,
        evaluate_fn: Callable,
        config: Optional[ArchitectureSearchConfig] = None,
    ):
        """
        Initialize mixed-precision search.

        Args:
            n_layers: Number of layers to assign precision
            quantize_fn: Function(model, precision_assignment) -> quantized_model
            evaluate_fn: Function(model) -> {"model_size", "accuracy_loss", "inference_speed"}
            config: Search configuration
        """
        self.n_layers = n_layers
        self.quantize_fn = quantize_fn
        self.evaluate_fn = evaluate_fn
        self.config = config or ArchitectureSearchConfig()

        # Precision choices: 0-3 for each layer
        xl = np.zeros(n_layers)
        xu = np.ones(n_layers) * 3

        super().__init__(
            n_var=n_layers,
            objectives=self.OBJECTIVES,
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate precision assignments."""
        f = np.zeros((len(x), self.n_obj))

        for i, xi in enumerate(x):
            precision_assignment = [int(p) for p in xi]

            try:
                quantized_model = self.quantize_fn(precision_assignment)
                metrics = self.evaluate_fn(quantized_model)

                f[i, 0] = metrics.get("model_size", 1e9)
                f[i, 1] = metrics.get("accuracy_loss", 1e6)
                f[i, 2] = metrics.get("inference_speed", 1e6)

            except Exception:
                f[i, :] = 1e9

        out["F"] = f


def search_precision_assignment(
    n_layers: int,
    quantize_fn: Callable,
    evaluate_fn: Callable,
    config: Optional[ArchitectureSearchConfig] = None,
) -> Dict[str, Any]:
    """
    Search for optimal mixed-precision assignment.

    Args:
        n_layers: Number of layers
        quantize_fn: Quantization function
        evaluate_fn: Evaluation function
        config: Search configuration

    Returns:
        Dictionary with Pareto front and best assignments
    """
    config = config or ArchitectureSearchConfig()

    problem = MixedPrecisionProblem(n_layers, quantize_fn, evaluate_fn, config)

    moo_config = MOOConfig(
        n_generations=config.n_generations,
        population_size=config.population_size,
    )

    runner = MOORunner(moo_config)
    return runner.optimize(problem)


# =============================================================================
# EXPERT COUNT SEARCH (Phase 7)
# =============================================================================

class ExpertCountProblem(AgentForgeMOOProblem):
    """
    Multi-objective expert count optimization.

    Objectives:
        - capacity: Maximize model capacity
        - efficiency: Minimize compute per forward pass
        - diversity: Maximize expert diversity

    Decision Variables:
        - n_experts: Number of experts (4-64)
        - top_k: Number of experts per token (1-8)
        - capacity_factor: Capacity multiplier (1.0-2.0)
    """

    OBJECTIVES = [
        ObjectiveDefinition("capacity", minimize=False, weight=1.0),  # Maximize
        ObjectiveDefinition("efficiency", minimize=True, weight=1.0),
        ObjectiveDefinition("diversity", minimize=False, weight=1.5),  # Maximize
    ]

    def __init__(
        self,
        evaluator: Callable,
        config: Optional[ArchitectureSearchConfig] = None,
    ):
        """
        Initialize expert count search.

        Args:
            evaluator: Function(n_experts, top_k, capacity_factor) -> metrics
            config: Search configuration
        """
        self.evaluator = evaluator
        self.config = config or ArchitectureSearchConfig()

        # [n_experts, top_k, capacity_factor]
        xl = np.array([4, 1, 1.0])
        xu = np.array([64, 8, 2.0])

        super().__init__(
            n_var=3,
            objectives=self.OBJECTIVES,
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate expert configurations."""
        f = np.zeros((len(x), self.n_obj))

        for i, xi in enumerate(x):
            n_experts = int(xi[0])
            top_k = int(xi[1])
            capacity_factor = xi[2]

            try:
                metrics = self.evaluator(n_experts, top_k, capacity_factor)

                # Note: capacity and diversity are maximized (negated)
                f[i, 0] = -metrics.get("capacity", 0)
                f[i, 1] = metrics.get("efficiency", 1e6)
                f[i, 2] = -metrics.get("diversity", 0)

            except Exception:
                f[i, :] = 1e9

        out["F"] = f


def search_expert_count(
    evaluator: Callable,
    config: Optional[ArchitectureSearchConfig] = None,
) -> Dict[str, Any]:
    """
    Search for optimal expert configuration.

    Args:
        evaluator: Function to evaluate expert configs
        config: Search configuration

    Returns:
        Dictionary with Pareto front and best configurations
    """
    config = config or ArchitectureSearchConfig()

    problem = ExpertCountProblem(evaluator, config)

    moo_config = MOOConfig(
        n_generations=config.n_generations,
        population_size=config.population_size,
    )

    runner = MOORunner(moo_config)
    return runner.optimize(problem)

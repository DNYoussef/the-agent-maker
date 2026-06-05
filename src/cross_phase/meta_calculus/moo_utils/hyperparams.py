"""
Hyperparameter Optimization MOO Utilities

Multi-objective hyperparameter search for training configurations.

Phase Applications:
    - Phase 1: Training hyperparameters (LR, warmup, weight decay)
    - Phase 3: Thought generation hyperparameters
    - Phase 5: Curriculum schedule optimization
    - Phase 6: A/B cycle interleaving optimization
"""

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import numpy as np

# Layer 1 import only
from ..moo_bridge import (
    AgentForgeMOOProblem,
    MOORunner,
    MOOConfig,
    ObjectiveDefinition,
)


@dataclass
class HyperparamSearchConfig:
    """Configuration for hyperparameter search."""

    # MOO settings
    n_generations: int = 30
    population_size: int = 50

    # Training trials
    n_trials_per_config: int = 1
    max_epochs_per_trial: int = 10


# =============================================================================
# TRAINING HYPERPARAMS (Phase 1)
# =============================================================================

class TrainingHyperparamProblem(AgentForgeMOOProblem):
    """
    Multi-objective training hyperparameter optimization.

    Objectives:
        - final_loss: Minimize final training loss
        - training_time: Minimize training time
        - generalization: Minimize generalization gap

    Decision Variables:
        - learning_rate: Log-scale (1e-5 to 1e-2)
        - warmup_ratio: 0 to 0.2
        - weight_decay: 0 to 0.1
        - batch_size_factor: 0.5 to 2.0
    """

    OBJECTIVES = [
        ObjectiveDefinition("final_loss", minimize=True, weight=2.0),
        ObjectiveDefinition("training_time", minimize=True, weight=1.0),
        ObjectiveDefinition("generalization", minimize=True, weight=1.5),
    ]

    def __init__(
        self,
        train_fn: Callable,
        config: Optional[HyperparamSearchConfig] = None,
    ):
        """
        Initialize training hyperparam search.

        Args:
            train_fn: Function(lr, warmup, wd, batch_factor) -> metrics dict
            config: Search configuration
        """
        self.train_fn = train_fn
        self.config = config or HyperparamSearchConfig()

        # [log_lr, warmup_ratio, weight_decay, batch_factor]
        # Note: log_lr is in log10 scale
        xl = np.array([-5, 0.0, 0.0, 0.5])
        xu = np.array([-2, 0.2, 0.1, 2.0])

        super().__init__(
            n_var=4,
            objectives=self.OBJECTIVES,
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate training configurations."""
        f = np.zeros((len(x), self.n_obj))

        for i, xi in enumerate(x):
            lr = 10 ** xi[0]  # Convert from log scale
            warmup_ratio = xi[1]
            weight_decay = xi[2]
            batch_factor = xi[3]

            try:
                metrics = self.train_fn(lr, warmup_ratio, weight_decay, batch_factor)

                f[i, 0] = metrics.get("final_loss", 1e6)
                f[i, 1] = metrics.get("training_time", 1e6)
                f[i, 2] = metrics.get("generalization", 1e6)

            except Exception:
                f[i, :] = 1e9

        out["F"] = f


def optimize_training_hyperparams(
    train_fn: Callable,
    config: Optional[HyperparamSearchConfig] = None,
) -> Dict[str, Any]:
    """
    Optimize training hyperparameters using MOO.

    Args:
        train_fn: Training function that returns metrics
        config: Search configuration

    Returns:
        Dictionary with Pareto front and best configurations
    """
    config = config or HyperparamSearchConfig()

    problem = TrainingHyperparamProblem(train_fn, config)

    moo_config = MOOConfig(
        n_generations=config.n_generations,
        population_size=config.population_size,
    )

    runner = MOORunner(moo_config)
    return runner.optimize(problem)


# =============================================================================
# THOUGHT HYPERPARAMS (Phase 3)
# =============================================================================

class ThoughtHyperparamProblem(AgentForgeMOOProblem):
    """
    Multi-objective thought generation hyperparameter optimization.

    Objectives:
        - coherence: Maximize thought coherence
        - diversity: Maximize thought diversity
        - compute_cost: Minimize compute cost
        - quality_improvement: Maximize task improvement from thoughts

    Decision Variables:
        - num_thoughts: 1-8
        - thought_length: 4-64 tokens
        - temperature: 0.1-2.0
        - top_p: 0.5-1.0
    """

    OBJECTIVES = [
        ObjectiveDefinition("coherence", minimize=False, weight=1.0),
        ObjectiveDefinition("diversity", minimize=False, weight=1.0),
        ObjectiveDefinition("compute_cost", minimize=True, weight=0.5),
        ObjectiveDefinition("quality_improvement", minimize=False, weight=2.0),
    ]

    def __init__(
        self,
        thought_evaluator: Callable,
        config: Optional[HyperparamSearchConfig] = None,
    ):
        """
        Initialize thought hyperparam search.

        Args:
            thought_evaluator: Function(n_thoughts, length, temp, top_p) -> metrics
            config: Search configuration
        """
        self.thought_evaluator = thought_evaluator
        self.config = config or HyperparamSearchConfig()

        # [num_thoughts, thought_length, temperature, top_p]
        xl = np.array([1, 4, 0.1, 0.5])
        xu = np.array([8, 64, 2.0, 1.0])

        super().__init__(
            n_var=4,
            objectives=self.OBJECTIVES,
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate thought configurations."""
        f = np.zeros((len(x), self.n_obj))

        for i, xi in enumerate(x):
            n_thoughts = int(xi[0])
            thought_length = int(xi[1])
            temperature = xi[2]
            top_p = xi[3]

            try:
                metrics = self.thought_evaluator(
                    n_thoughts, thought_length, temperature, top_p
                )

                # Note: coherence, diversity, quality are maximized (negated)
                f[i, 0] = -metrics.get("coherence", 0)
                f[i, 1] = -metrics.get("diversity", 0)
                f[i, 2] = metrics.get("compute_cost", 1e6)
                f[i, 3] = -metrics.get("quality_improvement", 0)

            except Exception:
                f[i, :] = 1e9

        out["F"] = f


def optimize_thought_hyperparams(
    thought_evaluator: Callable,
    config: Optional[HyperparamSearchConfig] = None,
) -> Dict[str, Any]:
    """
    Optimize thought generation hyperparameters.

    Args:
        thought_evaluator: Function to evaluate thought configurations
        config: Search configuration

    Returns:
        Dictionary with Pareto front and best configurations
    """
    config = config or HyperparamSearchConfig()

    problem = ThoughtHyperparamProblem(thought_evaluator, config)

    moo_config = MOOConfig(
        n_generations=config.n_generations,
        population_size=config.population_size,
    )

    runner = MOORunner(moo_config)
    return runner.optimize(problem)


# =============================================================================
# CURRICULUM SCHEDULE (Phase 5)
# =============================================================================

class CurriculumScheduleProblem(AgentForgeMOOProblem):
    """
    Multi-objective curriculum schedule optimization.

    Objectives:
        - learning_efficiency: Maximize learning rate per step
        - final_performance: Maximize final performance
        - stability: Minimize training instability

    Decision Variables:
        - stage_durations: Duration ratios for each stage (normalized)
        - difficulty_curve: Difficulty progression parameters
    """

    OBJECTIVES = [
        ObjectiveDefinition("learning_efficiency", minimize=False, weight=1.0),
        ObjectiveDefinition("final_performance", minimize=False, weight=2.0),
        ObjectiveDefinition("stability", minimize=True, weight=1.0),
    ]

    def __init__(
        self,
        n_stages: int,
        curriculum_evaluator: Callable,
        config: Optional[HyperparamSearchConfig] = None,
    ):
        """
        Initialize curriculum schedule search.

        Args:
            n_stages: Number of curriculum stages
            curriculum_evaluator: Function(stage_ratios, difficulty_params) -> metrics
            config: Search configuration
        """
        self.n_stages = n_stages
        self.curriculum_evaluator = curriculum_evaluator
        self.config = config or HyperparamSearchConfig()

        # [stage_ratio_1, ..., stage_ratio_n, difficulty_start, difficulty_end]
        n_vars = n_stages + 2
        xl = np.zeros(n_vars)
        xl[-2:] = [0.5, 1.0]  # difficulty range

        xu = np.ones(n_vars)
        xu[-2:] = [1.5, 3.0]  # difficulty range

        super().__init__(
            n_var=n_vars,
            objectives=self.OBJECTIVES,
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate curriculum schedules."""
        f = np.zeros((len(x), self.n_obj))

        for i, xi in enumerate(x):
            # Normalize stage ratios to sum to 1
            stage_ratios = xi[:self.n_stages]
            stage_ratios = stage_ratios / (stage_ratios.sum() + 1e-10)

            difficulty_start = xi[-2]
            difficulty_end = xi[-1]

            try:
                metrics = self.curriculum_evaluator(
                    stage_ratios.tolist(),
                    {"start": difficulty_start, "end": difficulty_end}
                )

                # Note: efficiency, performance maximized (negated)
                f[i, 0] = -metrics.get("learning_efficiency", 0)
                f[i, 1] = -metrics.get("final_performance", 0)
                f[i, 2] = metrics.get("stability", 1e6)

            except Exception:
                f[i, :] = 1e9

        out["F"] = f


def optimize_curriculum_schedule(
    n_stages: int,
    curriculum_evaluator: Callable,
    config: Optional[HyperparamSearchConfig] = None,
) -> Dict[str, Any]:
    """
    Optimize curriculum schedule using MOO.

    Args:
        n_stages: Number of curriculum stages
        curriculum_evaluator: Function to evaluate schedules
        config: Search configuration

    Returns:
        Dictionary with Pareto front and best schedules
    """
    config = config or HyperparamSearchConfig()

    problem = CurriculumScheduleProblem(n_stages, curriculum_evaluator, config)

    moo_config = MOOConfig(
        n_generations=config.n_generations,
        population_size=config.population_size,
    )

    runner = MOORunner(moo_config)
    return runner.optimize(problem)


# =============================================================================
# A/B CYCLE INTERLEAVING (Phase 6)
# =============================================================================

class ABCycleInterleavingProblem(AgentForgeMOOProblem):
    """
    Multi-objective A/B cycle interleaving optimization.

    Objectives:
        - tool_performance: Maximize tool-use performance
        - persona_consistency: Maximize persona consistency
        - interference: Minimize interference between cycles

    Decision Variables:
        - a_cycle_ratio: Proportion of A-cycle steps
        - interleave_frequency: How often to switch cycles
        - half_baking_ratio: Strength of half-baking
    """

    OBJECTIVES = [
        ObjectiveDefinition("tool_performance", minimize=False, weight=1.0),
        ObjectiveDefinition("persona_consistency", minimize=False, weight=1.0),
        ObjectiveDefinition("interference", minimize=True, weight=1.5),
    ]

    def __init__(
        self,
        baking_evaluator: Callable,
        config: Optional[HyperparamSearchConfig] = None,
    ):
        """
        Initialize A/B cycle search.

        Args:
            baking_evaluator: Function(a_ratio, interleave_freq, half_bake) -> metrics
            config: Search configuration
        """
        self.baking_evaluator = baking_evaluator
        self.config = config or HyperparamSearchConfig()

        # [a_cycle_ratio, interleave_frequency, half_baking_ratio]
        xl = np.array([0.2, 1, 0.3])
        xu = np.array([0.8, 10, 0.7])

        super().__init__(
            n_var=3,
            objectives=self.OBJECTIVES,
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate A/B cycle configurations."""
        f = np.zeros((len(x), self.n_obj))

        for i, xi in enumerate(x):
            a_ratio = xi[0]
            interleave_freq = int(xi[1])
            half_bake_ratio = xi[2]

            try:
                metrics = self.baking_evaluator(a_ratio, interleave_freq, half_bake_ratio)

                # Note: performance metrics maximized (negated)
                f[i, 0] = -metrics.get("tool_performance", 0)
                f[i, 1] = -metrics.get("persona_consistency", 0)
                f[i, 2] = metrics.get("interference", 1e6)

            except Exception:
                f[i, :] = 1e9

        out["F"] = f


def optimize_ab_interleaving(
    baking_evaluator: Callable,
    config: Optional[HyperparamSearchConfig] = None,
) -> Dict[str, Any]:
    """
    Optimize A/B cycle interleaving using MOO.

    Args:
        baking_evaluator: Function to evaluate interleaving strategies
        config: Search configuration

    Returns:
        Dictionary with Pareto front and best strategies
    """
    config = config or HyperparamSearchConfig()

    problem = ABCycleInterleavingProblem(baking_evaluator, config)

    moo_config = MOOConfig(
        n_generations=config.n_generations,
        population_size=config.population_size,
    )

    runner = MOORunner(moo_config)
    return runner.optimize(problem)

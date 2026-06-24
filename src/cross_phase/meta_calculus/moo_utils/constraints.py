"""
Constraint Handling Utilities (Layer 2)

Utilities for creating and combining constraints for MOO problems.
These are MOO-only utilities with no meta-calculus dependencies.

Functions:
    make_param_constraint: Constraint on parameter count
    make_latency_constraint: Constraint on inference latency
    make_memory_constraint: Constraint on memory usage
    make_accuracy_constraint: Constraint on accuracy threshold
    combine_constraints: Combine multiple constraints
    apply_constraints: Apply constraints to solutions

Design:
    - Returns constraint functions compatible with pymoo
    - No dependencies on k_formula or bigeometric
    - Uses standard inequality constraint format: g(x) <= 0
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Type alias for constraint functions
# Constraint function signature: (x: np.ndarray) -> float
# Returns <= 0 if satisfied, > 0 if violated
ConstraintFunc = Callable[[np.ndarray], float]


@dataclass
class ConstraintConfig:
    """Configuration for constraint creation."""

    # Tolerance for constraint satisfaction
    tolerance: float = 1e-6

    # Scaling for constraint normalization
    normalize: bool = True

    # Penalty weight for soft constraints
    penalty_weight: float = 1.0

    # Whether constraints are "hard" (must satisfy) or "soft" (penalty)
    hard: bool = True


@dataclass
class Constraint:
    """A single constraint with metadata."""

    name: str
    func: ConstraintFunc
    bound: float
    constraint_type: str  # "upper", "lower", "equality"
    config: ConstraintConfig = field(default_factory=ConstraintConfig)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate constraint. Returns <= 0 if satisfied."""
        value = self.func(x)

        if self.constraint_type == "upper":
            # value <= bound  =>  value - bound <= 0
            return value - self.bound
        elif self.constraint_type == "lower":
            # value >= bound  =>  bound - value <= 0
            return self.bound - value
        elif self.constraint_type == "equality":
            # value == bound  =>  |value - bound| - tolerance <= 0
            return abs(value - self.bound) - self.config.tolerance
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")

    def is_satisfied(self, x: np.ndarray) -> bool:
        """Check if constraint is satisfied."""
        return self.evaluate(x) <= self.config.tolerance

    def violation(self, x: np.ndarray) -> float:
        """Get violation amount (0 if satisfied)."""
        return max(0.0, self.evaluate(x))


def make_param_constraint(
    max_params: int,
    param_counter: Optional[Callable[[np.ndarray], int]] = None,
    config: Optional[ConstraintConfig] = None,
) -> Constraint:
    """
    Create constraint on total parameter count.

    Args:
        max_params: Maximum allowed parameters
        param_counter: Function to count params from decision vector.
                      If None, assumes x[0] = param count.
        config: Constraint configuration

    Returns:
        Constraint object

    Example:
        >>> # Limit to 25M parameters
        >>> constraint = make_param_constraint(25_000_000)
        >>> problem.add_constraint(constraint)
    """
    config = config or ConstraintConfig()

    if param_counter is None:
        # Default: first variable is param count
        def param_counter(x: np.ndarray) -> float:
            return float(x[0]) if len(x) > 0 else 0.0

    return Constraint(
        name="param_count",
        func=param_counter,
        bound=float(max_params),
        constraint_type="upper",
        config=config,
    )


def make_latency_constraint(
    max_latency_ms: float,
    latency_estimator: Optional[Callable[[np.ndarray], float]] = None,
    config: Optional[ConstraintConfig] = None,
) -> Constraint:
    """
    Create constraint on inference latency.

    Args:
        max_latency_ms: Maximum latency in milliseconds
        latency_estimator: Function to estimate latency from decision vector.
                          If None, assumes x[1] = latency.
        config: Constraint configuration

    Returns:
        Constraint object

    Example:
        >>> # Limit to 100ms inference time
        >>> constraint = make_latency_constraint(100.0)
    """
    config = config or ConstraintConfig()

    if latency_estimator is None:

        def latency_estimator(x: np.ndarray) -> float:
            return float(x[1]) if len(x) > 1 else 0.0

    return Constraint(
        name="latency",
        func=latency_estimator,
        bound=max_latency_ms,
        constraint_type="upper",
        config=config,
    )


def make_memory_constraint(
    max_memory_mb: float,
    memory_estimator: Optional[Callable[[np.ndarray], float]] = None,
    config: Optional[ConstraintConfig] = None,
) -> Constraint:
    """
    Create constraint on memory usage.

    Args:
        max_memory_mb: Maximum memory in megabytes
        memory_estimator: Function to estimate memory from decision vector.
                         If None, assumes x[2] = memory.
        config: Constraint configuration

    Returns:
        Constraint object

    Example:
        >>> # Limit to 6GB VRAM
        >>> constraint = make_memory_constraint(6144.0)
    """
    config = config or ConstraintConfig()

    if memory_estimator is None:

        def memory_estimator(x: np.ndarray) -> float:
            return float(x[2]) if len(x) > 2 else 0.0

    return Constraint(
        name="memory",
        func=memory_estimator,
        bound=max_memory_mb,
        constraint_type="upper",
        config=config,
    )


def make_accuracy_constraint(
    min_accuracy: float,
    accuracy_evaluator: Optional[Callable[[np.ndarray], float]] = None,
    config: Optional[ConstraintConfig] = None,
) -> Constraint:
    """
    Create constraint on minimum accuracy.

    Args:
        min_accuracy: Minimum required accuracy (e.g., 0.85 for 85%)
        accuracy_evaluator: Function to evaluate accuracy from decision vector.
                           If None, assumes x[3] = accuracy.
        config: Constraint configuration

    Returns:
        Constraint object

    Example:
        >>> # Require at least 85% accuracy
        >>> constraint = make_accuracy_constraint(0.85)
    """
    config = config or ConstraintConfig()

    if accuracy_evaluator is None:

        def accuracy_evaluator(x: np.ndarray) -> float:
            return float(x[3]) if len(x) > 3 else 1.0

    return Constraint(
        name="accuracy",
        func=accuracy_evaluator,
        bound=min_accuracy,
        constraint_type="lower",
        config=config,
    )


def make_custom_constraint(
    name: str,
    func: ConstraintFunc,
    bound: float,
    constraint_type: str = "upper",
    config: Optional[ConstraintConfig] = None,
) -> Constraint:
    """
    Create a custom constraint.

    Args:
        name: Constraint name for identification
        func: Function mapping decision vector to constraint value
        bound: Constraint bound value
        constraint_type: "upper" (value <= bound), "lower" (value >= bound),
                        or "equality" (value == bound)
        config: Constraint configuration

    Returns:
        Constraint object

    Example:
        >>> # Custom constraint: layer count <= 12
        >>> constraint = make_custom_constraint(
        ...     name="layer_count",
        ...     func=lambda x: x[4],  # 5th variable is layer count
        ...     bound=12.0,
        ...     constraint_type="upper",
        ... )
    """
    config = config or ConstraintConfig()

    return Constraint(
        name=name,
        func=func,
        bound=bound,
        constraint_type=constraint_type,
        config=config,
    )


@dataclass
class CombinedConstraint:
    """Multiple constraints combined into one."""

    constraints: List[Constraint]
    aggregation: str = "max"  # "max", "sum", or "all"

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate combined constraint.

        Returns:
            - "max": Maximum violation (most violated constraint)
            - "sum": Sum of violations (total constraint pressure)
            - "all": Returns max (same as "max" for scalar return)
        """
        violations = [c.evaluate(x) for c in self.constraints]

        if self.aggregation == "max":
            return max(violations) if violations else 0.0
        elif self.aggregation == "sum":
            return sum(max(0.0, v) for v in violations)
        elif self.aggregation == "all":
            return max(violations) if violations else 0.0
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def evaluate_all(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all constraints and return array."""
        return np.array([c.evaluate(x) for c in self.constraints])

    def is_satisfied(self, x: np.ndarray) -> bool:
        """Check if all constraints are satisfied."""
        return all(c.is_satisfied(x) for c in self.constraints)

    def violations(self, x: np.ndarray) -> Dict[str, float]:
        """Get violations by constraint name."""
        return {c.name: c.violation(x) for c in self.constraints}

    def satisfied_constraints(self, x: np.ndarray) -> List[str]:
        """Get names of satisfied constraints."""
        return [c.name for c in self.constraints if c.is_satisfied(x)]

    def violated_constraints(self, x: np.ndarray) -> List[str]:
        """Get names of violated constraints."""
        return [c.name for c in self.constraints if not c.is_satisfied(x)]


def combine_constraints(
    *constraints: Constraint,
    aggregation: str = "max",
) -> CombinedConstraint:
    """
    Combine multiple constraints into one.

    Args:
        *constraints: Constraint objects to combine
        aggregation: How to combine:
            - "max": Use maximum violation (strictest)
            - "sum": Sum all violations (total pressure)
            - "all": Same as "max" for compatibility

    Returns:
        CombinedConstraint object

    Example:
        >>> c1 = make_param_constraint(25_000_000)
        >>> c2 = make_latency_constraint(100.0)
        >>> c3 = make_memory_constraint(6144.0)
        >>> combined = combine_constraints(c1, c2, c3)
        >>> is_feasible = combined.is_satisfied(solution)
    """
    return CombinedConstraint(
        constraints=list(constraints),
        aggregation=aggregation,
    )


def apply_constraints(
    solutions: np.ndarray,
    constraints: Union[Constraint, CombinedConstraint, List[Constraint]],
    return_feasible_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply constraints to a set of solutions.

    Args:
        solutions: Array of solutions (n_solutions x n_variables)
        constraints: Constraint(s) to apply
        return_feasible_only: If True, only return feasible solutions

    Returns:
        Tuple of (solutions, violations) where violations is (n_solutions,)

    Example:
        >>> combined = combine_constraints(c1, c2, c3)
        >>> feasible_solutions, violations = apply_constraints(
        ...     pareto_solutions,
        ...     combined,
        ...     return_feasible_only=True,
        ... )
    """
    # Normalize to CombinedConstraint
    if isinstance(constraints, list):
        combined = combine_constraints(*constraints)
    elif isinstance(constraints, Constraint):
        combined = combine_constraints(constraints)
    else:
        combined = constraints

    # Ensure 2D
    if solutions.ndim == 1:
        solutions = solutions.reshape(1, -1)

    # Evaluate violations
    violations = np.array([combined.evaluate(x) for x in solutions])

    if return_feasible_only:
        feasible_mask = violations <= 0
        return solutions[feasible_mask], violations[feasible_mask]

    return solutions, violations


def create_pymoo_constraint_handler(
    constraints: Union[Constraint, CombinedConstraint, List[Constraint]],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a constraint handler function for pymoo problems.

    Args:
        constraints: Constraint(s) to apply

    Returns:
        Function suitable for pymoo's _evaluate G parameter

    Example:
        >>> combined = combine_constraints(c1, c2, c3)
        >>> handler = create_pymoo_constraint_handler(combined)
        >>> # In pymoo Problem._evaluate:
        >>> out["G"] = handler(X)
    """
    # Normalize to CombinedConstraint
    if isinstance(constraints, list):
        combined = combine_constraints(*constraints)
    elif isinstance(constraints, Constraint):
        combined = combine_constraints(constraints)
    else:
        combined = constraints

    def handler(X: np.ndarray) -> np.ndarray:
        """Evaluate constraints for all solutions."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Return shape: (n_solutions, n_constraints)
        G = np.zeros((len(X), len(combined.constraints)))
        for i, x in enumerate(X):
            G[i] = combined.evaluate_all(x)

        return G

    return handler


# Convenience function for common Agent Forge constraints
def create_agent_forge_constraints(
    max_params: int = 25_000_000,
    max_latency_ms: float = 100.0,
    max_memory_mb: float = 6144.0,  # 6GB
    min_accuracy: Optional[float] = None,
    custom: Optional[List[Constraint]] = None,
) -> CombinedConstraint:
    """
    Create standard Agent Forge constraints.

    Args:
        max_params: Maximum parameter count (default 25M)
        max_latency_ms: Maximum inference latency in ms
        max_memory_mb: Maximum GPU memory in MB
        min_accuracy: Minimum accuracy (optional)
        custom: Additional custom constraints

    Returns:
        Combined constraint for Agent Forge optimization

    Example:
        >>> constraints = create_agent_forge_constraints(
        ...     max_params=25_000_000,
        ...     max_memory_mb=6144.0,  # 6GB VRAM
        ... )
        >>> handler = create_pymoo_constraint_handler(constraints)
    """
    constraints = [
        make_param_constraint(max_params),
        make_latency_constraint(max_latency_ms),
        make_memory_constraint(max_memory_mb),
    ]

    if min_accuracy is not None:
        constraints.append(make_accuracy_constraint(min_accuracy))

    if custom:
        constraints.extend(custom)

    return combine_constraints(*constraints, aggregation="max")

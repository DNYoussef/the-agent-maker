"""
Pareto Selection Strategies (Layer 2)

Utilities for selecting solutions from Pareto fronts returned by MOO.
These are MOO-only utilities with no meta-calculus dependencies.

Functions:
    select_balanced: Select solution balancing all objectives
    select_knee_point: Find knee point (max curvature) on Pareto front
    select_by_constraint: Select best under constraint
    select_by_preference: Select based on weighted preferences
    analyze_tradeoffs: Analyze tradeoffs between objectives
    compute_hypervolume: Compute hypervolume indicator

Design:
    - Works with pymoo Result objects or raw numpy arrays
    - No dependencies on k_formula or bigeometric
    - Only imports from Layer 1 (moo_bridge.py) if needed
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    from pymoo.indicators.hv import HV
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


@dataclass
class SelectionConfig:
    """Configuration for Pareto selection strategies."""

    # For select_balanced
    normalize: bool = True  # Normalize objectives before balancing

    # For select_knee_point
    knee_method: str = "distance"  # "distance" or "angle"

    # For select_by_constraint
    strict: bool = True  # Reject if no feasible solution

    # For analyze_tradeoffs
    num_samples: int = 10  # Number of samples for tradeoff analysis

    # For hypervolume
    reference_point: Optional[List[float]] = None  # Auto-compute if None


@dataclass
class SelectionResult:
    """Result from a selection operation."""

    index: int  # Index of selected solution in Pareto front
    solution: np.ndarray  # Decision variables (X)
    objectives: np.ndarray  # Objective values (F)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_objectives(F: np.ndarray) -> np.ndarray:
    """Normalize objectives to [0, 1] range."""
    F_min = F.min(axis=0)
    F_max = F.max(axis=0)
    F_range = F_max - F_min
    # Avoid division by zero
    F_range = np.where(F_range == 0, 1.0, F_range)
    return (F - F_min) / F_range


def _extract_pareto_data(
    result: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract X and F from pymoo Result or raw arrays."""
    if hasattr(result, 'X') and hasattr(result, 'F'):
        # pymoo Result object
        X = result.X
        F = result.F
    elif isinstance(result, dict):
        # Dict from pymoo-like code or MOORunner.optimize().
        X = np.array(result.get('X', result.get('x', result.get('pareto_solutions', []))))
        F = np.array(result.get('F', result.get('f', result.get('pareto_front', []))))
    elif isinstance(result, tuple) and len(result) == 2:
        # Tuple of (X, F)
        X, F = result
        X = np.array(X)
        F = np.array(F)
    else:
        raise ValueError(
            f"Cannot extract Pareto data from type {type(result)}. "
            "Expected pymoo Result, dict with X/F keys, or (X, F) tuple."
        )

    if X.size == 0 or F.size == 0:
        raise ValueError("Pareto front is empty")

    # Handle single solution case
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if F.ndim == 1:
        F = F.reshape(1, -1)

    return X, F


def select_balanced(
    result: Any,
    config: Optional[SelectionConfig] = None,
) -> SelectionResult:
    """
    Select solution that balances all objectives equally.

    Uses the "ideal point" approach - finds solution closest to
    the point where all objectives are at their minimum.

    Args:
        result: Pareto front (pymoo Result, dict, or tuple)
        config: Selection configuration

    Returns:
        SelectionResult with balanced solution

    Example:
        >>> result = run_moo_optimization("phase2", evaluator)
        >>> balanced = select_balanced(result)
        >>> print(f"Selected index: {balanced.index}")
    """
    config = config or SelectionConfig()
    X, F = _extract_pareto_data(result)

    if config.normalize:
        F_norm = _normalize_objectives(F)
    else:
        F_norm = F

    # Compute distance to ideal point (origin in normalized space)
    distances = np.sqrt(np.sum(F_norm ** 2, axis=1))

    # Select closest to ideal
    best_idx = int(np.argmin(distances))

    return SelectionResult(
        index=best_idx,
        solution=X[best_idx],
        objectives=F[best_idx],
        metadata={
            "method": "balanced",
            "distance_to_ideal": float(distances[best_idx]),
            "normalized": config.normalize,
        }
    )


def select_knee_point(
    result: Any,
    config: Optional[SelectionConfig] = None,
) -> SelectionResult:
    """
    Select knee point on Pareto front (maximum curvature).

    The knee point represents the best tradeoff - moving away from
    it in any direction gives diminishing returns.

    Args:
        result: Pareto front (pymoo Result, dict, or tuple)
        config: Selection configuration

    Returns:
        SelectionResult with knee point solution

    Example:
        >>> result = search_architecture(evaluator)
        >>> knee = select_knee_point(result)
        >>> print(f"Knee point objectives: {knee.objectives}")
    """
    config = config or SelectionConfig()
    X, F = _extract_pareto_data(result)

    if len(F) <= 2:
        # Too few points - return balanced instead
        return select_balanced(result, config)

    # Normalize for consistent distance calculation
    F_norm = _normalize_objectives(F)

    if config.knee_method == "distance":
        # Distance-based knee detection
        # Sort by first objective
        sorted_indices = np.argsort(F_norm[:, 0])
        F_sorted = F_norm[sorted_indices]

        # Line from first to last point
        p1 = F_sorted[0]
        p2 = F_sorted[-1]
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            # All points same - return first
            best_idx = 0
        else:
            line_unit = line_vec / line_len

            # Distance of each point to the line
            distances = []
            for i, p in enumerate(F_sorted):
                vec_to_p = p - p1
                proj_len = np.dot(vec_to_p, line_unit)
                proj = p1 + proj_len * line_unit
                dist = np.linalg.norm(p - proj)
                distances.append(dist)

            # Knee is point with maximum distance
            knee_sorted_idx = int(np.argmax(distances))
            best_idx = sorted_indices[knee_sorted_idx]

    elif config.knee_method == "angle":
        # Angle-based knee detection
        sorted_indices = np.argsort(F_norm[:, 0])
        F_sorted = F_norm[sorted_indices]

        angles = []
        for i in range(1, len(F_sorted) - 1):
            v1 = F_sorted[i] - F_sorted[i-1]
            v2 = F_sorted[i+1] - F_sorted[i]

            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
            )
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)

        if angles:
            # Minimum angle = sharpest turn = knee
            knee_local_idx = int(np.argmin(angles)) + 1
            best_idx = sorted_indices[knee_local_idx]
        else:
            best_idx = 0
    else:
        raise ValueError(f"Unknown knee method: {config.knee_method}")

    return SelectionResult(
        index=best_idx,
        solution=X[best_idx],
        objectives=F[best_idx],
        metadata={
            "method": "knee_point",
            "knee_method": config.knee_method,
        }
    )


def select_by_constraint(
    result: Any,
    constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
    primary_objective: int = 0,
    config: Optional[SelectionConfig] = None,
) -> Optional[SelectionResult]:
    """
    Select best solution under constraints.

    Args:
        result: Pareto front
        constraints: Dict mapping objective index to (min, max) bounds.
                    Use None for unbounded.
        primary_objective: Index of objective to minimize after constraints
        config: Selection configuration

    Returns:
        SelectionResult or None if no feasible solution (when strict=True)

    Example:
        >>> # Select best perplexity with param_count <= 25M
        >>> result = search_architecture(evaluator)
        >>> selection = select_by_constraint(
        ...     result,
        ...     constraints={0: (None, 25_000_000)},  # obj 0 = params
        ...     primary_objective=1,  # obj 1 = perplexity
        ... )
    """
    config = config or SelectionConfig()
    X, F = _extract_pareto_data(result)

    # Find feasible solutions
    feasible_mask = np.ones(len(F), dtype=bool)

    for obj_idx, (lb, ub) in constraints.items():
        if lb is not None:
            feasible_mask &= F[:, obj_idx] >= lb
        if ub is not None:
            feasible_mask &= F[:, obj_idx] <= ub

    feasible_indices = np.where(feasible_mask)[0]

    if len(feasible_indices) == 0:
        if config.strict:
            return None
        else:
            # Return least violating solution
            # Sum of constraint violations
            violations = np.zeros(len(F))
            for obj_idx, (lb, ub) in constraints.items():
                if lb is not None:
                    violations += np.maximum(0, lb - F[:, obj_idx])
                if ub is not None:
                    violations += np.maximum(0, F[:, obj_idx] - ub)
            best_idx = int(np.argmin(violations))
            return SelectionResult(
                index=best_idx,
                solution=X[best_idx],
                objectives=F[best_idx],
                metadata={
                    "method": "by_constraint",
                    "feasible": False,
                    "violation": float(violations[best_idx]),
                }
            )

    # Among feasible, select best on primary objective
    feasible_F = F[feasible_indices]
    local_best = int(np.argmin(feasible_F[:, primary_objective]))
    best_idx = feasible_indices[local_best]

    return SelectionResult(
        index=best_idx,
        solution=X[best_idx],
        objectives=F[best_idx],
        metadata={
            "method": "by_constraint",
            "feasible": True,
            "num_feasible": len(feasible_indices),
            "constraints_used": constraints,
        }
    )


def select_by_preference(
    result: Any,
    weights: List[float],
    config: Optional[SelectionConfig] = None,
) -> SelectionResult:
    """
    Select solution based on weighted preferences.

    Uses weighted sum (scalarization) to combine objectives.
    All objectives are assumed to be minimized.

    Args:
        result: Pareto front
        weights: Weight for each objective (will be normalized)
        config: Selection configuration

    Returns:
        SelectionResult

    Example:
        >>> # Prefer accuracy over speed (3:1 ratio)
        >>> result = search_architecture(evaluator)
        >>> selection = select_by_preference(
        ...     result,
        ...     weights=[1.0, 3.0, 1.0, 1.0],  # 4 objectives
        ... )
    """
    config = config or SelectionConfig()
    X, F = _extract_pareto_data(result)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Normalize objectives if requested
    if config.normalize:
        F_norm = _normalize_objectives(F)
    else:
        F_norm = F

    # Weighted sum
    scores = np.sum(F_norm * weights, axis=1)
    best_idx = int(np.argmin(scores))

    return SelectionResult(
        index=best_idx,
        solution=X[best_idx],
        objectives=F[best_idx],
        metadata={
            "method": "by_preference",
            "weights": weights.tolist(),
            "score": float(scores[best_idx]),
        }
    )


@dataclass
class TradeoffAnalysis:
    """Result of tradeoff analysis between objectives."""

    objective_i: int
    objective_j: int
    correlation: float  # Pearson correlation (-1 to 1)
    tradeoff_ratio: float  # Avg units of j lost per unit of i gained
    extreme_solutions: Dict[str, int]  # Indices of extreme solutions
    samples: List[Tuple[int, int]]  # Pairs of (idx_i_better, idx_j_better)


def analyze_tradeoffs(
    result: Any,
    objective_pairs: Optional[List[Tuple[int, int]]] = None,
    config: Optional[SelectionConfig] = None,
) -> List[TradeoffAnalysis]:
    """
    Analyze tradeoffs between objective pairs.

    Args:
        result: Pareto front
        objective_pairs: List of (i, j) pairs to analyze.
                        If None, analyzes all pairs.
        config: Selection configuration

    Returns:
        List of TradeoffAnalysis for each pair

    Example:
        >>> result = search_architecture(evaluator)
        >>> tradeoffs = analyze_tradeoffs(result)
        >>> for t in tradeoffs:
        ...     print(f"Obj {t.objective_i} vs {t.objective_j}: r={t.correlation:.2f}")
    """
    config = config or SelectionConfig()
    X, F = _extract_pareto_data(result)

    n_objectives = F.shape[1]

    if objective_pairs is None:
        # Analyze all pairs
        objective_pairs = [
            (i, j) for i in range(n_objectives) for j in range(i+1, n_objectives)
        ]

    results = []

    for obj_i, obj_j in objective_pairs:
        fi = F[:, obj_i]
        fj = F[:, obj_j]

        # Correlation
        if len(fi) > 1:
            corr = np.corrcoef(fi, fj)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        # Tradeoff ratio (slope of Pareto front)
        if len(fi) > 1:
            # Fit line through Pareto points
            slope, _ = np.polyfit(fi, fj, 1)
            tradeoff_ratio = abs(slope)
        else:
            tradeoff_ratio = 0.0

        # Extreme solutions
        extremes = {
            f"best_obj_{obj_i}": int(np.argmin(fi)),
            f"best_obj_{obj_j}": int(np.argmin(fj)),
            f"worst_obj_{obj_i}": int(np.argmax(fi)),
            f"worst_obj_{obj_j}": int(np.argmax(fj)),
        }

        # Sample pairs showing tradeoff
        samples = []
        if len(F) >= 2:
            n_samples = min(config.num_samples, len(F) // 2)
            sorted_by_i = np.argsort(fi)
            sorted_by_j = np.argsort(fj)
            for k in range(n_samples):
                if k < len(sorted_by_i) and k < len(sorted_by_j):
                    samples.append((
                        int(sorted_by_i[k]),
                        int(sorted_by_j[k])
                    ))

        results.append(TradeoffAnalysis(
            objective_i=obj_i,
            objective_j=obj_j,
            correlation=float(corr),
            tradeoff_ratio=float(tradeoff_ratio),
            extreme_solutions=extremes,
            samples=samples,
        ))

    return results


def compute_hypervolume(
    result: Any,
    reference_point: Optional[List[float]] = None,
    config: Optional[SelectionConfig] = None,
) -> float:
    """
    Compute hypervolume indicator for Pareto front quality.

    Higher hypervolume = better Pareto front coverage.

    Args:
        result: Pareto front
        reference_point: Reference point for hypervolume.
                        If None, uses max of each objective + 10%.
        config: Selection configuration

    Returns:
        Hypervolume value

    Example:
        >>> result = search_architecture(evaluator)
        >>> hv = compute_hypervolume(result)
        >>> print(f"Hypervolume: {hv:.4f}")
    """
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo is required for hypervolume computation")

    config = config or SelectionConfig()
    X, F = _extract_pareto_data(result)

    # Determine reference point
    if reference_point is not None:
        ref = np.array(reference_point)
    elif config.reference_point is not None:
        ref = np.array(config.reference_point)
    else:
        # Auto-compute: max + 10%
        F_max = F.max(axis=0)
        F_range = F.max(axis=0) - F.min(axis=0)
        ref = F_max + 0.1 * F_range

    # Compute hypervolume
    hv = HV(ref_point=ref)
    return float(hv(F))

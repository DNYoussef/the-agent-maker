"""
GlobalMOO + pymoo Hybrid Adapter

Provides seamless integration with GlobalMOO cloud API while falling back
to pymoo for local optimization when GlobalMOO is unavailable.

This adapter is based on meta-calculus-toolkit's moo_integration.py but
adapted for Agent Forge V2's phase-specific optimization needs.

Usage:
    from src.cross_phase.meta_calculus.moo_utils import HybridMOORunner

    # Automatically uses GlobalMOO if API key is set, else pymoo
    runner = HybridMOORunner(problem)
    result = runner.run()

    # Force specific backend
    runner = HybridMOORunner(problem, backend="globalmoo")  # or "pymoo"

Environment:
    GLOBALMOO_API_KEY: API key from https://app.globalmoo.com
    GLOBALMOO_API_URL: API URL (default: https://app.globalmoo.com/api/)

References:
    - GlobalMOO Docs: https://globalmoo.gitbook.io/globalmoo-documentation
    - GlobalMOO SDK: pip install globalmoo-sdk
    - GitHub: https://github.com/globalMOO/gmoo-sdk-suite
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

# Check for pymoo
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.core.problem import Problem as PymooProblem
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

# Check for GlobalMOO SDK
try:
    from globalmoo.client import Client
    from globalmoo.credentials import Credentials
    from globalmoo.request.create_model import CreateModel
    from globalmoo.request.create_project import CreateProject
    from globalmoo.request.load_output_cases import LoadOutputCases
    from globalmoo.request.load_objectives import LoadObjectives
    from globalmoo.request.suggest_inverse import SuggestInverse
    from globalmoo.request.load_inversed_output import LoadInversedOutput
    GLOBALMOO_AVAILABLE = True
except ImportError:
    GLOBALMOO_AVAILABLE = False

# Environment variables
GLOBALMOO_API_KEY = os.environ.get('GLOBALMOO_API_KEY', '')
GLOBALMOO_API_URL = os.environ.get('GLOBALMOO_API_URL', 'https://app.globalmoo.com/api/')


@dataclass
class HybridMOOConfig:
    """Configuration for hybrid MOO runner."""

    # Backend selection
    backend: str = "auto"  # "auto", "globalmoo", or "pymoo"

    # GlobalMOO settings
    project_name: str = "AgentForge_Optimization"
    model_name: str = "phase_optimization"
    max_api_retries: int = 3
    api_timeout: int = 60

    # pymoo settings
    n_generations: int = 50
    pop_size: int = 30
    seed: int = 42

    # Fallback behavior
    fallback_on_error: bool = True  # Fall back to pymoo if GlobalMOO fails

    # Objective configuration
    objective_names: List[str] = field(default_factory=list)
    objective_minimize: List[bool] = field(default_factory=list)


@dataclass
class HybridMOOResult:
    """Result from hybrid MOO optimization."""

    X: np.ndarray  # Decision variables (Pareto solutions)
    F: np.ndarray  # Objective values
    backend_used: str  # "globalmoo" or "pymoo"
    n_evaluations: int
    runtime_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)

    def best_by_objective(self, obj_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get solution with best value for given objective."""
        minimize = self.metadata.get("objective_minimize", [])
        is_minimize = minimize[obj_idx] if obj_idx < len(minimize) else True
        best_idx = np.argmin(self.F[:, obj_idx]) if is_minimize else np.argmax(self.F[:, obj_idx])
        return self.X[best_idx], self.F[best_idx]


class GlobalMOOClient:
    """
    Client for GlobalMOO cloud API.

    Based on meta-calculus-toolkit implementation but simplified for
    Agent Forge use case.
    """

    def __init__(self, config: HybridMOOConfig):
        if not GLOBALMOO_AVAILABLE:
            raise ImportError(
                "GlobalMOO SDK not installed. Install with: pip install globalmoo-sdk"
            )
        if not GLOBALMOO_API_KEY:
            raise ValueError(
                "GLOBALMOO_API_KEY environment variable not set. "
                "Get your key from https://app.globalmoo.com"
            )

        self.config = config
        self.credentials = Credentials(api_key=GLOBALMOO_API_KEY)
        self.client = Client(self.credentials)
        self.project_id = None
        self.model_id = None

    def setup_project(
        self,
        n_inputs: int,
        n_outputs: int,
        input_bounds: List[Tuple[float, float]],
    ) -> str:
        """Create or get project in GlobalMOO."""
        # Create project
        project_req = CreateProject(
            name=self.config.project_name,
            description=f"Agent Forge Phase Optimization - {self.config.model_name}",
        )
        response = self.client.send(project_req)
        self.project_id = response.get('project_id', response.get('id'))

        # Create model with input/output specs
        model_req = CreateModel(
            project_id=self.project_id,
            name=self.config.model_name,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            input_specs=[
                {
                    "name": f"x{i}",
                    "min": input_bounds[i][0],
                    "max": input_bounds[i][1],
                }
                for i in range(n_inputs)
            ],
            output_specs=[
                {
                    "name": self.config.objective_names[i] if i < len(self.config.objective_names) else f"obj{i}",
                    "minimize": self.config.objective_minimize[i] if i < len(self.config.objective_minimize) else True,
                }
                for i in range(n_outputs)
            ],
        )
        response = self.client.send(model_req)
        self.model_id = response.get('model_id', response.get('id'))

        return self.project_id

    def submit_evaluations(
        self,
        X: np.ndarray,
        F: np.ndarray,
    ) -> None:
        """Submit evaluated solutions to GlobalMOO."""
        # Load output cases (evaluated solutions)
        cases_req = LoadOutputCases(
            project_id=self.project_id,
            model_id=self.model_id,
            cases=[
                {
                    "inputs": X[i].tolist(),
                    "outputs": F[i].tolist(),
                }
                for i in range(len(X))
            ],
        )
        self.client.send(cases_req)

    def get_suggestions(
        self,
        n_suggestions: int = 10,
    ) -> np.ndarray:
        """Get suggested points from GlobalMOO."""
        # Request inverse suggestions
        suggest_req = SuggestInverse(
            project_id=self.project_id,
            model_id=self.model_id,
            n_suggestions=n_suggestions,
        )
        response = self.client.send(suggest_req)

        # Load the suggestions
        load_req = LoadInversedOutput(
            project_id=self.project_id,
            model_id=self.model_id,
        )
        suggestions = self.client.send(load_req)

        return np.array([s['inputs'] for s in suggestions])

    def get_pareto_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current Pareto front from GlobalMOO."""
        # Load objectives (Pareto-optimal solutions)
        obj_req = LoadObjectives(
            project_id=self.project_id,
            model_id=self.model_id,
        )
        response = self.client.send(obj_req)

        X = np.array([s['inputs'] for s in response])
        F = np.array([s['outputs'] for s in response])

        return X, F


class PymooRunner:
    """
    Local pymoo NSGA-II runner.

    Used as fallback when GlobalMOO is unavailable.
    """

    def __init__(
        self,
        problem: PymooProblem,
        config: HybridMOOConfig,
    ):
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo not installed. Install with: pip install pymoo")

        self.problem = problem
        self.config = config

    def run(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run NSGA-II optimization."""
        algorithm = NSGA2(pop_size=self.config.pop_size)

        result = minimize(
            self.problem,
            algorithm,
            ('n_gen', self.config.n_generations),
            seed=self.config.seed,
            verbose=False,
        )

        n_evals = result.algorithm.evaluator.n_eval

        return result.X, result.F, n_evals


class HybridMOORunner:
    """
    Hybrid MOO runner that uses GlobalMOO when available, pymoo as fallback.

    This provides the best of both worlds:
    - GlobalMOO: Cloud-based, handles complex search spaces, proven algorithms
    - pymoo: Local, fast, no API dependency

    Usage:
        # With pymoo Problem
        from pymoo.core.problem import Problem
        class MyProblem(Problem):
            ...

        runner = HybridMOORunner(MyProblem())
        result = runner.run()

        # With evaluator function
        def evaluator(x):
            return [obj1, obj2, obj3]

        runner = HybridMOORunner.from_evaluator(
            evaluator,
            n_vars=4,
            n_objs=3,
            xl=[0, 0, 0, 0],
            xu=[1, 1, 1, 1],
        )
        result = runner.run()
    """

    def __init__(
        self,
        problem: PymooProblem,
        config: Optional[HybridMOOConfig] = None,
        backend: str = "auto",
    ):
        self.problem = problem
        self.config = config or HybridMOOConfig()
        self.config.backend = backend

    @classmethod
    def from_evaluator(
        cls,
        evaluator: Callable[[np.ndarray], List[float]],
        n_vars: int,
        n_objs: int,
        xl: List[float],
        xu: List[float],
        config: Optional[HybridMOOConfig] = None,
        backend: str = "auto",
    ) -> "HybridMOORunner":
        """
        Create runner from evaluator function.

        Args:
            evaluator: Function (x) -> [obj1, obj2, ...]
            n_vars: Number of decision variables
            n_objs: Number of objectives
            xl: Lower bounds for variables
            xu: Upper bounds for variables
            config: Optional configuration
            backend: "auto", "globalmoo", or "pymoo"

        Returns:
            HybridMOORunner instance
        """
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo required for from_evaluator")

        runner_config = config or HybridMOOConfig()

        class EvaluatorProblem(PymooProblem):
            def __init__(self):
                super().__init__(
                    n_var=n_vars,
                    n_obj=n_objs,
                    xl=np.array(xl),
                    xu=np.array(xu),
                )
                self.evaluator_fn = evaluator

            def _evaluate(self, X, out, *args, **kwargs):
                F = np.array([self.evaluator_fn(x) for x in X], dtype=float)
                minimize = runner_config.objective_minimize
                if minimize and len(minimize) == F.shape[1]:
                    for j, is_minimize in enumerate(minimize):
                        if not is_minimize:
                            F[:, j] = -F[:, j]
                out["F"] = F

        return cls(EvaluatorProblem(), runner_config, backend)

    def _select_backend(self) -> str:
        """Select which backend to use."""
        if self.config.backend == "globalmoo":
            if not GLOBALMOO_AVAILABLE or not GLOBALMOO_API_KEY:
                if self.config.fallback_on_error:
                    return "pymoo"
                raise ValueError("GlobalMOO requested but not available")
            return "globalmoo"

        elif self.config.backend == "pymoo":
            if not PYMOO_AVAILABLE:
                raise ValueError("pymoo requested but not installed")
            return "pymoo"

        else:  # auto
            if GLOBALMOO_AVAILABLE and GLOBALMOO_API_KEY:
                return "globalmoo"
            elif PYMOO_AVAILABLE:
                return "pymoo"
            else:
                raise ValueError("No MOO backend available")

    def run(
        self,
        n_generations: Optional[int] = None,
        pop_size: Optional[int] = None,
        **kwargs,
    ) -> HybridMOOResult:
        """
        Run optimization using selected backend.

        Args:
            n_generations: Optional per-call override for legacy callers.
            pop_size: Optional per-call population-size override for legacy callers.

        Returns:
            HybridMOOResult with Pareto front and metadata
        """
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported HybridMOORunner.run arguments: {unknown}")
        if n_generations is not None:
            self.config.n_generations = n_generations
        if pop_size is not None:
            self.config.pop_size = pop_size

        backend = self._select_backend()
        start_time = time.time()

        if backend == "globalmoo":
            try:
                X, F, n_evals = self._run_globalmoo()
            except Exception as e:
                if self.config.fallback_on_error:
                    print(f"GlobalMOO failed ({e}), falling back to pymoo")
                    backend = "pymoo"
                    X, F, n_evals = self._run_pymoo()
                else:
                    raise
        else:
            X, F, n_evals = self._run_pymoo()

        minimize = runner_config.objective_minimize
        if minimize and len(minimize) == F.shape[1]:
            F = F.copy()
            for j, is_minimize in enumerate(minimize):
                if not is_minimize:
                    F[:, j] = -F[:, j]

        runtime = time.time() - start_time

        return HybridMOOResult(
            X=X,
            F=F,
            backend_used=backend,
            n_evaluations=n_evals,
            runtime_seconds=runtime,
            metadata={
                "config": {
                    "n_generations": self.config.n_generations,
                    "pop_size": self.config.pop_size,
                },
                "objective_minimize": list(self.config.objective_minimize),
            },
        )

    def _run_pymoo(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run using pymoo backend."""
        runner = PymooRunner(self.problem, self.config)
        return runner.run()

    def _run_globalmoo(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Run using GlobalMOO backend.

        Uses an iterative approach:
        1. Generate initial population with pymoo
        2. Evaluate and submit to GlobalMOO
        3. Get suggestions from GlobalMOO
        4. Evaluate suggestions and repeat
        5. Return final Pareto front
        """
        client = GlobalMOOClient(self.config)

        # Get problem bounds
        xl = self.problem.xl
        xu = self.problem.xu
        n_vars = self.problem.n_var
        n_objs = self.problem.n_obj

        # Setup project
        input_bounds = [(xl[i], xu[i]) for i in range(n_vars)]
        client.setup_project(n_vars, n_objs, input_bounds)

        # Generate initial population (Latin Hypercube Sampling)
        from pymoo.operators.sampling.lhs import LHS
        sampling = LHS()
        X_init = sampling(self.problem, self.config.pop_size).get("X")

        # Evaluate initial population
        F_init = np.zeros((len(X_init), n_objs))
        for i, x in enumerate(X_init):
            out = {}
            self.problem._evaluate(x.reshape(1, -1), out)
            F_init[i] = out["F"][0]

        # Submit to GlobalMOO
        client.submit_evaluations(X_init, F_init)

        n_evals = len(X_init)
        total_gens = self.config.n_generations // 5  # Fewer iterations for API

        # Iterative optimization
        for gen in range(total_gens):
            # Get suggestions
            X_suggest = client.get_suggestions(self.config.pop_size // 2)

            if len(X_suggest) == 0:
                break

            # Evaluate suggestions
            F_suggest = np.zeros((len(X_suggest), n_objs))
            for i, x in enumerate(X_suggest):
                out = {}
                self.problem._evaluate(x.reshape(1, -1), out)
                F_suggest[i] = out["F"][0]

            # Submit to GlobalMOO
            client.submit_evaluations(X_suggest, F_suggest)
            n_evals += len(X_suggest)

        # Get final Pareto front
        X, F = client.get_pareto_front()

        return X, F, n_evals


def create_hybrid_runner(
    evaluator: Callable[[np.ndarray], List[float]],
    n_vars: int,
    n_objs: int,
    bounds: List[Tuple[float, float]],
    objective_names: Optional[List[str]] = None,
    backend: str = "auto",
    **config_kwargs,
) -> HybridMOORunner:
    """
    Convenience function to create a hybrid MOO runner.

    Args:
        evaluator: Function (x) -> [obj1, obj2, ...]
        n_vars: Number of decision variables
        n_objs: Number of objectives
        bounds: List of (min, max) tuples for each variable
        objective_names: Names for objectives
        backend: "auto", "globalmoo", or "pymoo"
        **config_kwargs: Additional HybridMOOConfig arguments

    Returns:
        Configured HybridMOORunner

    Example:
        >>> def evaluate(x):
        ...     return [x[0]**2, x[1]**2, x[0] + x[1]]
        >>> runner = create_hybrid_runner(
        ...     evaluate,
        ...     n_vars=2,
        ...     n_objs=3,
        ...     bounds=[(0, 1), (0, 1)],
        ...     objective_names=["obj1", "obj2", "obj3"],
        ... )
        >>> result = runner.run()
    """
    xl = [b[0] for b in bounds]
    xu = [b[1] for b in bounds]

    config = HybridMOOConfig(
        objective_names=objective_names or [],
        **config_kwargs,
    )

    return HybridMOORunner.from_evaluator(
        evaluator,
        n_vars=n_vars,
        n_objs=n_objs,
        xl=xl,
        xu=xu,
        config=config,
        backend=backend,
    )


# Check availability
def check_moo_backends() -> Dict[str, bool]:
    """Check which MOO backends are available."""
    return {
        "pymoo": PYMOO_AVAILABLE,
        "globalmoo_sdk": GLOBALMOO_AVAILABLE,
        "globalmoo_configured": bool(GLOBALMOO_API_KEY),
    }


__all__ = [
    "HybridMOOConfig",
    "HybridMOOResult",
    "HybridMOORunner",
    "GlobalMOOClient",
    "PymooRunner",
    "create_hybrid_runner",
    "check_moo_backends",
    "GLOBALMOO_AVAILABLE",
    "PYMOO_AVAILABLE",
]

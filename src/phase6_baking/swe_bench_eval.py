"""
Phase 6: SWE-Bench Evaluator

SWE-Bench evaluates code generation on real GitHub issues.
SWE-Bench Lite contains 300 curated tasks from real repositories.

Each task has:
- Problem statement (GitHub issue description)
- Ground truth patch (the actual fix)
- Test patch (tests that verify the fix)

Reference: https://github.com/princeton-nlp/SWE-bench
Paper: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
from pathlib import Path
from enum import Enum
import json
import ast
import re


class EvaluationMode(Enum):
    """Evaluation strictness levels."""
    EXACT = "exact"           # Exact patch match
    SEMANTIC = "semantic"     # Semantically equivalent
    PARTIAL = "partial"       # Partial line overlap
    SYNTAX = "syntax"         # Just check syntax validity


@dataclass
class SWEBenchTask:
    """Single SWE-Bench evaluation task."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    patch: str = ""              # Ground truth patch
    test_patch: str = ""         # Tests for the patch
    fail_to_pass: List[str] = field(default_factory=list)  # Tests that should pass after fix
    pass_to_pass: List[str] = field(default_factory=list)  # Tests that should still pass
    environment_setup_commit: str = ""
    version: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SWEBenchTask':
        """Create task from dictionary."""
        return cls(
            instance_id=data.get('instance_id', ''),
            repo=data.get('repo', ''),
            base_commit=data.get('base_commit', ''),
            problem_statement=data.get('problem_statement', ''),
            hints_text=data.get('hints_text', ''),
            patch=data.get('patch', ''),
            test_patch=data.get('test_patch', ''),
            fail_to_pass=data.get('fail_to_pass', []),
            pass_to_pass=data.get('pass_to_pass', []),
            environment_setup_commit=data.get('environment_setup_commit', ''),
            version=data.get('version', '')
        )


@dataclass
class EvaluationResult:
    """Result of evaluating a single task."""
    instance_id: str
    exact_match: float       # 1.0 if prediction matches patch exactly
    partial_match: float     # Fraction of lines matching (Jaccard similarity)
    syntax_valid: float      # 1.0 if syntactically valid Python
    semantic_similar: float  # Cosine similarity of code embeddings (if available)
    prediction: str = ""
    ground_truth: str = ""
    error: Optional[str] = None
    evaluation_time_ms: float = 0.0

    @property
    def composite_score(self) -> float:
        """Weighted composite score."""
        return (
            0.4 * self.exact_match +
            0.3 * self.partial_match +
            0.2 * self.syntax_valid +
            0.1 * self.semantic_similar
        )


class SWEBenchEvaluator:
    """
    Evaluator for SWE-Bench Lite (300 curated tasks).

    Provides structured evaluation of code generation capabilities
    on real-world GitHub issues.

    Usage:
        evaluator = SWEBenchEvaluator("data/swe_bench_lite.json")
        tasks = evaluator.load_tasks(max_tasks=50)

        for task in tasks:
            prediction = model.generate(task.problem_statement)
            result = evaluator.evaluate_single(task, prediction)
            print(f"{task.instance_id}: {result.partial_match:.2%}")

        metrics = evaluator.aggregate_results()
        print(f"Overall: {metrics['composite_score']:.2%}")
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        subset: str = "lite",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize SWE-Bench evaluator.

        Args:
            data_path: Path to SWE-Bench JSON data file
            subset: "lite" (300 tasks), "test" (2294 tasks), or "dev" (225 tasks)
            cache_dir: Directory to cache downloaded data
        """
        self.data_path = Path(data_path) if data_path else None
        self.subset = subset
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/swe_bench_cache")
        self.tasks: List[SWEBenchTask] = []
        self.results: List[EvaluationResult] = []
        self._loaded = False

    def load_tasks(
        self,
        max_tasks: Optional[int] = None,
        repos: Optional[List[str]] = None,
        min_difficulty: Optional[int] = None
    ) -> List[SWEBenchTask]:
        """
        Load SWE-Bench tasks from JSON file.

        Args:
            max_tasks: Maximum number of tasks to load
            repos: Filter to specific repositories (e.g., ["django/django"])
            min_difficulty: Minimum difficulty level (based on patch size)

        Returns:
            List of loaded tasks
        """
        if self.data_path and self.data_path.exists():
            return self._load_from_file(max_tasks, repos, min_difficulty)
        else:
            print(f"Warning: SWE-Bench data not found at {self.data_path}")
            print("Download from: https://github.com/princeton-nlp/SWE-bench")
            print("Using synthetic tasks for testing...")
            return self._create_synthetic_tasks(max_tasks or 10)

    def _load_from_file(
        self,
        max_tasks: Optional[int],
        repos: Optional[List[str]],
        min_difficulty: Optional[int]
    ) -> List[SWEBenchTask]:
        """Load tasks from JSON file."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, dict):
            items = list(data.values())
        else:
            items = data

        # Filter by repos
        if repos:
            items = [item for item in items if item.get('repo') in repos]

        # Filter by difficulty (patch size as proxy)
        if min_difficulty:
            items = [
                item for item in items
                if len(item.get('patch', '').split('\n')) >= min_difficulty
            ]

        # Limit count
        if max_tasks:
            items = items[:max_tasks]

        # Convert to SWEBenchTask objects
        for item in items:
            task = SWEBenchTask.from_dict(item)
            self.tasks.append(task)

        self._loaded = True
        return self.tasks

    def _create_synthetic_tasks(self, count: int) -> List[SWEBenchTask]:
        """Create synthetic tasks for testing without real data."""
        synthetic_tasks = [
            SWEBenchTask(
                instance_id="synthetic-001",
                repo="test/repo",
                base_commit="abc123",
                problem_statement="Fix the off-by-one error in the loop that iterates over user indices.",
                patch="for i in range(len(users)):",  # Changed from range(len(users) + 1)
            ),
            SWEBenchTask(
                instance_id="synthetic-002",
                repo="test/repo",
                base_commit="def456",
                problem_statement="Handle the case where the input list is empty to prevent IndexError.",
                patch="if not items:\n    return []\nresult = items[0]",
            ),
            SWEBenchTask(
                instance_id="synthetic-003",
                repo="test/repo",
                base_commit="ghi789",
                problem_statement="Fix the SQL injection vulnerability by using parameterized queries.",
                patch='cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
            ),
            SWEBenchTask(
                instance_id="synthetic-004",
                repo="test/repo",
                base_commit="jkl012",
                problem_statement="Add proper error handling for file not found exceptions.",
                patch="try:\n    with open(path) as f:\n        data = f.read()\nexcept FileNotFoundError:\n    return None",
            ),
            SWEBenchTask(
                instance_id="synthetic-005",
                repo="test/repo",
                base_commit="mno345",
                problem_statement="Fix the race condition by adding proper locking.",
                patch="with self.lock:\n    self.counter += 1",
            ),
        ]

        # Repeat to reach count
        while len(synthetic_tasks) < count:
            base = synthetic_tasks[len(synthetic_tasks) % 5]
            new_task = SWEBenchTask(
                instance_id=f"synthetic-{len(synthetic_tasks):03d}",
                repo=base.repo,
                base_commit=f"commit{len(synthetic_tasks)}",
                problem_statement=base.problem_statement,
                patch=base.patch
            )
            synthetic_tasks.append(new_task)

        self.tasks = synthetic_tasks[:count]
        self._loaded = True
        return self.tasks

    def evaluate_single(
        self,
        task: SWEBenchTask,
        prediction: str,
        mode: EvaluationMode = EvaluationMode.PARTIAL
    ) -> EvaluationResult:
        """
        Evaluate a single prediction against ground truth.

        Args:
            task: The SWE-Bench task
            prediction: Model's predicted patch
            mode: Evaluation strictness level

        Returns:
            EvaluationResult with scores
        """
        import time
        start_time = time.time()

        try:
            result = EvaluationResult(
                instance_id=task.instance_id,
                exact_match=self._exact_match(prediction, task.patch),
                partial_match=self._line_overlap(prediction, task.patch),
                syntax_valid=self._check_syntax(prediction),
                semantic_similar=self._semantic_similarity(prediction, task.patch),
                prediction=prediction,
                ground_truth=task.patch,
                evaluation_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            result = EvaluationResult(
                instance_id=task.instance_id,
                exact_match=0.0,
                partial_match=0.0,
                syntax_valid=0.0,
                semantic_similar=0.0,
                error=str(e),
                evaluation_time_ms=(time.time() - start_time) * 1000
            )

        self.results.append(result)
        return result

    def _exact_match(self, pred: str, target: str) -> float:
        """Check for exact match after normalizing whitespace."""
        def normalize(s: str) -> str:
            # Remove leading/trailing whitespace, normalize line endings
            lines = [line.strip() for line in s.strip().split('\n')]
            return '\n'.join(line for line in lines if line)

        return 1.0 if normalize(pred) == normalize(target) else 0.0

    def _line_overlap(self, pred: str, target: str) -> float:
        """Compute Jaccard similarity of line sets."""
        pred_lines = set(
            line.strip() for line in pred.split('\n')
            if line.strip() and not line.strip().startswith('#')
        )
        target_lines = set(
            line.strip() for line in target.split('\n')
            if line.strip() and not line.strip().startswith('#')
        )

        if not target_lines:
            return 0.0

        intersection = len(pred_lines & target_lines)
        union = len(pred_lines | target_lines)

        return intersection / union if union > 0 else 0.0

    def _check_syntax(self, code: str) -> float:
        """Check if code is syntactically valid Python."""
        try:
            # Try to parse as Python
            ast.parse(code)
            return 1.0
        except SyntaxError:
            # Maybe it's a diff/patch format
            if code.startswith('+') or code.startswith('-') or code.startswith('@@'):
                # Extract actual code lines from diff
                code_lines = []
                for line in code.split('\n'):
                    if line.startswith('+') and not line.startswith('+++'):
                        code_lines.append(line[1:])
                    elif not line.startswith('-') and not line.startswith('@@'):
                        code_lines.append(line)

                try:
                    ast.parse('\n'.join(code_lines))
                    return 0.8  # Partial credit for valid diff
                except SyntaxError:
                    return 0.0
            return 0.0

    def _semantic_similarity(self, pred: str, target: str) -> float:
        """
        Compute semantic similarity between code snippets.

        Uses simple token overlap as a proxy. For production, consider
        using code embeddings (CodeBERT, GraphCodeBERT, etc.).
        """
        # Tokenize by splitting on whitespace and punctuation
        def tokenize(code: str) -> set:
            # Remove comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            # Split on non-alphanumeric
            tokens = re.findall(r'\w+', code.lower())
            return set(tokens)

        pred_tokens = tokenize(pred)
        target_tokens = tokenize(target)

        if not target_tokens:
            return 0.0

        # Jaccard similarity of tokens
        intersection = len(pred_tokens & target_tokens)
        union = len(pred_tokens | target_tokens)

        return intersection / union if union > 0 else 0.0

    def aggregate_results(self) -> Dict[str, float]:
        """
        Compute aggregate metrics across all evaluated tasks.

        Returns:
            Dictionary with aggregate metrics
        """
        if not self.results:
            return {
                'exact_match': 0.0,
                'partial_match': 0.0,
                'syntax_valid': 0.0,
                'semantic_similar': 0.0,
                'composite_score': 0.0,
                'count': 0,
                'error_count': 0
            }

        n = len(self.results)
        error_count = sum(1 for r in self.results if r.error is not None)

        return {
            'exact_match': sum(r.exact_match for r in self.results) / n,
            'partial_match': sum(r.partial_match for r in self.results) / n,
            'syntax_valid': sum(r.syntax_valid for r in self.results) / n,
            'semantic_similar': sum(r.semantic_similar for r in self.results) / n,
            'composite_score': sum(r.composite_score for r in self.results) / n,
            'count': n,
            'error_count': error_count,
            'avg_eval_time_ms': sum(r.evaluation_time_ms for r in self.results) / n
        }

    def run_evaluation(
        self,
        generate_fn: Callable[[str], str],
        max_tasks: int = 50,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run full evaluation using a generation function.

        Args:
            generate_fn: Function that takes problem_statement and returns prediction
            max_tasks: Maximum tasks to evaluate
            verbose: Print progress

        Returns:
            Aggregate metrics dictionary
        """
        tasks = self.load_tasks(max_tasks)

        for i, task in enumerate(tasks):
            if verbose:
                print(f"Evaluating {i+1}/{len(tasks)}: {task.instance_id}")

            try:
                prediction = generate_fn(task.problem_statement)
                result = self.evaluate_single(task, prediction)

                if verbose:
                    print(f"  Partial match: {result.partial_match:.2%}")
                    print(f"  Syntax valid: {result.syntax_valid:.0%}")

            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                self.results.append(EvaluationResult(
                    instance_id=task.instance_id,
                    exact_match=0.0,
                    partial_match=0.0,
                    syntax_valid=0.0,
                    semantic_similar=0.0,
                    error=str(e)
                ))

        metrics = self.aggregate_results()

        if verbose:
            print(f"\n{'='*50}")
            print(f"EVALUATION COMPLETE")
            print(f"{'='*50}")
            print(f"Tasks evaluated: {metrics['count']}")
            print(f"Exact match: {metrics['exact_match']:.2%}")
            print(f"Partial match: {metrics['partial_match']:.2%}")
            print(f"Syntax valid: {metrics['syntax_valid']:.2%}")
            print(f"Composite score: {metrics['composite_score']:.2%}")

        return metrics

    def get_failed_tasks(self, threshold: float = 0.5) -> List[Tuple[SWEBenchTask, EvaluationResult]]:
        """Get tasks where the model scored below threshold."""
        failed = []
        for task, result in zip(self.tasks, self.results):
            if result.composite_score < threshold:
                failed.append((task, result))
        return failed

    def reset(self):
        """Reset evaluator state for new evaluation run."""
        self.results.clear()


__all__ = [
    'SWEBenchEvaluator',
    'SWEBenchTask',
    'EvaluationResult',
    'EvaluationMode',
]

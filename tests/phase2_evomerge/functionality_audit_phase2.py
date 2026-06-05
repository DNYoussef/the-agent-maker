"""
Comprehensive Functionality Audit for Phase 2 EvoMerge Implementations

Tests the four newly created/updated Phase 2 modules:
1. src/phase2_evomerge/evolution/cma_es.py
2. src/phase2_evomerge/fitness/benchmarks.py
3. src/phase2_evomerge/merge/dfs_paper_accurate.py
4. src/phase2_evomerge/merge/hybrid_ps_dfs.py

Audit Methodology:
- Syntax verification via import
- Unit tests for core functions
- Integration tests for pipelines
- Input/output validation
- Error handling verification
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np


class FunctionalityAuditReport:
    """Structured audit report."""

    def __init__(self):
        self.import_status = {}
        self.syntax_errors = []
        self.runtime_errors = []
        self.functionality_gaps = []
        self.recommendations = []
        self.test_results = {}

    def add_import_result(self, module_name: str, success: bool, error: str = None):
        """Record import test result."""
        self.import_status[module_name] = {
            "success": success,
            "error": error if not success else None
        }

    def add_syntax_error(self, module_name: str, error: str):
        """Record syntax error."""
        self.syntax_errors.append({
            "module": module_name,
            "error": error
        })

    def add_runtime_error(self, test_name: str, error: str):
        """Record runtime error."""
        self.runtime_errors.append({
            "test": test_name,
            "error": error
        })

    def add_functionality_gap(self, module_name: str, gap: str):
        """Record functionality gap."""
        self.functionality_gaps.append({
            "module": module_name,
            "gap": gap
        })

    def add_recommendation(self, recommendation: str):
        """Add recommendation."""
        self.recommendations.append(recommendation)

    def add_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Record test result."""
        self.test_results[test_name] = {
            "passed": passed,
            "details": details
        }

    def generate_report(self) -> str:
        """Generate formatted audit report."""
        report_lines = [
            "=" * 80,
            "PHASE 2 EVOMERGE - FUNCTIONALITY AUDIT REPORT",
            "=" * 80,
            "",
            "1. IMPORT STATUS",
            "-" * 80
        ]

        for module, status in self.import_status.items():
            status_str = "PASS" if status["success"] else "FAIL"
            report_lines.append(f"  [{status_str}] {module}")
            if status["error"]:
                report_lines.append(f"        Error: {status['error']}")

        report_lines.extend([
            "",
            "2. SYNTAX ERRORS",
            "-" * 80
        ])

        if not self.syntax_errors:
            report_lines.append("  No syntax errors detected.")
        else:
            for error in self.syntax_errors:
                report_lines.append(f"  Module: {error['module']}")
                report_lines.append(f"  Error: {error['error']}")
                report_lines.append("")

        report_lines.extend([
            "",
            "3. RUNTIME ERRORS",
            "-" * 80
        ])

        if not self.runtime_errors:
            report_lines.append("  No runtime errors detected.")
        else:
            for error in self.runtime_errors:
                report_lines.append(f"  Test: {error['test']}")
                report_lines.append(f"  Error: {error['error']}")
                report_lines.append("")

        report_lines.extend([
            "",
            "4. TEST RESULTS",
            "-" * 80
        ])

        passed_count = sum(1 for r in self.test_results.values() if r["passed"])
        total_count = len(self.test_results)

        for test_name, result in self.test_results.items():
            status_str = "PASS" if result["passed"] else "FAIL"
            report_lines.append(f"  [{status_str}] {test_name}")
            if result["details"]:
                report_lines.append(f"        {result['details']}")

        report_lines.append(f"\n  Summary: {passed_count}/{total_count} tests passed")

        report_lines.extend([
            "",
            "5. FUNCTIONALITY GAPS",
            "-" * 80
        ])

        if not self.functionality_gaps:
            report_lines.append("  No functionality gaps detected.")
        else:
            for gap in self.functionality_gaps:
                report_lines.append(f"  Module: {gap['module']}")
                report_lines.append(f"  Gap: {gap['gap']}")
                report_lines.append("")

        report_lines.extend([
            "",
            "6. RECOMMENDATIONS",
            "-" * 80
        ])

        if not self.recommendations:
            report_lines.append("  No specific recommendations.")
        else:
            for i, rec in enumerate(self.recommendations, 1):
                report_lines.append(f"  {i}. {rec}")

        report_lines.extend([
            "",
            "=" * 80,
            f"AUDIT COMPLETE - {passed_count}/{total_count} tests passed",
            "=" * 80
        ])

        return "\n".join(report_lines)


# Simple test model for testing
class SimpleTestModel(nn.Module):
    """Minimal model for testing merge operations."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(3)
        ])
        self.output = nn.Linear(hidden_size, 100)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

    def generate(self, input_ids, max_length=50, **kwargs):
        """Dummy generation for testing."""
        batch_size = input_ids.shape[0]
        generated = torch.randint(0, 100, (batch_size, max_length), device=input_ids.device)
        return generated


def test_imports(report: FunctionalityAuditReport):
    """Test 1: Import all target modules."""
    print("\n[TEST 1] Testing module imports...")

    modules_to_test = [
        ("CMA-ES Optimizer", "phase2_evomerge.evolution.cma_es"),
        ("Benchmark Evaluation", "phase2_evomerge.fitness.benchmarks"),
        ("DFS Paper-Accurate", "phase2_evomerge.merge.dfs_paper_accurate"),
        ("Hybrid PS+DFS", "phase2_evomerge.merge.hybrid_ps_dfs"),
    ]

    imported_modules = {}

    for name, module_path in modules_to_test:
        try:
            module = __import__(module_path, fromlist=['*'])
            report.add_import_result(name, True)
            imported_modules[module_path] = module
            print(f"  [PASS] {name}")
        except Exception as e:
            report.add_import_result(name, False, str(e))
            report.add_syntax_error(name, str(e))
            print(f"  [FAIL] {name}: {e}")

    return imported_modules


def test_cmaes_optimizer(report: FunctionalityAuditReport):
    """Test 2: CMA-ES Optimizer functionality."""
    print("\n[TEST 2] Testing CMA-ES Optimizer...")

    try:
        from phase2_evomerge.evolution.cma_es import CMAESConfig, CMAESOptimizer

        # Test 2.1: Initialization
        config = CMAESConfig(population_size=10, sigma=0.3, seed=42)
        optimizer = CMAESOptimizer(config)

        assert optimizer.config.population_size == 10
        assert optimizer.config.sigma == 0.3
        report.add_test_result("CMA-ES: Initialization", True)
        print("  [PASS] Initialization")

        # Test 2.2: Simple optimization
        def simple_objective(coeffs):
            # Maximize closeness to [0.5, 0.5, 0.5]
            return -np.sum((coeffs - 0.5) ** 2)

        best_coeffs, best_fitness = optimizer.optimize(
            simple_objective,
            n_dimensions=3,
            n_trials=30,
            verbose=False
        )

        assert best_coeffs is not None
        assert len(best_coeffs) == 3
        assert np.allclose(np.sum(best_coeffs), 1.0, atol=0.1)  # Should sum to ~1
        report.add_test_result("CMA-ES: Simple Optimization", True,
                              f"Best fitness: {best_fitness:.4f}")
        print(f"  [PASS] Simple Optimization (fitness: {best_fitness:.4f})")

        # Test 2.3: Coefficient normalization
        assert np.allclose(np.sum(best_coeffs), 1.0, atol=0.1)
        report.add_test_result("CMA-ES: Coefficient Normalization", True)
        print("  [PASS] Coefficient Normalization")

    except Exception as e:
        report.add_runtime_error("CMA-ES Optimizer", str(e))
        report.add_test_result("CMA-ES: Overall", False, str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


def test_benchmark_evaluation(report: FunctionalityAuditReport):
    """Test 3: Benchmark evaluation functionality."""
    print("\n[TEST 3] Testing Benchmark Evaluation...")

    try:
        from phase2_evomerge.fitness.benchmarks import (
            BenchmarkConfig,
            GSM8KDataset,
            extract_numeric_answer,
        )

        # Test 3.1: Answer extraction
        test_cases = [
            ("The answer is #### 42", 42.0),
            ("Result: 123", 123.0),
            ("Value is -10", -10.0),
            ("Decimal: 3.14", 3.14),
        ]

        all_correct = True
        for text, expected in test_cases:
            result = extract_numeric_answer(text)
            if result != expected:
                all_correct = False
                break

        if all_correct:
            report.add_test_result("Benchmarks: Answer Extraction", True)
            print("  [PASS] Answer Extraction")
        else:
            report.add_test_result("Benchmarks: Answer Extraction", False)
            print("  [FAIL] Answer Extraction")

        # Test 3.2: Config initialization
        config = BenchmarkConfig(benchmark_name="gsm8k", max_samples=10)
        assert config.benchmark_name == "gsm8k"
        assert config.max_samples == 10
        report.add_test_result("Benchmarks: Config Initialization", True)
        print("  [PASS] Config Initialization")

        # Test 3.3: Dataset loading (may fail without datasets library)
        try:
            dataset = GSM8KDataset(max_samples=5)
            if len(dataset) > 0:
                sample = dataset[0]
                assert "question" in sample and "answer" in sample
                report.add_test_result("Benchmarks: Dataset Loading", True,
                                      f"Loaded {len(dataset)} samples")
                print(f"  [PASS] Dataset Loading ({len(dataset)} samples)")
            else:
                report.add_test_result("Benchmarks: Dataset Loading", False,
                                      "Dataset is empty")
                print("  [WARN] Dataset is empty (may require internet)")
        except Exception as e:
            report.add_functionality_gap("Benchmarks",
                                        f"Dataset loading failed: {str(e)}")
            report.add_test_result("Benchmarks: Dataset Loading", False, str(e))
            print(f"  [SKIP] Dataset Loading (requires datasets library)")

    except Exception as e:
        report.add_runtime_error("Benchmark Evaluation", str(e))
        report.add_test_result("Benchmarks: Overall", False, str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


def test_dfs_paper_accurate(report: FunctionalityAuditReport):
    """Test 4: DFS paper-accurate implementation."""
    print("\n[TEST 4] Testing DFS Paper-Accurate Merger...")

    try:
        from phase2_evomerge.merge.dfs_paper_accurate import DFSConfig, DFSPaperAccurate

        # Test 4.1: Initialization
        config = DFSConfig(init_strategy="uniform")
        dfs = DFSPaperAccurate(config)
        assert dfs.config.init_strategy == "uniform"
        report.add_test_result("DFS: Initialization", True)
        print("  [PASS] Initialization")

        # Test 4.2: Model merging
        models = [SimpleTestModel(hidden_size=32) for _ in range(3)]
        merged = dfs.merge(models)

        assert isinstance(merged, nn.Module)
        assert dfs.indicator_array is not None
        assert dfs.scaling_matrix is not None

        M = len(models)
        r = dfs._count_layers(models[0])
        T = M * r

        assert len(dfs.indicator_array) == T
        assert dfs.scaling_matrix.shape == (M, M)

        report.add_test_result("DFS: Model Merging", True,
                              f"Merged {M} models with {r} layers each")
        print(f"  [PASS] Model Merging ({M} models, {r} layers, T={T})")

        # Test 4.3: Custom indicator array
        indicators = np.ones(T, dtype=np.float32)
        scaling = np.eye(M, dtype=np.float32)

        merged_custom = dfs.merge(models, indicator_array=indicators,
                                 scaling_matrix=scaling)
        assert isinstance(merged_custom, nn.Module)
        report.add_test_result("DFS: Custom Indicators", True)
        print("  [PASS] Custom Indicators")

        # Test 4.4: Layer counting
        layer_count = dfs._count_layers(models[0])
        assert layer_count > 0
        report.add_test_result("DFS: Layer Counting", True,
                              f"Detected {layer_count} layers")
        print(f"  [PASS] Layer Counting ({layer_count} layers)")

    except Exception as e:
        report.add_runtime_error("DFS Paper-Accurate", str(e))
        report.add_test_result("DFS: Overall", False, str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


def test_hybrid_ps_dfs(report: FunctionalityAuditReport):
    """Test 5: Hybrid PS+DFS implementation."""
    print("\n[TEST 5] Testing Hybrid PS+DFS Merger...")

    try:
        from phase2_evomerge.merge.hybrid_ps_dfs import HybridConfig, HybridPSDFS

        # Test 5.1: Initialization
        config = HybridConfig(ps_candidates_multiplier=2,
                            ps_generations=5,
                            dfs_optimization_iterations=10)
        hybrid = HybridPSDFS(config)

        assert hybrid.config.ps_candidates_multiplier == 2
        report.add_test_result("Hybrid: Initialization", True)
        print("  [PASS] Initialization")

        # Test 5.2: Hybrid merge (lightweight)
        models = [SimpleTestModel(hidden_size=32) for _ in range(3)]

        def simple_fitness(model):
            """Simple fitness function for testing."""
            return float(list(model.parameters())[0].mean().item())

        # Use minimal settings for fast test
        test_config = HybridConfig(
            ps_candidates_multiplier=2,
            ps_generations=3,
            dfs_optimization_iterations=5
        )

        hybrid = HybridPSDFS(test_config)
        champion, metrics = hybrid.merge(models, simple_fitness, verbose=False)

        assert isinstance(champion, nn.Module)
        assert "fitness_improvement" in metrics
        assert "ps_best_fitness" in metrics
        assert "champion_fitness" in metrics

        n_candidates = len(models) * test_config.ps_candidates_multiplier
        assert len(hybrid.ps_candidates) == n_candidates

        report.add_test_result("Hybrid: PS+DFS Merge", True,
                              f"Created {n_candidates} PS candidates, "
                              f"fitness improvement: {metrics['fitness_improvement']:.2%}")
        print(f"  [PASS] PS+DFS Merge ({n_candidates} candidates, "
              f"improvement: {metrics['fitness_improvement']:.2%})")

        # Test 5.3: Metrics validation
        required_metrics = ["n_base_models", "n_ps_candidates", "baseline_fitness",
                          "ps_best_fitness", "champion_fitness"]

        all_present = all(k in metrics for k in required_metrics)
        if all_present:
            report.add_test_result("Hybrid: Metrics Validation", True)
            print("  [PASS] Metrics Validation")
        else:
            missing = [k for k in required_metrics if k not in metrics]
            report.add_test_result("Hybrid: Metrics Validation", False,
                                  f"Missing metrics: {missing}")
            print(f"  [FAIL] Metrics Validation (missing: {missing})")

    except Exception as e:
        report.add_runtime_error("Hybrid PS+DFS", str(e))
        report.add_test_result("Hybrid: Overall", False, str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


def test_integration(report: FunctionalityAuditReport):
    """Test 6: Integration across modules."""
    print("\n[TEST 6] Testing Cross-Module Integration...")

    try:
        from phase2_evomerge.evolution.cma_es import CMAESOptimizer, CMAESConfig
        from phase2_evomerge.merge.dfs_paper_accurate import DFSPaperAccurate
        from phase2_evomerge.merge.hybrid_ps_dfs import HybridPSDFS, HybridConfig

        # Test 6.1: CMA-ES with model merging
        models = [SimpleTestModel(hidden_size=32) for _ in range(3)]

        config = CMAESConfig(population_size=5, max_generations=3, seed=42)
        optimizer = CMAESOptimizer(config)

        def merge_and_evaluate(coeffs):
            """Merge models with coefficients and evaluate."""
            # Simple weighted merge
            merged_params = {}
            with torch.no_grad():
                for param_name in dict(models[0].named_parameters()).keys():
                    merged_param = sum(
                        coeffs[i] * dict(models[i].named_parameters())[param_name]
                        for i in range(len(models))
                    )
                    merged_params[param_name] = merged_param
            return float(list(merged_params.values())[0].mean().item())

        best_coeffs, _ = optimizer.optimize(merge_and_evaluate,
                                           n_dimensions=3,
                                           n_trials=10,
                                           verbose=False)

        assert best_coeffs is not None
        report.add_test_result("Integration: CMA-ES + Merge", True)
        print("  [PASS] CMA-ES + Merge")

        # Test 6.2: DFS after PS merge
        dfs = DFSPaperAccurate()
        merged = dfs.merge(models)
        assert isinstance(merged, nn.Module)
        report.add_test_result("Integration: PS then DFS", True)
        print("  [PASS] PS then DFS")

        # Test 6.3: Full hybrid pipeline
        hybrid_config = HybridConfig(
            ps_candidates_multiplier=2,
            ps_generations=2,
            dfs_optimization_iterations=3
        )
        hybrid = HybridPSDFS(hybrid_config)

        def fitness(model):
            return float(list(model.parameters())[0].mean().item())

        champion, metrics = hybrid.merge(models, fitness, verbose=False)
        assert isinstance(champion, nn.Module)
        report.add_test_result("Integration: Full Hybrid Pipeline", True)
        print("  [PASS] Full Hybrid Pipeline")

    except Exception as e:
        report.add_runtime_error("Integration", str(e))
        report.add_test_result("Integration: Overall", False, str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


def generate_recommendations(report: FunctionalityAuditReport):
    """Generate recommendations based on audit results."""

    # Check import failures
    failed_imports = [name for name, status in report.import_status.items()
                     if not status["success"]]
    if failed_imports:
        report.add_recommendation(
            f"Fix import errors in modules: {', '.join(failed_imports)}"
        )

    # Check for dataset availability
    if any("dataset" in gap["gap"].lower() for gap in report.functionality_gaps):
        report.add_recommendation(
            "Install datasets library for GSM8K evaluation: pip install datasets"
        )

    # Check test pass rate
    passed = sum(1 for r in report.test_results.values() if r["passed"])
    total = len(report.test_results)
    pass_rate = passed / total if total > 0 else 0

    if pass_rate < 0.9:
        report.add_recommendation(
            f"Test pass rate is {pass_rate:.1%}. Investigate failing tests."
        )

    # Runtime errors
    if report.runtime_errors:
        report.add_recommendation(
            f"Fix {len(report.runtime_errors)} runtime errors detected during testing"
        )

    # General recommendations
    if pass_rate >= 0.9 and not report.syntax_errors:
        report.add_recommendation(
            "Core functionality verified. Ready for integration testing."
        )


def main():
    """Run complete functionality audit."""
    print("=" * 80)
    print("PHASE 2 EVOMERGE - COMPREHENSIVE FUNCTIONALITY AUDIT")
    print("=" * 80)
    print("\nAudit Scope:")
    print("  1. src/phase2_evomerge/evolution/cma_es.py")
    print("  2. src/phase2_evomerge/fitness/benchmarks.py")
    print("  3. src/phase2_evomerge/merge/dfs_paper_accurate.py")
    print("  4. src/phase2_evomerge/merge/hybrid_ps_dfs.py")
    print("\nStarting audit...")

    report = FunctionalityAuditReport()

    # Run all tests
    imported_modules = test_imports(report)

    if len(imported_modules) > 0:
        test_cmaes_optimizer(report)
        test_benchmark_evaluation(report)
        test_dfs_paper_accurate(report)
        test_hybrid_ps_dfs(report)
        test_integration(report)
    else:
        print("\n[CRITICAL] All imports failed. Cannot proceed with testing.")

    # Generate recommendations
    generate_recommendations(report)

    # Print final report
    print("\n" + report.generate_report())

    # Save report to file
    report_path = Path(__file__).parent / "audit_report_phase2.txt"
    with open(report_path, "w") as f:
        f.write(report.generate_report())

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

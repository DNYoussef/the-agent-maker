"""
Phase 3 Quiet-STaR Implementation Audit

Tests newly created Phase 3 implementations:
1. src/phase3_quietstar/architecture/parallel_thought_generator.py
2. src/phase3_quietstar/config_extensions.py
3. src/phase3_quietstar/examples/parallel_sampling_example.py

Author: Functionality Audit Agent
Date: 2025-12-02
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import torch.nn as nn


class AuditReport:
    """Structured audit reporting."""

    def __init__(self):
        self.results = {
            "imports": {},
            "instantiation": {},
            "methods": {},
            "integration": {},
            "bugs": [],
            "fixes": [],
        }

    def add_import_test(self, module_name, success, error=None):
        self.results["imports"][module_name] = {
            "status": "PASS" if success else "FAIL",
            "error": str(error) if error else None,
        }

    def add_instantiation_test(self, class_name, success, error=None):
        self.results["instantiation"][class_name] = {
            "status": "PASS" if success else "FAIL",
            "error": str(error) if error else None,
        }

    def add_method_test(self, method_name, success, result=None, error=None):
        self.results["methods"][method_name] = {
            "status": "PASS" if success else "FAIL",
            "result": result,
            "error": str(error) if error else None,
        }

    def add_integration_test(self, test_name, success, details=None):
        self.results["integration"][test_name] = {
            "status": "PASS" if success else "FAIL",
            "details": details,
        }

    def add_bug(self, bug_description):
        self.results["bugs"].append(bug_description)

    def add_fix(self, fix_description):
        self.results["fixes"].append(fix_description)

    def print_report(self):
        """Print formatted audit report."""
        print("\n" + "=" * 80)
        print("PHASE 3 QUIET-STAR FUNCTIONALITY AUDIT REPORT")
        print("=" * 80)

        # 1. Import Status
        print("\n1. IMPORT STATUS:")
        print("-" * 80)
        for module, result in self.results["imports"].items():
            status_symbol = "[PASS]" if result["status"] == "PASS" else "[FAIL]"
            print(f"  {status_symbol} {module}")
            if result["error"]:
                print(f"      Error: {result['error']}")

        # 2. Instantiation Status
        print("\n2. INSTANTIATION STATUS:")
        print("-" * 80)
        for cls, result in self.results["instantiation"].items():
            status_symbol = "[PASS]" if result["status"] == "PASS" else "[FAIL]"
            print(f"  {status_symbol} {cls}")
            if result["error"]:
                print(f"      Error: {result['error']}")

        # 3. Method Tests
        print("\n3. METHOD TESTS:")
        print("-" * 80)
        for method, result in self.results["methods"].items():
            status_symbol = "[PASS]" if result["status"] == "PASS" else "[FAIL]"
            print(f"  {status_symbol} {method}")
            if result["result"]:
                print(f"      Result: {result['result']}")
            if result["error"]:
                print(f"      Error: {result['error']}")

        # 4. Integration Tests
        print("\n4. COMPATIBILITY:")
        print("-" * 80)
        for test, result in self.results["integration"].items():
            status_symbol = "[PASS]" if result["status"] == "PASS" else "[FAIL]"
            print(f"  {status_symbol} {test}")
            if result["details"]:
                print(f"      Details: {result['details']}")

        # 5. Bugs Found
        print("\n5. BUGS FOUND:")
        print("-" * 80)
        if not self.results["bugs"]:
            print("  No bugs detected!")
        else:
            for i, bug in enumerate(self.results["bugs"], 1):
                print(f"  {i}. {bug}")

        # 6. Fixes Needed
        print("\n6. FIXES NEEDED:")
        print("-" * 80)
        if not self.results["fixes"]:
            print("  No fixes required!")
        else:
            for i, fix in enumerate(self.results["fixes"], 1):
                print(f"  {i}. {fix}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY:")
        total_imports = len(self.results["imports"])
        passed_imports = sum(1 for r in self.results["imports"].values() if r["status"] == "PASS")
        total_methods = len(self.results["methods"])
        passed_methods = sum(1 for r in self.results["methods"].values() if r["status"] == "PASS")

        print(f"  Imports: {passed_imports}/{total_imports} passed")
        print(f"  Methods: {passed_methods}/{total_methods} passed")
        print(f"  Bugs: {len(self.results['bugs'])} found")
        print(f"  Overall: {'PASS' if len(self.results['bugs']) == 0 else 'FAIL'}")
        print("=" * 80 + "\n")


def create_mock_model():
    """Create mock model for testing."""

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("obj", (object,), {"hidden_size": 512, "vocab_size": 50000})()
            self.embedding = nn.Embedding(50000, 512)
            self.lm_head = nn.Linear(512, 50000)

        def forward(self, input_ids, attention_mask=None):
            embeddings = self.embedding(input_ids)
            logits = self.lm_head(embeddings)
            return type("obj", (object,), {"logits": logits, "last_hidden_state": embeddings})()

    return MockModel()


@pytest.fixture
def report() -> AuditReport:
    return AuditReport()


def test_imports(report):
    """Test all imports."""
    print("\nTesting imports...")

    # Test 1: parallel_thought_generator
    try:
        from src.phase3_quietstar.architecture.parallel_thought_generator import (
            ParallelThoughtGenerator,
        )

        report.add_import_test("parallel_thought_generator", True)
    except Exception as e:
        report.add_import_test("parallel_thought_generator", False, e)
        report.add_bug(f"Cannot import ParallelThoughtGenerator: {e}")

    # Test 2: config_extensions
    try:
        from src.phase3_quietstar.config_extensions import (
            MetaTokenConfig,
            ParallelSamplingConfig,
            TeacherForcingConfig,
            create_complete_rl_config,
            extend_rl_config,
        )

        report.add_import_test("config_extensions", True)
    except Exception as e:
        report.add_import_test("config_extensions", False, e)
        report.add_bug(f"Cannot import from config_extensions: {e}")

    # Test 3: dataclasses
    try:
        from src.phase3_quietstar.architecture.dataclasses import ThoughtOutput

        report.add_import_test("dataclasses", True)
    except Exception as e:
        report.add_import_test("dataclasses", False, e)
        report.add_bug(f"Cannot import ThoughtOutput: {e}")

    # Test 4: existing thought_generator
    try:
        from src.phase3_quietstar.architecture.thought_generator import ThoughtGenerator

        report.add_import_test("thought_generator (existing)", True)
    except Exception as e:
        report.add_import_test("thought_generator (existing)", False, e)
        report.add_bug(f"Cannot import ThoughtGenerator: {e}")

    failures = [
        name for name, result in report.results["imports"].items() if result["status"] != "PASS"
    ]
    assert not failures, f"import failures: {failures}; bugs={report.results['bugs']}"


def test_instantiation(report):
    """Test class instantiation."""
    print("\nTesting instantiation...")

    # Create mock model
    try:
        mock_model = create_mock_model()
        report.add_instantiation_test("MockModel", True)
    except Exception as e:
        report.add_instantiation_test("MockModel", False, e)
        report.add_bug(f"Cannot create mock model: {e}")
        raise AssertionError(f"Cannot create mock model: {e}") from e

    # Test ParallelThoughtGenerator
    try:
        from src.phase3_quietstar.architecture.parallel_thought_generator import (
            ParallelThoughtGenerator,
        )

        generator = ParallelThoughtGenerator(mock_model, num_thoughts=4)
        report.add_instantiation_test("ParallelThoughtGenerator", True)
    except Exception as e:
        report.add_instantiation_test("ParallelThoughtGenerator", False, e)
        report.add_bug(f"Cannot instantiate ParallelThoughtGenerator: {e}")
        report.add_fix("Check __init__ method signature and base class compatibility")

    # Test config classes
    try:
        from src.phase3_quietstar.config_extensions import (
            MetaTokenConfig,
            ParallelSamplingConfig,
            TeacherForcingConfig,
        )

        parallel_cfg = ParallelSamplingConfig()
        teacher_cfg = TeacherForcingConfig()
        meta_cfg = MetaTokenConfig()
        report.add_instantiation_test("Config Classes", True)
    except Exception as e:
        report.add_instantiation_test("Config Classes", False, e)
        report.add_bug(f"Cannot instantiate config classes: {e}")

    failures = [
        name
        for name, result in report.results["instantiation"].items()
        if result["status"] != "PASS"
    ]
    assert not failures, f"instantiation failures: {failures}; bugs={report.results['bugs']}"


def test_methods(report):
    """Test key methods."""
    print("\nTesting methods...")

    # Setup
    try:
        from src.phase3_quietstar.architecture.parallel_thought_generator import (
            ParallelThoughtGenerator,
        )

        mock_model = create_mock_model()
        generator = ParallelThoughtGenerator(
            mock_model, num_thoughts=4, max_length=10, min_length=5
        )
    except Exception as e:
        report.add_method_test("setup", False, error=e)
        raise AssertionError(f"method setup failed: {e}") from e

    # Test 1: forward()
    try:
        input_ids = torch.randint(0, 50000, (2, 20))  # batch=2, seq_len=20
        output = generator(input_ids, position=10)

        # Validate output
        assert hasattr(output, "thoughts"), "Output missing 'thoughts' attribute"
        assert hasattr(output, "thought_ids"), "Output missing 'thought_ids' attribute"
        assert hasattr(output, "log_probs"), "Output missing 'log_probs' attribute"

        # Check shapes
        batch_size = 2
        num_thoughts = 4
        assert output.thoughts.dim() == 4, f"thoughts should be 4D, got {output.thoughts.dim()}D"
        assert output.thoughts.size(0) == batch_size, f"batch dimension should be {batch_size}"
        assert (
            output.thoughts.size(1) == num_thoughts
        ), f"num_thoughts dimension should be {num_thoughts}"

        report.add_method_test(
            "forward()",
            True,
            result=f"Output shape: {output.thoughts.shape}, log_probs: {output.log_probs.shape}",
        )
    except Exception as e:
        report.add_method_test("forward()", False, error=e)
        report.add_bug(f"forward() method failed: {e}")
        report.add_fix(
            "Check forward() implementation, especially tensor shapes and attention mask handling"
        )

    # Test 2: _create_diagonal_attention_mask()
    try:
        mask = generator._create_diagonal_attention_mask(
            batch_size=2, num_thoughts=4, seq_len=30, position=10, device="cpu"
        )

        # Validate mask
        expected_shape = (2 * 4, 30, 30)  # (batch * num_thoughts, seq_len, seq_len)
        assert mask.shape == expected_shape, f"Expected shape {expected_shape}, got {mask.shape}"

        # Check mask values
        unique_values = torch.unique(mask)
        has_zero = 0.0 in unique_values
        has_neginf = float("-inf") in unique_values or torch.isinf(mask).any()
        assert has_zero and has_neginf, "Mask should have 0.0 (attend) and -inf (mask) values"

        report.add_method_test(
            "_create_diagonal_attention_mask()",
            True,
            result=f"Mask shape: {mask.shape}, unique values: {unique_values.tolist()[:5]}...",
        )
    except Exception as e:
        report.add_method_test("_create_diagonal_attention_mask()", False, error=e)
        report.add_bug(f"_create_diagonal_attention_mask() failed: {e}")
        report.add_fix(
            "Review diagonal mask algorithm - check shared context and diagonal blocking logic"
        )

    # Test 3: _nucleus_sampling()
    try:
        logits = torch.randn(2, 50000)  # batch=2, vocab_size=50000
        probs = generator._nucleus_sampling(logits)

        # Validate
        assert probs.shape == logits.shape, f"Probs shape mismatch: {probs.shape} vs {logits.shape}"
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5), "Probs don't sum to 1"
        assert (probs >= 0).all(), "Probs contain negative values"

        report.add_method_test(
            "_nucleus_sampling()",
            True,
            result=f"Probs shape: {probs.shape}, sum: {probs.sum(dim=-1).tolist()}",
        )
    except Exception as e:
        report.add_method_test("_nucleus_sampling()", False, error=e)
        report.add_bug(f"_nucleus_sampling() failed: {e}")
        report.add_fix("Check nucleus sampling implementation - verify sorting and renormalization")

    # Test 4: compute_teacher_forced_loss()
    try:
        input_ids = torch.randint(0, 50000, (2, 20))
        thought_ids = [[100, 200, 300], [400, 500, 600], [700, 800, 900], [1000, 1100, 1200]]
        labels = torch.randint(0, 50000, (2, 30))

        loss = generator.compute_teacher_forced_loss(
            input_ids=input_ids, thought_ids=thought_ids, labels=labels, n_true=4
        )

        # Validate
        assert loss.dim() == 0, f"Loss should be scalar, got {loss.dim()}D"
        assert loss > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is inf"

        report.add_method_test(
            "compute_teacher_forced_loss()", True, result=f"Loss value: {loss.item():.4f}"
        )
    except Exception as e:
        report.add_method_test("compute_teacher_forced_loss()", False, error=e)
        report.add_bug(f"compute_teacher_forced_loss() failed: {e}")
        report.add_fix(
            "Review teacher forcing implementation - check label alignment and loss computation"
        )

    failures = [
        name for name, result in report.results["methods"].items() if result["status"] != "PASS"
    ]
    assert not failures, f"method failures: {failures}; bugs={report.results['bugs']}"


def test_integration(report):
    """Test integration with existing code."""
    print("\nTesting integration...")

    # Test 1: Compatibility with ThoughtGenerator
    try:
        from src.phase3_quietstar.architecture.parallel_thought_generator import (
            ParallelThoughtGenerator,
        )
        from src.phase3_quietstar.architecture.thought_generator import ThoughtGenerator

        mock_model = create_mock_model()

        # Create both generators
        parallel_gen = ParallelThoughtGenerator(
            mock_model, num_thoughts=4, max_length=10, min_length=5
        )
        sequential_gen = ThoughtGenerator(mock_model, num_thoughts=4, max_length=10, min_length=5)

        # Test same interface
        input_ids = torch.randint(0, 50000, (2, 20))

        output_parallel = parallel_gen(input_ids, position=10)
        output_sequential = sequential_gen(input_ids, position=10)

        # Check compatibility
        assert hasattr(output_parallel, "thoughts"), "Parallel output missing 'thoughts'"
        assert hasattr(output_sequential, "thoughts"), "Sequential output missing 'thoughts'"

        assert (
            output_parallel.thoughts.dim() == output_sequential.thoughts.dim()
        ), f"Dimension mismatch: parallel {output_parallel.thoughts.dim()}D vs sequential {output_sequential.thoughts.dim()}D"

        report.add_integration_test(
            "ThoughtGenerator compatibility",
            True,
            details="Both generators produce compatible ThoughtOutput",
        )
    except Exception as e:
        report.add_integration_test("ThoughtGenerator compatibility", False, details=str(e))
        report.add_bug(f"Integration with ThoughtGenerator failed: {e}")
        report.add_fix("Ensure ParallelThoughtGenerator has same interface as ThoughtGenerator")

    # Test 2: Config extensions integration
    try:
        from src.phase3_quietstar.config import QuietSTaRRLConfig
        from src.phase3_quietstar.config_extensions import (
            create_complete_rl_config,
            extend_rl_config,
        )

        # Test extend_rl_config
        base_config = QuietSTaRRLConfig()
        extended_config = extend_rl_config(base_config)

        assert hasattr(extended_config, "meta_token_grad_scale"), "Missing meta_token_grad_scale"
        assert hasattr(extended_config, "n_true"), "Missing n_true"
        assert hasattr(
            extended_config, "use_parallel_generation"
        ), "Missing use_parallel_generation"

        # Test create_complete_rl_config
        complete_config = create_complete_rl_config()
        assert "base" in complete_config, "Missing 'base' in config"
        assert "parallel_sampling" in complete_config, "Missing 'parallel_sampling'"
        assert "teacher_forcing" in complete_config, "Missing 'teacher_forcing'"
        assert "meta_token" in complete_config, "Missing 'meta_token'"

        report.add_integration_test(
            "Config extensions integration",
            True,
            details="Config extensions properly extend base config",
        )
    except Exception as e:
        report.add_integration_test("Config extensions integration", False, details=str(e))
        report.add_bug(f"Config integration failed: {e}")
        report.add_fix("Check config_extensions functions work with existing QuietSTaRRLConfig")

    failures = [
        name for name, result in report.results["integration"].items() if result["status"] != "PASS"
    ]
    assert not failures, f"integration failures: {failures}; bugs={report.results['bugs']}"


def main():
    """Run full audit."""
    print("\n" + "=" * 80)
    print("Starting Phase 3 Quiet-STaR Implementation Audit...")
    print("=" * 80)

    report = AuditReport()

    try:
        test_imports(report)
        test_instantiation(report)
        test_methods(report)
        test_integration(report)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        report.add_bug(f"Fatal error during audit: {e}")

    report.print_report()


if __name__ == "__main__":
    main()

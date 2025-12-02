"""
ADAS Module Import Verification

Quick test to verify all imports work correctly after refactoring.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test all module imports."""
    print("Testing ADAS module imports...")

    try:
        # Test main imports
        from adas import ADASConfig, ADASOptimizer, ADASResult, Individual

        print("  Main imports: OK")

        # Test sub-module imports
        from adas.config import ADASConfig as Config
        from adas.evaluation import evaluate_individual
        from adas.nsga2 import assign_ranks, calculate_crowding_distance
        from adas.operators import crossover, mutate
        from adas.optimizer import ADASOptimizer as Optimizer

        print("  Sub-module imports: OK")

        # Test class instantiation
        config = ADASConfig(population_size=10)
        optimizer = ADASOptimizer(config)
        print("  Class instantiation: OK")

        # Test data structures
        individual = Individual(routing_weights=[0.5, 0.5], expert_configs={}, fitness_scores={})
        print("  Data structures: OK")

        print("\nAll imports successful!")
        print(f"  ADASConfig: {ADASConfig}")
        print(f"  Individual: {Individual}")
        print(f"  ADASResult: {ADASResult}")
        print(f"  ADASOptimizer: {ADASOptimizer}")

        return True

    except ImportError as e:
        print(f"  Import failed: {e}")
        return False
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return False


def test_module_structure():
    """Verify module structure."""
    print("\nVerifying module structure...")

    import adas

    expected_exports = [
        "ADASOptimizer",
        "ADASConfig",
        "ADASResult",
        "Individual",
        "assign_ranks",
        "calculate_crowding_distance",
        "crossover",
        "mutate",
        "evaluate_individual",
    ]

    for export in expected_exports:
        if hasattr(adas, export):
            print(f"  {export}: Found")
        else:
            print(f"  {export}: MISSING")
            return False

    print("\nModule structure verified!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ADAS Module Refactoring Verification")
    print("=" * 60)
    print()

    success = test_imports()
    if success:
        success = test_module_structure()

    print()
    print("=" * 60)
    if success:
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION FAILED")
    print("=" * 60)

    sys.exit(0 if success else 1)

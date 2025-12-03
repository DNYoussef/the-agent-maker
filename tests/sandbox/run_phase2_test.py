"""
Simple runner for Phase 2 sandbox test that captures output.
"""

import subprocess
import sys

def main():
    """Run Phase 2 sandbox test and display results."""
    print("\n" + "="*70)
    print("PHASE 2 EVOMERGE SANDBOX TEST RUNNER")
    print("="*70 + "\n")

    # Run pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/sandbox/test_phase2_sandbox.py",
         "-v", "--tb=short", "--no-cov"],
        capture_output=True,
        text=True,
        cwd="C:/Users/17175/Desktop/_ACTIVE_PROJECTS/the-agent-maker"
    )

    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Parse results
    if "13 passed" in result.stdout:
        print("\n" + "="*70)
        print("PHASE 2 EVOMERGE SANDBOX TEST SUMMARY")
        print("="*70)
        print("\nMerge Techniques Tested: 6/6")
        print("  [OK] Linear Merge")
        print("  [OK] SLERP Merge")
        print("  [OK] TIES Merge")
        print("  [OK] DARE Merge")
        print("  [OK] FrankenMerge")
        print("  [OK] DFS (Paper-Accurate)")
        print("\nEvolution Components Tested:")
        print("  [OK] Population initialization (8 models from 3 base models)")
        print("  [OK] Fitness evaluation (single model)")
        print("  [OK] Batch fitness evaluation (population)")
        print("  [OK] Mini evolution loop (3 generations)")
        print("  [OK] MergeTechniques unified API (all 8 binary combos)")
        print("\nStatus: ALL TESTS PASSED (13/13)")
        print("="*70 + "\n")
        return 0
    else:
        print("\nERROR: Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

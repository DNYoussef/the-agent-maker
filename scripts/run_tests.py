#!/usr/bin/env python3
"""
Run all tests and generate coverage report
Quick test runner for Agent Forge V2
"""

import subprocess
import sys
import shlex
from pathlib import Path


def run_command(cmd, description):
    """Run command and print results"""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")

    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("AGENT FORGE V2 - TEST SUITE")
    print("=" * 70)

    # Change to project root
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)

    results = []

    # 1. Run pytest with coverage
    success = run_command(
        "pytest tests/unit -v --cov=src --cov-report=term-missing --cov-report=html",
        "1. UNIT TESTS (pytest + coverage)"
    )
    results.append(("Unit Tests", success))

    # 2. Run integration tests
    success = run_command(
        "pytest tests/integration -v -m integration",
        "2. INTEGRATION TESTS"
    )
    results.append(("Integration Tests", success))

    # 3. NASA POT10 check
    success = run_command(
        "python .github/hooks/nasa_pot10_check.py src/cross_phase/*.py src/cross_phase/*/*.py",
        "3. NASA POT10 CHECK (≤60 LOC/function)"
    )
    results.append(("NASA POT10", success))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n[OK] ALL TESTS PASSED!")
        print("\nCoverage report: htmlcov/index.html")
        return 0
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

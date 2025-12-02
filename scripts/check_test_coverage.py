#!/usr/bin/env python3
"""
Test Coverage Checker for Agent Forge V2

Ensures new code has corresponding test files
Not a replacement for pytest-cov, but a simple pre-commit check

Usage:
    python scripts/check_test_coverage.py [files...]
    pre-commit hook: automatic on commit

Exit codes:
    0 - All files have corresponding tests or are exempt
    1 - One or more files missing tests
"""

import sys
from pathlib import Path
from typing import List, Set
import argparse


# Files/patterns exempt from requiring tests
EXEMPT_PATTERNS = [
    '__init__.py',
    '__main__.py',
    'setup.py',
    'conftest.py',
]

EXEMPT_DIRECTORIES = [
    'scripts',
    'docs',
    'examples',
    'migrations',
]


def is_exempt(filepath: Path) -> bool:
    """Check if file is exempt from test coverage requirements"""
    # Check filename patterns
    if filepath.name in EXEMPT_PATTERNS:
        return True

    # Check if in exempt directory
    for part in filepath.parts:
        if part in EXEMPT_DIRECTORIES:
            return True

    # CLI scripts (single file applications)
    if filepath.stem.startswith('cli_'):
        return True

    return False


def find_test_file(source_file: Path) -> List[Path]:
    """
    Find corresponding test file(s) for a source file

    Searches for:
    1. tests/test_{filename}.py (standard pattern)
    2. tests/{module}/test_{filename}.py (nested pattern)
    3. {module}/tests/test_{filename}.py (adjacent pattern)

    Returns:
        List of potential test file paths (may not exist)
    """
    test_files = []

    # Extract module path (e.g., src/phase1/model.py -> phase1)
    if 'src' in source_file.parts:
        src_index = source_file.parts.index('src')
        relative_parts = source_file.parts[src_index + 1:]
    else:
        relative_parts = source_file.parts

    # Remove filename, keep module path
    module_parts = relative_parts[:-1]
    filename = source_file.stem

    # Pattern 1: tests/test_{filename}.py
    test_files.append(Path('tests') / f'test_{filename}.py')

    # Pattern 2: tests/{module}/test_{filename}.py
    if module_parts:
        test_files.append(Path('tests') / Path(*module_parts) / f'test_{filename}.py')

    # Pattern 3: src/{module}/tests/test_{filename}.py
    if 'src' in source_file.parts:
        src_path = Path('src')
        if module_parts:
            src_path = src_path / Path(*module_parts)
        test_files.append(src_path / 'tests' / f'test_{filename}.py')

    return test_files


def check_coverage(filepaths: List[Path], verbose: bool = False) -> int:
    """
    Check if source files have corresponding tests

    Args:
        filepaths: List of Python source files to check
        verbose: Print detailed information

    Returns:
        Exit code: 0 if all have tests, 1 if missing tests
    """
    missing_tests: List[tuple[Path, List[Path]]] = []
    exempt_files: List[Path] = []

    print(f"🧪 Checking test coverage for {len(filepaths)} file(s)")
    print()

    for filepath in filepaths:
        # Check if exempt
        if is_exempt(filepath):
            exempt_files.append(filepath)
            if verbose:
                print(f"⚪ {filepath}: Exempt from test requirement")
            continue

        # Find potential test files
        test_files = find_test_file(filepath)
        test_exists = any(tf.exists() for tf in test_files)

        if test_exists:
            if verbose:
                found_test = next(tf for tf in test_files if tf.exists())
                print(f"✅ {filepath}: Test found at {found_test}")
        else:
            missing_tests.append((filepath, test_files))
            print(f"❌ {filepath}: No test file found")
            if verbose:
                print(f"   └─ Searched: {', '.join(str(tf) for tf in test_files)}")

    # Summary
    print()
    print("─" * 70)

    if missing_tests:
        print(f"❌ MISSING TESTS: {len(missing_tests)} file(s) without test coverage")
        print()
        print("💡 Expected test file locations:")
        for source_file, test_files in missing_tests:
            print(f"   {source_file}:")
            for test_file in test_files[:2]:  # Show first 2 options
                print(f"     • {test_file}")
        print()
        print("To bypass this check (not recommended):")
        print("  1. Add file to EXEMPT_PATTERNS in scripts/check_test_coverage.py")
        print("  2. Or use: git commit --no-verify")
        return 1
    else:
        total_checked = len(filepaths) - len(exempt_files)
        print(f"✅ PASSED: All {total_checked} file(s) have corresponding tests")
        if exempt_files and verbose:
            print(f"   ({len(exempt_files)} file(s) exempt)")
        return 0


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Check if Python source files have corresponding test files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check specific files
  python scripts/check_test_coverage.py src/phase1/model.py src/phase2/trainer.py

  # Check all staged files (pre-commit hook)
  python scripts/check_test_coverage.py $(git diff --cached --name-only --diff-filter=ACM | grep '\\.py$')

  # Verbose mode
  python scripts/check_test_coverage.py --verbose src/phase1/*.py

Test file patterns searched:
  1. tests/test_{filename}.py
  2. tests/{module}/test_{filename}.py
  3. src/{module}/tests/test_{filename}.py
        """
    )
    parser.add_argument(
        'files',
        nargs='*',
        type=Path,
        help='Python files to check'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Filter for Python source files (not test files themselves)
    filepaths = [
        f for f in args.files
        if f.suffix == '.py'
        and f.exists()
        and not f.name.startswith('test_')
        and 'tests' not in f.parts
    ]

    if not filepaths:
        if args.verbose:
            print("⚠️  No source files to check (test files excluded)")
        return 0

    return check_coverage(filepaths, args.verbose)


if __name__ == '__main__':
    sys.exit(main())

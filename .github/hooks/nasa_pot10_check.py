#!/usr/bin/env python3
"""
NASA POT10 Pre-Commit Hook
Enforces the Power of Ten rule: All functions must be ≤60 lines of code

NASA JPL Power of Ten Rules for Safety-Critical Code:
https://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code

Rule 3: "Do not use functions longer than what can be printed on a single
sheet of paper in a standard reference format with one line per statement
and one line per declaration." (Interpreted as ≤60 LOC)
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


def count_function_lines(node: ast.FunctionDef) -> int:
    """Count lines in function (excluding docstring)"""
    # Get function body start
    if (node.body and
        isinstance(node.body[0], ast.Expr) and
        isinstance(node.body[0].value, ast.Constant)):
        # Skip docstring
        start = node.body[1].lineno if len(node.body) > 1 else node.end_lineno
    else:
        start = node.body[0].lineno if node.body else node.lineno

    end = node.end_lineno
    return end - start + 1


def find_violations(file_path: Path) -> List[Tuple[str, int, int]]:
    """Find functions violating POT10 (>60 LOC)"""
    violations = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines = count_function_lines(node)
                if lines > 60:
                    violations.append((node.name, node.lineno, lines))

    except SyntaxError as e:
        print(f"[WARN] Syntax error in {file_path}: {e}")
    except Exception as e:
        print(f"[WARN] Error parsing {file_path}: {e}")

    return violations


def check_files(file_paths: List[str]) -> bool:
    """Check files for POT10 violations"""
    all_violations = []

    for file_path_str in file_paths:
        file_path = Path(file_path_str)

        # Skip non-Python files
        if file_path.suffix != '.py':
            continue

        # Skip __init__.py files
        if file_path.name == '__init__.py':
            continue

        violations = find_violations(file_path)

        if violations:
            all_violations.append((file_path, violations))

    return all_violations


def print_report(violations: List) -> None:
    """Print violation report"""
    if not violations:
        print("\n[OK] NASA POT10 CHECK PASSED")
        print("All functions are <=60 lines of code")
        return

    print("\n" + "=" * 70)
    print("NASA POT10 VIOLATION DETECTED")
    print("=" * 70)
    print("\nRule: All functions must be ≤60 lines of code")
    print("(excluding docstrings)\n")

    total_violations = 0

    for file_path, file_violations in violations:
        print(f"\n{file_path}:")
        for func_name, lineno, lines in file_violations:
            print(f"  Line {lineno}: {func_name}() - {lines} lines (exceeds by {lines - 60})")
            total_violations += 1

    print("\n" + "=" * 70)
    print(f"Total violations: {total_violations}")
    print("=" * 70)
    print("\nFix: Refactor large functions into smaller helper functions")
    print("Each function should do ONE thing and be ≤60 LOC\n")


def main() -> int:
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: nasa_pot10_check.py <file1.py> <file2.py> ...")
        return 0

    file_paths = sys.argv[1:]
    violations = check_files(file_paths)

    print_report(violations)

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())

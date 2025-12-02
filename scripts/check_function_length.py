#!/usr/bin/env python3
"""
NASA POT10 Function Length Checker

Enforces NASA's Power of Ten rule #4: "Restrict functions to a single printed page"
Interpreted as: No function shall exceed 60 lines of code (excluding comments/docstrings)

Usage:
    python scripts/check_function_length.py --max-lines=60 [files...]
    pre-commit hook: automatic on commit

Exit codes:
    0 - All functions comply
    1 - One or more functions exceed limit
"""

import ast
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional


class FunctionLengthChecker(ast.NodeVisitor):
    """AST visitor that checks function length against NASA POT10 standard"""

    def __init__(self, filename: str, max_lines: int = 60):
        self.filename = filename
        self.max_lines = max_lines
        self.violations: List[Tuple[str, int, int, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition and check length"""
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition and check length"""
        self._check_function(node)
        self.generic_visit(node)

    def _check_function(self, node: ast.FunctionDef) -> None:
        """Check if function exceeds maximum line count"""
        # Calculate function body length (excluding decorators and docstring)
        start_line = node.lineno
        end_line = self._get_end_line(node)

        # Adjust for docstring if present
        docstring_lines = self._get_docstring_lines(node)

        # Calculate actual code lines (total - docstring)
        total_lines = end_line - start_line + 1
        code_lines = total_lines - docstring_lines

        if code_lines > self.max_lines:
            self.violations.append(
                (node.name, start_line, code_lines, self.max_lines)
            )

    def _get_end_line(self, node: ast.FunctionDef) -> int:
        """Get the last line of the function"""
        if not node.body:
            return node.lineno

        # Find the last statement's line number
        last_stmt = node.body[-1]
        if hasattr(last_stmt, 'end_lineno') and last_stmt.end_lineno:
            return last_stmt.end_lineno
        return last_stmt.lineno

    def _get_docstring_lines(self, node: ast.FunctionDef) -> int:
        """Count docstring lines if present"""
        if not node.body:
            return 0

        first_stmt = node.body[0]
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
            if isinstance(first_stmt.value.value, str):
                # Docstring found - count its lines
                docstring = first_stmt.value.value
                return len(docstring.splitlines())

        return 0


def check_file(filepath: Path, max_lines: int = 60) -> List[Tuple[str, int, int, int]]:
    """
    Check a single Python file for NASA POT10 compliance

    Args:
        filepath: Path to Python file
        max_lines: Maximum allowed lines per function

    Returns:
        List of violations: (function_name, line_number, actual_lines, max_lines)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))
        checker = FunctionLengthChecker(str(filepath), max_lines)
        checker.visit(tree)

        return checker.violations

    except SyntaxError as e:
        print(f"⚠️  Syntax error in {filepath}:{e.lineno}: {e.msg}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"⚠️  Error checking {filepath}: {e}", file=sys.stderr)
        return []


def format_violation(filename: str, func_name: str, line_no: int,
                     actual: int, limit: int) -> str:
    """Format violation message for output"""
    excess = actual - limit
    return (
        f"❌ {filename}:{line_no} - Function '{func_name}' exceeds limit\n"
        f"   └─ {actual} lines (limit: {limit}, excess: {excess} lines)"
    )


def check_files(filepaths: List[Path], max_lines: int = 60,
                verbose: bool = False) -> int:
    """
    Check multiple files for NASA POT10 compliance

    Args:
        filepaths: List of Python files to check
        max_lines: Maximum allowed lines per function
        verbose: Print detailed information

    Returns:
        Exit code: 0 if all pass, 1 if violations found
    """
    total_violations = 0
    total_files = len(filepaths)
    files_with_violations = 0

    print(f"🔍 Checking {total_files} file(s) for NASA POT10 compliance (max {max_lines} LOC/function)")
    print()

    for filepath in filepaths:
        violations = check_file(filepath, max_lines)

        if violations:
            files_with_violations += 1
            total_violations += len(violations)

            print(f"📄 {filepath}:")
            for func_name, line_no, actual, limit in violations:
                print(f"   {format_violation(str(filepath), func_name, line_no, actual, limit)}")
            print()

        elif verbose:
            print(f"✅ {filepath}: All functions comply")

    # Summary
    print("─" * 70)
    if total_violations > 0:
        print(f"❌ FAILED: {total_violations} violation(s) in {files_with_violations} file(s)")
        print()
        print("💡 Suggestions:")
        print("   • Extract helper functions to reduce complexity")
        print("   • Split large functions into smaller, single-purpose functions")
        print("   • Move complex logic to separate modules")
        print("   • Use composition over long procedural code")
        return 1
    else:
        print(f"✅ PASSED: All {total_files} file(s) comply with NASA POT10")
        return 0


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Check Python files for NASA POT10 function length compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check specific files
  python scripts/check_function_length.py src/phase1/model.py src/phase2/trainer.py

  # Check with custom limit
  python scripts/check_function_length.py --max-lines=50 src/*.py

  # Check all Python files in src/
  python scripts/check_function_length.py src/**/*.py

  # Verbose mode (show compliant files)
  python scripts/check_function_length.py --verbose src/*.py

Pre-commit hook usage:
  This script is automatically run by pre-commit on staged Python files.
  Configure in .pre-commit-config.yaml
        """
    )
    parser.add_argument(
        'files',
        nargs='*',
        type=Path,
        help='Python files to check (if none, checks all in src/)'
    )
    parser.add_argument(
        '--max-lines',
        type=int,
        default=60,
        help='Maximum lines per function (default: 60)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output (show compliant files)'
    )

    args = parser.parse_args()

    # Determine which files to check
    if args.files:
        filepaths = [f for f in args.files if f.suffix == '.py' and f.exists()]
    else:
        # Default: check all Python files in src/
        src_dir = Path('src')
        if src_dir.exists():
            filepaths = list(src_dir.rglob('*.py'))
        else:
            print("❌ No files specified and src/ directory not found", file=sys.stderr)
            return 1

    if not filepaths:
        print("⚠️  No Python files found to check", file=sys.stderr)
        return 0

    return check_files(filepaths, args.max_lines, args.verbose)


if __name__ == '__main__':
    sys.exit(main())

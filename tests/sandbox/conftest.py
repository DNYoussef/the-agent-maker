"""
Pytest configuration for sandbox directory.

Sandbox tests are standalone scripts meant to be run with `python` directly,
not as pytest tests. This conftest.py tells pytest to skip collecting them.
"""

import pytest


def pytest_collect_file(file_path, parent):
    """Skip collecting test files in sandbox directory."""
    # Return None to skip collection of sandbox test files
    return None


collect_ignore = [
    "test_phase5_sandbox.py",
    "test_phase6_sandbox.py",
    "test_phase7_sandbox.py",
    "test_phase7_sandbox_v2.py",
    "test_phase8_sandbox.py",
]

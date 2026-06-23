"""Shared correctness evaluation (Wave 1 "measurement truth")."""

from .evaluator import DEFAULT_TASKS, Task, evaluate, matches

__all__ = ["evaluate", "matches", "Task", "DEFAULT_TASKS"]

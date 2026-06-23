"""Shared correctness evaluator - Wave 1 root ("measurement truth").

ONE real signal the phases can share, replacing the proxy/random/substring scorers
that the audit found everywhere (phase2 fitness, phase5 _check_correctness=random,
phase6 substring tool-match, phase7 ADAS no-evaluator).

Run the model on a small DETERMINISTIC set of (prompt, answer) tasks and score
exact-answer correctness in [0, 1]. The scoring/extraction is the load-bearing,
testable core; how a model produces text is injected via generate_fn so this works
across the heterogeneous phase models (HF GPT2, TRM, etc.).

Fail-closed: if no generation is possible, raise rather than fabricate a score -
the whole point is to stop faking the signal.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch


@dataclass
class Task:
    prompt: str
    answer: str


# Small, deterministic, exact-answer set. Cheap to run, hard to pass by luck.
DEFAULT_TASKS: List[Task] = [
    Task("What is 2 + 2? Answer: ", "4"),
    Task("What is 10 - 3? Answer: ", "7"),
    Task("What is 3 * 4? Answer: ", "12"),
    Task("What is 9 + 6? Answer: ", "15"),
    Task("What is 100 / 4? Answer: ", "25"),
]


def _extract(text: str) -> str:
    """Pull the candidate answer: last integer if any, else last word."""
    nums = re.findall(r"-?\d+", text)
    if nums:
        return nums[-1]
    words = re.findall(r"[A-Za-z]+", text)
    return words[-1] if words else text.strip()


def matches(generated: str, answer: str) -> bool:
    """True if the generated continuation contains/extracts the exact answer."""
    return _extract(generated).lower() == answer.lower() or answer.lower() in generated.lower()


@torch.no_grad()
def _hf_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    if not hasattr(model, "generate"):
        raise RuntimeError(
            "evaluate() requires a model with .generate() (or an injected "
            "generate_fn); refusing to fabricate a correctness score."
        )
    if hasattr(model, "eval"):
        model.eval()
    enc = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):] if text.startswith(prompt) else text


def evaluate(
    model: Any,
    tokenizer: Any,
    tasks: Optional[List[Task]] = None,
    max_new_tokens: int = 8,
    generate_fn: Optional[Callable[[Any, Any, str, int], str]] = None,
) -> float:
    """Return measured accuracy in [0, 1] over the task set.

    generate_fn(model, tokenizer, prompt, max_new_tokens) -> continuation text.
    Defaults to the HF generate path. Inject for non-HF models or for testing.
    """
    gen = generate_fn or _hf_generate
    tasks = tasks or DEFAULT_TASKS
    if not tasks:
        raise ValueError("evaluate() needs at least one task")
    correct = sum(1 for t in tasks if matches(gen(model, tokenizer, t.prompt, max_new_tokens), t.answer))
    return correct / len(tasks)

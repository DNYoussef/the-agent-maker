"""
Tokenizer Utilities - ISS-016
DEPRECATED: Use cross_phase.utils.MockTokenizer instead

This module re-exports from the canonical location for backwards compatibility.
"""

from typing import Any

# DEPRECATED: Import from canonical location
from cross_phase.utils import MockTokenizer, get_tokenizer

__all__ = ["MockTokenizer", "get_tokenizer", "get_tokenizer_for_model"]


def get_tokenizer_for_model(model_name: str = "gpt2") -> Any:
    """Alias for get_tokenizer for backwards compatibility."""
    return get_tokenizer(model_name)

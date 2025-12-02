"""Cross-phase utility modules."""

# Import from parent utils.py (ISS-016 unified MockTokenizer)
# Note: This requires accessing ../utils.py (sibling to utils/ directory)
import importlib.util
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path

from .checkpoint_utils import SAFETENSORS_AVAILABLE, load_checkpoint, save_checkpoint

# Load utils.py from parent directory
_utils_file = Path(__file__).parent.parent / "utils.py"
_spec = importlib.util.spec_from_file_location("cross_phase.utils_module", _utils_file)
_utils = importlib.util.module_from_spec(_spec)
sys.modules["cross_phase.utils_module"] = _utils
_spec.loader.exec_module(_utils)

MockTokenizer = _utils.MockTokenizer
get_tokenizer = _utils.get_tokenizer

# ISS-001: Export utility functions required by tests
get_model_size = _utils.get_model_size
calculate_safe_batch_size = _utils.calculate_safe_batch_size
validate_model_diversity = _utils.validate_model_diversity
detect_training_divergence = _utils.detect_training_divergence
compute_population_diversity = _utils.compute_population_diversity

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "SAFETENSORS_AVAILABLE",
    "MockTokenizer",
    "get_tokenizer",
    # ISS-001: Added exports
    "get_model_size",
    "calculate_safe_batch_size",
    "validate_model_diversity",
    "detect_training_divergence",
    "compute_population_diversity",
]

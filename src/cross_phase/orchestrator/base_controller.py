"""
PhaseController Abstract Base Class
All phases implement this interface for orchestration

ISS-016: Uses unified get_tokenizer() for all phases
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ISS-015/ISS-022: Import constants and validation thresholds. Re-exported here
# for the phase controllers (e.g. phase2 imports ValidationThresholds from this
# module); kept despite being unused locally.
from src.cross_phase.constants import (  # noqa: F401
    CURRICULUM_LEVELS,
    EVOMERGE_GENERATIONS,
    MIN_EXPERTS,
    ValidationThresholds,
)

# ISS-016: Import unified tokenizer utility. Re-exported for the phase
# controllers (phase3/5/6/7/8 import get_tokenizer from this module).
from src.cross_phase.utils import MockTokenizer, get_tokenizer  # noqa: F401


@dataclass
class PhaseResult:
    """
    Standard result returned by all phases

    This is the standardized PhaseResult interface from the GraphViz flows
    """

    success: bool
    phase_name: str
    model: object  # torch.nn.Module
    metrics: Dict
    duration: float  # seconds
    artifacts: Dict  # e.g., {'checkpoint_path': str, 'logs': str}
    config: Dict
    error: Optional[str] = None
    # E0 handoff contract: the tokenizer that goes WITH this model. The orchestrator
    # carries it to the next phase so phases stop fabricating their own gpt2 tokenizer.
    tokenizer: object = None


class PhaseController(ABC):
    """
    Abstract base class for phase implementations

    All phases (1-8) must inherit from this and implement:
    - execute() - Main phase logic
    - validate_input() - Validate input from previous phase
    - validate_output() - Validate output before handoff
    """

    def __init__(self, config: Dict, session_id: str):
        self.config = config
        self.session_id = session_id
        self.phase_name = self.__class__.__name__.replace("Controller", "").lower()
        # E0 handoff contract: the orchestrator injects the prior phase's tokenizer here
        # before execute(). Phases should consume this instead of hardcoding gpt2 (E2).
        self.input_tokenizer = None

    @abstractmethod
    def execute(self, input_models: Optional[List[Any]] = None) -> PhaseResult:
        """
        Execute phase logic

        Args:
            input_models: Models from previous phase (None for Phase 1)

        Returns:
            PhaseResult with success flag, model, metrics
        """

    @abstractmethod
    def validate_input(self, input_models: Optional[List[Any]] = None) -> bool:
        """
        Validate input from previous phase

        Args:
            input_models: Models to validate

        Returns:
            True if valid, raises error otherwise
        """

    @abstractmethod
    def validate_output(self, result: PhaseResult) -> bool:
        """
        Validate output before handoff to next phase

        Args:
            result: PhaseResult to validate

        Returns:
            True if valid, raises error otherwise
        """

    def get_metrics_config(self) -> Dict:
        """Get W&B metrics configuration for this phase"""
        return {}  # Override in subclass

    def cleanup(self) -> None:
        """Cleanup resources after phase completion"""

"""
PhaseController Abstract Base Class
All phases implement this interface for orchestration

ISS-016: Uses unified get_tokenizer() for all phases
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass

# ISS-016: Import unified tokenizer utility
from cross_phase.utils import get_tokenizer, MockTokenizer
# ISS-015/ISS-022: Import constants and validation thresholds
from cross_phase.constants import (
    ValidationThresholds,
    EVOMERGE_GENERATIONS,
    MIN_EXPERTS,
    CURRICULUM_LEVELS
)


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

    @abstractmethod
    def execute(self, input_models: list = None) -> PhaseResult:
        """
        Execute phase logic

        Args:
            input_models: Models from previous phase (None for Phase 1)

        Returns:
            PhaseResult with success flag, model, metrics
        """
        pass

    @abstractmethod
    def validate_input(self, input_models: list = None) -> bool:
        """
        Validate input from previous phase

        Args:
            input_models: Models to validate

        Returns:
            True if valid, raises error otherwise
        """
        pass

    @abstractmethod
    def validate_output(self, result: PhaseResult) -> bool:
        """
        Validate output before handoff to next phase

        Args:
            result: PhaseResult to validate

        Returns:
            True if valid, raises error otherwise
        """
        pass

    def get_metrics_config(self) -> Dict:
        """Get W&B metrics configuration for this phase"""
        return {}  # Override in subclass

    def cleanup(self):
        """Cleanup resources after phase completion"""
        pass

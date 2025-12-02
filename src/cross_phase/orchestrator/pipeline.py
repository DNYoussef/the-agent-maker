"""
Pipeline Orchestrator
Coordinates execution of all 8 phases with handoff validation
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..storage.model_registry import ModelRegistry
from .phase_controller import PhaseController, PhaseResult


class PipelineOrchestrator:
    """
    Main pipeline controller that sequences phases and manages handoffs

    Features:
    - Phase sequencing (Phase 1 → 2 → 3 → ... → 8)
    - Handoff validation between phases
    - Error recovery and rollback
    - Progress tracking via model registry
    """

    def __init__(self, config: dict, session_id: Optional[str] = None):
        self.config = config
        self.session_id = session_id or self._generate_session_id()

        # Initialize model registry
        self.registry = ModelRegistry(
            config.get("registry_path", "./storage/registry/model_registry.db")
        )

        # Create session
        self.registry.create_session(self.session_id, config)

        # Phase controllers (will be instantiated as needed)
        self.phase_controllers: List[PhaseController] = []

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_full_pipeline(self) -> dict:
        """
        Execute all 8 phases in sequence

        Returns:
            Final metrics dict
        """
        results = {}
        current_models = None

        for phase_num in range(1, 9):
            phase_name = f"phase{phase_num}"

            print(f"\n{'='*60}")
            print(f"Starting {phase_name.upper()}")
            print(f"{'='*60}\n")

            # Get phase controller
            controller = self._get_phase_controller(phase_num)

            # Validate input
            if not controller.validate_input(current_models):
                raise ValueError(f"{phase_name} input validation failed")

            # Execute phase
            start_time = time.time()
            result = controller.execute(current_models)
            duration = time.time() - start_time

            # Validate output
            if not controller.validate_output(result):
                raise ValueError(f"{phase_name} output validation failed")

            # Update progress
            progress_percent = (phase_num / 8) * 100
            self.registry.update_session_progress(self.session_id, phase_name, progress_percent)

            # Store results
            results[phase_name] = {
                "success": result.success,
                "duration": duration,
                "metrics": result.metrics,
            }

            # Pass model(s) to next phase
            if isinstance(result.model, list):
                current_models = result.model
            else:
                current_models = [result.model] if result.model else None

            print(f"\n✅ {phase_name.upper()} Complete")
            print(f"   Duration: {duration/60:.1f} minutes")
            print(f"   Metrics: {result.metrics}\n")

        # Final summary
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}\n")

        total_duration = sum(r["duration"] for r in results.values())
        print(f"Total Duration: {total_duration/3600:.1f} hours")

        return results

    def run_single_phase(self, phase_num: int, input_models: list = None) -> PhaseResult:
        """
        Run a single phase (for testing or resuming)

        Args:
            phase_num: Phase number (1-8)
            input_models: Input models (None for Phase 1)

        Returns:
            PhaseResult
        """
        controller = self._get_phase_controller(phase_num)

        if not controller.validate_input(input_models):
            raise ValueError(f"Phase {phase_num} input validation failed")

        result = controller.execute(input_models)

        if not controller.validate_output(result):
            raise ValueError(f"Phase {phase_num} output validation failed")

        return result

    def _get_phase_controller(self, phase_num: int) -> PhaseController:
        """
        Get phase controller instance

        Args:
            phase_num: Phase number (1-8)

        Returns:
            PhaseController instance
        """
        # Import phase controllers
        from .phase_controller import (
            Phase1Controller,
            Phase2Controller,
            Phase3Controller,
            Phase4Controller,
        )

        # Map phase numbers to controllers
        controller_classes = {
            1: Phase1Controller,
            2: Phase2Controller,
            3: Phase3Controller,
            4: Phase4Controller,
            # TODO: Add Phase 5-8 controllers when implemented
        }

        controller_class = controller_classes.get(phase_num)
        if not controller_class:
            raise NotImplementedError(f"Phase {phase_num} not yet implemented")

        # Get phase-specific config
        phase_config = self.config.get("phases", {}).get(f"phase{phase_num}", {})

        return controller_class(phase_config, self.session_id)

    def rollback_to_phase(self, phase_num: int) -> None:
        """
        Rollback to a previous phase checkpoint

        Args:
            phase_num: Phase to rollback to (1-7)
        """
        print(f"🔄 Rolling back to Phase {phase_num}...")

        # Load checkpoint from registry
        phase_name = f"phase{phase_num}"
        model_info = self.registry.get_model(session_id=self.session_id, phase_name=phase_name)

        print(f"✅ Loaded checkpoint from {model_info['created_at']}")
        print(f"   Model: {model_info['model_path']}")

        return model_info

    def cleanup(self) -> Any:
        """Cleanup resources"""
        self.registry.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

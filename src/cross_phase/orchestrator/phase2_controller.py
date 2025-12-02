"""Phase 2: EvoMerge - Evolve 3 models into 1"""

from typing import Any, List, Optional

from .base_controller import PhaseController, PhaseResult, ValidationThresholds


class Phase2Controller(PhaseController):
    """Phase 2: EvoMerge - Evolve 3 models into 1"""

    def execute(self, input_models: Optional[List[Any]] = None) -> PhaseResult:
        """Execute Phase 2: 50-generation evolution.

        Uses evolutionary optimization with 6 merge techniques to evolve
        3 input models from Phase 1 into 1 champion model.
        """
        import time

        from phase2_evomerge.phase2_pipeline import EvolutionConfig, Phase2Pipeline

        print("\n" + "=" * 60)
        print("PHASE 2: EVOMERGE - INITIALIZING")
        print("=" * 60 + "\n")

        start_time = time.time()

        try:
            # Validate input
            self.validate_input(input_models)

            # Create evolution config from phase config
            evo_config = (
                EvolutionConfig.from_dict(self.config) if self.config else EvolutionConfig()
            )

            # Create and run pipeline
            pipeline = Phase2Pipeline(config=evo_config)
            champion = pipeline.run(input_models, session_id=self.session_id)

            duration = time.time() - start_time

            return PhaseResult(
                success=True,
                phase_name="phase2",
                model=champion,
                metrics=pipeline.get_metrics(),
                duration=duration,
                artifacts={
                    "fitness_history": pipeline.fitness_history,
                    "best_fitness": pipeline.best_fitness,
                },
                config=self.config,
                error=None,
            )

        except Exception as e:
            duration = time.time() - start_time
            return PhaseResult(
                success=False,
                phase_name="phase2",
                model=None,
                metrics={},
                duration=duration,
                artifacts={},
                config=self.config,
                error=str(e),
            )

    def validate_input(self, input_models: Optional[List[Any]] = None) -> bool:
        """Validate 3 input models from Phase 1"""
        if not input_models or len(input_models) != 3:
            raise ValueError(
                f"Phase 2 requires 3 input models, got {len(input_models) if input_models else 0}"
            )
        return True

    def validate_output(self, result: PhaseResult) -> bool:
        """
        Validate Phase 2 output (ISS-022).

        Checks:
        - Champion model produced
        - Fitness gain exceeds threshold
        - Generations completed
        """
        if not result.success:
            return False

        if result.metrics:
            # Check fitness gain
            fitness_gain = result.metrics.get("fitness_gain", 0.0)
            if fitness_gain < ValidationThresholds.PHASE2_MIN_FITNESS_GAIN:
                return False

            # Check generations completed
            generations = result.metrics.get("generations", 0)
            if generations < 1:
                return False

        return True

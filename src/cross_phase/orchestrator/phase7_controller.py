"""Phase 7: Self-Guided Experts - Model-driven expert discovery"""

from typing import Any, List, Optional

from .base_controller import PhaseController, PhaseResult, get_tokenizer


class Phase7Controller(PhaseController):
    """Phase 7: Self-Guided Experts - Model-driven expert discovery."""

    def execute(self, input_models: Optional[List[Any]] = None) -> PhaseResult:
        """Execute Phase 7: Expert discovery, SVF training, ADAS optimization.

        Args:
            input_models: [baked_model] from Phase 6

        Returns:
            PhaseResult with expert-enhanced model
        """
        import time

        start_time = time.time()

        print("\n" + "=" * 60)
        print("PHASE 7: SELF-GUIDED EXPERTS")
        print("=" * 60 + "\n")

        try:
            # Validate input
            self.validate_input(input_models)
            if not input_models:
                raise ValueError("input_models cannot be None")
            baked_model = input_models[0]

            # Get tokenizer
            tokenizer = self._get_tokenizer()

            # Create experts config
            from phase7_experts import ExpertsConfig, ExpertsEngine

            config = ExpertsConfig(
                min_experts=self.config.get("min_experts", 3) if self.config else 3,
                max_experts=self.config.get("max_experts", 10) if self.config else 10,
                svf_epochs=self.config.get("svf_epochs", 5) if self.config else 5,
                adas_population=self.config.get("adas_population", 50) if self.config else 50,
                adas_generations=self.config.get("adas_generations", 100) if self.config else 100,
            )

            # Run experts engine
            engine = ExpertsEngine(config=config)
            result = engine.run(model=baked_model, tokenizer=tokenizer)

            duration = time.time() - start_time

            return PhaseResult(
                success=result.success,
                phase_name="phase7",
                model=result.model,
                metrics={
                    "num_experts": result.num_experts,
                    "routing_config": result.routing_config,
                    "duration_seconds": duration,
                    **result.metrics,
                },
                duration=duration,
                artifacts=result.artifacts,
                config=self.config,
                error=result.error,
            )

        except Exception as e:
            duration = time.time() - start_time
            return PhaseResult(
                success=False,
                phase_name="phase7",
                model=None,
                metrics={},
                duration=duration,
                artifacts={},
                config=self.config,
                error=str(e),
            )

    def _get_tokenizer(self) -> Any:
        """Get tokenizer using unified utility (ISS-016)."""
        return get_tokenizer("gpt2")

    def validate_input(self, input_models: Optional[List[Any]] = None) -> bool:
        """Validate 1 input model from Phase 6."""
        if not input_models or len(input_models) != 1:
            raise ValueError(
                f"Phase 7 requires 1 input model, got {len(input_models) if input_models else 0}"
            )
        return True

    def validate_output(self, result: PhaseResult) -> bool:
        """Validate Phase 7 output (experts discovered)."""
        if result.metrics:
            num_experts = result.metrics.get("num_experts", 0)
            return bool(num_experts >= 1)
        return True

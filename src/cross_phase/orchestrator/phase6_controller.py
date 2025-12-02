"""Phase 6: Tool & Persona Baking - A/B cycle optimization"""

from .base_controller import PhaseController, PhaseResult, get_tokenizer


class Phase6Controller(PhaseController):
    """Phase 6: Tool & Persona Baking - A/B cycle optimization."""

    def execute(self, input_models: list = None) -> PhaseResult:
        """Execute Phase 6: A/B baking cycles.

        Args:
            input_models: [specialized_model] from Phase 5

        Returns:
            PhaseResult with baked model
        """
        import time
        start_time = time.time()

        print("\n" + "=" * 60)
        print("PHASE 6: TOOL & PERSONA BAKING")
        print("=" * 60 + "\n")

        try:
            # Validate input
            self.validate_input(input_models)
            specialized_model = input_models[0]

            # Get tokenizer
            tokenizer = self._get_tokenizer()

            # Create baking config
            from phase6_baking import BakingEngine, BakingConfig

            config = BakingConfig(
                a_cycle_iterations=self.config.get('a_cycle_iterations', 5) if self.config else 5,
                b_cycle_iterations=self.config.get('b_cycle_iterations', 5) if self.config else 5,
                half_bake_strength=self.config.get('half_bake_strength', 0.5) if self.config else 0.5,
                max_total_iterations=self.config.get('max_iterations', 20) if self.config else 20
            )

            # Run baking engine
            engine = BakingEngine(config=config)
            result = engine.run(
                model=specialized_model,
                tokenizer=tokenizer
            )

            duration = time.time() - start_time

            return PhaseResult(
                success=result.success,
                phase_name="phase6",
                model=result.model,
                metrics={
                    'total_iterations': result.total_iterations,
                    'a_cycle_count': result.a_cycle_count,
                    'b_cycle_count': result.b_cycle_count,
                    'final_tool_score': result.final_tool_score,
                    'final_persona_score': result.final_persona_score,
                    'duration_seconds': duration
                },
                duration=duration,
                artifacts=result.artifacts,
                config=self.config,
                error=result.error
            )

        except Exception as e:
            duration = time.time() - start_time
            return PhaseResult(
                success=False,
                phase_name="phase6",
                model=None,
                metrics={},
                duration=duration,
                artifacts={},
                config=self.config,
                error=str(e)
            )

    def _get_tokenizer(self):
        """Get tokenizer using unified utility (ISS-016)."""
        return get_tokenizer("gpt2")

    def validate_input(self, input_models: list = None) -> bool:
        """Validate 1 input model from Phase 5."""
        if not input_models or len(input_models) != 1:
            raise ValueError(f"Phase 6 requires 1 input model, got {len(input_models) if input_models else 0}")
        return True

    def validate_output(self, result: PhaseResult) -> bool:
        """Validate Phase 6 output (baking iterations completed)."""
        if result.metrics:
            iterations = result.metrics.get('total_iterations', 0)
            return iterations >= 1
        return True

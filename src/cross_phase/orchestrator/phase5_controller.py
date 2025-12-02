"""Phase 5: Curriculum Learning - Specialized agent training"""

from .base_controller import PhaseController, PhaseResult, get_tokenizer


class Phase5Controller(PhaseController):
    """Phase 5: Curriculum Learning - Specialized agent training.

    Implements 7-stage curriculum learning pipeline:
    1. Assessment - Edge-of-chaos detection
    2. Curriculum Generation - 20,000 questions
    3. Training Loop - Variants/hints
    4. Prompt Baking - Moral compass
    5. Self-Modeling - Temperature prediction
    6. Dream Consolidation - Memory preservation
    7. Level Progression - 10 levels
    """

    def execute(self, input_models: list = None) -> PhaseResult:
        """Execute Phase 5: Curriculum-based specialization training.

        Args:
            input_models: [quantized_model] from Phase 4

        Returns:
            PhaseResult with specialized model
        """
        import time
        start_time = time.time()

        print("\n" + "=" * 60)
        print("PHASE 5: CURRICULUM LEARNING - INITIALIZING")
        print("=" * 60 + "\n")

        try:
            # Validate input
            self.validate_input(input_models)
            quantized_model = input_models[0]

            # Get tokenizer
            tokenizer = self._get_tokenizer()

            # Create curriculum config
            from phase5_curriculum import CurriculumEngine, CurriculumConfig, SpecializationType

            specialization = self.config.get('specialization', 'coding') if self.config else 'coding'
            spec_type = SpecializationType(specialization)

            config = CurriculumConfig(
                num_levels=self.config.get('num_levels', 10) if self.config else 10,
                questions_per_level=self.config.get('questions_per_level', 2000) if self.config else 2000,
                specialization=spec_type
            )

            # Create and run engine
            engine = CurriculumEngine(config=config)
            result = engine.run(
                model=quantized_model,
                tokenizer=tokenizer,
                frontier_client=None,  # Would connect to OpenRouter
                coding_env=None  # Would connect to sandbox
            )

            duration = time.time() - start_time

            return PhaseResult(
                success=result.success,
                phase_name="phase5",
                model=result.model,
                metrics={
                    'levels_completed': result.levels_completed,
                    'specialization': result.specialization.value,
                    'duration_seconds': duration,
                    **result.metrics
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
                phase_name="phase5",
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
        """Validate 1 input model from Phase 4."""
        if not input_models or len(input_models) != 1:
            raise ValueError(f"Phase 5 requires 1 input model, got {len(input_models) if input_models else 0}")
        return True

    def validate_output(self, result: PhaseResult) -> bool:
        """Validate Phase 5 output (specialization achieved)."""
        if result.metrics:
            levels = result.metrics.get('levels_completed', 0)
            return levels >= 1  # At least one level completed
        return True

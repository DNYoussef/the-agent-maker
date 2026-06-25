"""Phase 5: Curriculum Learning - Specialized agent training"""

from typing import Any, List, Optional

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

    def execute(self, input_models: Optional[List[Any]] = None) -> PhaseResult:
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
            result = self._run_curriculum(input_models)

            duration = time.time() - start_time

            return PhaseResult(
                success=result.success,
                phase_name="phase5",
                model=result.model,
                metrics={
                    "levels_completed": result.levels_completed,
                    "specialization": result.specialization.value,
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
                phase_name="phase5",
                model=None,
                metrics={},
                duration=duration,
                artifacts={},
                config=self.config,
                error=str(e),
            )

    def _run_curriculum(self, input_models: Optional[List[Any]]) -> Any:
        """Validate input, build the curriculum engine and run it.

        Returns:
            The engine's CurriculumResult.
        """
        from phase5_curriculum import (
            CurriculumConfig,
            CurriculumEngine,
            OpenRouterClient,
            SpecializationType,
        )
        from phase5_curriculum.docker_sandbox import DockerSandbox

        # Validate input
        self.validate_input(input_models)
        if not input_models:
            raise ValueError("input_models cannot be None")
        quantized_model = input_models[0]

        # Get tokenizer
        tokenizer = self._get_tokenizer()

        # Create curriculum config
        specialization = self.config.get("specialization", "coding") if self.config else "coding"
        spec_type = SpecializationType(specialization)

        config = CurriculumConfig(
            num_levels=self.config.get("num_levels", 10) if self.config else 10,
            questions_per_level=self.config.get("questions_per_level", 2000)
            if self.config
            else 2000,
            specialization=spec_type,
        )

        # Initialize OpenRouter; the Docker sandbox is OPT-IN. By default coding_env is
        # None so the curriculum uses the deterministic heuristic validator. The sandbox
        # path (coding_env.execute/.check) is currently broken (async called sync, missing
        # .check) and walls every question at level 1 - enable only once E6 fixes it.
        frontier_client = OpenRouterClient()
        coding_env = DockerSandbox() if (self.config or {}).get("use_docker_sandbox") else None

        # Create and run engine
        engine = CurriculumEngine(
            config=config,
            wandb_integration=self._build_wandb_integration(),
            session_id=self.session_id,
        )
        return engine.run(
            model=quantized_model,
            tokenizer=tokenizer,
            frontier_client=frontier_client,
            coding_env=coding_env,
        )

    def _build_wandb_integration(self) -> Any:
        """Build a WandBIntegration from this phase's config."""
        from cross_phase.monitoring.wandb_integration import WandBIntegration

        return WandBIntegration(
            project_name=(
                self.config.get("wandb_project", "agent-forge-v2")
                if self.config
                else "agent-forge-v2"
            ),
            mode=(self.config.get("wandb_mode", "auto") if self.config else "auto"),
        )

    def _get_tokenizer(self) -> Any:
        """Get tokenizer using unified utility (ISS-016)."""
        return get_tokenizer("gpt2")

    def validate_input(self, input_models: Optional[List[Any]] = None) -> bool:
        """Validate 1 input model from Phase 4."""
        if not input_models or len(input_models) != 1:
            raise ValueError(
                f"Phase 5 requires 1 input model, got {len(input_models) if input_models else 0}"
            )
        return True

    def validate_output(self, result: PhaseResult) -> bool:
        """Validate Phase 5 output (specialization achieved)."""
        if result.metrics:
            levels = result.metrics.get("levels_completed", 0)
            return bool(levels >= 1)  # At least one level completed
        return True

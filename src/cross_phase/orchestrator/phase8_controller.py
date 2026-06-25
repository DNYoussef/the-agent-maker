"""Phase 8: Final Compression - Triple compression pipeline"""

from typing import Any, List, Optional

from .base_controller import PhaseController, PhaseResult, get_tokenizer


class Phase8Controller(PhaseController):
    """Phase 8: Final Compression - Triple compression pipeline."""

    def execute(self, input_models: Optional[List[Any]] = None) -> PhaseResult:
        """Execute Phase 8: SeedLM + VPTQ + Hypercompression.

        Args:
            input_models: [expert_model] from Phase 7

        Returns:
            PhaseResult with compressed model
        """
        import time

        start_time = time.time()

        print("\n" + "=" * 60)
        print("PHASE 8: FINAL COMPRESSION")
        print("=" * 60 + "\n")

        try:
            result = self._run_compression(input_models)

            duration = time.time() - start_time

            return PhaseResult(
                success=result.success,
                phase_name="phase8",
                model=result.model,
                metrics={
                    "original_size_mb": result.original_size_mb,
                    "final_size_mb": result.final_size_mb,
                    "total_compression": result.total_compression,
                    "retention_score": result.retention_score,
                    "stage_results": result.stage_results,
                    "benchmark_results": result.benchmark_results,
                    "duration_seconds": duration,
                },
                duration=duration,
                artifacts={"rollback_stage": result.rollback_stage},
                config=self.config,
                error=result.error,
            )

        except Exception as e:
            duration = time.time() - start_time
            return PhaseResult(
                success=False,
                phase_name="phase8",
                model=None,
                metrics={},
                duration=duration,
                artifacts={},
                config=self.config,
                error=str(e),
            )

    def _run_compression(self, input_models: Optional[List[Any]]) -> Any:
        """Validate input, build the compression engine and run it.

        Returns:
            The engine's CompressionResult.
        """
        from phase8_compression import CompressionConfig, CompressionEngine

        # Validate input
        self.validate_input(input_models)
        if not input_models:
            raise ValueError("input_models cannot be None")
        expert_model = input_models[0]

        # Get tokenizer
        tokenizer = self._get_tokenizer()

        # Create compression config
        config = CompressionConfig(
            seedlm_enabled=self.config.get("seedlm_enabled", True) if self.config else True,
            vptq_enabled=self.config.get("vptq_enabled", True) if self.config else True,
            hyper_enabled=self.config.get("hyper_enabled", True) if self.config else True,
            min_retention_final=self.config.get("min_retention", 0.84) if self.config else 0.84,
            run_benchmarks=self.config.get("run_benchmarks", True) if self.config else True,
        )

        # Run compression engine
        engine = CompressionEngine(config=config)
        return engine.run(model=expert_model, tokenizer=tokenizer)

    def _get_tokenizer(self) -> Any:
        """Use the tokenizer threaded from the prior phase (E0/E2); fall back to gpt2 only
        when run standalone with no upstream tokenizer."""
        if self.input_tokenizer is not None:
            return self.input_tokenizer
        return get_tokenizer("gpt2")

    def validate_input(self, input_models: Optional[List[Any]] = None) -> bool:
        """Validate 1 input model from Phase 7."""
        if not input_models or len(input_models) != 1:
            raise ValueError(
                f"Phase 8 requires 1 input model, got {len(input_models) if input_models else 0}"
            )
        return True

    def validate_output(self, result: PhaseResult) -> bool:
        """Validate Phase 8 output. E9: require a REAL compression (>1.0x, not the old
        >=1.0 which greened a no-op) AND result.success (a rolled-back/corrupted run used to
        pass), and reject empty metrics (used to return True)."""
        if not result.success or not result.metrics:
            return False
        compression = result.metrics.get("total_compression", 0)
        retention = result.metrics.get("retention_score", 0)
        return bool(compression > 1.0 and retention >= 0.5)

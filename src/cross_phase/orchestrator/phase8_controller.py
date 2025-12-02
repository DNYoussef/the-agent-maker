"""Phase 8: Final Compression - Triple compression pipeline"""

from .base_controller import PhaseController, PhaseResult, get_tokenizer


class Phase8Controller(PhaseController):
    """Phase 8: Final Compression - Triple compression pipeline."""

    def execute(self, input_models: list = None) -> PhaseResult:
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
            # Validate input
            self.validate_input(input_models)
            expert_model = input_models[0]

            # Get tokenizer
            tokenizer = self._get_tokenizer()

            # Create compression config
            from phase8_compression import CompressionConfig, CompressionEngine

            config = CompressionConfig(
                seedlm_enabled=self.config.get("seedlm_enabled", True) if self.config else True,
                vptq_enabled=self.config.get("vptq_enabled", True) if self.config else True,
                hyper_enabled=self.config.get("hyper_enabled", True) if self.config else True,
                min_retention_final=self.config.get("min_retention", 0.84) if self.config else 0.84,
                run_benchmarks=self.config.get("run_benchmarks", True) if self.config else True,
            )

            # Run compression engine
            engine = CompressionEngine(config=config)
            result = engine.run(model=expert_model, tokenizer=tokenizer)

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

    def _get_tokenizer(self):
        """Get tokenizer using unified utility (ISS-016)."""
        return get_tokenizer("gpt2")

    def validate_input(self, input_models: list = None) -> bool:
        """Validate 1 input model from Phase 7."""
        if not input_models or len(input_models) != 1:
            raise ValueError(
                f"Phase 8 requires 1 input model, got {len(input_models) if input_models else 0}"
            )
        return True

    def validate_output(self, result: PhaseResult) -> bool:
        """Validate Phase 8 output (compression achieved)."""
        if result.metrics:
            compression = result.metrics.get("total_compression", 0)
            retention = result.metrics.get("retention_score", 0)
            return compression >= 1.0 and retention >= 0.5
        return True

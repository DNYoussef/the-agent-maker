"""Phase 3: Quiet-STaR - Add reasoning via prompt baking + RL"""

from typing import Any, List, Optional

from .base_controller import PhaseController, PhaseResult, get_tokenizer


class Phase3Controller(PhaseController):
    """Phase 3: Quiet-STaR - Add reasoning via prompt baking + RL"""

    def execute(self, input_models: Optional[List[Any]] = None) -> PhaseResult:
        """Execute Phase 3: Prompt Baking (Step 1) + Quiet-STaR RL (Step 2).

        Two-step process:
        1. Prompt Baking: Embed reasoning strategies into model weights
        2. REINFORCE RL: Optimize thought generation with KL regularization

        Args:
            input_models: [champion_model] from Phase 2

        Returns:
            PhaseResult with reasoning-enhanced model
        """
        import time

        start_time = time.time()

        print("\n" + "=" * 60)
        print("PHASE 3: QUIET-STAR - REASONING ENHANCEMENT")
        print("=" * 60 + "\n")

        try:
            enhanced_model, baked_model, rl_completed, anti_theater_results = self._run_phase(
                input_models
            )
            duration = time.time() - start_time

            return PhaseResult(
                success=True,
                phase_name="phase3",
                model=enhanced_model,
                metrics={
                    "baking_completed": True,
                    "rl_completed": rl_completed,  # AGM-004: Honest reporting
                    "anti_theater_passed": anti_theater_results.get("all_passed", False),
                    "duration_seconds": duration,
                },
                duration=duration,
                artifacts={
                    "anti_theater_results": anti_theater_results,
                    "baked_model": baked_model,
                },
                config=self.config,
                error=None,
            )

        except Exception as e:
            duration = time.time() - start_time
            return PhaseResult(
                success=False,
                phase_name="phase3",
                model=None,
                metrics={},
                duration=duration,
                artifacts={},
                config=self.config,
                error=str(e),
            )

    def _run_phase(self, input_models: Optional[List[Any]]) -> tuple:
        """Run the 3 reasoning-enhancement steps and return their outputs.

        Returns:
            (enhanced_model, baked_model, rl_completed, anti_theater_results)
        """
        # Validate input
        self.validate_input(input_models)
        champion_model = input_models[0]

        # Get tokenizer (try to load, fall back to mock)
        tokenizer = self._get_tokenizer()

        # Step 1: Prompt Baking
        print("--- Step 1: Prompt Baking ---")
        baked_model = self._run_prompt_baking(champion_model, tokenizer)

        # Step 2: Quiet-STaR RL
        # AGM-004: Returns tuple (model, rl_completed) for honest reporting
        print("\n--- Step 2: Quiet-STaR RL ---")
        enhanced_model, rl_completed = self._run_quietstar_rl(
            baked_model, champion_model, tokenizer
        )

        # Step 3: Anti-theater validation
        print("\n--- Step 3: Anti-Theater Validation ---")
        anti_theater_results = self._validate_anti_theater(enhanced_model, tokenizer)

        return enhanced_model, baked_model, rl_completed, anti_theater_results

    def _get_tokenizer(self) -> Any:
        """Use the tokenizer threaded from the prior phase (E0/E2); fall back to gpt2 only
        when run standalone with no upstream tokenizer."""
        if self.input_tokenizer is not None:
            return self.input_tokenizer
        return get_tokenizer("gpt2")

    def _run_prompt_baking(self, model, tokenizer) -> None:
        """Run Step 1: Prompt Baking to embed reasoning strategies."""
        from cross_phase.prompt_baking.baker import PromptBaker, PromptBakingConfig

        config = PromptBakingConfig(
            lora_r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            num_epochs=self.config.get("baking_epochs", 3),
            batch_size=self.config.get("batch_size", 8),
            learning_rate=self.config.get("learning_rate", 1e-4),
        )

        baker = PromptBaker(config)

        # Reasoning prompt for baking
        reasoning_prompt = """You are a careful reasoning assistant. When solving problems:
1. Break down complex problems into smaller steps
2. Consider multiple approaches before choosing one
3. Verify your intermediate results
4. State your assumptions explicitly
5. Double-check your final answer"""

        # Calibration data (simplified for MVP)
        calibration_data = [
            "What is 2 + 2?",
            "Explain why the sky is blue.",
            "What are the factors of 12?",
            "How does photosynthesis work?",
            "What is the capital of France?",
        ]

        print("  Baking reasoning prompt into model...")
        baked_model = baker.bake_prompt(
            model=model,
            prompt=reasoning_prompt,
            tokenizer=tokenizer,
            calibration_data=calibration_data,
            half_bake=False,
        )

        print("  Prompt baking complete")
        return baked_model

    def _run_quietstar_rl(self, baked_model, baseline_model, tokenizer) -> tuple:
        """
        Run Step 2: Quiet-STaR RL training.

        AGM-004: Returns (model, rl_completed) tuple with honest reporting.
        Integrates full REINFORCETrainer when enable_full_rl=True in config.
        """
        import torch

        enable_full_rl = self.config.get("enable_full_rl", False)
        rl_episodes = self.config.get("rl_episodes", 1000)

        if enable_full_rl:
            # AGM-004: Full RL implementation using REINFORCETrainer
            try:
                from phase3_quietstar.config import QuietSTaRConfig
                from phase3_quietstar.step2_rl import REINFORCETrainer

                print(f"  Running full REINFORCE RL training ({rl_episodes} episodes)...")

                # Build config for trainer
                rl_config = QuietSTaRConfig()
                rl_config.rl.num_episodes = rl_episodes
                rl_config.rl.num_thoughts = self.config.get("num_thoughts", 4)
                rl_config.rl.max_thought_length = self.config.get("max_thought_length", 64)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                trainer = REINFORCETrainer(
                    model=baked_model,
                    baked_model=baseline_model,
                    tokenizer=tokenizer,
                    config=rl_config,
                    device=device,
                )

                # P3: train() REQUIRES (train_dl, val_dl) and returns METRICS, not a model.
                # The old `trainer.train()` raised TypeError (no dataloaders) -> silent
                # fallback, so RL never ran. Build dataloaders and read the in-place model.
                train_dl, val_dl = self._build_rl_dataloaders(tokenizer)
                trainer.train(train_dl, val_dl, num_episodes=rl_episodes)
                enhanced_model = trainer.model  # trained in place (best restored in train())
                print("  Full RL training complete")
                return enhanced_model, True

            except Exception as e:
                print(f"  RL training failed: {e}, falling back to baked model")
                return baked_model, False
        else:
            # AGM-004: Skip RL with honest reporting
            print("  RL step skipped (enable_full_rl=False)")
            print("  Using baked model as output (RL training is compute-intensive)")
            return baked_model, False

    def _build_rl_dataloaders(self, tokenizer):
        """P3: small train/val dataloaders of tokenized reasoning prompts for the RL loop.
        The controller used to pass none, so train() raised and RL silently fell back. Real
        reasoning text (not random noise); swap in a real dataset for production runs."""
        import torch
        from torch.utils.data import DataLoader, Dataset

        prompts = [
            "Question: What is 2+2? Answer: Let's think step by step.",
            "Question: A train goes 60 km in 1 hour; how far in 3 hours? Answer:",
            "Question: What are the factors of 12? Answer: Step by step,",
            "Question: Why is the sky blue? Answer: Reasoning:",
        ]
        seq_len = 64

        def _ids(text):
            enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
            ids = enc["input_ids"][0][:seq_len]
            if ids.shape[0] < seq_len:  # pad to fixed length so default collate can stack
                pad = torch.zeros(seq_len - ids.shape[0], dtype=ids.dtype)
                ids = torch.cat([ids, pad])
            return ids

        class _DS(Dataset):
            def __init__(self):
                self.items = [{"input_ids": (i := _ids(t)), "labels": i.clone()} for t in prompts]

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

        ds = _DS()
        return DataLoader(ds, batch_size=2), DataLoader(ds, batch_size=2)

    def _validate_anti_theater(self, model, tokenizer) -> Any:
        """Validate model outputs are genuine, not theatrical. E4: divergence + consistency
        are REAL probes now (were hardcoded True / fake-hash); the unimplemented ablation
        test is no longer reported as a pass; a model that cannot generate fails honestly
        (we cannot prove genuineness, so we do not claim it)."""
        import torch

        print("  Running anti-theater validation...")
        results = {"divergence_test": False, "consistency_test": False, "all_passed": False}

        if not hasattr(model, "generate"):
            print("  Anti-theater validation: FAILED (model has no generate())")
            return results

        def _gen(text):
            enc = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding=True)
            return model.generate(**enc, max_new_tokens=10, do_sample=False)[0].tolist()

        try:
            model.eval()
            with torch.no_grad():
                test_inputs = ["Hello", "Goodbye", "What is 2+2?", "Tell me a story"]
                outputs = [_gen(t) for t in test_inputs]
                # Divergence: different inputs must produce different outputs.
                results["divergence_test"] = len(set(str(o) for o in outputs)) > 1
                # Consistency: the same input (greedy) must reproduce its output.
                results["consistency_test"] = _gen(test_inputs[0]) == outputs[0]

            results["all_passed"] = results["divergence_test"] and results["consistency_test"]
            print(f"  Anti-theater validation: {'PASSED' if results['all_passed'] else 'FAILED'}")
        except Exception as e:
            print(f"  Anti-theater validation error: {e}")
            results["all_passed"] = False

        return results

    def validate_input(self, input_models: Optional[List[Any]] = None) -> bool:
        """Validate 1 input model from Phase 2"""
        if not input_models or len(input_models) != 1:
            raise ValueError(
                f"Phase 3 requires 1 input model, got {len(input_models) if input_models else 0}"
            )
        return True

    def validate_output(self, result: PhaseResult) -> bool:
        """Validate Phase 3 output (anti-theater tests pass)"""
        if result.artifacts and "anti_theater_results" in result.artifacts:
            return result.artifacts["anti_theater_results"].get("all_passed", False)
        return True

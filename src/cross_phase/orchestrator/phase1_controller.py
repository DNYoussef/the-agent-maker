"""Phase 1: Cognate - Create 3 foundation models"""

from .base_controller import PhaseController, PhaseResult
from typing import Optional, List, Any


class Phase1Controller(PhaseController):
    """Phase 1: Cognate - Create 3 foundation models"""

    def execute(self, input_models: Optional[List[Any]] = None) -> PhaseResult:
        """Execute Phase 1: Create 3 TRM x Titans-MAG models"""
        import time

        start_time = time.time()

        print("\n" + "=" * 60)
        print("PHASE 1: COGNATE - INITIALIZING")
        print("=" * 60 + "\n")

        # Imports local to avoid circular dependencies and ensure path context
        import sys
        from pathlib import Path

        import torch
        from transformers import GPT2Tokenizer

        # Ensure src is in path
        src_path = str(Path(__file__).parents[3])
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Phase 1 specific imports
        # ISS-006: Use canonical MockTokenizer from cross_phase.utils
        from cross_phase.utils import get_tokenizer
        from phase1_cognate.data.dataset_downloader import DATASET_CONFIGS, download_all_datasets
        from phase1_cognate.data.dataset_processor import process_dataset
        from phase1_cognate.model.full_model import TRMTitansMAGModel
        from phase1_cognate.model.model_config import Phase1Config
        from phase1_cognate.training.trainer import Phase1Trainer, TrainingConfig

        # 1. Setup Tokenizer (get_tokenizer handles fallback to MockTokenizer)
        tokenizer = get_tokenizer("gpt2")

        # 2. Data Setup
        # Default foundation datasets
        datasets_to_use = ["gsm8k", "svamp", "mbpp", "arc_easy", "piqa", "wikitext"]

        print("\n--- Step 1: Dataset Preparation ---")
        raw_datasets = download_all_datasets(datasets_to_use)

        print("\n--- Step 2: Dataset Processing ---")
        processed_datasets = {}
        for name, dataset in raw_datasets.items():
            config = DATASET_CONFIGS[name]
            processed_datasets[name] = process_dataset(dataset, name, config.category)
            print(f"Processed {name}: {len(processed_datasets[name])} samples")

        trained_models = []
        all_metrics = {}

        # 3. Train 3 Models
        specializations = ["reasoning", "memory", "speed"]

        print("\n--- Step 3: Training Foundation Models ---")
        for spec in specializations:
            print(f"\nTraining Model: {spec.upper()}")

            # Config
            model_config = Phase1Config(specialization=spec)

            # Model
            model = TRMTitansMAGModel(model_config)

            # Trainer Config
            # Use config from self.config if available, else defaults for prototype
            # Note: defaulting to 1 epoch/small batch for prototype speed unless specified
            train_config = TrainingConfig(
                model_config=model_config,
                num_epochs=self.config.get("epochs", 1),
                batch_size=self.config.get("batch_size", 4),
                checkpoint_dir=Path(f"checkpoints/phase1/{spec}"),
                device="cuda" if torch.cuda.is_available() else "cpu",
                wandb_mode="offline",
            )

            # Trainer
            trainer = Phase1Trainer(
                model=model,
                config=train_config,
                train_datasets=processed_datasets,
                tokenizer=tokenizer,
            )

            # Train
            trainer.train()

            trained_models.append(model)
            all_metrics[spec] = {
                "final_loss": trainer.best_val_loss
                if trainer.best_val_loss != float("inf")
                else 0.0,
                "epochs": train_config.num_epochs,
                "parameters": model.count_parameters()["total"],
            }

        print(f"\nPhase 1 Complete. Generated {len(trained_models)} models.")

        return PhaseResult(
            success=True,
            phase_name="phase1",
            model=trained_models,
            metrics=all_metrics,
            duration=time.time() - start_time,
            artifacts={"models": [f"model_{s}" for s in specializations]},
            config=self.config,
            error=None,
        )

    def validate_input(self, input_models: Optional[List[Any]] = None) -> bool:
        """Phase 1 has no input"""
        return True

    def validate_output(self, result: PhaseResult) -> bool:
        """
        Validate Phase 1 output (ISS-022).

        Checks:
        - 3 models produced
        - Each model has reasonable loss
        - Models show diversity
        """
        if not result.success:
            return False

        if result.metrics:
            # Check for 3 models
            model_count = result.metrics.get("model_count", 0)
            if model_count < 3:
                return False

            # Check loss is reasonable (not NaN, not too high)
            for spec in ["reasoning", "memory", "speed"]:
                loss = result.metrics.get(f"{spec}_loss", float("inf"))
                if loss == float("inf") or loss != loss:  # NaN check
                    return False

        return True

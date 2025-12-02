"""
Phase 1 (Cognate) Training Script

Train all 3 specialized TRM × Titans-MAG models:
- Model 1: Reasoning (ACT=0.95, LTM=4096)
- Model 2: Memory (ACT=0.90, LTM=8192)
- Model 3: Speed (ACT=0.99, LTM=2048)

Usage:
    # Train single model
    python train_phase1.py --model reasoning

    # Train all 3 models
    python train_phase1.py --all

    # Quick test on CPU
    python train_phase1.py --model reasoning --test --epochs 1
"""

import argparse
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from phase1_cognate.data.dataset_downloader import download_all_datasets, DATASET_CONFIGS
from phase1_cognate.data.dataset_processor import process_dataset
from phase1_cognate.training.trainer import Phase1Trainer, TrainingConfig
from cross_phase.utils import get_tokenizer


def get_tokenizer_phase1():
    """Get GPT-2 tokenizer with Phase 1 specific settings"""
    try:
        import os
        from transformers import GPT2Tokenizer

        # Temporarily disable offline mode for tokenizer download
        old_offline = os.environ.get('HF_DATASETS_OFFLINE', None)
        os.environ['HF_DATASETS_OFFLINE'] = '0'

        try:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=False)
            tokenizer.pad_token = tokenizer.eos_token
            print("Loaded GPT-2 tokenizer")
            return tokenizer
        finally:
            # Restore offline mode setting
            if old_offline:
                os.environ['HF_DATASETS_OFFLINE'] = old_offline

    except (ImportError, OSError) as e:
        print(f"WARNING: Could not load GPT-2 tokenizer ({e}), using mock tokenizer")
        return get_tokenizer("gpt2")


def download_and_process_datasets(dataset_names, cache_dir=None):
    """
    Download and process datasets

    Args:
        dataset_names: List of dataset names
        cache_dir: Cache directory

    Returns:
        Dict of processed datasets
    """
    print("\n" + "="*70)
    print("STEP 1: DOWNLOAD DATASETS")
    print("="*70 + "\n")

    # Download
    raw_datasets = download_all_datasets(dataset_names, cache_dir)

    print("\n" + "="*70)
    print("STEP 2: PROCESS DATASETS")
    print("="*70 + "\n")

    # Process
    processed_datasets = {}
    for name, dataset in raw_datasets.items():
        config = DATASET_CONFIGS[name]
        processed = process_dataset(dataset, name, config.category)
        processed_datasets[name] = processed
        print(f"Processed {name}: {len(processed)} samples")

    return processed_datasets


def train_single_model(
    specialization: str,
    datasets: dict,
    tokenizer,
    args
):
    """
    Train a single model

    Args:
        specialization: Model type (reasoning/memory/speed)
        datasets: Processed datasets
        tokenizer: Tokenizer
        args: Command line arguments
    """
    print("\n" + "="*70)
    print(f"TRAINING MODEL: {specialization.upper()}")
    print("="*70 + "\n")

    # Create model config
    model_config = Phase1Config(specialization=specialization)

    # Create model
    model = TRMTitansMAGModel(model_config)

    # Print parameter count
    param_counts = model.count_parameters()
    print(f"\nModel Parameters:")
    for component, count in param_counts.items():
        print(f"  {component}: {count:,}")
    print()

    # Create training config
    train_config = TrainingConfig(
        model_config=model_config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=Path(args.checkpoint_dir) / specialization,
        wandb_mode=args.wandb_mode,
        device=args.device
    )

    # Create trainer
    trainer = Phase1Trainer(
        model=model,
        config=train_config,
        train_datasets=datasets,
        val_datasets=datasets if args.validate else None,
        tokenizer=tokenizer
    )

    # Train
    trainer.train()

    print(f"\n✓ {specialization.upper()} model training complete!\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 (Cognate) Training")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["reasoning", "memory", "speed"],
        help="Train single model"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all 3 models sequentially"
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to use (default: foundation datasets)"
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="Dataset cache directory")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/phase1"))

    # Logging
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--no-validate", dest="validate", action="store_false", help="Disable validation")

    # Testing
    parser.add_argument("--test", action="store_true", help="Quick test mode (CPU, 1 epoch, small datasets)")

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.device = "cpu"
        args.epochs = 1
        args.batch_size = 2
        args.datasets = ["gsm8k", "piqa"]  # Just 2 datasets
        print("\n*** TEST MODE: CPU, 1 epoch, 2 datasets ***\n")

    # Default datasets (foundation stage)
    if args.datasets is None:
        args.datasets = [
            "gsm8k", "svamp", "mbpp", "arc_easy", "piqa", "wikitext"
        ]

    # Get tokenizer
    tokenizer = get_tokenizer_phase1()

    # Download and process datasets
    datasets = download_and_process_datasets(args.datasets, args.cache_dir)

    # Train models
    if args.all:
        # Train all 3 sequentially
        for spec in ["reasoning", "memory", "speed"]:
            train_single_model(spec, datasets, tokenizer, args)
    elif args.model:
        # Train single model
        train_single_model(args.model, datasets, tokenizer, args)
    else:
        print("Error: Must specify --model or --all")
        return 1

    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())

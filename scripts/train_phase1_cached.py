"""
Phase 1 Training with Pre-Cached Datasets

This script uses the 6 datasets already cached at D:\AIVillage\hf_cache
to start training immediately without re-downloading.
"""

import os
import sys
from pathlib import Path

# Set cache directory to D:/AIVillage/hf_cache
os.environ['HF_DATASETS_CACHE'] = 'D:/AIVillage/hf_cache/datasets'
os.environ['HF_HOME'] = 'D:/AIVillage/hf_cache'

# Force online mode BEFORE importing datasets
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'

# Configure W&B for online mode (dashboard at wandb.ai)
os.environ['WANDB_MODE'] = 'offline'  # Changed from 'offline' - enables dashboard syncing
os.environ['WANDB_DIR'] = str(Path.cwd() / 'wandb')
os.environ['WANDB_PROJECT'] = 'agent-forge-v2'  # Project name for dashboard

# Add src to path (scripts/ is one level down from root)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

# Import datasets with online mode enabled
from datasets import load_dataset
from datasets.config import HF_DATASETS_OFFLINE
print(f"HuggingFace Offline Mode: {HF_DATASETS_OFFLINE}")

# Import Phase 1 components
from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from phase1_cognate.training.trainer import Phase1Trainer, TrainingConfig
from phase1_cognate.data.dataset_processor import process_dataset
from cross_phase.utils import MockTokenizer, get_tokenizer


def load_cached_datasets():
    """Load and process the 6 cached datasets"""
    print("\n" + "="*70)
    print("LOADING CACHED DATASETS")
    print("="*70 + "\n")

    datasets = {}

    # Dataset configurations (name -> (hf_id, subset, split, type))
    dataset_configs = {
        "gsm8k": ("gsm8k", "main", "train", "math"),
        "svamp": ("ChilleD/SVAMP", None, "train", "math"),
        "mbpp": ("mbpp", "full", "train", "code"),
        "arc_easy": ("ai2_arc", "ARC-Easy", "train", "science"),
        "arc_challenge": ("ai2_arc", "ARC-Challenge", "train", "science"),
        "hellaswag": ("hellaswag", None, "train", "commonsense"),
    }

    for name, (hf_id, subset, split, dataset_type) in dataset_configs.items():
        try:
            # Load from cache (DON'T specify cache_dir - let HF find it automatically)
            if subset:
                dataset = load_dataset(hf_id, subset, split=split)
            else:
                dataset = load_dataset(hf_id, split=split)

            # Limit samples if needed
            max_samples = 10000  # Limit large datasets
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))

            # Process to standardized format
            print(f"  Processing {name}... ", end="", flush=True)
            processed = process_dataset(dataset, name, dataset_type)
            datasets[name] = processed
            print(f"OK - {len(processed)} samples", flush=True)

        except Exception as e:
            print(f"  ERROR {name}: {e}")

    print(f"\nLoaded and processed {len(datasets)}/6 datasets")
    print("="*70 + "\n")

    return datasets


def get_tokenizer_cached():
    """Get GPT-2 tokenizer or mock tokenizer"""
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=False)
        tokenizer.pad_token = tokenizer.eos_token
        print("Loaded GPT-2 tokenizer\n")
        return tokenizer
    except Exception as e:
        print(f"Using mock tokenizer (GPT-2 failed: {e})\n")
        return get_tokenizer("gpt2")


def train_model(specialization, datasets, tokenizer, device="cuda"):
    """Train a single model"""
    print("\n" + "="*70)
    print(f"TRAINING: {specialization.upper()} MODEL")
    print("="*70 + "\n")

    # Create config
    print("[1/6] Creating model configuration...")
    config = Phase1Config(specialization=specialization)

    # Create model
    print("[2/6] Instantiating TRM × Titans-MAG model...")
    model = TRMTitansMAGModel(config)

    # Print parameters
    print("[3/6] Counting model parameters...")
    params = model.count_parameters()
    print(f"Model Parameters: {params['total']:,}\n")

    # Training config
    print("[4/6] Creating training configuration...")
    train_config = TrainingConfig(
        model_config=config,
        num_epochs=10,
        batch_size=4,  # Reduced from 16 to fit 32M param model in 8GB VRAM
        checkpoint_dir=Path("checkpoints/phase1") / specialization,
        wandb_mode="online",  # Enable dashboard syncing to wandb.ai
        device=device
    )

    # Create trainer
    print("[5/6] Creating Phase1Trainer...")
    trainer = Phase1Trainer(
        model=model,
        config=train_config,
        train_datasets=datasets,
        val_datasets=datasets,
        tokenizer=tokenizer
    )

    # Check for existing checkpoints and resume if available
    print("[6/6] Checking for existing checkpoints...")
    checkpoint_dir = Path("checkpoints/phase1") / specialization
    latest_checkpoint = None

    if checkpoint_dir.exists():
        # Find latest epoch checkpoint
        checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
        if checkpoints:
            # Sort by epoch number (epoch_2.pt, epoch_4.pt, etc.)
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
            print(f"Found checkpoint: {latest_checkpoint}")

            # Load checkpoint
            if trainer.load_checkpoint(latest_checkpoint):
                print("Successfully resumed from checkpoint!\n")
            else:
                print("Failed to load checkpoint, starting from scratch\n")
        else:
            print("No checkpoints found, starting from scratch\n")
    else:
        print("No checkpoint directory found, starting from scratch\n")

    # Train
    print("Starting training loop...\n")
    trainer.train()

    print(f"\n{specialization.upper()} model training complete!\n")


def cleanup_gpu():
    """Force GPU cleanup before training"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory Cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def main():
    # Clean up GPU memory from previous runs
    cleanup_gpu()

    print("\n" + "="*70)
    print("PHASE 1 COGNATE - TRAINING WITH CACHED DATASETS")
    print("="*70)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load datasets
    datasets = load_cached_datasets()

    if len(datasets) == 0:
        print("ERROR: No datasets loaded. Cannot proceed with training.")
        return 1

    # Get tokenizer
    tokenizer = get_tokenizer_cached()

    # Train all 3 models
    for spec in ["reasoning", "memory", "speed"]:
        train_model(spec, datasets, tokenizer, device)

    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

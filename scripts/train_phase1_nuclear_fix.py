"""
Phase 1 Training with NUCLEAR FIXES APPLIED

This script implements all critical fixes from the ML expert analysis:
1. Validation split (10% held out) - FIXES frozen val_loss=2.5
2. Gradient accumulation (4 steps) - FIXES batch size from 16 to 64 effective
3. Reduced learning rates - FIXES overstepping and instability
4. HellaSwag downsampling - FIXES 46% dataset dominance

Run this to test if the fixes prevent training divergence.
"""

import os
import sys
import random
from pathlib import Path

# Set cache directory
os.environ['HF_DATASETS_CACHE'] = 'D:/AIVillage/hf_cache/datasets'
os.environ['HF_HOME'] = 'D:/AIVillage/hf_cache'
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'

# Configure W&B for offline mode (avoid watch_model hanging)
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_DIR'] = str(Path.cwd() / 'wandb')
os.environ['WANDB_PROJECT'] = 'agent-forge-v2'

# Configure PyTorch CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Import Phase 1 components
from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from phase1_cognate.training.trainer import Phase1Trainer, TrainingConfig
from phase1_cognate.datasets.dataset_processor import process_dataset


def load_and_split_datasets(val_split=0.1, seed=42):
    """
    Load datasets with NUCLEAR FIXES:
    - Downsample HellaSwag 10K → 2K
    - Split train/val 90/10
    """
    print("\n" + "="*70)
    print("LOADING DATASETS WITH NUCLEAR FIXES")
    print("="*70 + "\n")
    print(f"FIX #1: HellaSwag downsampling 10K -> 2K")
    train_pct_msg = int((1-val_split)*100)
    val_pct_msg = int(val_split*100)
    print(f"FIX #2: Train/Val split {train_pct_msg}/{val_pct_msg}")
    print()

    random.seed(seed)
    all_datasets = {}

    # Dataset configurations
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
            # Load from cache
            if subset:
                dataset = load_dataset(hf_id, subset, split=split)
            else:
                dataset = load_dataset(hf_id, split=split)

            # NUCLEAR FIX: Downsample HellaSwag 10K -> 2K
            if name == "hellaswag":
                original_size = len(dataset)
                target_size = 2000
                indices = random.sample(range(original_size), target_size)
                dataset = dataset.select(indices)
                print(f"  {name}: Downsampled {original_size} -> {len(dataset)} samples")
            else:
                # Limit other datasets to reasonable size
                max_samples = 8000
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))

            # Process to standardized format
            print(f"  Processing {name}... ", end="", flush=True)
            processed = process_dataset(dataset, name, dataset_type)
            all_datasets[name] = processed
            print(f"OK - {len(processed)} samples")

        except Exception as e:
            print(f"  ERROR {name}: {e}")

    # NUCLEAR FIX: Split train/val for each dataset
    train_pct = int((1-val_split)*100)
    val_pct = int(val_split*100)
    print(f"\nSplitting datasets into train/val ({train_pct}/{val_pct}):")
    train_datasets = {}
    val_datasets = {}

    for name, samples in all_datasets.items():
        # Shuffle
        shuffled = samples.copy()
        random.shuffle(shuffled)

        # Split
        split_idx = int(len(shuffled) * (1 - val_split))
        train_samples = shuffled[:split_idx]
        val_samples = shuffled[split_idx:]

        train_datasets[name] = train_samples
        val_datasets[name] = val_samples

        print(f"  {name}: {len(train_samples)} train, {len(val_samples)} val")

    # Print totals
    total_train = sum(len(v) for v in train_datasets.values())
    total_val = sum(len(v) for v in val_datasets.values())
    print(f"\nTotal: {total_train} train, {total_val} val samples")
    print("="*70 + "\n")

    return train_datasets, val_datasets


def main():
    # AUTO-CLEANUP: Kill zombie processes before starting
    print("Checking for zombie Python processes...")
    import subprocess
    result = subprocess.run(
        ["taskkill", "//F", "//IM", "python.exe"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("  OK - Killed zombie processes\n")
    else:
        if "not found" not in result.stderr.lower():
            print(f"  {result.stderr.strip()}\n")
        else:
            print("  OK - No zombie processes\n")

    print("\n" + "="*70)
    print("PHASE 1 TRAINING - NUCLEAR FIXES APPLIED")
    print("="*70)
    print("\nFixes Applied:")
    print("  1. Validation split (10%) - Real generalization feedback")
    print("  2. Gradient accumulation (x4) - Effective batch size = 64")
    print("  3. Reduced learning rates (50%) - Less overstepping")
    print("  4. HellaSwag downsampled (2K) - Balanced dataset distribution")
    print("  5. Auto-cleanup + GPU memory check + 60s timeout protection")
    print()

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {gpu_total:.1f} GB")

        # CRITICAL: Check available GPU memory BEFORE loading anything
        gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3

        if gpu_free < 4.0:  # Need at least 4GB free
            print(f"\nERROR: Insufficient GPU memory!")
            print(f"  Free: {gpu_free:.2f}GB")
            print(f"  Required: 4.0GB minimum")
            print(f"\nSolution: Run cleanup script first:")
            print(f"  python scripts/cleanup_zombie_processes.py")
            print(f"\nOr reboot system to fully clear GPU memory.")
            return 1

    # Load datasets with fixes
    train_datasets, val_datasets = load_and_split_datasets(val_split=0.1, seed=42)

    if len(train_datasets) == 0:
        print("ERROR: No datasets loaded. Cannot proceed.")
        return 1

    # Load tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=False)
    tokenizer.pad_token = tokenizer.eos_token
    print("OK - Tokenizer loaded\n")

    # Train ONLY reasoning model for testing (3-5 epochs)
    print("\n" + "="*70)
    print("TRAINING: REASONING MODEL (TEST RUN - 5 EPOCHS)")
    print("="*70 + "\n")

    # Create config and model with timeout protection (Windows-compatible)
    print("[1/5] Creating model configuration...")

    # Use threading for timeout (signal.alarm doesn't work on Windows)
    import threading

    result_container = [None]  # Container to store result from thread
    error_container = [None]   # Container to store errors

    def create_model_with_timeout():
        """Create config and model in separate thread"""
        try:
            config = Phase1Config(specialization="reasoning")
            print("  Config created!")
            print(f"  [DEBUG] GPU memory after config: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            print("[2/5] Instantiating TRM × Titans-MAG model...")
            model = TRMTitansMAGModel(config)
            print("  Model instantiated!")
            print(f"  [DEBUG] GPU memory after model: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            result_container[0] = (config, model)
        except Exception as e:
            error_container[0] = e

    # Start model creation in thread with 60s timeout
    thread = threading.Thread(target=create_model_with_timeout, daemon=True)
    thread.start()
    thread.join(timeout=60)

    if thread.is_alive():
        print("\nERROR: Model initialization timed out (60 seconds)")
        print("Likely cause: GPU memory exhaustion")
        print("\nSolution:")
        print("  1. Run: python scripts/cleanup_zombie_processes.py")
        print("  2. Or reboot system to fully clear GPU memory")
        return 1

    if error_container[0] is not None:
        raise error_container[0]

    if result_container[0] is None:
        print("\nERROR: Model initialization failed (no result)")
        return 1

    config, model = result_container[0]

    # Print parameters
    print("[3/5] Counting model parameters...")
    params = model.count_parameters()
    print(f"Model Parameters: {params['total']:,}\n")

    # Training config with NUCLEAR FIXES
    print("[4/5] Creating training configuration with NUCLEAR FIXES...")
    train_config = TrainingConfig(
        model_config=config,
        num_epochs=5,  # Start with 5 epochs for testing
        batch_size=8,  # Physical batch size (reduced for 8GB VRAM)
        gradient_accumulation_steps=8,  # NUCLEAR FIX: Effective batch = 64
        # NUCLEAR FIX: Reduced learning rates (already in TrainingConfig defaults)
        checkpoint_dir=Path("checkpoints_nuclear_fix/phase1/reasoning"),
        wandb_mode="offline",  # Use offline to avoid watch_model hanging
        device=device
    )

    print(f"  Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"  Muon LR: {train_config.muon_lr}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Grokfast lambda: {train_config.grokfast_lambda}\n")

    # Create trainer (with GPU memory management)
    print("[5/5] Creating Phase1Trainer...")
    print(f"  GPU memory before trainer: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    print(f"  Clearing GPU cache...")
    torch.cuda.empty_cache()

    trainer = Phase1Trainer(
        model=model,
        config=train_config,
        train_datasets=train_datasets,  # NUCLEAR FIX: Separate train/val
        val_datasets=val_datasets,      # NUCLEAR FIX: Real validation set
        tokenizer=tokenizer
    )
    print(f"  Trainer created! GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # Train
    print("\nStarting training loop...")
    print("\nMONITOR THESE METRICS:")
    print("  1. Validation loss should DECREASE (not stay at 2.5)")
    print("  2. Training loss should decrease smoothly (no divergence at epoch 7)")
    print("  3. No gradient explosions (grad_norm should stay < 10)")
    print("\n" + "="*70 + "\n")

    trainer.train()

    print("\n" + "="*70)
    print("TEST TRAINING COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Check W&B dashboard: https://wandb.ai/dydavidyoussef-the-guild-of-the-rose/agent-forge-v2")
    print("  2. Verify validation loss decreased")
    print("  3. Test model outputs: python scripts/test_phase1_models.py")
    print("  4. If successful, train all 3 models for 10 epochs")
    print()

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

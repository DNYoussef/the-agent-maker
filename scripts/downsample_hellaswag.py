"""
Downsample HellaSwag dataset from 10K to 2K samples

This addresses the dataset imbalance issue where HellaSwag dominates
46% of the training data, causing format conflicts with Q&A tasks.
"""

import json
import random
from pathlib import Path

def downsample_hellaswag(
    input_file: Path,
    output_file: Path,
    target_size: int = 2000,
    seed: int = 42
):
    """
    Downsample HellaSwag to target size

    Args:
        input_file: Path to original hellaswag.json (10K samples)
        output_file: Path to save downsampled version (2K samples)
        target_size: Target number of samples
        seed: Random seed for reproducibility
    """
    print(f"Downsampling HellaSwag: {input_file}")

    # Load original data
    with open(input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    original_size = len(samples)
    print(f"  Original size: {original_size} samples")

    # Downsample
    random.seed(seed)
    downsampled = random.sample(samples, min(target_size, original_size))

    print(f"  Downsampled to: {len(downsampled)} samples ({len(downsampled)/original_size*100:.1f}%)")

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(downsampled, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_file}")

    return len(downsampled)

if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "processed_datasets"

    input_file = processed_dir / "hellaswag.json"
    output_file = processed_dir / "hellaswag_downsampled.json"

    if not input_file.exists():
        print(f"ERROR: {input_file} not found!")
        print("Run download_datasets.py first to create processed datasets.")
        exit(1)

    # Downsample
    final_size = downsample_hellaswag(
        input_file=input_file,
        output_file=output_file,
        target_size=2000,
        seed=42
    )

    print(f"\nSuccess! HellaSwag downsampled: 10,000 -> {final_size}")
    print(f"\nTo use the downsampled version, modify your training script to load:")
    print(f"  'hellaswag_downsampled.json' instead of 'hellaswag.json'")

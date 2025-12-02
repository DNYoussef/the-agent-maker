"""
Download all 16 HuggingFace datasets for Phase 1 training

This script downloads and caches all required datasets.
Run this before starting model training.
"""

import os
import sys
from pathlib import Path

# Force online mode (disable HF_DATASETS_OFFLINE)
os.environ['HF_DATASETS_OFFLINE'] = '0'
print("Enabled online mode for dataset downloads")

# Add src to path (scripts/ is one level down from root)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_cognate.datasets.dataset_downloader import (
    download_all_datasets,
    get_dataset_stats,
    DATASET_CONFIGS
)


def main():
    print("\n" + "="*70)
    print("PHASE 1 COGNATE - DATASET DOWNLOAD")
    print("="*70)
    print("\nThis will download 16 HuggingFace datasets:")
    print("  - Math: GSM8K, SVAMP, ASDiv")
    print("  - Code: MBPP, CodeXGLUE")
    print("  - Science: ARC-Easy, ARC-Challenge")
    print("  - Multi-Hop: HotpotQA, DROP, StrategyQA")
    print("  - Commonsense: PIQA, HellaSwag, BoolQ")
    print("  - Language: WikiText")
    print("\nDatasets will be cached in: ~/.cache/huggingface/datasets/")
    print("Estimated download size: ~2-3 GB")
    print("Estimated time: 10-30 minutes (depending on connection)")
    print("\n" + "="*70 + "\n")

    # Download all datasets
    all_dataset_names = list(DATASET_CONFIGS.keys())

    print(f"Starting download of {len(all_dataset_names)} datasets...\n")

    datasets = download_all_datasets(all_dataset_names)

    # Print statistics
    if datasets:
        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)

        stats = get_dataset_stats(datasets)

        print("\nDataset Statistics by Category:")
        category_order = ["math", "code", "science", "multihop", "commonsense", "language"]
        for category in category_order:
            if category in stats:
                print(f"  {category.capitalize():12s}: {stats[category]:>6,} samples")

        total = sum(stats.values())
        print(f"  {'-'*12}   {'-'*6}")
        print(f"  {'Total':12s}: {total:>6,} samples")

        print("\n" + "="*70)
        print(f"SUCCESS: Downloaded {len(datasets)}/{len(all_dataset_names)} datasets")
        print("="*70)

        # List any failed downloads
        failed = set(all_dataset_names) - set(datasets.keys())
        if failed:
            print("\nWARNING: Failed to download:")
            for name in failed:
                print(f"  - {name}")
            print("\nYou can retry failed downloads or continue with available datasets.")

        print("\nNext steps:")
        print("  1. Verify datasets loaded correctly")
        print("  2. Start training: python src/phase1_cognate/train_phase1.py --all")
        print("\n")
    else:
        print("\n" + "="*70)
        print("ERROR: No datasets downloaded successfully")
        print("="*70)
        print("\nPlease check your internet connection and try again.")
        print("If problems persist, check HuggingFace status: https://status.huggingface.co/")
        sys.exit(1)


if __name__ == "__main__":
    main()

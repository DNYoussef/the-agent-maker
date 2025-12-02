"""Debug script to check import paths and datasets availability"""

import sys
from pathlib import Path

print("="*70)
print("IMPORT DEBUG")
print("="*70)

print("\n1. Python executable:")
print(f"   {sys.executable}")

print("\n2. Python paths:")
for i, p in enumerate(sys.path[:10]):
    print(f"   [{i}] {p}")

print("\n3. Testing 'datasets' import:")
try:
    from datasets import load_dataset
    print("   [OK] SUCCESS: datasets library imported")
    print(f"   Location: {load_dataset.__module__}")
except ImportError as e:
    print(f"   [FAIL] {e}")

print("\n4. Adding 'src' to path and importing dataset_downloader:")
sys.path.insert(0, str(Path(__file__).parent / 'src'))
try:
    from phase1_cognate.datasets.dataset_downloader import DATASETS_AVAILABLE
    print(f"   DATASETS_AVAILABLE = {DATASETS_AVAILABLE}")
except Exception as e:
    print(f"   [FAIL] {e}")

print("\n5. Testing actual dataset load:")
try:
    from datasets import load_dataset
    ds = load_dataset('gsm8k', 'main', split='train[:5]')
    print(f"   [OK] SUCCESS: Loaded {len(ds)} samples from GSM8K")
except Exception as e:
    print(f"   [FAIL] {e}")

print("\n" + "="*70)

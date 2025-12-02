"""
Cleanup Zombie Training Processes

Kills all zombie Python processes and clears GPU memory.
Run this before starting training to ensure clean GPU state.

Usage:
    python scripts/cleanup_zombie_processes.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*70)
    print("CLEANUP: Killing Zombie Training Processes")
    print("="*70)
    print()

    # Step 1: Kill all Python processes
    print("Step 1: Killing all Python processes...")
    try:
        result = subprocess.run(
            ["taskkill", "//F", "//IM", "python.exe"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  OK - Python processes killed")
        else:
            if "not found" in result.stderr.lower():
                print("  OK - No Python processes running")
            else:
                print(f"  {result.stderr.strip()}")
    except Exception as e:
        print(f"  Warning: {e}")

    # Step 2: Clear GPU memory
    print("\nStep 2: Clearing GPU memory...")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Check GPU memory
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_mem_free = gpu_mem_total - gpu_mem_used

            print(f"  GPU Memory:")
            print(f"    Used: {gpu_mem_used:.2f}GB")
            print(f"    Free: {gpu_mem_free:.2f}GB")
            print(f"    Total: {gpu_mem_total:.1f}GB")

            if gpu_mem_free < 4.0:
                print()
                print("  WARNING: Less than 4GB free!")
                print("  Recommendation: Reboot system to fully clear GPU memory")
        else:
            print("  No CUDA GPU detected")
    except ImportError:
        print("  PyTorch not installed, skipping GPU check")
    except Exception as e:
        print(f"  Warning: {e}")

    print()
    print("="*70)
    print("CLEANUP COMPLETE!")
    print("="*70)
    print()
    print("Next step:")
    print("  python scripts/train_phase1_nuclear_fix.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCleanup interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

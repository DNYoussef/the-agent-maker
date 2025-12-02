#!/usr/bin/env python3
"""
Quick test script for Streamlit UI
Verifies all pages can be imported and basic functionality works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_ui_imports():
    """Test that all UI modules can be imported"""
    print("=" * 70)
    print("TEST: UI MODULE IMPORTS")
    print("=" * 70)

    results = []

    try:
        from ui.pages import pipeline_overview
        print("[OK] pipeline_overview imported")
        results.append(("pipeline_overview", True))
    except Exception as e:
        print(f"[FAIL] pipeline_overview import failed: {e}")
        results.append(("pipeline_overview", False))

    try:
        from ui.pages import phase_details
        print("[OK] phase_details imported")
        results.append(("phase_details", True))
    except Exception as e:
        print(f"[FAIL] phase_details import failed: {e}")
        results.append(("phase_details", False))

    try:
        from ui.pages import model_browser
        print("[OK] model_browser imported")
        results.append(("model_browser", True))
    except Exception as e:
        print(f"[FAIL] model_browser import failed: {e}")
        results.append(("model_browser", False))

    try:
        from ui.pages import system_monitor
        print("[OK] system_monitor imported")
        results.append(("system_monitor", True))
    except Exception as e:
        print(f"[FAIL] system_monitor import failed: {e}")
        results.append(("system_monitor", False))

    try:
        from ui.pages import config_editor
        print("[OK] config_editor imported")
        results.append(("config_editor", True))
    except Exception as e:
        print(f"[FAIL] config_editor import failed: {e}")
        results.append(("config_editor", False))

    return results


def test_dependencies():
    """Test that required dependencies are installed"""
    print("\n" + "=" * 70)
    print("TEST: DEPENDENCIES")
    print("=" * 70)

    results = []

    try:
        import streamlit
        print(f"[OK] streamlit version {streamlit.__version__}")
        results.append(("streamlit", True))
    except ImportError:
        print("[FAIL] streamlit not installed")
        results.append(("streamlit", False))

    try:
        import yaml
        print("[OK] pyyaml installed")
        results.append(("pyyaml", True))
    except ImportError:
        print("[FAIL] pyyaml not installed")
        results.append(("pyyaml", False))

    try:
        import psutil
        print(f"[OK] psutil version {psutil.__version__}")
        results.append(("psutil", True))
    except ImportError:
        print("[FAIL] psutil not installed")
        results.append(("psutil", False))

    try:
        import pandas
        print(f"[OK] pandas version {pandas.__version__}")
        results.append(("pandas", True))
    except ImportError:
        print("[FAIL] pandas not installed")
        results.append(("pandas", False))

    try:
        import torch
        print(f"[OK] torch version {torch.__version__} (GPU monitoring available)")
        results.append(("torch", True))
    except ImportError:
        print("[WARN] torch not installed (GPU monitoring unavailable)")
        results.append(("torch", False))

    return results


def test_config_loading():
    """Test configuration loading"""
    print("\n" + "=" * 70)
    print("TEST: CONFIGURATION LOADING")
    print("=" * 70)

    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        print("[OK] Configuration loaded successfully")

        # Check required sections
        required_sections = ["wandb", "phases", "hardware", "cleanup", "registry"]
        missing = [s for s in required_sections if s not in config]

        if missing:
            print(f"[WARN] Missing sections: {missing}")
            return False
        else:
            print("[OK] All required sections present")
            return True

    except FileNotFoundError:
        print(f"[FAIL] Configuration file not found: {config_path}")
        return False
    except Exception as e:
        print(f"[FAIL] Configuration loading failed: {e}")
        return False


def test_system_monitoring():
    """Test system monitoring functions"""
    print("\n" + "=" * 70)
    print("TEST: SYSTEM MONITORING")
    print("=" * 70)

    try:
        import psutil

        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        print(f"[OK] CPU usage: {cpu_percent:.1f}%")

        # RAM
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024 ** 3)
        ram_total_gb = ram.total / (1024 ** 3)
        print(f"[OK] RAM usage: {ram_used_gb:.1f} / {ram_total_gb:.1f} GB ({ram.percent:.1f}%)")

        # Disk
        disk = psutil.disk_usage('.')
        disk_used_gb = disk.used / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)
        print(f"[OK] Disk usage: {disk_used_gb:.1f} / {disk_total_gb:.1f} GB ({disk.percent:.1f}%)")

        return True

    except Exception as e:
        print(f"[FAIL] System monitoring failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("STREAMLIT UI - QUICK TEST SUITE")
    print("Agent Forge V2")
    print("=" * 70)

    all_results = {}

    # Run tests
    all_results["UI Imports"] = test_ui_imports()
    all_results["Dependencies"] = test_dependencies()
    all_results["Config Loading"] = test_config_loading()
    all_results["System Monitoring"] = test_system_monitoring()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_tests = 0
    passed_tests = 0

    for test_name, results in all_results.items():
        print(f"\n{test_name}:")

        if isinstance(results, list):
            for name, passed in results:
                status = "[OK]" if passed else "[FAIL]"
                print(f"  {status} {name}")
                total_tests += 1
                if passed:
                    passed_tests += 1
        elif isinstance(results, bool):
            status = "[OK]" if results else "[FAIL]"
            print(f"  {status}")
            total_tests += 1
            if results:
                passed_tests += 1

    print("\n" + "=" * 70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed:      {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Failed:      {total_tests - passed_tests}")
    print("=" * 70)

    if passed_tests == total_tests:
        print("\n[OK] ALL TESTS PASSED - UI Ready for Launch!")
        print("\nTo launch dashboard:")
        print("  streamlit run src/ui/app.py")
        return 0
    else:
        print(f"\n[WARN] {total_tests - passed_tests} TEST(S) FAILED")
        print("\nInstall missing dependencies:")
        print("  pip install -r requirements-ui.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

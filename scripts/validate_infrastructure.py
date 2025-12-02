#!/usr/bin/env python3
"""
Infrastructure Validation Script
Tests all core Agent Forge V2 infrastructure components
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def validate_imports():
    """Test that all modules can be imported"""
    print("=" * 60)
    print("VALIDATING IMPORTS")
    print("=" * 60)

    try:
        from cross_phase.storage.model_registry import ModelRegistry
        print("[OK] ModelRegistry imported")
    except Exception as e:
        print(f"[FAIL] ModelRegistry import failed: {e}")
        return False

    try:
        from cross_phase.orchestrator.pipeline import PipelineOrchestrator
        print("[OK] PipelineOrchestrator imported")
    except Exception as e:
        print(f"[FAIL] PipelineOrchestrator import failed: {e}")
        return False

    try:
        from cross_phase.orchestrator.phase_controller import (
            PhaseController, PhaseResult,
            Phase1Controller, Phase2Controller
        )
        print("[OK] PhaseControllers imported")
    except Exception as e:
        print(f"[FAIL] PhaseControllers import failed: {e}")
        return False

    try:
        from cross_phase.mugrokfast.optimizer import (
            MuonGrokfast, create_optimizer_from_phase
        )
        print("[OK] MuonGrokfast imported")
    except Exception as e:
        print(f"[FAIL] MuonGrokfast import failed: {e}")
        return False

    try:
        from cross_phase.monitoring.wandb_integration import (
            WandBIntegration, MetricContinuityTracker
        )
        print("[OK] WandBIntegration imported")
    except Exception as e:
        print(f"[FAIL] WandBIntegration import failed: {e}")
        return False

    try:
        from cross_phase.prompt_baking.baker import (
            PromptBaker, bake_prompt
        )
        print("[OK] PromptBaker imported")
    except Exception as e:
        print(f"[FAIL] PromptBaker import failed: {e}")
        return False

    try:
        from cross_phase.utils import (
            get_model_size, calculate_safe_batch_size
        )
        print("[OK] Utils imported")
    except Exception as e:
        print(f"[FAIL] Utils import failed: {e}")
        return False

    return True


def validate_model_registry():
    """Test Model Registry functionality"""
    print("\n" + "=" * 60)
    print("VALIDATING MODEL REGISTRY")
    print("=" * 60)

    try:
        from cross_phase.storage.model_registry import ModelRegistry

        # Create test registry
        test_db = "./test_registry.db"
        registry = ModelRegistry(test_db)
        print("[OK] Registry created with WAL mode")

        # Create test session
        registry.create_session("test_session", {"test": "config"})
        print("[OK] Session created")

        # Update session progress
        registry.update_session_progress("test_session", "phase1", 25.0)
        print("[OK] Session progress updated")

        # Cleanup
        registry.close()
        if os.path.exists(test_db):
            os.remove(test_db)
        if os.path.exists(test_db + "-wal"):
            os.remove(test_db + "-wal")
        if os.path.exists(test_db + "-shm"):
            os.remove(test_db + "-shm")
        print("[OK] Registry cleaned up")

        return True
    except Exception as e:
        print(f"[FAIL] Registry validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_configuration():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("VALIDATING CONFIGURATION")
    print("=" * 60)

    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(f"[OK] Config loaded")

        # Check required sections
        assert "wandb" in config, "Missing wandb section"
        assert "phases" in config, "Missing phases section"
        assert "hardware" in config, "Missing hardware section"
        print("[OK] All required sections present")

        # Check all 8 phases configured
        for phase_num in range(1, 9):
            phase_key = f"phase{phase_num}"
            assert phase_key in config["phases"], f"Missing {phase_key}"
        print("[OK] All 8 phases configured")

        return True
    except Exception as e:
        print(f"[FAIL] Config validation failed: {e}")
        return False


def validate_mugrokfast():
    """Test MuGrokfast optimizer"""
    print("\n" + "=" * 60)
    print("VALIDATING MUGROKFAST OPTIMIZER")
    print("=" * 60)

    try:
        from cross_phase.mugrokfast.config import MuGrokConfig

        # Test phase presets
        for phase_num in [1, 3, 5, 6, 7]:
            config = MuGrokConfig.from_phase(phase_num)
            print(f"[OK] Phase {phase_num} preset: muon_lr={config.muon_lr}, lambda={config.grokfast_lambda}")

        return True
    except Exception as e:
        print(f"[FAIL] MuGrokfast validation failed: {e}")
        return False


def validate_prompt_templates():
    """Test prompt templates"""
    print("\n" + "=" * 60)
    print("VALIDATING PROMPT TEMPLATES")
    print("=" * 60)

    try:
        from cross_phase.prompt_baking.prompts import PromptManager

        phase3 = PromptManager.get_phase3_prompts()
        print(f"[OK] Phase 3: {len(phase3)} prompts")

        phase5 = PromptManager.get_phase5_prompts()
        print(f"[OK] Phase 5: {len(phase5)} prompts")

        phase6 = PromptManager.get_phase6_prompts()
        print(f"[OK] Phase 6: {len(phase6)} prompts (9 personas + SWE-Bench)")

        return True
    except Exception as e:
        print(f"[FAIL] Prompt template validation failed: {e}")
        return False


def main():
    """Run all validation tests"""
    print("\n" + "=" * 60)
    print("AGENT FORGE V2 - INFRASTRUCTURE VALIDATION")
    print("=" * 60 + "\n")

    results = {
        "Imports": validate_imports(),
        "Model Registry": validate_model_registry(),
        "Configuration": validate_configuration(),
        "MuGrokfast": validate_mugrokfast(),
        "Prompt Templates": validate_prompt_templates()
    }

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{test_name:20} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] ALL VALIDATIONS PASSED - Infrastructure Ready!")
    else:
        print("[FAIL] SOME VALIDATIONS FAILED - Review errors above")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

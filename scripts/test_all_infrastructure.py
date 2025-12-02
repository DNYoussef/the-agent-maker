#!/usr/bin/env python3
"""
Comprehensive Infrastructure Testing Script
Tests all Agent Forge V2 components with real operations
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Dummy PyTorch if not installed
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("[WARN] PyTorch not installed - using mock objects")
    TORCH_AVAILABLE = False
    # Create mock torch module
    class MockModule:
        def parameters(self):
            return []
        def named_parameters(self):
            return []
        def named_modules(self):
            return []
    class MockTorch:
        nn = type('nn', (), {'Module': MockModule})
        @staticmethod
        def zeros_like(x):
            return 0
        @staticmethod
        def randn(*args, **kwargs):
            return None
    torch = MockTorch()


def test_model_registry():
    """Test SQLite Model Registry with real operations"""
    print("\n" + "=" * 70)
    print("TEST 1: MODEL REGISTRY")
    print("=" * 70)

    from cross_phase.storage.model_registry import ModelRegistry

    test_db = "./test_infrastructure_registry.db"
    results = []

    try:
        # Test 1.1: Create registry
        print("\n[TEST 1.1] Creating registry with WAL mode...")
        registry = ModelRegistry(test_db)
        results.append(("[OK]", "Registry created"))

        # Test 1.2: Create session
        print("[TEST 1.2] Creating session...")
        registry.create_session("test_session_001", {
            "pipeline": "agent-forge-v2",
            "version": "1.0.0"
        })
        results.append(("[OK]", "Session created"))

        # Test 1.3: Update progress
        print("[TEST 1.3] Updating session progress...")
        registry.update_session_progress("test_session_001", "phase1", 12.5)
        results.append(("[OK]", "Progress updated to 12.5%"))

        registry.update_session_progress("test_session_001", "phase2", 25.0)
        results.append(("[OK]", "Progress updated to 25.0%"))

        # Test 1.4: Register model (without actual file)
        print("[TEST 1.4] Registering model metadata...")
        # Create dummy file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            dummy_model_path = f.name
            f.write(b"dummy model data" * 1000)  # ~16KB

        model_id = registry.register_model(
            session_id="test_session_001",
            phase_name="phase1",
            model_name="model1_reasoning",
            model_path=dummy_model_path,
            metadata={'parameters': 25_000_000, 'architecture': 'TRM'}
        )
        results.append(("[OK]", f"Model registered: {model_id}"))

        # Test 1.5: Retrieve model
        print("[TEST 1.5] Retrieving model metadata...")
        model_info = registry.get_model(model_id=model_id)
        assert model_info['model_id'] == model_id
        assert model_info['phase_name'] == "phase1"
        assert model_info['parameters'] == 25_000_000
        results.append(("[OK]", f"Model retrieved: {model_info['size_mb']:.2f} MB"))

        # Test 1.6: WAL checkpoint
        print("[TEST 1.6] Performing WAL checkpoint...")
        registry.checkpoint_wal()
        results.append(("[OK]", "WAL checkpoint successful"))

        # Test 1.7: Incremental vacuum
        print("[TEST 1.7] Running incremental vacuum...")
        registry.vacuum_incremental(pages=10)
        results.append(("[OK]", "Incremental vacuum successful"))

        # Cleanup
        registry.close()
        os.remove(dummy_model_path)
        if os.path.exists(test_db):
            os.remove(test_db)
        if os.path.exists(test_db + "-wal"):
            os.remove(test_db + "-wal")
        if os.path.exists(test_db + "-shm"):
            os.remove(test_db + "-shm")
        results.append(("[OK]", "Cleanup successful"))

    except Exception as e:
        results.append(("[FAIL]", f"Registry test failed: {e}"))
        import traceback
        traceback.print_exc()

    return results


def test_mugrokfast_optimizer():
    """Test MuGrokfast Optimizer"""
    print("\n" + "=" * 70)
    print("TEST 2: MUGROKFAST OPTIMIZER")
    print("=" * 70)

    from cross_phase.mugrokfast.optimizer import MuonGrokfast, create_optimizer_from_phase
    from cross_phase.mugrokfast.config import MuGrokConfig

    results = []

    try:
        # Test 2.1: Load all phase presets
        print("\n[TEST 2.1] Loading phase presets...")
        for phase_num in [1, 3, 5, 6, 7]:
            config = MuGrokConfig.from_phase(phase_num)
            results.append((
                "[OK]",
                f"Phase {phase_num}: lr={config.muon_lr}, lambda={config.grokfast_lambda}, kl={config.kl_coefficient}"
            ))

        # Test 2.2: Create optimizer (if torch available)
        if TORCH_AVAILABLE:
            print("[TEST 2.2] Creating optimizer with dummy model...")
            model = nn.Linear(512, 512)
            optimizer = create_optimizer_from_phase(model, phase_num=1)
            results.append(("[OK]", f"Optimizer created: {type(optimizer).__name__}"))

            # Test 2.3: Get metrics
            print("[TEST 2.3] Getting optimizer metrics...")
            muon_lr = optimizer.get_muon_lr()
            mu_norm = optimizer.get_mu_norm()
            results.append(("[OK]", f"Metrics: muon_lr={muon_lr}, mu_norm={mu_norm:.4f}"))
        else:
            results.append(("[SKIP]", "PyTorch not available - skipping optimizer instantiation"))

    except Exception as e:
        results.append(("[FAIL]", f"MuGrokfast test failed: {e}"))
        import traceback
        traceback.print_exc()

    return results


def test_utils():
    """Test model-size-agnostic utilities"""
    print("\n" + "=" * 70)
    print("TEST 3: MODEL-SIZE UTILITIES")
    print("=" * 70)

    results = []

    try:
        if TORCH_AVAILABLE:
            from cross_phase.utils import get_model_size, calculate_safe_batch_size

            # Test 3.1: Get model size
            print("\n[TEST 3.1] Testing get_model_size...")
            model = nn.Linear(512, 512)  # ~262K params
            size_info = get_model_size(model)

            assert 'params' in size_info
            assert 'size_mb' in size_info
            assert 'size_category' in size_info

            results.append((
                "[OK]",
                f"Model size: {size_info['params']:,} params, "
                f"{size_info['size_mb']:.2f} MB, category={size_info['size_category']}"
            ))

            # Test 3.2: Calculate batch size
            print("[TEST 3.2] Testing calculate_safe_batch_size...")
            batch_size, acc_steps = calculate_safe_batch_size(model, device_vram_gb=6)

            assert batch_size >= 1
            assert acc_steps >= 1

            results.append((
                "[OK]",
                f"Batch config: batch_size={batch_size}, accumulation_steps={acc_steps}"
            ))
        else:
            results.append(("[SKIP]", "PyTorch not available - skipping utils tests"))

    except Exception as e:
        results.append(("[FAIL]", f"Utils test failed: {e}"))
        import traceback
        traceback.print_exc()

    return results


def test_prompt_baking():
    """Test Prompt Baking System"""
    print("\n" + "=" * 70)
    print("TEST 4: PROMPT BAKING SYSTEM")
    print("=" * 70)

    from cross_phase.prompt_baking.prompts import PromptManager
    from cross_phase.prompt_baking.baker import PromptBakingConfig

    results = []

    try:
        # Test 4.1: Load prompts
        print("\n[TEST 4.1] Loading prompt templates...")
        phase3 = PromptManager.get_phase3_prompts()
        phase5 = PromptManager.get_phase5_prompts()
        phase6 = PromptManager.get_phase6_prompts()

        results.append(("[OK]", f"Phase 3: {len(phase3)} prompts loaded"))
        results.append(("[OK]", f"Phase 5: {len(phase5)} prompts loaded"))
        results.append(("[OK]", f"Phase 6: {len(phase6)} prompts loaded (9 personas + tools)"))

        # Test 4.2: Verify prompt content
        print("[TEST 4.2] Verifying prompt content...")
        assert len(phase3[0]) > 50, "Phase 3 prompt too short"
        assert "reasoning" in phase3[0].lower(), "Phase 3 prompt missing 'reasoning'"
        results.append(("[OK]", "Phase 3 prompt content valid"))

        # Test 4.3: Create baking config
        print("[TEST 4.3] Creating baking configuration...")
        config = PromptBakingConfig(
            lora_r=16,
            num_epochs=3,
            half_baking=True
        )
        assert config.lora_r == 16
        assert config.half_baking == True
        results.append(("[OK]", f"Baking config: r={config.lora_r}, epochs={config.num_epochs}, half={config.half_baking}"))

    except Exception as e:
        results.append(("[FAIL]", f"Prompt baking test failed: {e}"))
        import traceback
        traceback.print_exc()

    return results


def test_wandb_integration():
    """Test W&B Integration"""
    print("\n" + "=" * 70)
    print("TEST 5: WEIGHTS & BIASES INTEGRATION")
    print("=" * 70)

    from cross_phase.monitoring.wandb_integration import WandBIntegration, MetricContinuityTracker

    results = []

    try:
        # Test 5.1: Create W&B integration
        print("\n[TEST 5.1] Creating W&B integration (offline mode)...")
        wandb_integration = WandBIntegration(
            project_name="test-agent-forge",
            mode="disabled"  # Don't actually init wandb
        )
        results.append(("[OK]", "W&B integration created (offline mode)"))

        # Test 5.2: Check metrics count
        print("[TEST 5.2] Verifying metrics configuration...")
        total_metrics = sum(WandBIntegration.METRICS_COUNT.values())
        assert total_metrics == 676, f"Expected 676 metrics, got {total_metrics}"
        results.append(("[OK]", f"Total metrics: {total_metrics} (Phase 1: 37, Phase 2: 370, ...)"))

        # Test 5.3: Test metric continuity tracker
        print("[TEST 5.3] Testing metric continuity tracker...")
        tracker = MetricContinuityTracker()
        tracker.record_phase('phase1', {
            'accuracy': 0.12,
            'perplexity': 15.2,
            'model_size_mb': 95.4,
            'inference_latency_ms': 45
        })
        tracker.record_phase('phase2', {
            'accuracy': 0.15,
            'perplexity': 13.2,
            'model_size_mb': 95.4,
            'inference_latency_ms': 42
        })
        assert len(tracker.phase_names) == 2
        assert len(tracker.metrics['accuracy']) == 2
        results.append(("[OK]", f"Metric continuity: {len(tracker.phase_names)} phases tracked"))

    except Exception as e:
        results.append(("[FAIL]", f"W&B integration test failed: {e}"))
        import traceback
        traceback.print_exc()

    return results


def test_configuration():
    """Test Configuration System"""
    print("\n" + "=" * 70)
    print("TEST 6: CONFIGURATION SYSTEM")
    print("=" * 70)

    results = []

    try:
        import yaml

        # Test 6.1: Load config
        print("\n[TEST 6.1] Loading pipeline configuration...")
        config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        results.append(("[OK]", "Configuration loaded"))

        # Test 6.2: Check sections
        print("[TEST 6.2] Validating configuration sections...")
        required_sections = ["wandb", "registry", "hardware", "phases", "cleanup"]
        for section in required_sections:
            assert section in config, f"Missing section: {section}"
        results.append(("[OK]", f"All {len(required_sections)} required sections present"))

        # Test 6.3: Check all phases
        print("[TEST 6.3] Validating phase configurations...")
        for phase_num in range(1, 9):
            phase_key = f"phase{phase_num}"
            assert phase_key in config["phases"], f"Missing {phase_key}"
            phase_config = config["phases"][phase_key]
            assert "wandb_metrics" in phase_config, f"{phase_key} missing wandb_metrics"
        results.append(("[OK]", "All 8 phases configured with metrics"))

        # Test 6.4: Validate optimizer configs
        print("[TEST 6.4] Validating optimizer configurations...")
        for phase_num in [1, 2, 3, 4]:
            phase_config = config["phases"][f"phase{phase_num}"]
            if "optimizer" in phase_config:
                optimizer_config = phase_config["optimizer"]
                assert "type" in optimizer_config
                results.append(("[OK]", f"Phase {phase_num} optimizer: {optimizer_config['type']}"))

    except Exception as e:
        results.append(("[FAIL]", f"Configuration test failed: {e}"))
        import traceback
        traceback.print_exc()

    return results


def main():
    """Run all comprehensive tests"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE INFRASTRUCTURE TESTING")
    print("Agent Forge V2 - All Components")
    print("=" * 70)

    all_results = {}

    # Run all tests
    all_results["Model Registry"] = test_model_registry()
    all_results["MuGrokfast Optimizer"] = test_mugrokfast_optimizer()
    all_results["Model-Size Utilities"] = test_utils()
    all_results["Prompt Baking"] = test_prompt_baking()
    all_results["W&B Integration"] = test_wandb_integration()
    all_results["Configuration"] = test_configuration()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0

    for component, results in all_results.items():
        print(f"\n{component}:")
        for status, message in results:
            print(f"  {status} {message}")
            total_tests += 1
            if status == "[OK]":
                passed_tests += 1
            elif status == "[FAIL]":
                failed_tests += 1
            elif status == "[SKIP]":
                skipped_tests += 1

    print("\n" + "=" * 70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed:      {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Failed:      {failed_tests}")
    print(f"Skipped:     {skipped_tests}")
    print("=" * 70)

    if failed_tests == 0:
        print("\n[OK] ALL TESTS PASSED - Infrastructure Fully Functional!")
        return 0
    else:
        print(f"\n[FAIL] {failed_tests} TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

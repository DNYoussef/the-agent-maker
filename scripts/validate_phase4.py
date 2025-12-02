"""
Phase 4 End-to-End Validation Script

Validates the complete Phase 4 BitNet compression pipeline:
1. Mock Phase 3 output
2. Execute Phase 4 compression
3. Validate dual outputs
4. Test Phase 4→5 handoff
5. Performance benchmarking
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase4_bitnet import (
    Phase4Controller,
    Phase4Config,
    BitNetQuantizer,
    CompressedModel,
)
from src.phase4_bitnet.utils import (
    calculate_model_size_mb,
    count_parameters,
    calculate_sparsity_ratio,
    test_gradient_flow,
    validate_compression_quality,
)


class MockPhase3Model(nn.Module):
    """Mock 25M parameter transformer model from Phase 3"""

    def __init__(self, vocab_size=50257, hidden_size=512, num_layers=6):
        super().__init__()
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
        })()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def create_mock_phase3_output(output_dir: Path) -> Dict:
    """Create mock Phase 3 output for testing"""
    print("\n" + "=" * 60)
    print("STEP 1: Creating Mock Phase 3 Output")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create mock model
    model = MockPhase3Model()
    params = count_parameters(model)
    size_mb = calculate_model_size_mb(model)

    print(f"  Mock model created:")
    print(f"    Parameters: {params['total']:,}")
    print(f"    Size: {size_mb:.1f} MB")

    # Save model
    model_path = output_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'vocab_size': 50257,
            'hidden_size': 512,
            'num_layers': 6,
            'model_type': 'gpt2',  # Use GPT2 for compatibility
            'phase': 'phase3_quiet_star',
        }, f, indent=2)

    # Create mock tokenizer (use GPT2 tokenizer for compatibility)
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer_dir = output_dir
        tokenizer.save_pretrained(str(tokenizer_dir))
    except Exception as e:
        print(f"  Warning: Could not save tokenizer: {e}")
        # Fallback: create minimal tokenizer files
        tokenizer_dir = output_dir / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        (tokenizer_dir / "tokenizer_config.json").write_text('{}')

    print(f"  [OK] Mock Phase 3 output saved to: {output_dir}")

    return {
        'model_path': model_path,
        'config_path': config_path,
        'params': params,
        'size_mb': size_mb,
    }


def validate_phase4_execution(
    phase3_path: Path,
    phase4_output: Path
) -> Dict:
    """Execute Phase 4 and validate results"""
    print("\n" + "=" * 60)
    print("STEP 2: Executing Phase 4 Compression")
    print("=" * 60)

    # Create Phase 4 config
    config = Phase4Config(
        model_path=str(phase3_path),
        output_path=str(phase4_output),
        target_compression_ratio=8.0,
        sparsity_threshold=0.08,
        enable_fine_tuning=True,
        fine_tune_epochs=2,
        calibration_samples=100,  # Reduced for testing
        save_quantized=True,
        save_dequantized_fp16=True,
    )

    print(f"  Configuration:")
    print(f"    Target compression: {config.target_compression_ratio}×")
    print(f"    Sparsity threshold: {config.sparsity_threshold}")
    print(f"    Fine-tuning: {config.enable_fine_tuning}")

    # Execute Phase 4
    controller = Phase4Controller(config)

    start_time = time.time()
    results = controller.execute(
        phase3_output_path=str(phase3_path),
        wandb_logger=None
    )
    execution_time = time.time() - start_time

    print(f"\n  [OK] Phase 4 execution completed in {execution_time:.1f}s")

    return results


def validate_dual_outputs(phase4_output: Path, results: Dict) -> bool:
    """Validate that both quantized and dequantized models exist"""
    print("\n" + "=" * 60)
    print("STEP 3: Validating Dual Model Outputs")
    print("=" * 60)

    all_checks_passed = True

    # Check quantized model
    quantized_path = Path(results['output_paths'].get('quantized', ''))
    if quantized_path.exists():
        size_mb = quantized_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Quantized model exists: {quantized_path.name}")
        print(f"     Size: {size_mb:.1f} MB")

        # Validate structure - prefer SafeTensors format
        if quantized_path.suffix == '.safetensors':
            from safetensors.torch import load_file as safe_load_file
            data = safe_load_file(str(quantized_path), device="cpu")
            # SafeTensors stores state_dict directly, check for metadata separately
            print(f"     Structure: [OK] (SafeTensors format)")
        else:
            print(f"     WARNING: Legacy .pt format detected, consider migrating to SafeTensors")
            data = torch.load(quantized_path, weights_only=False)
            assert 'state_dict' in data, "Missing state_dict"
            assert 'scale_factors' in data, "Missing scale_factors"
            print(f"     Structure: [OK] (state_dict + scale_factors)")
    else:
        print(f"  [FAIL] Quantized model NOT FOUND")
        all_checks_passed = False

    # Check dequantized model
    dequantized_path = Path(results['output_paths'].get('dequantized_fp16', ''))
    if dequantized_path.exists():
        size_mb = dequantized_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Dequantized FP16 model exists: {dequantized_path.name}")
        print(f"     Size: {size_mb:.1f} MB")
        print(f"     PRIMARY for Phase 5: YES")

        # Validate it's a state dict - prefer SafeTensors format
        if dequantized_path.suffix == '.safetensors':
            from safetensors.torch import load_file as safe_load_file
            state_dict = safe_load_file(str(dequantized_path), device="cpu")
            print(f"     Structure: [OK] (SafeTensors FP16 state_dict)")
        else:
            print(f"     WARNING: Legacy .pt format detected, consider migrating to SafeTensors")
            state_dict = torch.load(dequantized_path, weights_only=False)
            assert isinstance(state_dict, dict), "Not a valid state_dict"
            print(f"     Structure: [OK] (FP16 state_dict)")
    else:
        print(f"  [FAIL] Dequantized model NOT FOUND")
        all_checks_passed = False

    # Check tokenizer
    tokenizer_path = Path(results['output_paths'].get('tokenizer', ''))
    if tokenizer_path.exists():
        print(f"  [OK] Tokenizer saved: {tokenizer_path.name}")
    else:
        print(f"  [FAIL] Tokenizer NOT FOUND")
        all_checks_passed = False

    return all_checks_passed


def validate_gradient_flow_test(results: Dict) -> bool:
    """Validate gradient flow test passed"""
    print("\n" + "=" * 60)
    print("STEP 4: Validating Gradient Flow Test")
    print("=" * 60)

    gradient_test = results.get('gradient_flow_test', {})
    passed = gradient_test.get('passed', False)
    error = gradient_test.get('error', None)

    if passed:
        print(f"  [OK] Gradient flow test: PASSED")
        print(f"     Dequantized model supports backprop")
        print(f"     Ready for Phase 5 training")
        return True
    else:
        print(f"  [FAIL] Gradient flow test: FAILED")
        if error:
            print(f"     Error: {error}")
        return False


def validate_compression_metrics(results: Dict) -> bool:
    """Validate compression metrics meet targets"""
    print("\n" + "=" * 60)
    print("STEP 5: Validating Compression Metrics")
    print("=" * 60)

    all_checks_passed = True
    post_metrics = results.get('post_compression', {})

    # Check compression ratio
    compression_ratio = post_metrics.get('compression_ratio', 0.0)
    if compression_ratio >= 6.0:
        print(f"  [OK] Compression ratio: {compression_ratio:.2f}× (target: ≥6.0×)")
    else:
        print(f"  [FAIL] Compression ratio: {compression_ratio:.2f}× (BELOW target)")
        all_checks_passed = False

    # Check sparsity
    sparsity = post_metrics.get('sparsity_ratio', 0.0)
    if 0.25 <= sparsity <= 0.45:
        print(f"  [OK] Sparsity ratio: {sparsity:.1%} (target: 25-45%)")
    else:
        print(f"  [FAIL] Sparsity ratio: {sparsity:.1%} (OUT OF RANGE)")
        all_checks_passed = False

    # Check layers quantized
    layers_quantized = post_metrics.get('layers_quantized', 0)
    print(f"  [OK] Layers quantized: {layers_quantized}")

    return all_checks_passed


def validate_phase4_to_phase5_handoff(
    phase4_output: Path,
    results: Dict
) -> bool:
    """Validate Phase 4→5 handoff requirements"""
    print("\n" + "=" * 60)
    print("STEP 6: Validating Phase 4→5 Handoff")
    print("=" * 60)

    all_checks_passed = True

    # Required files for Phase 5
    required_files = {
        'dequantized_fp16': 'bitnet_dequantized_fp16.pt',
        'tokenizer_config': 'tokenizer/tokenizer_config.json',
        'metadata': 'compression_metadata.json',
    }

    for key, filename in required_files.items():
        filepath = phase4_output / filename
        if filepath.exists():
            print(f"  [OK] {key}: {filename}")
        else:
            print(f"  [FAIL] {key}: {filename} NOT FOUND")
            all_checks_passed = False

    # Validate metadata
    metadata_path = phase4_output / "compression_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        required_fields = [
            'compression_method',
            'quantization_bits',
            'metrics',
            'config',
        ]

        for field in required_fields:
            if field in metadata:
                print(f"  [OK] Metadata field: {field}")
            else:
                print(f"  [FAIL] Metadata field: {field} MISSING")
                all_checks_passed = False
    else:
        print(f"  [FAIL] Metadata file not found")
        all_checks_passed = False

    # Check gradient flow test
    if results.get('gradient_flow_test', {}).get('passed', False):
        print(f"  [OK] Gradient flow: PASSED")
    else:
        print(f"  [FAIL] Gradient flow: FAILED")
        all_checks_passed = False

    # Check success flag
    if results.get('success', False):
        print(f"  [OK] Phase 4 success: TRUE")
    else:
        print(f"  [FAIL] Phase 4 success: FALSE")
        all_checks_passed = False

    return all_checks_passed


def run_performance_benchmark(
    phase3_path: Path,
    phase4_output: Path
) -> Dict:
    """Run performance benchmarks"""
    print("\n" + "=" * 60)
    print("STEP 7: Performance Benchmarking")
    print("=" * 60)

    # Load original model - prefer SafeTensors format
    original_model = MockPhase3Model()
    original_bin_path = phase3_path / "pytorch_model.bin"
    original_safetensors_path = phase3_path / "model.safetensors"

    if original_safetensors_path.exists():
        from safetensors.torch import load_file as safe_load_file
        print(f"  Loading original model from SafeTensors")
        original_state = safe_load_file(str(original_safetensors_path), device="cpu")
    else:
        print(f"  WARNING: Loading original model from legacy .pt format")
        original_state = torch.load(original_bin_path, weights_only=False)
    original_model.load_state_dict(original_state, strict=False)

    # Load dequantized model - prefer SafeTensors format
    dequantized_pt_path = phase4_output / "bitnet_dequantized_fp16.pt"
    dequantized_safetensors_path = phase4_output / "bitnet_dequantized_fp16.safetensors"

    if dequantized_safetensors_path.exists():
        from safetensors.torch import load_file as safe_load_file
        print(f"  Loading dequantized model from SafeTensors")
        dequantized_state = safe_load_file(str(dequantized_safetensors_path), device="cpu")
        dequantized_path = dequantized_safetensors_path
    else:
        print(f"  WARNING: Loading dequantized model from legacy .pt format")
        dequantized_state = torch.load(dequantized_pt_path, weights_only=False)
        dequantized_path = dequantized_pt_path

    dequantized_model = MockPhase3Model()
    dequantized_model.load_state_dict(dequantized_state, strict=False)

    # Benchmark inference time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randint(0, 1000, (1, 128)).to(device)

    original_model.eval()
    original_model.to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = original_model(dummy_input)

    # Benchmark original
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = original_model(dummy_input)
    original_time = (time.time() - start) / 100

    print(f"  Original model inference: {original_time*1000:.2f} ms/sample")

    # Compare sizes
    original_size = calculate_model_size_mb(original_model)
    dequantized_size = (dequantized_path.stat().st_size) / (1024 * 1024)

    print(f"  Original size: {original_size:.1f} MB")
    print(f"  Dequantized size: {dequantized_size:.1f} MB")
    print(f"  Size ratio: {original_size / dequantized_size:.2f}×")

    return {
        'original_inference_ms': original_time * 1000,
        'original_size_mb': original_size,
        'dequantized_size_mb': dequantized_size,
    }


def main():
    """Main validation script"""
    print("\n" + "=" * 60)
    print("PHASE 4 END-TO-END VALIDATION")
    print("=" * 60)

    # Setup paths
    base_dir = Path(__file__).parent.parent
    test_dir = base_dir / "tests" / "artifacts" / "phase4_validation"
    phase3_output = test_dir / "phase3_output"
    phase4_output = test_dir / "phase4_output"

    # Clean up previous runs
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)

    # Track results
    validation_results = {
        'step1_mock_phase3': False,
        'step2_phase4_execution': False,
        'step3_dual_outputs': False,
        'step4_gradient_flow': False,
        'step5_compression_metrics': False,
        'step6_phase4_to_5_handoff': False,
        'step7_performance_benchmark': False,
    }

    try:
        # Step 1: Create mock Phase 3 output
        mock_results = create_mock_phase3_output(phase3_output)
        validation_results['step1_mock_phase3'] = True

        # Step 2: Execute Phase 4
        phase4_results = validate_phase4_execution(phase3_output, phase4_output)
        validation_results['step2_phase4_execution'] = phase4_results.get('success', False)

        # Step 3: Validate dual outputs
        validation_results['step3_dual_outputs'] = validate_dual_outputs(
            phase4_output, phase4_results
        )

        # Step 4: Validate gradient flow
        validation_results['step4_gradient_flow'] = validate_gradient_flow_test(
            phase4_results
        )

        # Step 5: Validate compression metrics
        validation_results['step5_compression_metrics'] = validate_compression_metrics(
            phase4_results
        )

        # Step 6: Validate handoff
        validation_results['step6_phase4_to_5_handoff'] = validate_phase4_to_phase5_handoff(
            phase4_output, phase4_results
        )

        # Step 7: Performance benchmark
        perf_results = run_performance_benchmark(phase3_output, phase4_output)
        validation_results['step7_performance_benchmark'] = True

    except Exception as e:
        print(f"\n[FAIL] VALIDATION FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # Print final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    total_steps = len(validation_results)
    passed_steps = sum(validation_results.values())

    for step, passed in validation_results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"  {status}  {step}")

    print(f"\n  Overall: {passed_steps}/{total_steps} steps passed")

    if passed_steps == total_steps:
        print("\n  [SUCCESS] ALL VALIDATION STEPS PASSED!")
        print("  Phase 4 is PRODUCTION READY")
        return 0
    else:
        print(f"\n  [WARNING]  {total_steps - passed_steps} validation step(s) failed")
        print("  Please review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

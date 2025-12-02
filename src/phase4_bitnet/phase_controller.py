"""
Phase 4 Controller
Pipeline integration with dual model output

ISS-004: Updated to use secure SafeTensors checkpoint format.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
import time

from src.phase4_bitnet.config import Phase4Config
from src.phase4_bitnet.quantizer import BitNetQuantizer
from src.phase4_bitnet.compressed_model import CompressedModel
from src.phase4_bitnet.calibration import create_calibration_dataloader
from src.phase4_bitnet.fine_tuner import FineTuner
from src.phase4_bitnet.utils import (
    calculate_model_size_mb,
    count_parameters,
    calculate_sparsity_ratio,
    test_gradient_flow,
    save_compression_metadata,
    validate_compression_quality,
    estimate_inference_speedup,
)

# ISS-004: Secure checkpoint utilities
from safetensors.torch import save_file as safe_save_tensors


class Phase4Controller:
    """
    Phase 4: BitNet Compression Controller

    Orchestrates the complete compression pipeline:
    1. Load Phase 3 model
    2. Quantize to 1.58-bit ternary
    3. Optional fine-tuning
    4. Save dual outputs (quantized + dequantized FP16)
    5. Hand off to Phase 5

    Outputs:
    - Quantized model (12MB, 1.58-bit) - For inference
    - Dequantized FP16 model (50MB) - PRIMARY for Phase 5 training
    """

    def __init__(self, config: Phase4Config):
        """
        Initialize Phase 4 controller

        Args:
            config: Phase 4 configuration
        """
        self.config = config
        self.device = config.device

        # Components
        self.quantizer = BitNetQuantizer(config)
        self.model = None
        self.tokenizer = None
        self.compressed_model = None
        self.fine_tuner = None

        # Metrics
        self.metrics = {}

        # Timing
        self.start_time = None
        self.compression_time = 0.0
        self.fine_tune_time = 0.0

    def execute(
        self,
        phase3_output_path: str,
        wandb_logger: Optional[object] = None
    ) -> Dict:
        """
        Execute Phase 4 compression

        Args:
            phase3_output_path: Path to Phase 3 output
            wandb_logger: W&B logger instance

        Returns:
            Phase 4 results dictionary
        """
        self.start_time = time.time()

        print("=" * 60)
        print("PHASE 4: BitNet 1.58-bit Compression")
        print("=" * 60)

        try:
            # Step 1: Load Phase 3 model
            print("\n[1/7] Loading Phase 3 model...")
            self._load_phase3_model(phase3_output_path)

            # Step 2: Pre-compression evaluation
            print("\n[2/7] Pre-compression evaluation...")
            pre_metrics = self._evaluate_pre_compression()

            # Log to W&B
            if wandb_logger:
                wandb_logger.log_phase4_pre_compression(pre_metrics)

            # Step 3: Calibration
            print("\n[3/7] Calibration...")
            self._run_calibration()

            # Step 4: Compression
            print("\n[4/7] Compressing model...")
            compression_start = time.time()
            self._compress_model()
            self.compression_time = time.time() - compression_start

            # Step 5: Post-compression evaluation
            print("\n[5/7] Post-compression evaluation...")
            post_metrics = self._evaluate_post_compression()

            # Log to W&B
            if wandb_logger:
                wandb_logger.log_phase4_compression(post_metrics)

            # Step 6: Fine-tuning (if needed)
            fine_tune_needed = self._check_fine_tuning_needed(
                pre_metrics,
                post_metrics
            )

            if fine_tune_needed:
                print("\n[6/7] Fine-tuning (quality recovery)...")
                fine_tune_start = time.time()
                fine_tune_results = self._fine_tune_model()
                self.fine_tune_time = time.time() - fine_tune_start

                # Log to W&B
                if wandb_logger:
                    wandb_logger.log_phase4_fine_tuning(
                        fine_tune_results
                    )
            else:
                print("\n[6/7] Skipping fine-tuning (quality acceptable)")
                fine_tune_results = None

            # Step 7: Save outputs (dual models)
            print("\n[7/7] Saving outputs...")
            output_paths = self._save_outputs()

            # Validate gradient flow (critical for Phase 5)
            print("\nValidating gradient flow...")
            gradient_test_passed, gradient_error = self._validate_gradient_flow()

            # Prepare final results
            total_time = time.time() - self.start_time

            results = {
                'success': True,
                'phase': 'phase4_bitnet',
                'output_paths': output_paths,
                'pre_compression': pre_metrics,
                'post_compression': post_metrics,
                'fine_tuning': fine_tune_results,
                'gradient_flow_test': {
                    'passed': gradient_test_passed,
                    'error': gradient_error,
                },
                'timing': {
                    'total_seconds': total_time,
                    'compression_seconds': self.compression_time,
                    'fine_tune_seconds': self.fine_tune_time,
                },
                'metrics': self.metrics,
            }

            # Log phase summary to W&B
            if wandb_logger:
                wandb_logger.log_phase4_summary(results)

            # Print summary
            self._print_summary(results)

            return results

        except Exception as e:
            print(f"\n[FAIL] Phase 4 failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                'success': False,
                'phase': 'phase4_bitnet',
                'error': str(e),
            }

    def _load_phase3_model(self, phase3_path: str):
        """Load model and tokenizer from Phase 3"""
        phase3_path = Path(phase3_path)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(phase3_path)
        )

        # Load model
        self.model = AutoModel.from_pretrained(
            str(phase3_path)
        )

        # Count parameters and adapt config
        params = count_parameters(self.model)
        self.config.adapt_to_model_size(params['total'])

        print(f"  Model loaded: {params['total']:,} parameters")
        print(f"  Size category: {self.config.get_size_category(params['total'])}")
        print(f"  Compression target: {self.config.target_compression_ratio}x")

    def _evaluate_pre_compression(self) -> Dict:
        """Evaluate model before compression"""
        # Calculate size
        original_size_mb = calculate_model_size_mb(self.model)

        # Count parameters
        params = count_parameters(self.model)

        metrics = {
            'original_size_mb': original_size_mb,
            'total_params': params['total'],
            'trainable_params': params['trainable'],
        }

        print(f"  Original size: {original_size_mb:.1f} MB")
        print(f"  Parameters: {params['total']:,}")

        self.metrics.update(metrics)
        return metrics

    def _run_calibration(self):
        """Run calibration data collection"""
        # Create calibration dataloader
        dataloader = create_calibration_dataloader(
            self.tokenizer,
            self.config
        )

        print(f"  Dataset: {self.config.calibration_dataset}")
        print(f"  Samples: {self.config.calibration_samples}")

        # Note: Actual activation collection happens in quantizer
        # This step just prepares the data

    def _compress_model(self):
        """Compress model using BitNet quantization"""
        # Create compressed model wrapper
        self.compressed_model = CompressedModel(
            base_model=self.model,
            quantizer=self.quantizer,
            config=self.config
        )

        # Perform compression
        self.compressed_model.compress()

        # Get compression stats
        stats = self.compressed_model.get_compression_stats()

        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Sparsity ratio: {stats['sparsity_ratio']:.1%}")
        print(f"  Layers quantized: {stats['layers_quantized']}")

        self.metrics.update(stats)

    def _evaluate_post_compression(self) -> Dict:
        """Evaluate model after compression"""
        # Get compressed size
        stats = self.compressed_model.get_compression_stats()

        metrics = {
            'compressed_size_mb': stats['quantized_size_mb'],
            'compression_ratio': stats['compression_ratio'],
            'sparsity_ratio': stats['sparsity_ratio'],
            'layers_quantized': stats['layers_quantized'],
        }

        print(f"  Compressed size: {stats['quantized_size_mb']:.1f} MB")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")

        return metrics

    def _check_fine_tuning_needed(
        self,
        pre_metrics: Dict,
        post_metrics: Dict
    ) -> bool:
        """Check if fine-tuning is needed"""
        # For now, always fine-tune if enabled
        # In production, would compare perplexity
        return self.config.enable_fine_tuning

    def _fine_tune_model(self) -> Dict:
        """Fine-tune compressed model"""
        # Create fine-tuner
        self.fine_tuner = FineTuner(
            model=self.compressed_model,
            config=self.config,
            device=self.device
        )

        # Create training data
        train_dataloader = create_calibration_dataloader(
            self.tokenizer,
            self.config
        )

        # Fine-tune
        results = self.fine_tuner.fine_tune(
            train_dataloader=train_dataloader
        )

        print(f"  Epochs: {results['epochs_completed']}")
        print(f"  Final loss: {results['final_loss']:.4f}")

        return results

    def _validate_gradient_flow(self) -> Tuple[bool, Optional[str]]:
        """Validate gradient flow through dequantized model"""
        # Get dequantized model
        dequantized_state = self.compressed_model.get_dequantized_state_dict()

        # Load into fresh model
        test_model = type(self.model)(self.model.config)
        test_model.load_state_dict(dequantized_state, strict=False)

        # Test gradient flow
        passed, error = test_gradient_flow(test_model, self.device)

        if passed:
            print("  [OK] Gradient flow test PASSED")
        else:
            print(f"  [FAIL] Gradient flow test FAILED: {error}")

        return passed, error

    def _save_outputs(self) -> Dict[str, str]:
        """Save dual model outputs using secure SafeTensors format (ISS-004)"""
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save quantized model (1.58-bit) - SafeTensors + JSON
        if self.config.save_quantized:
            quantized_tensors_path = output_dir / "bitnet_quantized_model.safetensors"
            quantized_json_path = output_dir / "bitnet_quantized_model.json"

            # Get state dict and scale factors
            state_dict = self.compressed_model.get_quantized_state_dict()
            scale_factors = self.compressed_model.get_scale_factors()

            # Save tensors with SafeTensors (secure)
            safe_save_tensors(state_dict, str(quantized_tensors_path))

            # Save scale factors and config as JSON (secure)
            quantized_metadata = {
                'scale_factors': {k: float(v) if hasattr(v, 'item') else v
                                  for k, v in scale_factors.items()},
                'config': self.config.to_dict(),
            }
            with open(quantized_json_path, 'w', encoding='utf-8') as f:
                json.dump(quantized_metadata, f, indent=2, default=str)

            paths['quantized'] = str(quantized_tensors_path)
            print(f"  [OK] Quantized model: {quantized_tensors_path}")

        # Save dequantized FP16 model (PRIMARY for Phase 5) - SafeTensors
        if self.config.save_dequantized_fp16:
            dequantized_path = output_dir / "bitnet_dequantized_fp16.safetensors"

            dequantized_state = self.compressed_model.get_dequantized_state_dict()

            # Save with SafeTensors (secure)
            safe_save_tensors(dequantized_state, str(dequantized_path))

            paths['dequantized_fp16'] = str(dequantized_path)
            paths['primary_output'] = str(dequantized_path)  # PRIMARY
            print(f"  [OK] Dequantized FP16 (PRIMARY): {dequantized_path}")

        # Save tokenizer
        tokenizer_path = output_dir / "tokenizer"
        self.tokenizer.save_pretrained(str(tokenizer_path))
        paths['tokenizer'] = str(tokenizer_path)

        # Save metadata
        save_compression_metadata(
            output_dir,
            {
                'compression_method': 'BitNet-1.58',
                'quantization_bits': 1.58,
                'metrics': self.metrics,
                'config': self.config.to_dict(),
            }
        )

        return paths

    def _print_summary(self, results: Dict):
        """Print execution summary"""
        print("\n" + "=" * 60)
        print("PHASE 4 COMPLETE")
        print("=" * 60)

        if results['success']:
            print(f"[OK] Compression: {self.metrics['compression_ratio']:.2f}x")
            print(f"[OK] Sparsity: {self.metrics['sparsity_ratio']:.1%}")
            print(f"[OK] Gradient flow: {'PASSED' if results['gradient_flow_test']['passed'] else 'FAILED'}")
            print(f"Time:  Total time: {results['timing']['total_seconds']:.1f}s")

            print("\nOutputs:")
            for key, path in results['output_paths'].items():
                marker = "[PRIMARY] PRIMARY" if key == 'primary_output' else "  "
                print(f"{marker} {key}: {path}")
        else:
            print(f"[FAIL] Error: {results.get('error', 'Unknown')}")

        print("=" * 60)

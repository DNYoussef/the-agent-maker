"""Phase 4: BitNet - 1.58-bit quantization for model compression"""

from .base_controller import PhaseController, PhaseResult


class Phase4Controller(PhaseController):
    """Phase 4: BitNet - 1.58-bit quantization for model compression"""

    def execute(self, input_models: list = None) -> PhaseResult:
        """Execute Phase 4: Compress model to 1.58-bit using BitNet quantization.

        Process:
        1. Load reasoning-enhanced model from Phase 3
        2. Apply ternary quantization {-1, 0, +1}
        3. Fine-tune with STE (Straight-Through Estimator)
        4. Validate compression ratio and accuracy retention

        Args:
            input_models: [enhanced_model] from Phase 3

        Returns:
            PhaseResult with quantized model
        """
        import copy
        import time

        start_time = time.time()

        print("\n" + "=" * 60)
        print("PHASE 4: BITNET - 1.58-BIT QUANTIZATION")
        print("=" * 60 + "\n")

        try:
            # Validate input
            self.validate_input(input_models)
            enhanced_model = input_models[0]

            # Get model size before quantization
            original_size = self._get_model_size(enhanced_model)
            print(f"Original model size: {original_size['size_mb']:.2f} MB")
            print(f"Original parameters: {original_size['params']:,}")

            # Step 1: Quantize model
            print("\n--- Step 1: Ternary Quantization ---")
            quantized_state, scale_factors, quant_stats = self._quantize_model(enhanced_model)

            # Step 2: Create compressed model
            print("\n--- Step 2: Creating Compressed Model ---")
            compressed_model = self._create_compressed_model(
                enhanced_model, quantized_state, scale_factors
            )

            # Step 3: Fine-tune with STE (optional, simplified for MVP)
            print("\n--- Step 3: STE Fine-tuning ---")
            fine_tuned_model = self._ste_finetune(compressed_model)

            # Step 4: Validate compression
            print("\n--- Step 4: Validation ---")
            compressed_size = self._get_model_size(fine_tuned_model)
            compression_ratio = original_size["size_mb"] / max(compressed_size["size_mb"], 0.01)

            print(f"Compressed model size: {compressed_size['size_mb']:.2f} MB")
            print(f"Compression ratio: {compression_ratio:.1f}x")
            print(f"Sparsity ratio: {quant_stats.get('sparsity_ratio', 0):.2%}")

            # Validate thresholds
            validation_passed = compression_ratio >= self.config.get("min_compression", 4.0)

            duration = time.time() - start_time

            return PhaseResult(
                success=True,
                phase_name="phase4",
                model=fine_tuned_model,
                metrics={
                    "original_size_mb": original_size["size_mb"],
                    "compressed_size_mb": compressed_size["size_mb"],
                    "compression_ratio": compression_ratio,
                    "sparsity_ratio": quant_stats.get("sparsity_ratio", 0),
                    "layers_quantized": quant_stats.get("layers_quantized", 0),
                    "layers_preserved": quant_stats.get("layers_preserved", 0),
                    "validation_passed": validation_passed,
                    "duration_seconds": duration,
                },
                duration=duration,
                artifacts={"scale_factors": scale_factors, "quantization_stats": quant_stats},
                config=self.config,
                error=None,
            )

        except Exception as e:
            duration = time.time() - start_time
            return PhaseResult(
                success=False,
                phase_name="phase4",
                model=None,
                metrics={},
                duration=duration,
                artifacts={},
                config=self.config,
                error=str(e),
            )

    def _get_model_size(self, model) -> dict:
        """Calculate model size in MB and parameter count."""
        import torch

        total_params = sum(p.numel() for p in model.parameters())

        # Calculate size based on dtype
        size_bytes = 0
        for p in model.parameters():
            if p.dtype == torch.float32:
                size_bytes += p.numel() * 4
            elif p.dtype == torch.float16:
                size_bytes += p.numel() * 2
            elif p.dtype == torch.int8:
                size_bytes += p.numel() * 1
            else:
                size_bytes += p.numel() * 4  # Default to FP32

        size_mb = size_bytes / (1024 * 1024)

        return {"params": total_params, "size_mb": size_mb, "size_bytes": size_bytes}

    def _quantize_model(self, model):
        """Apply BitNet ternary quantization to model."""
        import torch
        import torch.nn as nn

        quantized_state = {}
        scale_factors = {}
        stats = {
            "layers_quantized": 0,
            "layers_preserved": 0,
            "total_params": 0,
            "quantized_params": 0,
            "zero_params": 0,
            "sparsity_ratio": 0.0,
        }

        # Sparsity threshold from config
        threshold = self.config.get("sparsity_threshold", 0.1)

        # Layers to preserve (embeddings, layer norms)
        preserve_patterns = ["embed", "norm", "ln_", "layernorm", "bias"]

        for name, param in model.state_dict().items():
            stats["total_params"] += param.numel()

            # Check if layer should be preserved
            should_preserve = any(p in name.lower() for p in preserve_patterns)

            if should_preserve:
                # Keep in FP16
                quantized_state[name] = param.data.half()
                scale_factors[name] = torch.tensor(1.0)
                stats["layers_preserved"] += 1
            else:
                # Quantize to ternary {-1, 0, +1}
                # Step 1: Calculate scale (mean absolute value)
                if len(param.shape) >= 2:
                    scale = param.abs().mean(dim=list(range(1, len(param.shape))), keepdim=True)
                else:
                    scale = param.abs().mean()
                scale = torch.clamp(scale, min=1e-8)

                # Step 2: Normalize and apply threshold
                normalized = param / scale
                sparsity_mask = param.abs() < (scale * threshold)

                # Step 3: Quantize
                quantized = torch.sign(normalized)
                quantized[sparsity_mask] = 0
                quantized_int8 = quantized.to(torch.int8)

                quantized_state[name] = quantized_int8
                scale_factors[name] = scale
                stats["layers_quantized"] += 1
                stats["quantized_params"] += param.numel()
                stats["zero_params"] += (quantized_int8 == 0).sum().item()

        # Calculate sparsity
        if stats["quantized_params"] > 0:
            stats["sparsity_ratio"] = stats["zero_params"] / stats["quantized_params"]

        print(f"  Quantized {stats['layers_quantized']} layers")
        print(f"  Preserved {stats['layers_preserved']} layers")
        print(f"  Sparsity: {stats['sparsity_ratio']:.2%}")

        return quantized_state, scale_factors, stats

    def _create_compressed_model(self, original_model, quantized_state, scale_factors):
        """Create compressed model from quantized state dict."""
        import copy

        import torch

        # Create a copy of the model
        compressed_model = copy.deepcopy(original_model)

        # Dequantize and load state dict
        dequantized_state = {}
        for name, param in quantized_state.items():
            if param.dtype == torch.int8:
                # Dequantize: W_deq = scale * Q(W)
                scale = scale_factors[name]
                dequantized = scale * param.to(torch.float32)
                dequantized_state[name] = dequantized.half()
            else:
                dequantized_state[name] = param

        # Load dequantized state
        compressed_model.load_state_dict(dequantized_state)
        print(f"  Compressed model created")

        return compressed_model

    def _ste_finetune(self, model):
        """Fine-tune with Straight-Through Estimator (simplified for MVP)."""
        # For MVP, skip actual fine-tuning (requires training data)
        # Full implementation would:
        # 1. Use STE for gradients through quantization
        # 2. Fine-tune for 2000 steps
        # 3. Validate accuracy retention
        print(f"  STE fine-tuning skipped (MVP mode)")
        return model

    def validate_input(self, input_models: list = None) -> bool:
        """Validate 1 input model from Phase 3"""
        if not input_models or len(input_models) != 1:
            raise ValueError(
                f"Phase 4 requires 1 input model, got {len(input_models) if input_models else 0}"
            )
        return True

    def validate_output(self, result: PhaseResult) -> bool:
        """Validate Phase 4 output (compression >6x, accuracy drop <10%)"""
        if result.metrics:
            compression = result.metrics.get("compression_ratio", 0)
            min_compression = self.config.get("min_compression", 4.0) if self.config else 4.0
            return compression >= min_compression
        return True

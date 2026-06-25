"""
Compressed Model Wrapper
STE-enabled wrapper for quantized models with BitLinear support
"""

from typing import Dict

import torch
import torch.nn as nn

from src.phase4_bitnet.bitlinear import BitLinear, replace_linear_with_bitlinear
from src.phase4_bitnet.config import Phase4Config
from src.phase4_bitnet.quantizer import BitNetQuantizer
from src.phase4_bitnet.utils import calculate_sparsity_ratio


class CompressedModel(nn.Module):
    """
    Wrapper for BitNet compressed models with two modes:

    Mode 1 (BitLinear Replacement - Recommended):
    - Replaces nn.Linear layers with BitLinear in-place
    - Automatic quantization during forward pass
    - Ternary weights (1.58-bit information content) stored as int8 -
      no bit-packing yet, so the real on-disk footprint is ~4x smaller
      (fp32 -> int8 + fp16 scale), NOT the ~20x of packed 1.58-bit
    - Preserves model architecture and gradient flow

    Mode 2 (Legacy Quantization):
    - Quantizes entire model state dict
    - Manual quantization/dequantization
    - Compatible with existing Phase 4 pipeline

    Features:
    - Transparent quantization/dequantization
    - Straight-Through Estimator (STE) for gradients
    - Compatible with standard PyTorch training
    - Maintains full-precision shadow weights (Mode 2 only)

    Forward Pass: Uses quantized weights
    Backward Pass: Gradients flow to full-precision weights (STE)
    """

    def __init__(
        self,
        base_model: nn.Module,
        quantizer: BitNetQuantizer,
        config: Phase4Config,
        use_bitlinear: bool = False,
    ):
        """
        Initialize compressed model

        Args:
            base_model: Original model to compress
            quantizer: BitNet quantizer instance
            config: Phase 4 configuration
            use_bitlinear: If True, use BitLinear replacement (recommended)
                          If False, use legacy quantization
        """
        super().__init__()

        self.config = config
        self.quantizer = quantizer
        self.use_bitlinear = use_bitlinear

        if use_bitlinear:
            # Mode 1: Replace nn.Linear with BitLinear
            self.base_model = replace_linear_with_bitlinear(
                base_model,
                weight_sparsity_threshold=config.sparsity_threshold,
                exclude_patterns=config.preserve_layers,
            )
            self.is_compressed = True
        else:
            # Mode 2: Legacy quantization
            self.base_model = base_model

            # Store quantized and scale factors
            self.quantized_state = {}
            self.scale_factors = {}

            # Shadow weights (full precision for gradients)
            self.shadow_weights = {}

            # Compression performed flag
            self.is_compressed = False

    def compress(self):
        """
        Compress the model using BitNet quantization (Mode 2 only)

        For Mode 1 (BitLinear), compression happens automatically in forward pass.
        """
        if self.use_bitlinear:
            # Already compressed via BitLinear replacement
            return

        # Mode 2: Legacy quantization
        # Quantize model
        quantized_dict, scales = self.quantizer.quantize_model(self.base_model)

        self.quantized_state = quantized_dict
        self.scale_factors = scales

        # Initialize shadow weights (full precision)
        for name, param in self.base_model.named_parameters():
            self.shadow_weights[name] = param.data.clone().detach()
            self.shadow_weights[name].requires_grad = True

        self.is_compressed = True

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Usability: delegate generation to the wrapped base model (it runs with the
        quantized weights). Without this the compressed model can't be used for inference."""
        return self.base_model.generate(*args, **kwargs)

    def get_input_embeddings(self):
        """Delegate the HF-style embedding accessor (lets a CompressedModel flow into
        downstream phases that read it, e.g. QuietSTaR)."""
        return self.base_model.get_input_embeddings()

    def forward(self, *args, **kwargs):
        """
        Forward pass with quantized weights

        Mode 1 (BitLinear): Automatic quantization in BitLinear layers
        Mode 2 (Legacy): Manual quantization with STE

        Uses Straight-Through Estimator:
        - Forward: Quantized weights
        - Backward: Gradients to full-precision weights
        """
        if self.use_bitlinear:
            # Mode 1: BitLinear handles quantization automatically
            return self.base_model(*args, **kwargs)

        # Mode 2: Legacy quantization
        if not self.is_compressed:
            # Not compressed, use base model directly
            return self.base_model(*args, **kwargs)

        # Apply quantized weights (with STE)
        with torch.no_grad():
            # Temporarily swap in quantized weights
            original_params = {}

            for name, param in self.base_model.named_parameters():
                # Store original
                original_params[name] = param.data.clone()

                # Check if layer was quantized
                if name in self.quantized_state:
                    quantized_tensor = self.quantized_state[name]
                    target_dtype = param.dtype

                    # Dequantize if int8
                    if quantized_tensor.dtype == torch.int8:
                        dequantized = self.quantizer.dequantize_tensor(
                            quantized_tensor, self.scale_factors[name]
                        )
                        param.data = dequantized.to(device=param.device, dtype=target_dtype)
                    else:
                        # Preserved layer
                        param.data = quantized_tensor.to(device=param.device, dtype=target_dtype)

        # Forward pass with quantized weights
        output = self.base_model(*args, **kwargs)

        # Restore original parameters for gradient computation (STE)
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in original_params:
                    param.data = original_params[name]

        return output

    def get_dequantized_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get dequantized FP16 state dict for saving

        Returns:
            Dequantized state dictionary
        """
        if self.use_bitlinear:
            # Mode 1: Save full model in FP16 (BitLinear stores FP32 weights)
            state_dict = self.base_model.state_dict()
            return {k: v.half() for k, v in state_dict.items()}

        # Mode 2: Legacy dequantization
        if not self.is_compressed:
            # Return original state dict in FP16
            state_dict = self.base_model.state_dict()
            return {k: v.half() for k, v in state_dict.items()}

        # Dequantize
        return self.quantizer.dequantize_model(self.quantized_state, self.scale_factors)

    def get_quantized_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get quantized (int8) state dict for saving

        Returns:
            Quantized state dictionary
        """
        if self.use_bitlinear:
            # Mode 1: Extract quantized state from BitLinear
            quantized_dict = {}
            for name, module in self.base_model.named_modules():
                if isinstance(module, BitLinear):
                    quant_state = module.get_quantized_state()
                    quantized_dict[f"{name}.quantized_weight"] = quant_state["quantized_weight"]
                    quantized_dict[f"{name}.scale_factor"] = quant_state["scale_factor"]
                    if quant_state["bias"] is not None:
                        quantized_dict[f"{name}.bias"] = quant_state["bias"]
            return quantized_dict

        # Mode 2: Legacy quantization
        if not self.is_compressed:
            raise RuntimeError("Model not compressed yet")

        return self.quantized_state.copy()

    def get_scale_factors(self) -> Dict[str, torch.Tensor]:
        """
        Get scale factors

        Returns:
            Scale factor dictionary
        """
        if self.use_bitlinear:
            # Mode 1: Extract from BitLinear
            scales = {}
            for name, module in self.base_model.named_modules():
                if isinstance(module, BitLinear):
                    quant_state = module.get_quantized_state()
                    scales[f"{name}.weight"] = quant_state["scale_factor"]
            return scales

        # Mode 2: Legacy
        if not self.is_compressed:
            raise RuntimeError("Model not compressed yet")

        return self.scale_factors.copy()

    def get_compression_stats(self) -> Dict:
        """
        Get compression statistics

        Returns:
            Statistics dictionary
        """
        if not self.is_compressed:
            return {
                "is_compressed": False,
                "compression_ratio": 1.0,
                "mode": "uncompressed",
            }

        if self.use_bitlinear:
            # Mode 1: Calculate from BitLinear layers
            total_original = 0
            total_quantized = 0
            num_bitlinear = 0

            for name, module in self.base_model.named_modules():
                if isinstance(module, BitLinear):
                    footprint = module.get_memory_footprint()
                    total_original += footprint["original_fp32"]
                    # int8 storage (1 byte/weight + fp16 scale) - ternary 1.58-bit
                    # values are NOT bit-packed, so this is the honest int8 on-disk
                    # size and the resulting ratio is ~4x, not the ~20x of packed
                    # 1.58-bit.
                    total_quantized += footprint["quantized_int8_bytes"]
                    num_bitlinear += 1
                elif isinstance(module, nn.Module) and list(module.parameters(recurse=False)):
                    # Non-BitLinear modules stored in FP16 for dequantized output
                    for param in module.parameters(recurse=False):
                        total_original += param.nelement() * 4
                        total_quantized += param.nelement() * 2

            sparsity_ratio = calculate_sparsity_ratio(self.base_model)

            return {
                "is_compressed": True,
                "mode": "bitlinear",
                # Honest: ternary weights carry 1.58-bit of information but are
                # stored as int8 (no bit-packing), so sizes/ratio reflect int8.
                "storage_dtype": "int8",
                "bit_packed": False,
                "num_bitlinear_layers": num_bitlinear,
                "layers_quantized": num_bitlinear,
                "layers_preserved": 0,
                "original_size_mb": total_original / (1024**2),
                "quantized_size_mb": total_quantized / (1024**2),
                "compression_ratio": total_original / total_quantized
                if total_quantized > 0
                else 1.0,
                "sparsity_ratio": sparsity_ratio,
            }

        # Mode 2: Legacy stats
        # Calculate sizes
        original_size_mb = self._calculate_state_dict_size(self.base_model.state_dict())

        quantized_size_mb = self._calculate_state_dict_size(self.quantized_state)

        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0

        stats = self.quantizer.get_stats()

        return {
            "is_compressed": True,
            "mode": "legacy",
            "original_size_mb": original_size_mb,
            "quantized_size_mb": quantized_size_mb,
            "compression_ratio": compression_ratio,
            **stats,
        }

    def _calculate_state_dict_size(self, state_dict: Dict[str, torch.Tensor]) -> float:
        """
        Calculate state dict size in MB

        Args:
            state_dict: State dictionary

        Returns:
            Size in MB
        """
        total_bytes = 0

        for tensor in state_dict.values():
            total_bytes += tensor.nelement() * tensor.element_size()

        return total_bytes / (1024**2)

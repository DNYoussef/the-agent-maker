"""
BitNet Quantizer
Core ternary quantization engine for 1.58-bit compression
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from src.phase4_bitnet.config import Phase4Config


class BitNetQuantizer:
    """
    BitNet 1.58-bit ternary quantizer

    Quantizes weights to {-1, 0, +1} with dynamic scaling
    and sparsity injection.

    Features:
    - Ternary quantization: sign(w) if |w| > threshold else 0
    - Per-channel dynamic scaling
    - Configurable sparsity threshold
    - Layer-wise precision preservation
    """

    def __init__(self, config: Phase4Config):
        """
        Initialize quantizer

        Args:
            config: Phase 4 configuration
        """
        self.config = config
        self.stats = {
            'layers_quantized': 0,
            'layers_preserved': 0,
            'total_params': 0,
            'quantized_params': 0,
            'sparsity_ratio': 0.0,
        }

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to ternary {-1, 0, +1}

        Args:
            tensor: Weight tensor to quantize
            threshold: Sparsity threshold (uses config if None)

        Returns:
            Tuple of (quantized_tensor, scale_factor)
        """
        if threshold is None:
            threshold = self.config.sparsity_threshold

        # Step 1: Calculate per-channel scale factor
        # Formula: α = mean(|W|)
        if len(tensor.shape) >= 2:
            # Per-output-channel scaling
            scale = tensor.abs().mean(
                dim=list(range(1, len(tensor.shape))),
                keepdim=True
            )
        else:
            # Scalar scaling for 1D tensors
            scale = tensor.abs().mean()

        # Prevent division by zero
        scale = torch.clamp(scale, min=1e-8)

        # Step 2: Normalize by scale
        normalized = tensor / scale

        # Step 3: Apply sparsity threshold
        # |w| < threshold * scale → 0
        sparsity_mask = tensor.abs() < (scale * threshold)

        # Step 4: Quantize to {-1, 0, +1}
        # Q(w) = sign(w) if |w| > τ else 0
        quantized = torch.sign(normalized)
        quantized[sparsity_mask] = 0

        # Convert to int8 for storage (memory efficiency)
        quantized_int8 = quantized.to(torch.int8)

        return quantized_int8, scale

    def dequantize_tensor(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize tensor back to FP32

        Args:
            quantized: Quantized int8 tensor
            scale: Scale factor tensor

        Returns:
            Dequantized FP32 tensor
        """
        # Convert int8 to float
        quantized_float = quantized.to(torch.float32)

        # Multiply by scale factor
        # W_deq = α * Q(W)
        dequantized = scale * quantized_float

        return dequantized

    def should_quantize_layer(self, layer_name: str) -> bool:
        """
        Check if layer should be quantized

        Args:
            layer_name: Name of the layer

        Returns:
            True if layer should be quantized
        """
        # Check against preserve patterns
        layer_name_lower = layer_name.lower()

        for pattern in self.config.preserve_layers:
            if pattern.lower() in layer_name_lower:
                return False

        # Quantize transformer layers
        quantize_patterns = [
            'attention',
            'self_attn',
            'mlp',
            'feed_forward',
            'linear',
            'conv',
        ]

        for pattern in quantize_patterns:
            if pattern in layer_name_lower:
                return True

        # Default: don't quantize
        return False

    def quantize_model(
        self,
        model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Quantize entire model

        Args:
            model: PyTorch model to quantize

        Returns:
            Tuple of (quantized_state_dict, scale_factors)
        """
        quantized_state_dict = {}
        scale_factors = {}

        # Reset stats
        self.stats = {
            'layers_quantized': 0,
            'layers_preserved': 0,
            'total_params': 0,
            'quantized_params': 0,
            'zero_params': 0,
        }

        original_state_dict = model.state_dict()

        for name, param in original_state_dict.items():
            # Count total parameters
            self.stats['total_params'] += param.numel()

            # Check if layer should be quantized
            if self.should_quantize_layer(name):
                # Quantize
                quantized, scale = self.quantize_tensor(param.data)

                quantized_state_dict[name] = quantized
                scale_factors[name] = scale

                # Update stats
                self.stats['layers_quantized'] += 1
                self.stats['quantized_params'] += param.numel()
                self.stats['zero_params'] += (quantized == 0).sum().item()

            else:
                # Preserve in FP16
                if self.config.preserve_embedding_precision:
                    quantized_state_dict[name] = param.data.half()
                else:
                    quantized_state_dict[name] = param.data

                scale_factors[name] = torch.tensor(1.0)

                # Update stats
                self.stats['layers_preserved'] += 1

        # Calculate sparsity ratio
        if self.stats['quantized_params'] > 0:
            self.stats['sparsity_ratio'] = (
                self.stats['zero_params'] / self.stats['quantized_params']
            )

        return quantized_state_dict, scale_factors

    def dequantize_model(
        self,
        quantized_state_dict: Dict[str, torch.Tensor],
        scale_factors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Dequantize model to FP16

        Args:
            quantized_state_dict: Quantized state dict
            scale_factors: Scale factors

        Returns:
            Dequantized FP16 state dict
        """
        dequantized_state_dict = {}

        for name, quantized_param in quantized_state_dict.items():
            if name in scale_factors:
                # Check if parameter was quantized
                if quantized_param.dtype == torch.int8:
                    # Dequantize
                    dequantized = self.dequantize_tensor(
                        quantized_param,
                        scale_factors[name]
                    )
                    # Convert to FP16
                    dequantized_state_dict[name] = dequantized.half()
                else:
                    # Already in higher precision
                    dequantized_state_dict[name] = quantized_param
            else:
                # No scale factor (preserved layer)
                dequantized_state_dict[name] = quantized_param

        return dequantized_state_dict

    def get_stats(self) -> Dict:
        """
        Get quantization statistics

        Returns:
            Statistics dictionary
        """
        return self.stats.copy()

"""
Compressed Model Wrapper
STE-enabled wrapper for quantized models
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from src.phase4_bitnet.quantizer import BitNetQuantizer
from src.phase4_bitnet.config import Phase4Config


class CompressedModel(nn.Module):
    """
    Wrapper for BitNet compressed models

    Features:
    - Transparent quantization/dequantization
    - Straight-Through Estimator (STE) for gradients
    - Compatible with standard PyTorch training
    - Maintains full-precision shadow weights

    Forward Pass: Uses quantized weights
    Backward Pass: Gradients flow to full-precision weights (STE)
    """

    def __init__(
        self,
        base_model: nn.Module,
        quantizer: BitNetQuantizer,
        config: Phase4Config
    ):
        """
        Initialize compressed model

        Args:
            base_model: Original model to compress
            quantizer: BitNet quantizer instance
            config: Phase 4 configuration
        """
        super().__init__()

        self.config = config
        self.quantizer = quantizer
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
        Compress the model using BitNet quantization
        """
        # Quantize model
        quantized_dict, scales = self.quantizer.quantize_model(
            self.base_model
        )

        self.quantized_state = quantized_dict
        self.scale_factors = scales

        # Initialize shadow weights (full precision)
        for name, param in self.base_model.named_parameters():
            self.shadow_weights[name] = param.data.clone().detach()
            self.shadow_weights[name].requires_grad = True

        self.is_compressed = True

    def forward(self, *args, **kwargs):
        """
        Forward pass with quantized weights

        Uses Straight-Through Estimator:
        - Forward: Quantized weights
        - Backward: Gradients to full-precision weights
        """
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

                    # Dequantize if int8
                    if quantized_tensor.dtype == torch.int8:
                        dequantized = self.quantizer.dequantize_tensor(
                            quantized_tensor,
                            self.scale_factors[name]
                        )
                        param.data = dequantized.to(param.device)
                    else:
                        # Preserved layer
                        param.data = quantized_tensor.to(param.device)

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
        if not self.is_compressed:
            # Return original state dict in FP16
            state_dict = self.base_model.state_dict()
            return {k: v.half() for k, v in state_dict.items()}

        # Dequantize
        return self.quantizer.dequantize_model(
            self.quantized_state,
            self.scale_factors
        )

    def get_quantized_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get quantized (int8) state dict for saving

        Returns:
            Quantized state dictionary
        """
        if not self.is_compressed:
            raise RuntimeError("Model not compressed yet")

        return self.quantized_state.copy()

    def get_scale_factors(self) -> Dict[str, torch.Tensor]:
        """
        Get scale factors

        Returns:
            Scale factor dictionary
        """
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
                'is_compressed': False,
                'compression_ratio': 1.0,
            }

        # Calculate sizes
        original_size_mb = self._calculate_state_dict_size(
            self.base_model.state_dict()
        )

        quantized_size_mb = self._calculate_state_dict_size(
            self.quantized_state
        )

        compression_ratio = original_size_mb / quantized_size_mb

        stats = self.quantizer.get_stats()

        return {
            'is_compressed': True,
            'original_size_mb': original_size_mb,
            'quantized_size_mb': quantized_size_mb,
            'compression_ratio': compression_ratio,
            **stats,
        }

    def _calculate_state_dict_size(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> float:
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

        return total_bytes / (1024 ** 2)

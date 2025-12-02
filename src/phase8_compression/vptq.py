"""
Phase 8: VPTQ (Vector Post-Training Quantization)

Second stage of triple compression pipeline.
Applies vector quantization to weight matrices.

Research: "VPTQ: Extreme Compression for LLMs"
Target: 20x compression with >95% retention.

Residual Quantization Process:
1. Quantize original vectors using first codebook
2. Compute residual = original - reconstructed
3. Quantize residual using second codebook
4. Repeat for N codebooks
5. Final reconstruction = sum of all codebook lookups

This multi-codebook approach progressively refines the approximation,
achieving higher quality than single-codebook quantization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import copy
import math

import torch
import torch.nn as nn


@dataclass
class ResidualQuantizationResult:
    """Result from residual quantization of a tensor."""
    indices_per_codebook: List[torch.Tensor]  # Indices for each codebook
    codebooks: List[torch.Tensor]              # The codebooks themselves
    per_codebook_retention: List[float]        # Retention after each level
    final_retention: float                     # Overall retention
    compression_ratio: float                   # Achieved compression


@dataclass
class VPTQConfig:
    """Configuration for VPTQ compression."""
    codebook_size: int = 256  # Number of codewords per codebook
    vector_dim: int = 8  # Dimension of each vector
    num_codebooks: int = 4  # Number of residual codebooks
    num_iterations: int = 50  # K-means iterations per codebook
    target_retention: float = 0.95
    preserve_layers: List[str] = None
    # Residual quantization settings
    use_residual: bool = True  # Enable multi-codebook residual quantization
    residual_scale: float = 1.0  # Scale factor for residuals
    shared_codebooks: bool = False  # Share codebooks across layers


@dataclass
class VPTQResult:
    """Result from VPTQ compression."""
    success: bool
    compressed_state: Dict[str, Any]
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    retention_score: float
    codebook_stats: Dict


class VPTQCompressor:
    """
    VPTQ: Vector Post-Training Quantization.

    Process:
    1. Reshape weights into vectors
    2. Learn codebook via k-means clustering
    3. Replace vectors with codebook indices
    4. Use residual quantization for accuracy

    Benefits:
    - 20x compression
    - Efficient codebook lookup at inference
    - Residual encoding preserves fine details
    """

    def __init__(self, config: VPTQConfig = None):
        """
        Initialize VPTQ compressor.

        Args:
            config: VPTQ configuration
        """
        self.config = config or VPTQConfig()
        if self.config.preserve_layers is None:
            self.config.preserve_layers = ['embed', 'norm', 'ln_', 'layernorm', 'bias']

        self.codebooks: Dict[str, torch.Tensor] = {}

    def compress(
        self,
        model: nn.Module,
        calibration_data: List[Any] = None,
        tokenizer: Any = None
    ) -> Tuple[nn.Module, VPTQResult]:
        """
        Compress model using VPTQ.

        Args:
            model: Model to compress
            calibration_data: Optional calibration samples
            tokenizer: Optional tokenizer

        Returns:
            Tuple of (compressed_model, VPTQResult)
        """
        print("  VPTQ Stage: Vector quantization")

        # Get original size
        original_state = model.state_dict()
        original_size = self._calculate_size(original_state)
        print(f"    Original size: {original_size:.2f} MB")

        # Compress each layer
        compressed_state = {}
        codebook_stats = {}

        for name, param in original_state.items():
            should_preserve = any(p in name.lower() for p in self.config.preserve_layers)

            if should_preserve or param.dim() < 2 or param.numel() < self.config.vector_dim:
                # Preserve layer
                compressed_state[name] = {
                    'type': 'preserved',
                    'data': param.half()
                }
            else:
                # Apply VPTQ
                indices, codebook, retention = self._vptq_compress(param)
                compressed_state[name] = {
                    'type': 'vptq',
                    'indices': indices,
                    'codebook': codebook,
                    'shape': param.shape
                }

                self.codebooks[name] = codebook

                codebook_stats[name] = {
                    'codebook_size': codebook.shape[0],
                    'vector_dim': codebook.shape[1],
                    'num_vectors': indices.numel(),
                    'retention': retention
                }

        # Calculate compressed size
        compressed_size = self._calculate_compressed_size(compressed_state)
        compression_ratio = original_size / max(compressed_size, 0.01)
        print(f"    Compressed size: {compressed_size:.2f} MB")
        print(f"    Compression ratio: {compression_ratio:.2f}x")

        # Create compressed model
        compressed_model = self._create_compressed_model(model, compressed_state)

        # Calculate retention
        retention = self._calculate_retention(codebook_stats)
        print(f"    Retention score: {retention:.2%}")

        return compressed_model, VPTQResult(
            success=True,
            compressed_state=compressed_state,
            original_size_mb=original_size,
            compressed_size_mb=compressed_size,
            compression_ratio=compression_ratio,
            retention_score=retention,
            codebook_stats=codebook_stats
        )

    def _vptq_compress(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compress tensor using vector quantization.

        Args:
            tensor: Weight tensor

        Returns:
            Tuple of (indices, codebook, retention)
        """
        # Use residual quantization if enabled
        if self.config.use_residual and self.config.num_codebooks > 1:
            result = self._residual_quantize(tensor)
            # Pack multi-codebook result into single format for compatibility
            return self._pack_residual_result(result, tensor.shape)

        # Single-codebook fallback
        return self._single_codebook_compress(tensor)

    def _single_codebook_compress(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Single-codebook vector quantization (original implementation).

        Args:
            tensor: Weight tensor

        Returns:
            Tuple of (indices, codebook, retention)
        """
        # Reshape to vectors
        flat = tensor.flatten()
        pad_size = (self.config.vector_dim - len(flat) % self.config.vector_dim) % self.config.vector_dim
        if pad_size > 0:
            flat = torch.cat([flat, torch.zeros(pad_size)])

        vectors = flat.view(-1, self.config.vector_dim)

        # Initialize codebook via k-means++
        codebook = self._init_codebook(vectors)

        # K-means optimization
        for _ in range(self.config.num_iterations):
            # Assign vectors to nearest codeword
            distances = torch.cdist(vectors, codebook)
            indices = distances.argmin(dim=1)

            # Update codebook
            new_codebook = torch.zeros_like(codebook)
            counts = torch.zeros(self.config.codebook_size)

            for i in range(self.config.codebook_size):
                mask = indices == i
                if mask.sum() > 0:
                    new_codebook[i] = vectors[mask].mean(dim=0)
                    counts[i] = mask.sum()
                else:
                    new_codebook[i] = codebook[i]

            codebook = new_codebook

        # Calculate retention
        reconstructed = codebook[indices]
        mse = ((vectors - reconstructed) ** 2).mean().item()
        original_var = vectors.var().item()
        retention = 1.0 - (mse / max(original_var, 1e-10))
        retention = max(0, min(1, retention))

        # Convert indices to uint8 if possible
        if self.config.codebook_size <= 256:
            indices = indices.to(torch.uint8)
        else:
            indices = indices.to(torch.int16)

        return indices, codebook.half(), retention

    def _residual_quantize(
        self,
        tensor: torch.Tensor
    ) -> ResidualQuantizationResult:
        """
        Multi-codebook residual vector quantization.

        Process:
        1. Quantize original vectors with codebook 1
        2. Compute residual = original - reconstruction_1
        3. Quantize residual with codebook 2
        4. Repeat until num_codebooks reached
        5. Final = sum of all reconstructions

        This progressively reduces quantization error at each level.

        Args:
            tensor: Weight tensor to quantize

        Returns:
            ResidualQuantizationResult with all codebooks and indices
        """
        # Reshape to vectors
        flat = tensor.flatten().float()
        pad_size = (self.config.vector_dim - len(flat) % self.config.vector_dim) % self.config.vector_dim
        if pad_size > 0:
            flat = torch.cat([flat, torch.zeros(pad_size)])

        vectors = flat.view(-1, self.config.vector_dim)
        original_vectors = vectors.clone()
        original_var = vectors.var().item()

        all_indices = []
        all_codebooks = []
        per_codebook_retention = []

        # Current signal to quantize (starts as original, becomes residual)
        current_signal = vectors.clone()
        cumulative_reconstruction = torch.zeros_like(vectors)

        for codebook_idx in range(self.config.num_codebooks):
            # Scale residuals for better quantization (optional)
            if codebook_idx > 0:
                scale = self.config.residual_scale
                current_signal = current_signal * scale

            # Initialize codebook for this level
            codebook = self._init_codebook(current_signal)

            # K-means optimization
            for _ in range(self.config.num_iterations):
                distances = torch.cdist(current_signal, codebook)
                indices = distances.argmin(dim=1)

                new_codebook = torch.zeros_like(codebook)
                for i in range(self.config.codebook_size):
                    mask = indices == i
                    if mask.sum() > 0:
                        new_codebook[i] = current_signal[mask].mean(dim=0)
                    else:
                        new_codebook[i] = codebook[i]
                codebook = new_codebook

            # Final assignment
            distances = torch.cdist(current_signal, codebook)
            indices = distances.argmin(dim=1)

            # Reconstruct
            reconstruction = codebook[indices]

            # Unscale if we scaled
            if codebook_idx > 0:
                reconstruction = reconstruction / self.config.residual_scale
                codebook = codebook / self.config.residual_scale

            # Update cumulative reconstruction
            cumulative_reconstruction = cumulative_reconstruction + reconstruction

            # Calculate retention at this level
            mse = ((original_vectors - cumulative_reconstruction) ** 2).mean().item()
            retention = 1.0 - (mse / max(original_var, 1e-10))
            retention = max(0, min(1, retention))
            per_codebook_retention.append(retention)

            # Compute residual for next level
            current_signal = original_vectors - cumulative_reconstruction

            # Store results
            if self.config.codebook_size <= 256:
                indices = indices.to(torch.uint8)
            else:
                indices = indices.to(torch.int16)

            all_indices.append(indices)
            all_codebooks.append(codebook.half())

        # Calculate compression ratio
        # Original: num_vectors * vector_dim * 4 bytes (float32)
        # Compressed: num_vectors * num_codebooks * 1 byte (uint8) +
        #            num_codebooks * codebook_size * vector_dim * 2 bytes (float16)
        num_vectors = vectors.shape[0]
        original_bytes = num_vectors * self.config.vector_dim * 4
        compressed_bytes = (
            num_vectors * self.config.num_codebooks * 1 +  # indices
            self.config.num_codebooks * self.config.codebook_size * self.config.vector_dim * 2  # codebooks
        )
        compression_ratio = original_bytes / max(compressed_bytes, 1)

        return ResidualQuantizationResult(
            indices_per_codebook=all_indices,
            codebooks=all_codebooks,
            per_codebook_retention=per_codebook_retention,
            final_retention=per_codebook_retention[-1] if per_codebook_retention else 0.0,
            compression_ratio=compression_ratio
        )

    def _pack_residual_result(
        self,
        result: ResidualQuantizationResult,
        original_shape: torch.Size
    ) -> Tuple[Any, Any, float]:
        """
        Pack residual quantization result for compatibility with existing interface.

        Args:
            result: ResidualQuantizationResult
            original_shape: Original tensor shape

        Returns:
            Tuple of (packed_indices, packed_codebooks, retention)
        """
        # Pack indices: concatenate along new dimension
        packed_indices = {
            'type': 'residual',
            'indices': result.indices_per_codebook,
            'num_codebooks': len(result.indices_per_codebook),
            'shape': original_shape
        }

        # Pack codebooks: list of codebooks
        packed_codebooks = {
            'type': 'residual',
            'codebooks': result.codebooks,
            'per_codebook_retention': result.per_codebook_retention
        }

        return packed_indices, packed_codebooks, result.final_retention

    def decompress_residual(
        self,
        packed_indices: Dict,
        packed_codebooks: Dict
    ) -> torch.Tensor:
        """
        Decompress residual-quantized tensor.

        Args:
            packed_indices: Packed indices from _pack_residual_result
            packed_codebooks: Packed codebooks from _pack_residual_result

        Returns:
            Decompressed tensor
        """
        indices_list = packed_indices['indices']
        codebooks_list = packed_codebooks['codebooks']
        original_shape = packed_indices['shape']

        # Accumulate reconstructions from all codebooks
        reconstruction = None

        for indices, codebook in zip(indices_list, codebooks_list):
            codebook = codebook.float()
            vectors = codebook[indices.long()]

            if reconstruction is None:
                reconstruction = vectors
            else:
                reconstruction = reconstruction + vectors

        # Flatten and reshape to original shape
        flat = reconstruction.flatten()
        original_size = 1
        for s in original_shape:
            original_size *= s

        return flat[:original_size].reshape(original_shape)

    def _init_codebook(self, vectors: torch.Tensor) -> torch.Tensor:
        """Initialize codebook using k-means++ algorithm."""
        n = len(vectors)
        k = self.config.codebook_size

        # First centroid: random
        idx = torch.randint(0, n, (1,)).item()
        centroids = [vectors[idx].clone()]

        for _ in range(1, k):
            # Calculate distances to nearest centroid
            centroid_tensor = torch.stack(centroids)
            distances = torch.cdist(vectors, centroid_tensor).min(dim=1).values

            # Sample proportional to distance squared
            probs = distances ** 2
            probs = probs / probs.sum()

            # Handle NaN
            if torch.isnan(probs).any():
                probs = torch.ones(n) / n

            idx = torch.multinomial(probs, 1).item()
            centroids.append(vectors[idx].clone())

        return torch.stack(centroids)

    def _calculate_size(self, state_dict: Dict) -> float:
        """Calculate size of state dict in MB."""
        total_bytes = 0
        for param in state_dict.values():
            if isinstance(param, torch.Tensor):
                if param.dtype == torch.float32:
                    total_bytes += param.numel() * 4
                elif param.dtype == torch.float16:
                    total_bytes += param.numel() * 2
                elif param.dtype in [torch.int8, torch.uint8]:
                    total_bytes += param.numel() * 1
                else:
                    total_bytes += param.numel() * 4
        return total_bytes / (1024 * 1024)

    def _calculate_compressed_size(self, compressed_state: Dict) -> float:
        """Calculate compressed state size in MB."""
        total_bytes = 0

        for name, data in compressed_state.items():
            if data['type'] == 'preserved':
                total_bytes += data['data'].numel() * 2  # FP16
            else:  # vptq
                # Indices
                indices = data['indices']
                if indices.dtype == torch.uint8:
                    total_bytes += indices.numel() * 1
                else:
                    total_bytes += indices.numel() * 2

                # Codebook (FP16)
                total_bytes += data['codebook'].numel() * 2

        return total_bytes / (1024 * 1024)

    def _calculate_retention(self, codebook_stats: Dict) -> float:
        """Calculate overall retention score."""
        retentions = [stats['retention'] for stats in codebook_stats.values()]
        return sum(retentions) / max(len(retentions), 1) if retentions else 1.0

    def _create_compressed_model(
        self,
        original_model: nn.Module,
        compressed_state: Dict
    ) -> nn.Module:
        """Create model with decompressed weights."""
        model = copy.deepcopy(original_model)

        decompressed_state = {}
        for name, data in compressed_state.items():
            if data['type'] == 'preserved':
                decompressed_state[name] = data['data'].float()
            else:
                # Decompress from indices
                decompressed = self._decompress_tensor(
                    data['indices'],
                    data['codebook'],
                    data['shape']
                )
                decompressed_state[name] = decompressed

        model.load_state_dict(decompressed_state)
        return model

    def _decompress_tensor(
        self,
        indices: torch.Tensor,
        codebook: torch.Tensor,
        shape: torch.Size
    ) -> torch.Tensor:
        """Decompress tensor from indices."""
        vectors = codebook[indices.long()]
        flat = vectors.flatten()

        # Calculate original size
        original_size = 1
        for s in shape:
            original_size *= s

        return flat[:original_size].reshape(shape).float()


__all__ = [
    'VPTQCompressor',
    'VPTQConfig',
    'VPTQResult',
    'ResidualQuantizationResult'
]

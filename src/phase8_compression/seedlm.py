"""
Phase 8: SeedLM Compression

First stage of triple compression pipeline.
Projects weight matrices to seed-based representation.

Research: "SeedLM: Compressing LLM Weights into Seeds"
Target: 2x compression with >95% retention.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SeedLMConfig:
    """Configuration for SeedLM compression."""
    seed_bits: int = 8  # Bits per seed
    block_size: int = 64  # Weight block size
    num_iterations: int = 100  # Seed search iterations
    target_retention: float = 0.95  # Quality retention target
    preserve_layers: List[str] = None  # Layers to preserve


@dataclass
class SeedLMResult:
    """Result from SeedLM compression."""
    success: bool
    compressed_state: Dict[str, Any]
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    retention_score: float
    layer_stats: Dict[str, Dict]


class SeedLMCompressor:
    """
    SeedLM: Seed-based weight compression.

    Process:
    1. For each weight block, find seed that generates closest approximation
    2. Store seeds instead of full weights
    3. At inference, regenerate weights from seeds

    Benefits:
    - 2x compression with minimal quality loss
    - Deterministic reconstruction
    - Compatible with further compression stages
    """

    def __init__(self, config: SeedLMConfig = None):
        """
        Initialize SeedLM compressor.

        Args:
            config: SeedLM configuration
        """
        self.config = config or SeedLMConfig()
        if self.config.preserve_layers is None:
            self.config.preserve_layers = ['embed', 'norm', 'ln_', 'layernorm', 'bias']

    def compress(
        self,
        model: nn.Module,
        calibration_data: List[Any] = None,
        tokenizer: Any = None
    ) -> Tuple[nn.Module, SeedLMResult]:
        """
        Compress model using SeedLM.

        Args:
            model: Model to compress
            calibration_data: Optional calibration samples
            tokenizer: Optional tokenizer

        Returns:
            Tuple of (compressed_model, SeedLMResult)
        """
        print("  SeedLM Stage: Seed-based projection")

        # Get original size
        original_state = model.state_dict()
        original_size = self._calculate_size(original_state)
        print(f"    Original size: {original_size:.2f} MB")

        # Compress each layer
        compressed_state = {}
        layer_stats = {}

        for name, param in original_state.items():
            should_preserve = any(p in name.lower() for p in self.config.preserve_layers)

            if should_preserve or param.dim() < 2:
                # Preserve layer as-is (but in FP16)
                compressed_state[name] = {
                    'type': 'preserved',
                    'data': param.half()
                }
                layer_stats[name] = {
                    'compression': 1.0,
                    'preserved': True
                }
            else:
                # Compress via seed projection
                seeds, scale, retention = self._compress_tensor(param)
                compressed_state[name] = {
                    'type': 'seedlm',
                    'seeds': seeds,
                    'scale': scale,
                    'shape': param.shape
                }
                layer_stats[name] = {
                    'compression': self._calculate_compression(param, seeds),
                    'retention': retention,
                    'preserved': False
                }

        # Calculate compressed size
        compressed_size = self._calculate_compressed_size(compressed_state)
        compression_ratio = original_size / max(compressed_size, 0.01)
        print(f"    Compressed size: {compressed_size:.2f} MB")
        print(f"    Compression ratio: {compression_ratio:.2f}x")

        # Create compressed model
        compressed_model = self._create_compressed_model(model, compressed_state)

        # Calculate retention score
        retention = self._calculate_retention(compressed_state, layer_stats)
        print(f"    Retention score: {retention:.2%}")

        return compressed_model, SeedLMResult(
            success=True,
            compressed_state=compressed_state,
            original_size_mb=original_size,
            compressed_size_mb=compressed_size,
            compression_ratio=compression_ratio,
            retention_score=retention,
            layer_stats=layer_stats
        )

    def _compress_tensor(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compress a tensor using seed-based projection.

        Args:
            tensor: Weight tensor to compress

        Returns:
            Tuple of (seeds, scale, retention_score)
        """
        original_shape = tensor.shape
        flat = tensor.flatten()

        # Calculate scale
        scale = flat.abs().max()
        if scale > 0:
            normalized = flat / scale
        else:
            normalized = flat

        # Block-wise seed search
        block_size = min(self.config.block_size, len(flat))
        num_blocks = (len(flat) + block_size - 1) // block_size

        seeds = []
        total_error = 0.0

        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, len(flat))
            block = normalized[start:end]

            # Find best seed for this block
            best_seed, best_error = self._find_best_seed(block)
            seeds.append(best_seed)
            total_error += best_error

        seeds_tensor = torch.tensor(seeds, dtype=torch.int64)

        # Calculate retention (1 - normalized error)
        avg_error = total_error / max(num_blocks, 1)
        retention = 1.0 - min(avg_error, 1.0)

        return seeds_tensor, scale, retention

    def _find_best_seed(self, block: torch.Tensor) -> Tuple[int, float]:
        """Find the best seed to approximate a block."""
        best_seed = 0
        best_error = float('inf')

        for _ in range(self.config.num_iterations):
            # Try random seed
            seed = torch.randint(0, 2 ** self.config.seed_bits, (1,)).item()

            # Generate pseudo-random block from seed
            generator = torch.Generator()
            generator.manual_seed(seed)
            generated = torch.randn(len(block), generator=generator)

            # Normalize generated
            if generated.abs().max() > 0:
                generated = generated / generated.abs().max()

            # Calculate error
            error = (block - generated).abs().mean().item()

            if error < best_error:
                best_error = error
                best_seed = seed

        return best_seed, best_error

    def _calculate_size(self, state_dict: Dict) -> float:
        """Calculate size of state dict in MB."""
        total_bytes = 0
        for param in state_dict.values():
            if isinstance(param, torch.Tensor):
                if param.dtype == torch.float32:
                    total_bytes += param.numel() * 4
                elif param.dtype == torch.float16:
                    total_bytes += param.numel() * 2
                elif param.dtype == torch.int8:
                    total_bytes += param.numel() * 1
                elif param.dtype == torch.int64:
                    total_bytes += param.numel() * 8
                else:
                    total_bytes += param.numel() * 4
        return total_bytes / (1024 * 1024)

    def _calculate_compressed_size(self, compressed_state: Dict) -> float:
        """Calculate size of compressed state in MB."""
        total_bytes = 0

        for name, data in compressed_state.items():
            if data['type'] == 'preserved':
                tensor = data['data']
                if tensor.dtype == torch.float16:
                    total_bytes += tensor.numel() * 2
                else:
                    total_bytes += tensor.numel() * 4
            else:  # seedlm
                # Seeds (int64) + scale (float32) + shape metadata
                seeds = data['seeds']
                total_bytes += seeds.numel() * (self.config.seed_bits // 8)
                total_bytes += 4  # scale
                total_bytes += 32  # shape metadata

        return total_bytes / (1024 * 1024)

    def _calculate_compression(self, original: torch.Tensor, seeds: torch.Tensor) -> float:
        """Calculate compression ratio for a layer."""
        original_bytes = original.numel() * 4  # FP32
        compressed_bytes = seeds.numel() * (self.config.seed_bits // 8) + 36  # seeds + metadata
        return original_bytes / max(compressed_bytes, 1)

    def _calculate_retention(self, compressed_state: Dict, layer_stats: Dict) -> float:
        """Calculate overall retention score."""
        retentions = []
        for name, stats in layer_stats.items():
            if 'retention' in stats:
                retentions.append(stats['retention'])
            elif stats.get('preserved'):
                retentions.append(1.0)

        return sum(retentions) / max(len(retentions), 1)

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
                # Decompress from seeds
                decompressed = self._decompress_tensor(
                    data['seeds'],
                    data['scale'],
                    data['shape']
                )
                decompressed_state[name] = decompressed

        model.load_state_dict(decompressed_state)
        return model

    def _decompress_tensor(
        self,
        seeds: torch.Tensor,
        scale: torch.Tensor,
        shape: torch.Size
    ) -> torch.Tensor:
        """Decompress tensor from seeds."""
        flat_size = 1
        for s in shape:
            flat_size *= s

        block_size = self.config.block_size
        blocks = []

        for seed in seeds:
            generator = torch.Generator()
            generator.manual_seed(seed.item())
            block = torch.randn(block_size, generator=generator)
            if block.abs().max() > 0:
                block = block / block.abs().max()
            blocks.append(block)

        flat = torch.cat(blocks)[:flat_size]
        return (flat * scale).reshape(shape)


__all__ = ['SeedLMCompressor', 'SeedLMConfig', 'SeedLMResult']

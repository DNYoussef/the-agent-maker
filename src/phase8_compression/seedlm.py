"""
Phase 8: SeedLM Compression

First stage of triple compression pipeline.
Projects weight matrices to seed-based representation.

Research: "SeedLM: Compressing LLM Weights into Seeds"
Target: 2x compression with >95% retention.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SeedLMConfig:
    """Configuration for SeedLM compression."""

    seed_bits: int = 8  # Bits per seed
    block_size: int = 64  # Weight block size
    # P8: seed-search count. The least-squares fit does the work, so a few seeds suffice;
    # 100 made compressing a large embedding take ~an hour (100 lstsq solves per block).
    num_iterations: int = 16  # Seed search iterations
    latent_dim: int = (
        4  # P8: least-squares coefficients per block (k << block_size = real compression)
    )
    search_seed: int = 1234  # deterministic seed for the seed-search RNG (reproducible compression)
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
        # Codex-final: deterministic RNG for the seed search so compression is reproducible.
        self._rng = torch.Generator().manual_seed(int(self.config.search_seed))
        if self.config.preserve_layers is None:
            # P8: "emb" (not just "embed") so the big tied token_emb/lm_head weight is
            # PRESERVED, not lossily seed-compressed - embeddings are sensitive lookup
            # tables that compress poorly and dominated the runtime.
            self.config.preserve_layers = ["emb", "norm", "ln_", "layernorm", "bias"]

    def compress(
        self, model: nn.Module, calibration_data: List[Any] = None, tokenizer: Any = None
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
                compressed_state[name] = {"type": "preserved", "data": param.half()}
                layer_stats[name] = {"compression": 1.0, "preserved": True}
            else:
                # Compress via least-squares seed projection
                seeds, coeffs, scale, retention = self._compress_tensor(param)
                compressed_state[name] = {
                    "type": "seedlm",
                    "seeds": seeds,
                    "coeffs": coeffs,
                    "scale": scale,
                    "shape": param.shape,
                    "block_size": self.config.block_size,
                }
                layer_stats[name] = {
                    "compression": self._calculate_compression(param, seeds, coeffs),
                    "retention": retention,
                    "preserved": False,
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
            layer_stats=layer_stats,
        )

    def _compress_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
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
        coeffs = []
        k = self.config.latent_dim
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, len(flat))
            block = normalized[start:end]

            # P8: fit the block by LEAST-SQUARES onto a seed-generated basis (real SeedLM),
            # not the old random-noise nearest-match.
            best_seed, best_coeffs = self._fit_block(block)
            seeds.append(best_seed)
            # pad coeffs to k so they stack into a uniform [num_blocks, k] tensor
            padded = torch.zeros(k)
            padded[: best_coeffs.numel()] = best_coeffs
            coeffs.append(padded)

        seeds_tensor = torch.tensor(seeds, dtype=torch.int64)
        coeffs_tensor = torch.stack(coeffs) if coeffs else torch.zeros(0, k)

        # Retention is reconstruction fidelity against the original tensor.
        reconstructed = self._reconstruct_from_seeds(
            seeds_tensor, scale, original_shape, block_size, coeffs_tensor
        )
        retention = self._calculate_reconstruction_fidelity(tensor, reconstructed)

        return seeds_tensor, coeffs_tensor, scale, retention

    def _basis(self, seed: int, block_len: int, k: int) -> torch.Tensor:
        """Deterministic pseudo-random basis [block_len, k] generated from an integer seed."""
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return torch.randn(block_len, k, generator=generator)

    def _fit_block(self, block: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """P8: search seeds; for each, least-squares fit block ~ basis(seed) @ coeffs. Keep
        the (seed, coeffs) with the lowest residual. This is real compression+reconstruction,
        replacing the old 'closest random vector' noise match.

        Codex-final: the fit runs on CPU (the seed RNG/basis are CPU; moving a CUDA block here
        avoids a device mismatch in lstsq), and the seed SEARCH uses a deterministic per-
        compressor generator so compression is reproducible (reconstruction always was)."""
        # Codex-final: cast to fp32 too (not just CPU) - _basis is fp32, so an fp16/bf16
        # block would dtype-mismatch in lstsq.
        block = block.detach().to(device="cpu", dtype=torch.float32)
        k = min(self.config.latent_dim, len(block))
        best_seed, best_coeffs, best_err = 0, torch.zeros(k), float("inf")
        target = block.unsqueeze(1)  # [B, 1]
        for _ in range(self.config.num_iterations):
            seed = int(
                torch.randint(0, 2**self.config.seed_bits, (1,), generator=self._rng).item()
            )
            basis = self._basis(seed, len(block), k)  # [B, k] on CPU
            coeffs = torch.linalg.lstsq(basis, target).solution.squeeze(1)  # [k]
            err = (basis @ coeffs - block).pow(2).mean().item()
            if err < best_err:
                best_err, best_seed, best_coeffs = err, seed, coeffs
        return best_seed, best_coeffs

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
            if data["type"] == "preserved":
                tensor = data["data"]
                if tensor.dtype == torch.float16:
                    total_bytes += tensor.numel() * 2
                else:
                    total_bytes += tensor.numel() * 4
            else:  # seedlm
                # Seeds are stored as actual tensors here; do not charge an
                # aspirational packed seed_bits size unless a packer exists.
                seeds = data["seeds"]
                total_bytes += self._seed_storage_bytes(seeds)
                coeffs = data.get("coeffs")
                if coeffs is not None:
                    total_bytes += coeffs.numel() * 4  # fp32 least-squares coefficients
                total_bytes += 4  # scale
                total_bytes += 32  # shape metadata

        return total_bytes / (1024 * 1024)

    def _calculate_compression(
        self, original: torch.Tensor, seeds: torch.Tensor, coeffs: Optional[torch.Tensor] = None
    ) -> float:
        """Calculate compression ratio for a layer: original fp32 vs (seeds + coeffs)."""
        original_bytes = original.numel() * 4  # FP32
        coeff_bytes = coeffs.numel() * 4 if coeffs is not None else 0  # fp32 coefficients
        compressed_bytes = self._seed_storage_bytes(seeds) + coeff_bytes + 36  # + metadata
        return original_bytes / max(compressed_bytes, 1)

    def _seed_storage_bytes(self, seeds: torch.Tensor) -> int:
        """Return bytes for the seed tensor representation currently stored."""
        return seeds.numel() * seeds.element_size()

    def _calculate_reconstruction_fidelity(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> float:
        """Compute fidelity as one minus relative RMSE, clamped to [0, 1]."""
        original_flat = original.flatten().float()
        reconstructed_flat = reconstructed.flatten().float()
        min_len = min(len(original_flat), len(reconstructed_flat))
        original_flat = original_flat[:min_len]
        reconstructed_flat = reconstructed_flat[:min_len]

        if min_len == 0:
            return 1.0

        rmse = torch.sqrt(((original_flat - reconstructed_flat) ** 2).mean()).item()
        reference = torch.sqrt((original_flat**2).mean()).item()

        if reference <= 1e-12:
            return 1.0 if rmse <= 1e-12 else 0.0

        fidelity = 1.0 - (rmse / reference)
        return max(0.0, min(1.0, fidelity))

    def _calculate_retention(self, compressed_state: Dict, layer_stats: Dict) -> float:
        """Calculate overall retention score."""
        retentions = []
        for name, stats in layer_stats.items():
            if "retention" in stats:
                retentions.append(stats["retention"])
            elif stats.get("preserved"):
                retentions.append(1.0)

        return sum(retentions) / max(len(retentions), 1)

    def _create_compressed_model(
        self, original_model: nn.Module, compressed_state: Dict
    ) -> nn.Module:
        """Create model with decompressed weights."""
        model = copy.deepcopy(original_model)

        decompressed_state = {}
        for name, data in compressed_state.items():
            if data["type"] == "preserved":
                decompressed_state[name] = data["data"].float()
            else:
                # Decompress from seeds
                decompressed = self._decompress_tensor(
                    data["seeds"],
                    data["scale"],
                    data["shape"],
                    data.get("block_size"),
                    data.get("coeffs"),
                )
                decompressed_state[name] = decompressed

        model.load_state_dict(decompressed_state)
        return model

    def _decompress_tensor(
        self,
        seeds: torch.Tensor,
        scale: torch.Tensor,
        shape: torch.Size,
        block_size: Optional[int] = None,
        coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decompress tensor from seeds + least-squares coefficients."""
        return self._reconstruct_from_seeds(seeds, scale, shape, block_size, coeffs)

    def _reconstruct_from_seeds(
        self,
        seeds: torch.Tensor,
        scale: torch.Tensor,
        shape: torch.Size,
        block_size: Optional[int] = None,
        coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """P8: reconstruct each block as basis(seed) @ coeffs (real seed-projection), not a
        scaled random vector. coeffs is [num_blocks, k]."""
        flat_size = 1
        for s in shape:
            flat_size *= s

        block_size = block_size or self.config.block_size
        blocks = []

        for i, seed in enumerate(seeds):
            remaining = flat_size - i * block_size
            block_len = min(block_size, remaining)
            if block_len <= 0:
                break
            # Codex-final: keep the basis/coeffs math on CPU (the seed RNG is CPU); a CUDA
            # scale would otherwise device-mismatch. Move to the target device at the end.
            # fp32 on CPU to match _basis (avoids fp16/bf16 dtype mismatch in the matmul).
            if coeffs is not None:
                c = coeffs[i].to(device="cpu", dtype=torch.float32)
            else:
                c = torch.zeros(self.config.latent_dim)
            k = min(c.numel(), block_len)
            basis = self._basis(int(seed.item()), block_len, k)  # [block_len, k] fp32 CPU
            blocks.append(basis @ c[:k])

        flat = torch.cat(blocks)[:flat_size] if blocks else torch.zeros(flat_size)
        # restore the original device AND dtype (the weight may be fp16/bf16).
        return (flat.to(scale.device) * scale).reshape(shape).to(scale.dtype)


__all__ = ["SeedLMCompressor", "SeedLMConfig", "SeedLMResult"]

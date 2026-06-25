"""Tier3-P8 gate - CRUCIBLE: SeedLM is real (least-squares), not random-noise matching.

Synthesis: SeedLM picked whichever random torch.randn vector was closest to a weight block
(_find_best_seed) and stored only seeds - reconstruction was noise and 'compression' didn't
reconstruct. P8 replaces it with least-squares coefficients onto a seed-generated basis:
block ~ basis(seed) @ coeffs. This reconstructs the weight AND compresses (k coeffs << block).
"""

import torch

from phase8_compression.seedlm import SeedLMCompressor, SeedLMConfig


def test_seedlm_reconstruction_explains_variance():
    torch.manual_seed(0)
    comp = SeedLMCompressor(SeedLMConfig(block_size=64, latent_dim=16, num_iterations=20))
    weight = torch.randn(8, 64)  # a 2-D weight matrix
    seeds, coeffs, scale, retention = comp._compress_tensor(weight)
    recon = comp._reconstruct_from_seeds(seeds, scale, weight.shape, comp.config.block_size, coeffs)
    mse = (recon - weight).pow(2).mean().item()
    var = weight.var().item()
    assert mse < var, f"least-squares recon must explain variance (mse {mse:.4f} < var {var:.4f})"
    # The LEAST-SQUARES coeffs must beat RANDOM coeffs on the SAME basis - this is what the
    # old noise-match lacked (it had no fitted coefficients at all). On random weights the
    # absolute retention is modest (~k/block energy); real structured weights retain more.
    rand_coeffs = torch.randn_like(coeffs)
    recon_rand = comp._reconstruct_from_seeds(seeds, scale, weight.shape, 64, rand_coeffs)
    mse_rand = (recon_rand - weight).pow(2).mean().item()
    assert mse < mse_rand, f"LS fit ({mse:.4f}) must beat random coeffs ({mse_rand:.4f})"
    assert retention > 0.1, f"retention should beat the noise-match baseline (~0); got {retention}"


def test_seedlm_actually_compresses():
    torch.manual_seed(0)
    comp = SeedLMCompressor(SeedLMConfig(block_size=64, latent_dim=4, num_iterations=10))
    weight = torch.randn(64, 64)
    seeds, coeffs, scale, _ = comp._compress_tensor(weight)
    ratio = comp._calculate_compression(weight, seeds, coeffs)
    assert ratio > 1.0, f"compression ratio must be > 1.0 (real compression); got {ratio:.2f}"


def test_seedlm_reconstruction_is_deterministic():
    torch.manual_seed(0)
    comp = SeedLMCompressor(SeedLMConfig(block_size=64, latent_dim=8, num_iterations=10))
    weight = torch.randn(4, 64)
    seeds, coeffs, scale, _ = comp._compress_tensor(weight)
    a = comp._reconstruct_from_seeds(seeds, scale, weight.shape, 64, coeffs)
    b = comp._reconstruct_from_seeds(seeds, scale, weight.shape, 64, coeffs)
    assert torch.equal(a, b), "reconstruction from seeds+coeffs must be deterministic"

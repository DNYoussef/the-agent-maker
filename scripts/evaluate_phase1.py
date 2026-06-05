#!/usr/bin/env python3
"""Evaluate Phase 1 trained models."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cross_phase.utils.checkpoint_utils import load_checkpoint
from phase1_cognate.model.model_config import Phase1Config
from phase1_cognate.model.full_model import TRMTitansMAGModel


CHECKPOINT_ROOT = Path("checkpoints/phase1_enhanced")


def secure_checkpoint_base(spec: str, checkpoint_root: Path = CHECKPOINT_ROOT) -> Path:
    """Return the SafeTensors checkpoint base, rejecting legacy pickle checkpoints."""
    base = checkpoint_root / spec / "best_model"
    safetensors_path = base.with_suffix(".safetensors")
    legacy_path = base.with_suffix(".pt")

    if safetensors_path.exists():
        return base

    if legacy_path.exists():
        raise RuntimeError(
            f"Refusing to load legacy pickle checkpoint: {legacy_path}. "
            "Convert it to SafeTensors with a trusted one-time migration before evaluation."
        )

    raise FileNotFoundError(f"SafeTensors checkpoint not found: {safetensors_path}")


def main():
    print("=" * 60)
    print("  PHASE 1 MODEL EVALUATION")
    print("  Enhanced Agent Maker with Meta Calculus")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    results = {}

    for spec in ["reasoning", "memory", "speed"]:
        print(f"\n{'=' * 50}")
        print(f"  Evaluating: {spec.upper()} Model")
        print(f"{'=' * 50}")

        config = Phase1Config(specialization=spec)
        model = TRMTitansMAGModel(config).to(device)
        checkpoint_base = secure_checkpoint_base(spec)
        checkpoint = load_checkpoint(model, checkpoint_base, device=device, strict=True)
        metadata = checkpoint.get("metadata", {})

        print(f"  Checkpoint epoch: {metadata.get('epoch', 'unknown')}")
        loss = float(metadata.get("loss", 0.0))
        print(f"  Training loss: {loss:.4f}")

        # Get metrics from training
        metrics = metadata.get("metrics", {})
        improvement = 0
        if metrics:
            losses = metrics.get("losses", [])
            gaps = metrics.get("spectral_gaps", [])
            k_vals = metrics.get("k_values", [])
            grad_norms = metrics.get("grad_norms", [])

            if losses:
                print(f"  Initial loss: {losses[0]:.4f}")
                print(f"  Final loss: {losses[-1]:.4f}")
                improvement = (losses[0] - losses[-1]) / losses[0] * 100
                print(f"  Improvement: {improvement:.2f}%")

            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                print(f"  Avg spectral gap: {avg_gap:.4f}")

            if k_vals:
                print(f"  k(L) range: {min(k_vals):.4f} - {max(k_vals):.4f}")

            if grad_norms:
                print(f"  Grad norm range: {min(grad_norms):.2f} - {max(grad_norms):.2f}")

        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total_params:,}")

        # Quick inference test
        with torch.no_grad():
            test_input = torch.randint(0, 1000, (1, 64)).to(device)
            output = model(test_input)
            if isinstance(output, dict):
                logits = output.get("logits", output.get("output"))
            else:
                logits = output
            print(f"  Output shape: {logits.shape}")

            # Compute perplexity on random data
            vocab_size = config.titans_config.vocab_size
            test_labels = torch.randint(0, vocab_size, (1, 64)).to(device)
            if logits.dim() == 3:
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = test_labels.view(-1)
            loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat)
            perplexity = torch.exp(loss).item()
            print(f"  Test perplexity: {perplexity:.2f}")

        results[spec] = {
            "loss": loss,
            "perplexity": perplexity,
            "params": total_params,
            "improvement": improvement,
        }

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{'Model':<12} {'Loss':<10} {'Perplexity':<12} {'Params':>12}")
    print("-" * 50)
    for spec, r in results.items():
        print(f"{spec:<12} {r['loss']:<10.4f} {r['perplexity']:<12.2f} {r['params']:>12,}")

    print(f"\n{'=' * 60}")
    print("  All models evaluated successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

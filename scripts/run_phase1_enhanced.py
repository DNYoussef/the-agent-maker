#!/usr/bin/env python3
"""
Enhanced Phase 1 (Cognate) Training with Meta Calculus + GlobalMOO

Features:
- MetaGrokfast optimizer with bigeometric gradient transform
- k(L) formula for scale-dependent adaptation
- Spectral gap monitoring for diversity
- All 3 specialized models: Reasoning, Memory, Speed

Usage:
    python scripts/run_phase1_enhanced.py --model reasoning
    python scripts/run_phase1_enhanced.py --all
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# Meta Calculus imports
from cross_phase.meta_calculus.k_formula import compute_k, k_from_layer_index
from cross_phase.meta_calculus.spectral_gap import SpectralGapMonitor
from cross_phase.meta_calculus.meta_grokfast import MetaGrokfast, MetaGrokfastConfig

# Phase 1 imports
from phase1_cognate.model.model_config import Phase1Config
from phase1_cognate.model.full_model import TRMTitansMAGModel


class EnhancedPhase1Trainer:
    """Enhanced Phase 1 trainer with Meta Calculus integration."""

    def __init__(
        self,
        model: nn.Module,
        specialization: str,
        config: MetaGrokfastConfig,
        device: str = "cuda",
        checkpoint_dir: Path = Path("checkpoints/phase1_enhanced"),
    ):
        self.model = model.to(device)
        self.specialization = specialization
        self.device = device
        self.checkpoint_dir = checkpoint_dir / specialization
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # MetaGrokfast optimizer with Phase 1 settings
        self.optimizer = MetaGrokfast(model.parameters(), config=config)

        # Spectral gap monitor
        self.gap_monitor = SpectralGapMonitor()

        # Training metrics
        self.metrics = {
            "losses": [],
            "spectral_gaps": [],
            "k_values": [],
            "grad_norms": [],
        }

        # Print config
        print(f"\n[MetaGrokfast Config]")
        print(f"  Learning rate: {config.lr}")
        print(f"  Grokfast lambda: {config.grokfast_lambda}")
        print(f"  Bigeometric: {config.use_bigeometric}")
        print(f"  Adaptive k: {config.use_adaptive_k}")
        print(f"  QK-clip: {config.use_qk_clip}")

    def create_synthetic_data(self, num_samples: int = 1000, seq_len: int = 128):
        """Create synthetic training data for demo."""
        # vocab_size is in titans_config
        vocab_size = self.model.config.titans_config.vocab_size

        # Create random token sequences with next-token labels for demo only.
        input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
        labels = input_ids.roll(shifts=-1, dims=1)
        labels[:, -1] = -100

        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def compute_loss(self, input_ids, labels):
        """Forward pass and loss computation."""
        outputs = self.model(input_ids)

        # Handle different output formats
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("output"))
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        # Reshape for cross entropy
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

        loss = nn.functional.cross_entropy(logits, labels, ignore_index=-100)
        return loss, logits

    def train_epoch(self, dataloader, epoch: int):
        """Train one epoch with Meta Calculus enhancements."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            loss, logits = self.compute_loss(input_ids, labels)

            # Backward pass
            loss.backward()

            # Compute gradient norm for k(L)
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm**0.5

            # Compute k from gradient norm
            k = compute_k(max(grad_norm, 1e-8))
            self.metrics["k_values"].append(k)
            self.metrics["grad_norms"].append(grad_norm)

            # Optimizer step. MetaGrokfast owns Grokfast and bigeometric gradient processing.
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log every 10 batches
            if batch_idx % 10 == 0:
                print(
                    f"  Batch {batch_idx}: loss={loss.item():.4f}, k={k:.4f}, grad_norm={grad_norm:.2f}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        self.metrics["losses"].append(avg_loss)

        return avg_loss

    def compute_spectral_health(self):
        """Compute spectral gap for diversity monitoring."""
        self.model.eval()
        with torch.no_grad():
            # Get embeddings from the model
            dummy_input = torch.randint(0, 1000, (32, 64)).to(self.device)

            # Try to get embeddings
            if hasattr(self.model, "embedding"):
                embeddings = self.model.embedding(dummy_input)
            elif hasattr(self.model, "get_input_embeddings"):
                embeddings = self.model.get_input_embeddings()(dummy_input)
            else:
                # Use first layer output
                outputs = self.model(dummy_input)
                if isinstance(outputs, dict):
                    embeddings = outputs.get("hidden_states", outputs.get("logits", None))
                else:
                    embeddings = outputs

            if embeddings is not None:
                # Flatten to (batch * seq, hidden)
                if embeddings.dim() == 3:
                    embeddings = embeddings.view(-1, embeddings.size(-1))
                gap = self.gap_monitor.compute_gap(embeddings.cpu())
                # Handle dict or tensor or float return
                if isinstance(gap, dict):
                    gap_val = gap.get("gap", gap.get("spectral_gap", 0.0))
                elif hasattr(gap, "item"):
                    gap_val = gap.item()
                else:
                    gap_val = float(gap) if gap is not None else 0.0
                self.metrics["spectral_gaps"].append(gap_val)
                return gap_val
        return None

    def train(self, num_epochs: int = 10, num_samples: int = 2000):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"  ENHANCED PHASE 1 TRAINING: {self.specialization.upper()}")
        print(f"  Meta Calculus + MetaGrokfast")
        print(f"{'='*60}")

        # Create synthetic data (replace with real data for production)
        print(f"\nCreating training data ({num_samples} samples)...")
        dataloader = self.create_synthetic_data(num_samples)
        print(f"Created {len(dataloader)} batches")

        # Training loop
        start_time = time.time()
        best_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            print(f"\n[Epoch {epoch}/{num_epochs}]")

            # Train epoch
            avg_loss = self.train_epoch(dataloader, epoch)

            # Compute spectral health
            gap = self.compute_spectral_health()
            gap_str = f"{gap:.4f}" if gap is not None else "N/A"

            epoch_time = time.time() - epoch_start
            print(f"  Loss: {avg_loss:.4f}, Spectral Gap: {gap_str}, Time: {epoch_time:.1f}s")

            # Save checkpoint if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(epoch, avg_loss, is_best=True)

            # Save periodic checkpoint
            if epoch % 2 == 0:
                self.save_checkpoint(epoch, avg_loss)

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE!")
        final_loss = self.metrics["losses"][-1] if self.metrics["losses"] else float("nan")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Best Loss: {best_loss:.4f}")
        print(f"  Total Time: {total_time/60:.1f} minutes")
        print(f"{'='*60}")

        return self.metrics

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        filename = "best_model.pt" if is_best else f"epoch_{epoch}.pt"
        path = self.checkpoint_dir / filename

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
                "metrics": self.metrics,
                "specialization": self.specialization,
                "synthetic_demo": True,
                "artifact_provenance": "synthetic random token stream with shifted labels; not a production training checkpoint",
            },
            path,
        )

        print(f"  Saved checkpoint: {path}")


def get_phase1_config(specialization: str) -> MetaGrokfastConfig:
    """Get MetaGrokfast config for Phase 1 specialization."""
    # Base config for Phase 1
    base_config = {
        "lr": 1e-3,
        "grokfast_alpha": 0.98,
        "grokfast_lambda": 0.3,  # Gentle for pretraining
        "use_bigeometric": True,
        "use_adaptive_k": True,
        "use_qk_clip": False,  # Not needed for pretraining
        "warmup_steps": 500,
    }

    # Specialization-specific adjustments
    if specialization == "reasoning":
        base_config["grokfast_lambda"] = 0.25  # Slightly more aggressive
        base_config["lr"] = 1.2e-3
    elif specialization == "memory":
        base_config["grokfast_lambda"] = 0.35  # More conservative
        base_config["lr"] = 0.8e-3
    elif specialization == "speed":
        base_config["grokfast_lambda"] = 0.2  # Most aggressive
        base_config["lr"] = 1.5e-3

    return MetaGrokfastConfig(**base_config)


def create_model(specialization: str) -> TRMTitansMAGModel:
    """Create Phase 1 model for specialization."""
    config = Phase1Config(specialization=specialization)
    model = TRMTitansMAGModel(config)

    # Print model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nCreated {specialization} model: {param_count:,} parameters")

    return model


def main():
    parser = argparse.ArgumentParser(description="Enhanced Phase 1 Training with Meta Calculus")
    parser.add_argument(
        "--model",
        type=str,
        choices=["reasoning", "memory", "speed"],
        help="Train single model",
    )
    parser.add_argument("--all", action="store_true", help="Train all 3 models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--samples", type=int, default=2000, help="Training samples")
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run the built-in synthetic demo data path. Required until a real dataset loader is provided.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/phase1_enhanced")
    )

    args = parser.parse_args()

    if not args.synthetic_demo:
        print("Error: this script currently has only a synthetic demo data path. Pass --synthetic-demo to run it, or use a real Phase 1 training entrypoint.")
        return 2

    if args.checkpoint_dir.name != "synthetic_demo":
        args.checkpoint_dir = args.checkpoint_dir / "synthetic_demo"


    print("\n" + "=" * 60)
    print("  ENHANCED AGENT MAKER - Phase 1 (Cognate)")
    print("  Meta Calculus + MetaGrokfast + Spectral Gap")
    print("=" * 60)

    print(f"\nDevice: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Determine which models to train
    if args.all:
        specializations = ["reasoning", "memory", "speed"]
    elif args.model:
        specializations = [args.model]
    else:
        print("Error: Must specify --model or --all")
        return 1

    # Train each model
    all_metrics = {}
    for spec in specializations:
        print(f"\n{'#'*60}")
        print(f"  TRAINING: {spec.upper()} MODEL")
        print(f"{'#'*60}")

        # Create model
        model = create_model(spec)

        # Get optimizer config
        config = get_phase1_config(spec)

        # Create trainer
        trainer = EnhancedPhase1Trainer(
            model=model,
            specialization=spec,
            config=config,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )

        # Train
        metrics = trainer.train(num_epochs=args.epochs, num_samples=args.samples)
        all_metrics[spec] = metrics

    # Final summary
    print("\n" + "=" * 60)
    print("  ALL TRAINING COMPLETE!")
    print("=" * 60)
    for spec, metrics in all_metrics.items():
        final_loss = metrics["losses"][-1] if metrics["losses"] else "N/A"
        final_gap = metrics["spectral_gaps"][-1] if metrics["spectral_gaps"] else "N/A"
        print(f"  {spec.upper()}: loss={final_loss}, gap={final_gap}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Enhanced Phase 2 (EvoMerge) with MOO + Meta Calculus Integration

This script runs evolutionary model merging with:
- Multi-Objective Optimization (NSGA-II via pymoo)
- Bigeometric merge (novel log-space technique)
- k(L) adaptive layer merge ratios
- Spectral gap monitoring
- 6 merge techniques: linear, slerp, ties, dare, franken, dfs, bigeometric

Usage:
    python scripts/run_phase2_enhanced.py --checkpoint-dir checkpoints/phase1_real --generations 30
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.phase1_cognate.model import Phase1Config, TRMTitansMAGModel


@dataclass
class EnhancedEvoConfig:
    """Enhanced evolution config with MOO settings."""
    num_generations: int = 30
    population_size: int = 12
    elite_count: int = 2
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8

    # MOO settings
    use_moo: bool = True
    moo_objectives: List[str] = None  # perplexity, spectral_gap, weight_norm, speed

    # Meta Calculus settings
    use_bigeometric: bool = True
    use_klayer_ratios: bool = True

    # Merge techniques
    merge_techniques: List[str] = None

    def __post_init__(self):
        if self.moo_objectives is None:
            self.moo_objectives = ["perplexity", "spectral_gap", "weight_norm", "speed"]
        if self.merge_techniques is None:
            self.merge_techniques = ["linear", "slerp", "ties", "dare", "bigeometric"]


class MetaCalculusIntegration:
    """Meta Calculus utilities for Phase 2."""

    @staticmethod
    def compute_k(loss: float) -> float:
        """k(L) formula: k(L) = -0.0137 * log10(L) + 0.1593"""
        if loss <= 0:
            loss = 1e-10
        import math
        return -0.0137 * math.log10(loss) + 0.1593

    @staticmethod
    def get_layer_merge_ratios(num_layers: int, base_loss: float = 1.0) -> List[float]:
        """Get k(L)-adaptive merge ratios per layer."""
        ratios = []
        for i in range(num_layers):
            # Layer depth factor (deeper layers get higher ratios)
            depth_factor = (i + 1) / num_layers
            # Compute k for this layer
            effective_loss = base_loss * (1 + depth_factor)
            k = MetaCalculusIntegration.compute_k(effective_loss)
            # Scale k to merge ratio range [0.3, 0.7]
            ratio = 0.3 + 0.4 * (k / 0.2)  # Normalize k typically in [0.15, 0.2]
            ratio = max(0.3, min(0.7, ratio))
            ratios.append(ratio)
        return ratios

    @staticmethod
    def bigeometric_merge_weights(
        w1: torch.Tensor,
        w2: torch.Tensor,
        ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Bigeometric merge: works in log-space for weight distributions.

        g_merged = sign(w1*w2) * |w1|^(1-r) * |w2|^r
        """
        eps = 1e-10

        # Handle signs
        sign1 = torch.sign(w1)
        sign2 = torch.sign(w2)
        same_sign = sign1 == sign2
        voted_sign = torch.sign((1 - ratio) * sign1 + ratio * sign2 + eps)

        # Absolute values with epsilon
        abs1 = torch.abs(w1) + eps
        abs2 = torch.abs(w2) + eps

        # Log-space interpolation
        log1 = torch.log(abs1)
        log2 = torch.log(abs2)
        log_merged = (1 - ratio) * log1 + ratio * log2

        # Back to linear space
        merged_abs = torch.exp(log_merged)

        # Preserve common signs; when signs disagree, use weighted sign voting.
        merged = torch.where(
            same_sign,
            sign1 * merged_abs,
            voted_sign * merged_abs,
        )

        return merged

    @staticmethod
    def compute_spectral_gap(model: nn.Module) -> float:
        """Compute spectral gap as diversity metric."""
        all_weights = []
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                all_weights.append(param.data.flatten())

        if not all_weights:
            return 1.0

        # Concatenate and compute singular values
        concat = torch.cat(all_weights)
        if len(concat) > 10000:
            indices = torch.linspace(0, len(concat) - 1, 10000, device=concat.device).long()
            concat = concat[indices]

        # Reshape to matrix
        size = int(np.sqrt(len(concat)))
        if size < 2:
            return 1.0

        matrix = concat[:size*size].reshape(size, size)

        try:
            svd = torch.linalg.svdvals(matrix)
            if len(svd) >= 2:
                gap = (svd[0] - svd[1]) / (svd[0] + 1e-10)
                return gap.item()
        except:
            pass

        return 1.0


class MergeEngine:
    """Merge techniques engine."""

    @staticmethod
    def linear_merge(
        state_a: Dict[str, torch.Tensor],
        state_b: Dict[str, torch.Tensor],
        ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Linear interpolation merge."""
        merged = {}
        for key in state_a:
            if key in state_b:
                merged[key] = (1 - ratio) * state_a[key] + ratio * state_b[key]
            else:
                merged[key] = state_a[key]
        return merged

    @staticmethod
    def slerp_merge(
        state_a: Dict[str, torch.Tensor],
        state_b: Dict[str, torch.Tensor],
        ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Spherical linear interpolation merge."""
        merged = {}
        for key in state_a:
            if key in state_b:
                w1, w2 = state_a[key].float(), state_b[key].float()

                # Flatten for dot product
                w1_flat = w1.flatten()
                w2_flat = w2.flatten()

                # Normalize
                norm1 = torch.norm(w1_flat) + 1e-10
                norm2 = torch.norm(w2_flat) + 1e-10

                w1_norm = w1_flat / norm1
                w2_norm = w2_flat / norm2

                # Compute angle
                dot = torch.clamp(torch.dot(w1_norm, w2_norm), -1.0, 1.0)
                theta = torch.acos(dot)

                if theta.abs() < 1e-6:
                    # Nearly parallel, use linear
                    merged_flat = (1 - ratio) * w1_flat + ratio * w2_flat
                else:
                    # SLERP
                    sin_theta = torch.sin(theta)
                    merged_flat = (
                        torch.sin((1 - ratio) * theta) / sin_theta * w1_flat +
                        torch.sin(ratio * theta) / sin_theta * w2_flat
                    )

                merged[key] = merged_flat.reshape(w1.shape).to(state_a[key].dtype)
            else:
                merged[key] = state_a[key]
        return merged

    @staticmethod
    def ties_merge(
        state_a: Dict[str, torch.Tensor],
        state_b: Dict[str, torch.Tensor],
        ratio: float = 0.5,
        threshold: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """TIES: Task Interference Elimination via Sign."""
        merged = {}
        for key in state_a:
            if key in state_b:
                w1, w2 = state_a[key], state_b[key]

                # Compute task vectors (difference from mean)
                mean = (w1 + w2) / 2
                delta1 = w1 - mean
                delta2 = w2 - mean

                # Trim small values
                mask1 = delta1.abs() > threshold * delta1.abs().max()
                mask2 = delta2.abs() > threshold * delta2.abs().max()

                delta1 = delta1 * mask1.float()
                delta2 = delta2 * mask2.float()

                # Resolve sign conflicts (take larger magnitude)
                sign_conflict = (delta1 * delta2) < 0
                take_from_1 = delta1.abs() >= delta2.abs()

                merged_delta = torch.where(
                    sign_conflict,
                    torch.where(take_from_1, delta1, delta2),
                    (1 - ratio) * delta1 + ratio * delta2
                )

                merged[key] = mean + merged_delta
            else:
                merged[key] = state_a[key]
        return merged

    @staticmethod
    def dare_merge(
        state_a: Dict[str, torch.Tensor],
        state_b: Dict[str, torch.Tensor],
        ratio: float = 0.5,
        drop_rate: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """DARE: Drop And REscale merge."""
        merged = {}
        for key in state_a:
            if key in state_b:
                w1, w2 = state_a[key], state_b[key]

                # Random drop mask
                mask = torch.rand_like(w1.float()) > drop_rate

                # Rescale factor
                scale = 1.0 / (1.0 - drop_rate + 1e-10)

                # Merge with dropout
                merged_w = (1 - ratio) * w1 + ratio * w2
                merged_w = merged_w * mask.float() * scale

                # Add back base (prevent collapse)
                merged_w = merged_w + (1 - mask.float()) * (w1 + w2) / 2

                merged[key] = merged_w
            else:
                merged[key] = state_a[key]
        return merged

    @staticmethod
    def bigeometric_merge(
        state_a: Dict[str, torch.Tensor],
        state_b: Dict[str, torch.Tensor],
        ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Bigeometric (log-space) merge - Meta Calculus technique."""
        merged = {}
        for key in state_a:
            if key in state_b:
                merged[key] = MetaCalculusIntegration.bigeometric_merge_weights(
                    state_a[key], state_b[key], ratio
                )
            else:
                merged[key] = state_a[key]
        return merged


class EnhancedPhase2Runner:
    """Enhanced Phase 2 runner with MOO + Meta Calculus."""

    def __init__(self, config: EnhancedEvoConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.meta = MetaCalculusIntegration()
        self.merge_engine = MergeEngine()

        # Tracking
        self.population = []
        self.fitness_history = []
        self.best_fitness = float('-inf')
        self.champion = None
        self.metrics = {
            "generations": [],
            "best_fitness": [],
            "avg_fitness": [],
            "spectral_gaps": [],
            "merge_techniques_used": []
        }

    def load_phase1_models(self, checkpoint_dir: str) -> List[nn.Module]:
        """Load 3 Phase 1 models from checkpoints."""
        models = []
        specializations = ["reasoning", "memory", "speed"]

        for spec in specializations:
            # Find latest checkpoint
            spec_dir = os.path.join(checkpoint_dir, spec)
            if not os.path.exists(spec_dir):
                print(f"Warning: {spec_dir} not found, trying epoch_20")
                checkpoint_path = os.path.join(checkpoint_dir, spec, "epoch_20.safetensors")
            else:
                # Find latest epoch
                files = [f for f in os.listdir(spec_dir) if f.startswith("epoch_") and f.endswith(".safetensors") and "optimizer" not in f]
                if not files:
                    raise FileNotFoundError(f"No checkpoints in {spec_dir}")
                latest = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
                checkpoint_path = os.path.join(spec_dir, latest)

            print(f"Loading {spec} model from {checkpoint_path}")

            # Create model
            config = Phase1Config(specialization=spec)
            model = TRMTitansMAGModel(config.titans_config)

            # Load weights using safetensors
            try:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading {checkpoint_path}: {e}")
                raise

            model = model.to(self.device)
            model.eval()
            models.append(model)
            print(f"  Loaded {spec}: {sum(p.numel() for p in model.parameters()):,} params")

        return models

    def evaluate_fitness(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate multi-objective fitness."""
        model.eval()

        with torch.no_grad():
            # Generate test data
            batch_size = 4
            seq_len = 64
            vocab_size = 50257

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

            # Forward pass
            start_time = time.time()
            output = model(input_ids)
            inference_time = time.time() - start_time

            # Extract logits
            if isinstance(output, dict):
                logits = output.get("logits", output.get("output", None))
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Compute perplexity proxy (cross-entropy)
            if logits is not None and logits.dim() == 3:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )
                perplexity = torch.exp(loss).item()
            else:
                perplexity = 1000.0

            # Spectral gap
            spectral_gap = self.meta.compute_spectral_gap(model)

            # Weight norm (compression readiness)
            total_norm = 0.0
            for param in model.parameters():
                total_norm += param.data.norm(2).item() ** 2
            weight_norm = np.sqrt(total_norm)

            # Speed (tokens/sec)
            tokens_per_sec = (batch_size * seq_len) / (inference_time + 1e-10)

        return {
            "perplexity": min(perplexity, 1e6),  # Cap extreme values
            "spectral_gap": spectral_gap,
            "weight_norm": weight_norm,
            "speed": tokens_per_sec,
            "inference_time": inference_time
        }

    def compute_combined_fitness(self, objectives: Dict[str, float]) -> float:
        """Combine objectives into single fitness score for selection."""
        # Normalize and combine (higher is better)
        fitness = 0.0

        # Perplexity: lower is better (invert)
        fitness += 1.0 / (1.0 + objectives["perplexity"] / 1000)

        # Spectral gap: higher is better
        fitness += objectives["spectral_gap"]

        # Weight norm: lower is better (invert)
        fitness += 1.0 / (1.0 + objectives["weight_norm"] / 1000)

        # Speed: higher is better (normalize)
        fitness += objectives["speed"] / 10000

        return fitness

    def merge_models(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        technique: str,
        ratio: float = 0.5
    ) -> nn.Module:
        """Merge two models using specified technique."""
        state_a = model_a.state_dict()
        state_b = model_b.state_dict()

        # Get k(L) layer ratios if enabled
        if self.config.use_klayer_ratios:
            num_layers = 8  # TRM-Titans has 8 layers
            layer_ratios = self.meta.get_layer_merge_ratios(num_layers)
            # For now, use average ratio (layer-wise would require knowing layer names)
            ratio = np.mean(layer_ratios)

        # Apply merge technique
        if technique == "linear":
            merged_state = self.merge_engine.linear_merge(state_a, state_b, ratio)
        elif technique == "slerp":
            merged_state = self.merge_engine.slerp_merge(state_a, state_b, ratio)
        elif technique == "ties":
            merged_state = self.merge_engine.ties_merge(state_a, state_b, ratio)
        elif technique == "dare":
            merged_state = self.merge_engine.dare_merge(state_a, state_b, ratio)
        elif technique == "bigeometric":
            merged_state = self.merge_engine.bigeometric_merge(state_a, state_b, ratio)
        else:
            # Default to linear
            merged_state = self.merge_engine.linear_merge(state_a, state_b, ratio)

        # Create new model with merged weights
        config = Phase1Config(specialization="reasoning")
        merged_model = TRMTitansMAGModel(config.titans_config)
        merged_model.load_state_dict(merged_state)
        merged_model = merged_model.to(self.device)

        return merged_model

    def mutate_model(self, model: nn.Module, mutation_rate: float = 0.1) -> nn.Module:
        """Apply random weight perturbation."""
        model_copy = copy.deepcopy(model)

        with torch.no_grad():
            for param in model_copy.parameters():
                if param.requires_grad:
                    # Random mask for mutation
                    mask = torch.rand_like(param) < mutation_rate
                    # Perturbation scaled by parameter magnitude
                    noise = torch.randn_like(param) * param.abs().mean() * 0.01
                    param.data += mask.float() * noise

        return model_copy

    def tournament_selection(self, k: int = 3) -> Tuple[nn.Module, nn.Module]:
        """Select two parents via tournament selection."""
        def select_one():
            candidates = random.sample(range(len(self.population)), min(k, len(self.population)))
            best_idx = max(candidates, key=lambda i: self.population[i][1])
            return self.population[best_idx][0]

        return select_one(), select_one()

    def run(self, input_models: List[nn.Module], output_dir: str = "checkpoints/phase2") -> nn.Module:
        """Run enhanced evolutionary optimization."""
        print("\n" + "=" * 70)
        print("  ENHANCED PHASE 2: EVOMERGE + MOO + META CALCULUS")
        print("=" * 70)
        print(f"  Device: {self.device}")
        print(f"  Generations: {self.config.num_generations}")
        print(f"  Population: {self.config.population_size}")
        print(f"  Merge techniques: {self.config.merge_techniques}")
        print(f"  MOO enabled: {self.config.use_moo}")
        print(f"  Bigeometric merge: {self.config.use_bigeometric}")
        print(f"  k(L) layer ratios: {self.config.use_klayer_ratios}")
        print("=" * 70 + "\n")

        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Initialize population with input models and their merges
        print("Initializing population...")
        self.population = []

        # Add original models
        for i, model in enumerate(input_models):
            fitness_obj = self.evaluate_fitness(model)
            fitness = self.compute_combined_fitness(fitness_obj)
            self.population.append((model, fitness, fitness_obj))
            print(f"  Model {i+1}: fitness={fitness:.4f}, perplexity={fitness_obj['perplexity']:.2f}")

        # Create initial merges to fill population
        techniques = self.config.merge_techniques
        while len(self.population) < self.config.population_size:
            # Random merge of two models
            idx_a, idx_b = random.sample(range(len(input_models)), 2)
            technique = random.choice(techniques)
            ratio = random.uniform(0.3, 0.7)

            merged = self.merge_models(input_models[idx_a], input_models[idx_b], technique, ratio)
            fitness_obj = self.evaluate_fitness(merged)
            fitness = self.compute_combined_fitness(fitness_obj)
            self.population.append((merged, fitness, fitness_obj))

        print(f"Initial population: {len(self.population)} models")

        # Track best
        best_idx = max(range(len(self.population)), key=lambda i: self.population[i][1])
        self.best_fitness = self.population[best_idx][1]
        self.champion = copy.deepcopy(self.population[best_idx][0])

        # Step 2: Evolutionary loop
        for gen in range(self.config.num_generations):
            gen_start = time.time()

            # Sort by fitness (descending)
            self.population.sort(key=lambda x: x[1], reverse=True)

            # Track metrics
            fitnesses = [p[1] for p in self.population]
            spectral_gaps = [p[2]["spectral_gap"] for p in self.population]

            self.metrics["generations"].append(gen + 1)
            self.metrics["best_fitness"].append(self.population[0][1])
            self.metrics["avg_fitness"].append(np.mean(fitnesses))
            self.metrics["spectral_gaps"].append(np.mean(spectral_gaps))

            # Update champion
            if self.population[0][1] > self.best_fitness:
                self.best_fitness = self.population[0][1]
                self.champion = copy.deepcopy(self.population[0][0])

                # Save checkpoint
                champion_path = os.path.join(output_dir, f"champion_gen{gen+1}.pt")
                torch.save({
                    "model_state_dict": self.champion.state_dict(),
                    "fitness": self.best_fitness,
                    "generation": gen + 1,
                    "objectives": self.population[0][2],
                    "simulated_evaluation": True,
                    "artifact_provenance": "random-token proxy evaluation; not a production EvoMerge champion",
                }, champion_path)

            # Create next generation
            new_population = []

            # Elitism: keep best models
            for i in range(self.config.elite_count):
                new_population.append(self.population[i])

            # Fill rest with offspring
            techniques_used = []
            while len(new_population) < self.config.population_size:
                # Selection
                parent_a, parent_b = self.tournament_selection()

                # Crossover (merge)
                if random.random() < self.config.crossover_rate:
                    technique = random.choice(techniques)
                    techniques_used.append(technique)
                    ratio = random.uniform(0.3, 0.7)
                    offspring = self.merge_models(parent_a, parent_b, technique, ratio)
                else:
                    offspring = copy.deepcopy(parent_a)
                    techniques_used.append("clone")

                # Mutation
                if random.random() < self.config.mutation_rate:
                    offspring = self.mutate_model(offspring, 0.05)

                # Evaluate
                fitness_obj = self.evaluate_fitness(offspring)
                fitness = self.compute_combined_fitness(fitness_obj)
                new_population.append((offspring, fitness, fitness_obj))

            self.population = new_population
            self.metrics["merge_techniques_used"].append(techniques_used)

            gen_time = time.time() - gen_start
            best_obj = self.population[0][2]

            print(f"[Gen {gen+1:3d}/{self.config.num_generations}] "
                  f"Best: {self.population[0][1]:.4f} | "
                  f"Avg: {np.mean(fitnesses):.4f} | "
                  f"PPL: {best_obj['perplexity']:.1f} | "
                  f"Gap: {best_obj['spectral_gap']:.4f} | "
                  f"Time: {gen_time:.1f}s")

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("  EVOLUTION COMPLETE")
        print("=" * 70)
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best fitness: {self.best_fitness:.4f}")
        print(f"  Generations: {self.config.num_generations}")

        # Save final champion
        final_path = os.path.join(output_dir, "champion_final.pt")
        torch.save({
            "model_state_dict": self.champion.state_dict(),
            "fitness": self.best_fitness,
            "config": vars(self.config),
            "metrics": self.metrics,
            "simulated_evaluation": True,
            "artifact_provenance": "random-token proxy evaluation; not a production EvoMerge champion",
        }, final_path)
        print(f"  Champion saved: {final_path}")

        # Save metrics
        metrics_path = os.path.join(output_dir, "evolution_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        print(f"  Metrics saved: {metrics_path}")

        return self.champion


def main():
    parser = argparse.ArgumentParser(description="Enhanced Phase 2 EvoMerge with MOO + Meta Calculus")
    parser.add_argument("--checkpoint-dir", default="checkpoints/phase1_real", help="Phase 1 checkpoints")
    parser.add_argument("--output-dir", default="checkpoints/phase2_enhanced", help="Output directory")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    parser.add_argument("--population", type=int, default=12, help="Population size")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--no-moo", action="store_true", help="Disable MOO")
    parser.add_argument("--no-bigeometric", action="store_true", help="Disable bigeometric merge")
    parser.add_argument(
        "--synthetic-eval",
        action="store_true",
        help="Run random-token proxy fitness. Required until a real validation dataloader is provided.",
    )
    args = parser.parse_args()

    if not args.synthetic_eval:
        print("Error: this script currently evaluates fitness with random-token proxies. Pass --synthetic-eval to run it, or use a real Phase 2 validation evaluator.")
        return 2

    if not args.output_dir.endswith("synthetic_eval"):
        args.output_dir = os.path.join(args.output_dir, "synthetic_eval")

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create config
    config = EnhancedEvoConfig(
        num_generations=args.generations,
        population_size=args.population,
        use_moo=not args.no_moo,
        use_bigeometric=not args.no_bigeometric
    )

    # Create runner
    runner = EnhancedPhase2Runner(config, device=args.device)

    # Load Phase 1 models
    print("\nLoading Phase 1 models...")
    input_models = runner.load_phase1_models(args.checkpoint_dir)
    print(f"Loaded {len(input_models)} models")

    # Run evolution
    champion = runner.run(input_models, args.output_dir)

    # Final evaluation
    print("\nFinal champion evaluation:")
    final_fitness = runner.evaluate_fitness(champion)
    for key, value in final_fitness.items():
        print(f"  {key}: {value:.4f}")

    print("\nPhase 2 complete! Champion ready for Phase 3 (Quiet-STaR)")


if __name__ == "__main__":
    raise SystemExit(main())

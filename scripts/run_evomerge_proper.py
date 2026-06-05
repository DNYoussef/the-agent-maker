#!/usr/bin/env python3
"""
Phase 2 EvoMerge - PROPER Implementation

Uses the full 6-technique 3-stage pipeline with 8 binary combinations:
- Stage 1 (Interpolation): Linear OR SLERP
- Stage 2 (Task Arithmetic): DARE OR TIES
- Stage 3 (Selection): FrankenMerge OR DFS

Binary encoding (3 bits = 8 combos):
  Bit 0 = Interpolation  (0=Linear, 1=SLERP)
  Bit 1 = Task Arithmetic (0=DARE, 1=TIES)
  Bit 2 = Selection      (0=FrankenMerge, 1=DFS)

Evolution Strategy:
- Population: 8 models (all binary combinations)
- Top 2 winners -> mutate 3x each = 6 children
- Bottom 6 losers -> 2 groups of 3 -> apply random combo = 2 children
- Fitness: 40% Perplexity + 30% Accuracy + 20% Speed + 10% Memory
"""

import copy
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from cross_phase.utils.checkpoint_utils import load_checkpoint, save_checkpoint


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvoMergeConfig:
    """Configuration for Phase 2 EvoMerge."""
    generations: int = 50
    population_size: int = 8  # Fixed: 8 binary combinations
    elite_count: int = 2

    # Mutation parameters
    mutation_sigma: float = 0.01  # Noise std
    mutation_rate: float = 0.01   # Fraction of weights to mutate

    # Fitness weights (from docs)
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'perplexity': 0.4,   # 40% - Language modeling quality
        'accuracy': 0.3,     # 30% - Task performance
        'speed': 0.2,        # 20% - Inference efficiency
        'memory': 0.1        # 10% - Resource usage
    })

    # Convergence
    convergence_threshold: float = 0.001
    convergence_patience: int = 5
    early_stopping: bool = True

    # Diversity
    min_diversity: float = 0.3
    diversity_reseed_threshold: float = 0.2

    # Benchmark settings
    benchmark_samples: int = 50  # Samples per benchmark

    # DARE parameters
    dare_drop_rate: float = 0.9  # Drop 90%, keep 10%

    # TIES parameters
    ties_trim_percent: float = 0.2  # Keep top 20%

    device: str = "cuda"


# ============================================================================
# MERGE TECHNIQUES (6 Total, 3 Stages)
# ============================================================================

class LinearMerge:
    """Stage 1: Simple weighted average of 3 models."""

    def merge(self, models: List[nn.Module], weights: Optional[List[float]] = None) -> nn.Module:
        """Merge 3 models via weighted average."""
        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        result = copy.deepcopy(models[0])

        with torch.no_grad():
            for name, param in result.named_parameters():
                weighted_sum = torch.zeros_like(param)
                for model, weight in zip(models, weights):
                    model_param = dict(model.named_parameters())[name]
                    weighted_sum += weight * model_param
                param.copy_(weighted_sum)

        return result


class SLERPMerge:
    """Stage 1: Spherical linear interpolation preserving magnitude."""

    def merge(self, models: List[nn.Module]) -> nn.Module:
        """Merge 3 models via pairwise SLERP."""
        # SLERP(m1, m2) -> intermediate, then SLERP(intermediate, m3) -> final
        intermediate = self._slerp_pair(models[0], models[1], t=0.5)
        final = self._slerp_pair(intermediate, models[2], t=0.33)
        return final

    def _slerp_pair(self, model_a: nn.Module, model_b: nn.Module, t: float) -> nn.Module:
        """SLERP between two models."""
        result = copy.deepcopy(model_a)

        with torch.no_grad():
            for name, param_a in model_a.named_parameters():
                param_b = dict(model_b.named_parameters())[name]
                result_param = dict(result.named_parameters())[name]

                flat_a = param_a.flatten().float()
                flat_b = param_b.flatten().float()

                norm_a = flat_a.norm()
                norm_b = flat_b.norm()

                if norm_a < 1e-8 or norm_b < 1e-8:
                    # Fallback to linear
                    result_param.copy_((1 - t) * param_a + t * param_b)
                    continue

                unit_a = flat_a / norm_a
                unit_b = flat_b / norm_b

                dot = torch.clamp(torch.dot(unit_a, unit_b), -1.0, 1.0)
                omega = torch.acos(dot)

                if omega.abs() < 1e-8:
                    # Nearly identical, use linear
                    result_param.copy_((1 - t) * param_a + t * param_b)
                    continue

                sin_omega = torch.sin(omega)
                interp = (torch.sin((1 - t) * omega) / sin_omega) * flat_a + \
                         (torch.sin(t * omega) / sin_omega) * flat_b

                result_param.copy_(interp.view(param_a.shape).to(param_a.dtype))

        return result


class DAREMerge:
    """Stage 2: Drop And REscale - sparse updates."""

    def __init__(self, drop_rate: float = 0.9):
        self.drop_rate = drop_rate
        self.keep_rate = 1.0 - drop_rate
        self.rescale = 1.0 / self.keep_rate  # 10x rescale for 90% drop

    def merge(self, model_merged: nn.Module, model_base: nn.Module) -> nn.Module:
        """Apply DARE: drop 90% of delta, rescale remaining 10%."""
        result = copy.deepcopy(model_base)

        with torch.no_grad():
            for name, base_param in model_base.named_parameters():
                merged_param = dict(model_merged.named_parameters())[name]
                result_param = dict(result.named_parameters())[name]

                # Compute delta
                delta = merged_param - base_param

                # Random mask (keep 10%)
                mask = torch.bernoulli(torch.full_like(delta, self.keep_rate)).bool()

                # Apply mask and rescale
                sparse_delta = torch.where(mask, delta * self.rescale, torch.zeros_like(delta))

                # Result = base + sparse_delta
                result_param.copy_(base_param + sparse_delta)

        return result


class TIESMerge:
    """Stage 2: TrIm, Elect Sign, Merge - conflict resolution."""

    def __init__(self, trim_percent: float = 0.2):
        self.trim_percent = trim_percent

    def merge(self, model_merged: nn.Module, models_ref: List[nn.Module]) -> nn.Module:
        """Apply TIES: trim, elect sign, merge matching."""
        result = copy.deepcopy(model_merged)

        with torch.no_grad():
            for name, merged_param in model_merged.named_parameters():
                ref_params = [dict(m.named_parameters())[name] for m in models_ref]
                result_param = dict(result.named_parameters())[name]

                # Compute deltas from merged
                deltas = [ref_param - merged_param for ref_param in ref_params]

                # Step 1: TRIM - Keep top 20% by magnitude
                trimmed_deltas = self._trim_deltas(deltas)

                # Step 2: ELECT - Vote on sign
                elected_sign = self._elect_sign(trimmed_deltas)

                # Step 3: MERGE - Average with matching sign
                merged_delta = self._merge_matching(trimmed_deltas, elected_sign)

                result_param.copy_(merged_param + merged_delta)

        return result

    def _trim_deltas(self, deltas: List[torch.Tensor]) -> List[torch.Tensor]:
        """Keep only top k% by magnitude."""
        trimmed = []
        for delta in deltas:
            magnitude = torch.abs(delta)
            flat_mag = magnitude.flatten()
            k = max(1, int(len(flat_mag) * self.trim_percent))
            threshold = torch.topk(flat_mag, k).values[-1]
            mask = magnitude >= threshold
            trimmed.append(torch.where(mask, delta, torch.zeros_like(delta)))
        return trimmed

    def _elect_sign(self, deltas: List[torch.Tensor]) -> torch.Tensor:
        """Vote on sign for each parameter."""
        stacked = torch.stack(deltas, dim=0)
        signs = torch.sign(stacked)
        elected = torch.sign(torch.mean(signs, dim=0))
        return elected

    def _merge_matching(self, deltas: List[torch.Tensor], elected_sign: torch.Tensor) -> torch.Tensor:
        """Average deltas matching elected sign."""
        matching = []
        for delta in deltas:
            delta_sign = torch.sign(delta)
            match_mask = (delta_sign == elected_sign) & (elected_sign != 0)
            matching.append(torch.where(match_mask, delta, torch.zeros_like(delta)))

        if matching:
            stacked = torch.stack(matching, dim=0)
            sum_deltas = torch.sum(stacked, dim=0)
            count = torch.sum(stacked != 0, dim=0).clamp(min=1)
            return sum_deltas / count
        return torch.zeros_like(deltas[0])


class FrankenMerge:
    """Stage 3: Layer-wise selection from best performers."""

    def merge(self, model_merged: nn.Module, models_ref: List[nn.Module]) -> nn.Module:
        """Select best layers from candidates."""
        result = copy.deepcopy(model_merged)
        candidates = [model_merged] + models_ref

        with torch.no_grad():
            # Group parameters by layer
            layer_params = {}
            for name, param in model_merged.named_parameters():
                # Extract layer identifier (e.g., "layers.0", "layers.1")
                parts = name.split('.')
                layer_key = '.'.join(parts[:2]) if len(parts) > 1 else parts[0]
                if layer_key not in layer_params:
                    layer_params[layer_key] = []
                layer_params[layer_key].append(name)

            # For each layer, pick from candidate with best gradient magnitude proxy
            for layer_key, param_names in layer_params.items():
                best_candidate_idx = 0
                best_score = -float('inf')

                for idx, candidate in enumerate(candidates):
                    # Score: variance (diversity) + magnitude (importance)
                    score = 0.0
                    for pname in param_names:
                        cparam = dict(candidate.named_parameters())[pname]
                        score += cparam.var().item() + cparam.abs().mean().item()

                    if score > best_score:
                        best_score = score
                        best_candidate_idx = idx

                # Copy best candidate's layer
                for pname in param_names:
                    src_param = dict(candidates[best_candidate_idx].named_parameters())[pname]
                    dst_param = dict(result.named_parameters())[pname]
                    dst_param.copy_(src_param)

        return result


class DFSMerge:
    """Stage 3: Deep Feature Selection - importance-weighted merge."""

    def merge(self, model_merged: nn.Module, models_ref: List[nn.Module]) -> nn.Module:
        """Merge weighted by inverse variance (stable = important)."""
        result = copy.deepcopy(model_merged)
        all_models = [model_merged] + models_ref

        with torch.no_grad():
            for name, merged_param in model_merged.named_parameters():
                all_params = [dict(m.named_parameters())[name] for m in all_models]
                result_param = dict(result.named_parameters())[name]

                # Stack parameters
                stacked = torch.stack(all_params, dim=0)

                # Importance = inverse variance (stable params are important)
                variance = torch.var(stacked, dim=0)
                importance = 1.0 / (variance + 1e-8)

                # Weighted average by importance
                weighted_sum = torch.zeros_like(merged_param)
                total_importance = torch.zeros_like(merged_param)

                for param in all_params:
                    weighted_sum += importance * param
                    total_importance += importance

                result_param.copy_(weighted_sum / total_importance)

        return result


# ============================================================================
# BINARY COMBINATION PIPELINE
# ============================================================================

class MergeTechniques:
    """
    Unified API for 3-stage merge pipeline with 8 binary combinations.

    Binary encoding:
      Bit 0 = Interpolation  (0=Linear, 1=SLERP)
      Bit 1 = Task Arithmetic (0=DARE, 1=TIES)
      Bit 2 = Selection      (0=FrankenMerge, 1=DFS)
    """

    def __init__(self, config: EvoMergeConfig):
        self.config = config
        self.linear = LinearMerge()
        self.slerp = SLERPMerge()
        self.dare = DAREMerge(drop_rate=config.dare_drop_rate)
        self.ties = TIESMerge(trim_percent=config.ties_trim_percent)
        self.frankenmerge = FrankenMerge()
        self.dfs = DFSMerge()

    def apply_combo(self, models: List[nn.Module], combo_id: int) -> nn.Module:
        """
        Apply 3-stage sequential merge pipeline.

        Args:
            models: List of 3 base models
            combo_id: Binary combination 0-7

        Returns:
            Merged model
        """
        if combo_id < 0 or combo_id > 7:
            raise ValueError(f"combo_id must be 0-7, got {combo_id}")
        if len(models) != 3:
            raise ValueError(f"Expected 3 models, got {len(models)}")

        # Decode binary
        bit0 = (combo_id >> 0) & 1  # Interpolation
        bit1 = (combo_id >> 1) & 1  # Task arithmetic
        bit2 = (combo_id >> 2) & 1  # Selection

        # Stage 1: Interpolation (3 models -> 1)
        if bit0 == 1:
            stage1 = self.slerp.merge(models)
        else:
            stage1 = self.linear.merge(models)

        # Stage 2: Task Arithmetic (refine merged model)
        if bit1 == 1:
            stage2 = self.ties.merge(stage1, models)
        else:
            stage2 = self.dare.merge(stage1, models[0])  # Use first as base

        # Stage 3: Selection (final refinement)
        if bit2 == 1:
            stage3 = self.dfs.merge(stage2, models)
        else:
            stage3 = self.frankenmerge.merge(stage2, models)

        return stage3

    def decode_combo(self, combo_id: int) -> str:
        """Human-readable combo name."""
        bit0 = (combo_id >> 0) & 1
        bit1 = (combo_id >> 1) & 1
        bit2 = (combo_id >> 2) & 1

        interp = "SLERP" if bit0 else "Linear"
        task = "TIES" if bit1 else "DARE"
        select = "DFS" if bit2 else "Franken"

        return f"{interp}+{task}+{select}"


# ============================================================================
# FITNESS EVALUATION
# ============================================================================

def evaluate_perplexity(model: nn.Module, tokenizer, device: str, num_samples: int = 20) -> float:
    """Evaluate perplexity on wikitext validation."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t for t in dataset["text"] if len(t.strip()) > 100][:num_samples]
    except Exception as e:
        print(f"    Perplexity dataset error: {e}")
        return 100.0  # Default high perplexity

    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(device)

            try:
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs["loss"]
                num_tokens = input_ids.numel()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
            except Exception:
                pass

    if total_tokens == 0:
        return 100.0

    avg_loss = total_loss / total_tokens
    perplexity = min(1000.0, np.exp(avg_loss))
    return perplexity


def evaluate_accuracy(model: nn.Module, tokenizer, device: str, num_samples: int = 30) -> float:
    """Evaluate accuracy on simple task (next token prediction)."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in dataset["text"] if len(t.strip()) > 50][:num_samples]
    except Exception:
        return 0.5  # Default accuracy

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)

            if input_ids.size(1) < 10:
                continue

            try:
                # Predict last token from context
                context = input_ids[:, :-1]
                target = input_ids[:, -1]

                outputs = model(input_ids=context)
                logits = outputs["logits"]
                pred = logits[:, -1, :].argmax(dim=-1)

                if pred.item() == target.item():
                    correct += 1
                total += 1
            except Exception:
                pass

    return correct / total if total > 0 else 0.5


def evaluate_speed(model: nn.Module, device: str) -> float:
    """Measure inference speed (tokens/sec)."""
    model.eval()

    # Warmup
    dummy = torch.randint(0, 1000, (1, 64)).to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids=dummy)

    # Measure
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    for _ in range(20):
        with torch.no_grad():
            _ = model(input_ids=dummy)

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - start

    tokens_per_sec = (20 * 64) / elapsed
    return tokens_per_sec


def evaluate_memory(model: nn.Module) -> float:
    """Estimate memory usage in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    # 4 bytes per float32 param, x2 for gradients/optimizer
    memory_mb = (total_params * 4 * 2) / (1024 * 1024)
    return memory_mb


def compute_composite_fitness(
    perplexity: float,
    accuracy: float,
    speed: float,
    memory: float,
    weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compute composite fitness score.

    Formula:
      fitness = w_ppl * (1/ppl) + w_acc * acc + w_spd * (spd/1200) + w_mem * (500/mem)
    """
    # Expected baselines for normalization
    expected_speed = 1200.0  # tokens/sec
    expected_memory = 500.0  # MB

    # Component scores
    ppl_score = 1.0 / max(1.0, perplexity)
    acc_score = accuracy
    spd_score = speed / expected_speed
    mem_score = expected_memory / max(1.0, memory)

    # Composite
    composite = (
        weights['perplexity'] * ppl_score +
        weights['accuracy'] * acc_score +
        weights['speed'] * spd_score +
        weights['memory'] * mem_score
    )

    return {
        'composite': composite,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'speed': speed,
        'memory': memory,
        'ppl_score': ppl_score,
        'acc_score': acc_score,
        'spd_score': spd_score,
        'mem_score': mem_score
    }


def evaluate_model_fitness(
    model: nn.Module,
    tokenizer,
    device: str,
    config: EvoMergeConfig,
    model_idx: int,
    gen: int
) -> Tuple[float, Dict]:
    """Full fitness evaluation for a model."""
    print(f"    Evaluating model {model_idx}/8 (gen {gen})...", end=" ", flush=True)
    start = time.time()

    perplexity = evaluate_perplexity(model, tokenizer, device, config.benchmark_samples // 2)
    accuracy = evaluate_accuracy(model, tokenizer, device, config.benchmark_samples)
    speed = evaluate_speed(model, device)
    memory = evaluate_memory(model)

    metrics = compute_composite_fitness(perplexity, accuracy, speed, memory, config.fitness_weights)

    elapsed = time.time() - start
    print(f"PPL={perplexity:.1f}, Acc={accuracy:.1%}, Spd={speed:.0f}, fit={metrics['composite']:.4f} ({elapsed:.1f}s)")

    return metrics['composite'], metrics


# ============================================================================
# EVOLUTION OPERATIONS
# ============================================================================

def mutate_model(model: nn.Module, sigma: float, rate: float, device: str) -> nn.Module:
    """
    Apply mutation: Gaussian noise to small fraction of weights.

    Args:
        sigma: Noise standard deviation (0.01 = 1% of typical weight)
        rate: Fraction of weights to mutate (0.01 = 1%)
    """
    mutated = copy.deepcopy(model)

    with torch.no_grad():
        for param in mutated.parameters():
            # Mask: which weights to mutate
            mask = torch.rand_like(param) < rate
            # Noise
            noise = torch.randn_like(param) * sigma
            # Apply
            param.add_(noise * mask.float())

    return mutated


def compute_diversity(population: List[nn.Module]) -> float:
    """
    Compute population diversity via pairwise parameter distance.

    Returns:
        Normalized diversity score (0 = identical, 1 = maximally diverse)
    """
    if len(population) < 2:
        return 1.0

    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            # Flatten both models
            flat_i = torch.cat([p.flatten() for p in population[i].parameters()])
            flat_j = torch.cat([p.flatten() for p in population[j].parameters()])

            # L2 distance
            dist = torch.norm(flat_i - flat_j).item()
            distances.append(dist)

    avg_dist = np.mean(distances) if distances else 0.0

    # Normalize (expected distance for 25M params with std 0.02 is ~4000)
    normalized = min(1.0, avg_dist / 4000.0)

    return normalized


# ============================================================================
# MAIN EVOLUTION LOOP
# ============================================================================

def run_evolution(
    base_models: List[nn.Module],
    tokenizer,
    config: EvoMergeConfig,
    output_dir: Path
) -> Tuple[nn.Module, Dict]:
    """
    Run complete Phase 2 evolution.

    Args:
        base_models: 3 Phase 1 models (reasoning, memory, speed)
        tokenizer: Tokenizer for benchmarks
        config: Evolution config
        output_dir: Where to save checkpoints

    Returns:
        (champion_model, metrics_dict)
    """
    device = config.device
    merger = MergeTechniques(config)

    print("\n" + "=" * 70)
    print("PHASE 2: EVOMERGE - PROPER 3-STAGE PIPELINE")
    print("=" * 70)
    print(f"Generations: {config.generations}")
    print(f"Population: {config.population_size} (8 binary combinations)")
    print(f"Elite count: {config.elite_count}")
    print(f"Fitness weights: {config.fitness_weights}")
    print("=" * 70)

    start_time = time.time()

    # =========================================================================
    # STEP 1: Initialize population with all 8 binary combinations
    # =========================================================================
    print("\n[Step 1] Creating initial population (8 binary combos)...")
    population = []
    for combo_id in range(8):
        combo_name = merger.decode_combo(combo_id)
        print(f"  Combo {combo_id} ({combo_name:20s})...", end=" ")
        try:
            merged = merger.apply_combo(base_models, combo_id)
            merged._combo_id = combo_id
            population.append(merged)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            # Fallback: copy first base model
            fallback = copy.deepcopy(base_models[0])
            fallback._combo_id = combo_id
            population.append(fallback)

    print(f"  Population initialized: {len(population)} models")

    # =========================================================================
    # STEP 2: Initial fitness evaluation
    # =========================================================================
    print("\n[Step 2] Evaluating initial population...")
    fitness_scores = []
    fitness_metrics = []
    for i, model in enumerate(population):
        fitness, metrics = evaluate_model_fitness(model, tokenizer, device, config, i + 1, 0)
        model._fitness = fitness
        model._metrics = metrics
        fitness_scores.append(fitness)
        fitness_metrics.append(metrics)

    # Sort by fitness
    sorted_indices = np.argsort(fitness_scores)[::-1]
    population = [population[i] for i in sorted_indices]
    fitness_scores = [fitness_scores[i] for i in sorted_indices]

    initial_best_fitness = fitness_scores[0]

    # Track champion
    champion = copy.deepcopy(population[0])
    champion_fitness = initial_best_fitness
    champion_metrics = population[0]._metrics
    fitness_history = [champion_fitness]

    print(f"\n  Initial champion: fitness={champion_fitness:.4f}")
    print(f"  Best combo: {merger.decode_combo(population[0]._combo_id)}")

    # Save initial checkpoint
    save_progress(output_dir, 0, champion, champion_fitness, champion_metrics, fitness_history, start_time)

    # =========================================================================
    # STEP 3: Evolution loop
    # =========================================================================
    convergence_reason = "max_generations"

    for gen in range(1, config.generations + 1):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen}/{config.generations}")
        print(f"{'='*60}")

        # ----- Elite Preservation: Top 2 -> 6 children via mutation -----
        print(f"\n  [Elite] Mutating top 2 -> 6 children...")
        elite1, elite2 = population[0], population[1]
        elite_children = []

        for elite in [elite1, elite2]:
            for mut_idx in range(3):  # 3 mutations per elite
                sigma = config.mutation_sigma * (1 + 0.5 * mut_idx)  # Increasing noise
                child = mutate_model(elite, sigma, config.mutation_rate, device)
                child._combo_id = elite._combo_id
                child._parent = "elite"
                elite_children.append(child)

        # ----- Loser Merging: Bottom 6 -> 2 children via combo merge -----
        print(f"  [Loser] Merging bottom 6 -> 2 children...")
        losers = population[-6:]

        # Split into 2 groups of 3
        group1 = losers[0:3]
        group2 = losers[3:6]

        # Random combo for each group
        combo1 = random.randint(0, 7)
        combo2 = random.randint(0, 7)

        print(f"    Group 1: combo {combo1} ({merger.decode_combo(combo1)})")
        print(f"    Group 2: combo {combo2} ({merger.decode_combo(combo2)})")

        loser_child1 = merger.apply_combo(group1, combo1)
        loser_child1._combo_id = combo1
        loser_child1._parent = "loser"

        loser_child2 = merger.apply_combo(group2, combo2)
        loser_child2._combo_id = combo2
        loser_child2._parent = "loser"

        loser_children = [loser_child1, loser_child2]

        # ----- New population: 6 elite + 2 loser = 8 -----
        population = elite_children + loser_children

        # ----- Evaluate new population -----
        print(f"\n  [Fitness] Evaluating generation {gen}...")
        fitness_scores = []
        for i, model in enumerate(population):
            fitness, metrics = evaluate_model_fitness(model, tokenizer, device, config, i + 1, gen)
            model._fitness = fitness
            model._metrics = metrics
            fitness_scores.append(fitness)

        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        gen_best_fitness = fitness_scores[0]

        # Update champion if improved
        if gen_best_fitness > champion_fitness:
            champion = copy.deepcopy(population[0])
            champion_fitness = gen_best_fitness
            champion_metrics = population[0]._metrics
            print(f"\n  *** NEW CHAMPION: fitness={champion_fitness:.4f} ***")

        fitness_history.append(champion_fitness)

        # ----- Diversity management -----
        diversity = compute_diversity(population)
        if diversity < config.diversity_reseed_threshold:
            print(f"  [Diversity] Low ({diversity:.3f}), reseeding bottom 2...")
            combo_a = random.randint(0, 7)
            combo_b = random.randint(0, 7)
            population[-2] = merger.apply_combo(base_models, combo_a)
            population[-1] = merger.apply_combo(base_models, combo_b)

        # ----- Progress summary -----
        elapsed = time.time() - start_time
        improvement = (champion_fitness / initial_best_fitness - 1) * 100 if initial_best_fitness > 0 else 0
        eta = (elapsed / gen) * (config.generations - gen) / 3600

        print(f"\n  Gen {gen} Summary:")
        print(f"    Best this gen: {gen_best_fitness:.4f}")
        print(f"    Champion: {champion_fitness:.4f} (+{improvement:.1f}%)")
        print(f"    Diversity: {diversity:.3f}")
        print(f"    Elapsed: {elapsed/3600:.2f}h, ETA: {eta:.2f}h")

        # ----- Checkpointing -----
        if gen % 5 == 0:
            save_progress(output_dir, gen, champion, champion_fitness, champion_metrics, fitness_history, start_time)

        # ----- Early stopping check -----
        if config.early_stopping and gen > config.convergence_patience:
            recent = fitness_history[-config.convergence_patience:]
            if max(recent) - min(recent) < config.convergence_threshold:
                print(f"\n  [Converged] Improvement < {config.convergence_threshold} for {config.convergence_patience} gens")
                convergence_reason = "threshold_met"
                break

    # =========================================================================
    # STEP 4: Final results
    # =========================================================================
    total_time = time.time() - start_time
    improvement_pct = (champion_fitness / initial_best_fitness - 1) * 100 if initial_best_fitness > 0 else 0

    final_metrics = {
        "initial_fitness": initial_best_fitness,
        "final_fitness": champion_fitness,
        "improvement_pct": improvement_pct,
        "generations_run": gen,
        "convergence_reason": convergence_reason,
        "total_hours": total_time / 3600,
        "champion_metrics": champion_metrics,
        "fitness_history": fitness_history
    }

    # Save final
    save_progress(output_dir, gen, champion, champion_fitness, champion_metrics, fitness_history, start_time, is_final=True)

    return champion, final_metrics


def save_progress(
    output_dir: Path,
    generation: int,
    champion: nn.Module,
    fitness: float,
    metrics: Dict,
    history: List[float],
    start_time: float,
    is_final: bool = False
):
    """Save checkpoint and progress."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    prefix = "final_champion" if is_final else f"champion_gen{generation:03d}"
    model_path = output_dir / prefix
    save_checkpoint(
        champion,
        model_path,
        metadata={
            "phase": 2,
            "type": "evomerge_proper",
            "generation": generation,
            "fitness": fitness,
            "timestamp": datetime.now().isoformat()
        }
    )

    # Save metrics
    progress = {
        "generation": generation,
        "fitness": fitness,
        "metrics": metrics,
        "history": history,
        "elapsed_hours": (time.time() - start_time) / 3600,
        "timestamp": datetime.now().isoformat()
    }

    metrics_file = output_dir / f"{prefix}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(progress, f, indent=2, default=str)

    print(f"  [Saved] {model_path}.safetensors")


# ============================================================================
# MAIN
# ============================================================================

def load_phase1_models(checkpoints_dir: Path, device: str) -> List[nn.Module]:
    """Load 3 Phase 1 models."""
    models = []
    for spec in ["reasoning", "memory", "speed"]:
        path = checkpoints_dir / f"phase1/{spec}/epoch_10.safetensors"
        print(f"  Loading {spec} model from {path}...")

        config = Phase1Config(specialization=spec)
        model = TRMTitansMAGModel(config)
        load_checkpoint(model, str(path), device=device)
        model = model.to(device)
        model.eval()
        models.append(model)

    return models


def main():
    print("=" * 70)
    print("PHASE 2: EVOMERGE - PROPER IMPLEMENTATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load Phase 1 models
    print("\nLoading Phase 1 models...")
    checkpoints_dir = Path("checkpoints")
    base_models = load_phase1_models(checkpoints_dir, device)
    print(f"Loaded {len(base_models)} models")

    # Configure evolution
    config = EvoMergeConfig(
        generations=50,
        benchmark_samples=30,  # Balance speed vs accuracy
        device=device
    )

    # Output directory
    output_dir = Path("checkpoints/phase2_proper")

    # Run evolution
    champion, metrics = run_evolution(base_models, tokenizer, config, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE!")
    print("=" * 70)
    print(f"Initial Fitness:     {metrics['initial_fitness']:.4f}")
    print(f"Final Fitness:       {metrics['final_fitness']:.4f}")
    print(f"Improvement:         +{metrics['improvement_pct']:.1f}%")
    print(f"Generations:         {metrics['generations_run']}")
    print(f"Convergence:         {metrics['convergence_reason']}")
    print(f"Total Time:          {metrics['total_hours']:.2f} hours")
    print(f"Output:              {output_dir}")
    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return champion


if __name__ == "__main__":
    main()

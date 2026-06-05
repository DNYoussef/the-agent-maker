#!/usr/bin/env python3
"""
Phase 2 EvoMerge - FULL PROPER Implementation with Real Benchmarks

This version does REAL benchmarking:
- 8 models per generation
- 50 generations (NO early stopping)
- Each model evaluated on GSM8K (50 samples) + Perplexity (100 samples)
- Expected time: 2-3 minutes per model = 8-12 hours total

3-Stage Pipeline:
- Stage 1 (Interpolation): Linear OR SLERP
- Stage 2 (Task Arithmetic): DARE OR TIES
- Stage 3 (Selection): FrankenMerge OR DFS

Evolution Strategy:
- Top 2 winners -> mutate 3x each = 6 children
- Bottom 6 losers -> 2 groups of 3 -> merge = 2 children
"""

import copy
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

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
    population_size: int = 8

    # NO early stopping - run all 50 generations
    early_stopping: bool = False

    # Mutation parameters
    mutation_sigma: float = 0.01
    mutation_rate: float = 0.01

    # Fitness weights (from docs)
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'perplexity': 0.4,
        'accuracy': 0.3,
        'speed': 0.2,
        'memory': 0.1
    })

    # REAL benchmark settings - enough samples for meaningful eval
    gsm8k_samples: int = 50      # 50 GSM8K problems per model
    perplexity_samples: int = 100  # 100 wikitext samples per model
    max_new_tokens: int = 100    # Generate 100 tokens for GSM8K

    # DARE/TIES parameters
    dare_drop_rate: float = 0.9
    ties_trim_percent: float = 0.2

    # Diversity
    diversity_reseed_threshold: float = 0.2

    device: str = "cuda"


# ============================================================================
# MERGE TECHNIQUES (6 Total)
# ============================================================================

class LinearMerge:
    """Stage 1: Weighted average of 3 models."""
    def merge(self, models: List[nn.Module]) -> nn.Module:
        weights = [1.0 / len(models)] * len(models)
        result = copy.deepcopy(models[0])
        with torch.no_grad():
            for name, param in result.named_parameters():
                weighted_sum = torch.zeros_like(param)
                for model, weight in zip(models, weights):
                    weighted_sum += weight * dict(model.named_parameters())[name]
                param.copy_(weighted_sum)
        return result


class SLERPMerge:
    """Stage 1: Spherical linear interpolation."""
    def merge(self, models: List[nn.Module]) -> nn.Module:
        intermediate = self._slerp_pair(models[0], models[1], 0.5)
        final = self._slerp_pair(intermediate, models[2], 0.33)
        return final

    def _slerp_pair(self, model_a: nn.Module, model_b: nn.Module, t: float) -> nn.Module:
        result = copy.deepcopy(model_a)
        with torch.no_grad():
            for name, param_a in model_a.named_parameters():
                param_b = dict(model_b.named_parameters())[name]
                result_param = dict(result.named_parameters())[name]

                flat_a = param_a.flatten().float()
                flat_b = param_b.flatten().float()
                norm_a, norm_b = flat_a.norm(), flat_b.norm()

                if norm_a < 1e-8 or norm_b < 1e-8:
                    result_param.copy_((1 - t) * param_a + t * param_b)
                    continue

                dot = torch.clamp(torch.dot(flat_a / norm_a, flat_b / norm_b), -1.0, 1.0)
                omega = torch.acos(dot)

                if omega.abs() < 1e-8:
                    result_param.copy_((1 - t) * param_a + t * param_b)
                    continue

                sin_omega = torch.sin(omega)
                interp = (torch.sin((1 - t) * omega) / sin_omega) * flat_a + \
                         (torch.sin(t * omega) / sin_omega) * flat_b
                result_param.copy_(interp.view(param_a.shape).to(param_a.dtype))
        return result


class DAREMerge:
    """Stage 2: Drop 90%, rescale 10%."""
    def __init__(self, drop_rate: float = 0.9):
        self.keep_rate = 1.0 - drop_rate
        self.rescale = 1.0 / self.keep_rate

    def merge(self, model_merged: nn.Module, model_base: nn.Module) -> nn.Module:
        result = copy.deepcopy(model_base)
        with torch.no_grad():
            for name, base_param in model_base.named_parameters():
                merged_param = dict(model_merged.named_parameters())[name]
                result_param = dict(result.named_parameters())[name]
                delta = merged_param - base_param
                mask = torch.bernoulli(torch.full_like(delta, self.keep_rate)).bool()
                sparse_delta = torch.where(mask, delta * self.rescale, torch.zeros_like(delta))
                result_param.copy_(base_param + sparse_delta)
        return result


class TIESMerge:
    """Stage 2: Trim, Elect Sign, Merge."""
    def __init__(self, trim_percent: float = 0.2):
        self.trim_percent = trim_percent

    def merge(self, model_merged: nn.Module, models_ref: List[nn.Module]) -> nn.Module:
        result = copy.deepcopy(model_merged)
        with torch.no_grad():
            for name, merged_param in model_merged.named_parameters():
                ref_params = [dict(m.named_parameters())[name] for m in models_ref]
                result_param = dict(result.named_parameters())[name]
                deltas = [ref - merged_param for ref in ref_params]
                trimmed = self._trim(deltas)
                elected = self._elect_sign(trimmed)
                merged_delta = self._merge_matching(trimmed, elected)
                result_param.copy_(merged_param + merged_delta)
        return result

    def _trim(self, deltas):
        trimmed = []
        for d in deltas:
            mag = torch.abs(d).flatten()
            k = max(1, int(len(mag) * self.trim_percent))
            thresh = torch.topk(mag, k).values[-1]
            mask = torch.abs(d) >= thresh
            trimmed.append(torch.where(mask, d, torch.zeros_like(d)))
        return trimmed

    def _elect_sign(self, deltas):
        stacked = torch.stack(deltas, dim=0)
        return torch.sign(torch.mean(torch.sign(stacked), dim=0))

    def _merge_matching(self, deltas, elected):
        matching = []
        for d in deltas:
            mask = (torch.sign(d) == elected) & (elected != 0)
            matching.append(torch.where(mask, d, torch.zeros_like(d)))
        stacked = torch.stack(matching, dim=0)
        return torch.sum(stacked, dim=0) / torch.sum(stacked != 0, dim=0).clamp(min=1)


class FrankenMerge:
    """Stage 3: Layer-wise selection."""
    def merge(self, model_merged: nn.Module, models_ref: List[nn.Module]) -> nn.Module:
        result = copy.deepcopy(model_merged)
        candidates = [model_merged] + models_ref

        # Group by layer
        layer_params = {}
        for name, _ in model_merged.named_parameters():
            key = '.'.join(name.split('.')[:2])
            layer_params.setdefault(key, []).append(name)

        with torch.no_grad():
            for layer_key, names in layer_params.items():
                best_idx, best_score = 0, -float('inf')
                for idx, cand in enumerate(candidates):
                    score = sum(dict(cand.named_parameters())[n].var().item() +
                               dict(cand.named_parameters())[n].abs().mean().item() for n in names)
                    if score > best_score:
                        best_score, best_idx = score, idx
                for n in names:
                    dict(result.named_parameters())[n].copy_(dict(candidates[best_idx].named_parameters())[n])
        return result


class DFSMerge:
    """Stage 3: Importance-weighted merge."""
    def merge(self, model_merged: nn.Module, models_ref: List[nn.Module]) -> nn.Module:
        result = copy.deepcopy(model_merged)
        all_models = [model_merged] + models_ref
        with torch.no_grad():
            for name, _ in model_merged.named_parameters():
                all_params = [dict(m.named_parameters())[name] for m in all_models]
                stacked = torch.stack(all_params, dim=0)
                importance = 1.0 / (torch.var(stacked, dim=0) + 1e-8)
                weighted = sum(importance * p for p in all_params)
                dict(result.named_parameters())[name].copy_(weighted / (importance * len(all_params)))
        return result


class MergeTechniques:
    """3-stage pipeline with 8 binary combinations."""
    def __init__(self, config: EvoMergeConfig):
        self.linear = LinearMerge()
        self.slerp = SLERPMerge()
        self.dare = DAREMerge(config.dare_drop_rate)
        self.ties = TIESMerge(config.ties_trim_percent)
        self.franken = FrankenMerge()
        self.dfs = DFSMerge()

    def apply_combo(self, models: List[nn.Module], combo_id: int) -> nn.Module:
        bit0, bit1, bit2 = (combo_id >> 0) & 1, (combo_id >> 1) & 1, (combo_id >> 2) & 1
        stage1 = self.slerp.merge(models) if bit0 else self.linear.merge(models)
        stage2 = self.ties.merge(stage1, models) if bit1 else self.dare.merge(stage1, models[0])
        stage3 = self.dfs.merge(stage2, models) if bit2 else self.franken.merge(stage2, models)
        return stage3

    def decode(self, combo_id: int) -> str:
        return f"{'SLERP' if (combo_id>>0)&1 else 'Linear'}+{'TIES' if (combo_id>>1)&1 else 'DARE'}+{'DFS' if (combo_id>>2)&1 else 'Franken'}"


# ============================================================================
# REAL BENCHMARKS (SLOW BUT ACCURATE)
# ============================================================================

def extract_answer(text: str):
    """Extract numeric answer from text."""
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(matches[-1]) if matches else None


def evaluate_gsm8k(model: nn.Module, tokenizer, device: str, num_samples: int, max_new_tokens: int) -> float:
    """
    REAL GSM8K evaluation with text generation.
    Takes 1-2 minutes for 50 samples.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main", split="test")
        samples = list(dataset)[:num_samples]
    except Exception as e:
        print(f"    GSM8K load error: {e}")
        return 0.0

    correct, total = 0, 0
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(samples):
            question = sample["question"]
            gold = extract_answer(sample["answer"])
            if gold is None:
                continue

            prompt = f"Question: {question}\nAnswer: Let me solve step by step.\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(device)

            try:
                # Generate tokens one by one (model doesn't have .generate())
                generated = input_ids.clone()
                for _ in range(max_new_tokens):
                    outputs = model(input_ids=generated)
                    logits = outputs["logits"]
                    next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                    generated = torch.cat([generated, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break

                text = tokenizer.decode(generated[0], skip_special_tokens=True)
                pred = extract_answer(text)

                if pred is not None and gold is not None and abs(pred - gold) < 0.01:
                    correct += 1
                total += 1

            except Exception:
                total += 1

            if (i + 1) % 10 == 0:
                print(f"      GSM8K: {i+1}/{num_samples}, acc={correct}/{total}")

    return correct / total if total > 0 else 0.0


def evaluate_perplexity(model: nn.Module, tokenizer, device: str, num_samples: int) -> float:
    """
    REAL perplexity evaluation on wikitext.
    Takes 30-60 seconds for 100 samples.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t for t in dataset["text"] if len(t.strip()) > 100][:num_samples]
    except Exception as e:
        print(f"    Perplexity load error: {e}")
        return 1000.0

    total_loss, total_tokens = 0, 0
    model.eval()

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)

            try:
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs["loss"]
                tokens = input_ids.numel()
                total_loss += loss.item() * tokens
                total_tokens += tokens
            except Exception:
                pass

    if total_tokens == 0:
        return 1000.0

    ppl = np.exp(total_loss / total_tokens)
    return min(1000.0, ppl)


def evaluate_speed(model: nn.Module, device: str) -> float:
    """Measure inference speed."""
    model.eval()
    dummy = torch.randint(0, 1000, (1, 64)).to(device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model(input_ids=dummy)

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            model(input_ids=dummy)
    torch.cuda.synchronize() if device == "cuda" else None

    return (10 * 64) / (time.time() - start)


def evaluate_memory(model: nn.Module) -> float:
    """Memory usage in MB."""
    params = sum(p.numel() for p in model.parameters())
    return (params * 4 * 2) / (1024 * 1024)


def compute_fitness(ppl, acc, speed, memory, weights) -> float:
    """Composite fitness: 40% PPL + 30% Acc + 20% Speed + 10% Memory."""
    ppl_score = 1.0 / max(1.0, ppl)
    spd_score = speed / 1200.0
    mem_score = 500.0 / max(1.0, memory)
    return weights['perplexity'] * ppl_score + weights['accuracy'] * acc + \
           weights['speed'] * spd_score + weights['memory'] * mem_score


def evaluate_model(model: nn.Module, tokenizer, device: str, config: EvoMergeConfig,
                   model_idx: int, gen: int) -> Tuple[float, Dict]:
    """
    FULL model evaluation - takes 2-3 minutes per model.
    """
    print(f"    Model {model_idx}/8 (gen {gen}):", flush=True)

    start = time.time()

    # GSM8K (1-2 min)
    print(f"      Running GSM8K ({config.gsm8k_samples} samples)...", flush=True)
    acc = evaluate_gsm8k(model, tokenizer, device, config.gsm8k_samples, config.max_new_tokens)

    # Perplexity (30-60 sec)
    print(f"      Running Perplexity ({config.perplexity_samples} samples)...", flush=True)
    ppl = evaluate_perplexity(model, tokenizer, device, config.perplexity_samples)

    # Speed (5 sec)
    speed = evaluate_speed(model, device)

    # Memory
    memory = evaluate_memory(model)

    # Composite fitness
    fitness = compute_fitness(ppl, acc, speed, memory, config.fitness_weights)

    elapsed = time.time() - start
    print(f"      DONE: PPL={ppl:.1f}, Acc={acc:.1%}, Spd={speed:.0f}, fit={fitness:.4f} ({elapsed:.1f}s)")

    return fitness, {'perplexity': ppl, 'accuracy': acc, 'speed': speed, 'memory': memory, 'fitness': fitness}


# ============================================================================
# EVOLUTION OPERATIONS
# ============================================================================

def mutate(model: nn.Module, sigma: float, rate: float) -> nn.Module:
    """Mutate model weights."""
    result = copy.deepcopy(model)
    with torch.no_grad():
        for p in result.parameters():
            mask = torch.rand_like(p) < rate
            p.add_(torch.randn_like(p) * sigma * mask.float())
    return result


def compute_diversity(population: List[nn.Module]) -> float:
    """Population diversity via pairwise distance."""
    if len(population) < 2:
        return 1.0
    dists = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            fi = torch.cat([p.flatten() for p in population[i].parameters()])
            fj = torch.cat([p.flatten() for p in population[j].parameters()])
            dists.append(torch.norm(fi - fj).item())
    return min(1.0, np.mean(dists) / 4000.0)


# ============================================================================
# MAIN EVOLUTION LOOP
# ============================================================================

def run_evolution(base_models: List[nn.Module], tokenizer, config: EvoMergeConfig, output_dir: Path):
    """Run full 50-generation evolution with real benchmarks."""
    device = config.device
    merger = MergeTechniques(config)

    print("\n" + "=" * 70)
    print("PHASE 2: EVOMERGE - FULL BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Generations: {config.generations} (NO early stopping)")
    print(f"Population: 8 models per generation")
    print(f"GSM8K samples: {config.gsm8k_samples} per model")
    print(f"Perplexity samples: {config.perplexity_samples} per model")
    print(f"Expected time: 8-12 hours")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Create initial population (8 binary combos)
    print("\n[1/3] Creating initial population...")
    population = []
    for combo_id in range(8):
        print(f"  Combo {combo_id} ({merger.decode(combo_id)})...", end=" ", flush=True)
        try:
            m = merger.apply_combo(base_models, combo_id)
            m._combo_id = combo_id
            population.append(m)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            m = copy.deepcopy(base_models[0])
            m._combo_id = combo_id
            population.append(m)

    # Step 2: Evaluate initial population
    print("\n[2/3] Evaluating initial population (this takes ~20 minutes)...")
    fitness_scores = []
    for i, model in enumerate(population):
        f, m = evaluate_model(model, tokenizer, device, config, i + 1, 0)
        model._fitness = f
        model._metrics = m
        fitness_scores.append(f)

    # Sort by fitness
    idx = np.argsort(fitness_scores)[::-1]
    population = [population[i] for i in idx]
    fitness_scores = [fitness_scores[i] for i in idx]

    initial_fitness = fitness_scores[0]
    champion = copy.deepcopy(population[0])
    champion_fitness = initial_fitness
    champion_metrics = population[0]._metrics
    history = [champion_fitness]

    print(f"\n  Initial champion: {merger.decode(population[0]._combo_id)}, fitness={champion_fitness:.4f}")
    save_checkpoint_with_metrics(output_dir, 0, champion, champion_fitness, champion_metrics, start_time)

    # Step 3: Evolution loop (ALL 50 generations)
    print("\n[3/3] Running evolution (50 generations)...")

    for gen in range(1, config.generations + 1):
        gen_start = time.time()
        print(f"\n{'='*60}")
        print(f"GENERATION {gen}/{config.generations}")
        print(f"{'='*60}")

        # Elite preservation: Top 2 -> 6 children
        print("  Creating 6 elite children...")
        elite_children = []
        for elite in population[:2]:
            for s in [0.005, 0.01, 0.02]:
                elite_children.append(mutate(elite, s, config.mutation_rate))

        # Loser merging: Bottom 6 -> 2 children
        print("  Creating 2 loser children...")
        losers = population[-6:]
        c1 = merger.apply_combo(losers[:3], random.randint(0, 7))
        c2 = merger.apply_combo(losers[3:], random.randint(0, 7))
        loser_children = [c1, c2]

        # New population
        population = elite_children + loser_children

        # Evaluate
        print(f"  Evaluating 8 models...")
        fitness_scores = []
        for i, model in enumerate(population):
            f, m = evaluate_model(model, tokenizer, device, config, i + 1, gen)
            model._fitness = f
            model._metrics = m
            fitness_scores.append(f)

        # Sort
        idx = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in idx]
        fitness_scores = [fitness_scores[i] for i in idx]

        # Update champion
        if fitness_scores[0] > champion_fitness:
            champion = copy.deepcopy(population[0])
            champion_fitness = fitness_scores[0]
            champion_metrics = population[0]._metrics
            print(f"\n  *** NEW CHAMPION: fitness={champion_fitness:.4f} ***")

        history.append(champion_fitness)

        # Diversity check
        diversity = compute_diversity(population)
        if diversity < config.diversity_reseed_threshold:
            print(f"  Reseeding (diversity={diversity:.3f})...")
            population[-2] = merger.apply_combo(base_models, random.randint(0, 7))
            population[-1] = merger.apply_combo(base_models, random.randint(0, 7))

        # Progress
        elapsed = time.time() - start_time
        gen_time = time.time() - gen_start
        eta = (elapsed / gen) * (config.generations - gen) / 3600
        improvement = (champion_fitness / initial_fitness - 1) * 100

        print(f"\n  Gen {gen} Summary:")
        print(f"    Best: {fitness_scores[0]:.4f}, Champion: {champion_fitness:.4f} (+{improvement:.1f}%)")
        print(f"    Diversity: {diversity:.3f}")
        print(f"    Gen time: {gen_time/60:.1f}min, Elapsed: {elapsed/3600:.2f}h, ETA: {eta:.2f}h")

        # Save every 5 generations
        if gen % 5 == 0:
            save_checkpoint_with_metrics(output_dir, gen, champion, champion_fitness, champion_metrics, start_time)

    # Final save
    total_time = time.time() - start_time
    save_checkpoint_with_metrics(output_dir, config.generations, champion, champion_fitness, champion_metrics, start_time, is_final=True)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE!")
    print("=" * 70)
    print(f"Initial Fitness: {initial_fitness:.4f}")
    print(f"Final Fitness:   {champion_fitness:.4f}")
    print(f"Improvement:     +{(champion_fitness/initial_fitness-1)*100:.1f}%")
    print(f"Total Time:      {total_time/3600:.2f} hours")
    print("=" * 70)

    return champion, {'initial': initial_fitness, 'final': champion_fitness, 'history': history, 'hours': total_time/3600}


def save_checkpoint_with_metrics(output_dir, gen, model, fitness, metrics, start_time, is_final=False):
    """Save model and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "final_champion" if is_final else f"checkpoint_gen{gen:03d}"

    save_checkpoint(model, output_dir / prefix, metadata={
        "phase": 2, "generation": gen, "fitness": fitness,
        "timestamp": datetime.now().isoformat()
    })

    with open(output_dir / f"{prefix}_metrics.json", "w") as f:
        json.dump({
            "generation": gen, "fitness": fitness, "metrics": metrics,
            "elapsed_hours": (time.time() - start_time) / 3600
        }, f, indent=2)

    print(f"  [Saved] {prefix}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 2: EVOMERGE - FULL OVERNIGHT RUN")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Tokenizer
    print("\nLoading tokenizer...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load Phase 1 models
    print("\nLoading Phase 1 models...")
    base_models = []
    for spec in ["reasoning", "memory", "speed"]:
        path = f"checkpoints/phase1/{spec}/epoch_10.safetensors"
        print(f"  {spec}: {path}")
        cfg = Phase1Config(specialization=spec)
        m = TRMTitansMAGModel(cfg)
        load_checkpoint(m, path, device=device)
        base_models.append(m.to(device).eval())

    # Config
    config = EvoMergeConfig(device=device)

    # Run
    output_dir = Path("checkpoints/phase2_full")
    champion, metrics = run_evolution(base_models, tokenizer, config, output_dir)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

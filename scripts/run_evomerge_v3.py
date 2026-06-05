#!/usr/bin/env python3
"""
Phase 2 EvoMerge V3 - MORE AGGRESSIVE DIVERSITY

Fixes from V2:
1. STRONGER mutations (5x sigma, 10x rate)
2. Random weight noise injection for losers
3. Occasional full random model from base (every 5 gens)
4. Temperature-scaled mutations (hotter early, cooler late)
5. Catastrophic mutation for bottom 2 models
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

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from cross_phase.utils.checkpoint_utils import load_checkpoint, save_checkpoint


@dataclass
class EvoMergeConfig:
    generations: int = 50
    population_size: int = 8
    early_stopping: bool = False

    # V3: MUCH stronger mutations
    mutation_sigma: float = 0.05      # 5x stronger (was 0.01)
    mutation_rate: float = 0.10       # 10x more weights (was 0.01)

    # V3: Catastrophic mutation for worst models
    catastrophic_sigma: float = 0.15  # Very strong noise
    catastrophic_rate: float = 0.30   # 30% of weights

    # V3: Temperature schedule (exploration -> exploitation)
    initial_temperature: float = 2.0
    final_temperature: float = 0.5

    # V3: Random injection frequency
    random_inject_every: int = 5  # Inject fresh random model every N gens

    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'perplexity': 0.4,
        'accuracy': 0.3,
        'speed': 0.2,
        'memory': 0.1
    })

    gsm8k_samples: int = 50
    perplexity_samples: int = 100
    max_new_tokens: int = 100

    dare_drop_rate: float = 0.9
    ties_trim_percent: float = 0.2
    diversity_reseed_threshold: float = 0.15  # Lower threshold

    device: str = "cuda"


# ============================================================================
# MERGE TECHNIQUES (Same as before)
# ============================================================================

class LinearMerge:
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
    def merge(self, model_merged: nn.Module, models_ref: List[nn.Module]) -> nn.Module:
        result = copy.deepcopy(model_merged)
        candidates = [model_merged] + models_ref
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
# V3: ENHANCED MUTATION FUNCTIONS
# ============================================================================

def mutate_model(model: nn.Module, sigma: float, rate: float, device: str) -> nn.Module:
    """Standard mutation with configurable strength."""
    mutated = copy.deepcopy(model)
    with torch.no_grad():
        for param in mutated.parameters():
            mask = torch.rand_like(param) < rate
            noise = torch.randn_like(param) * sigma
            param.add_(noise * mask.float())
    return mutated


def catastrophic_mutate(model: nn.Module, sigma: float, rate: float, device: str) -> nn.Module:
    """V3: Very aggressive mutation for stuck models."""
    mutated = copy.deepcopy(model)
    with torch.no_grad():
        for param in mutated.parameters():
            # Higher rate, stronger noise
            mask = torch.rand_like(param) < rate
            noise = torch.randn_like(param) * sigma
            param.add_(noise * mask.float())

            # Also randomly zero out some weights (pruning noise)
            zero_mask = torch.rand_like(param) < 0.05  # 5% zeroed
            param.mul_(~zero_mask)
    return mutated


def random_interpolate(model_a: nn.Module, model_b: nn.Module, device: str) -> nn.Module:
    """V3: Random per-layer interpolation between two models."""
    result = copy.deepcopy(model_a)
    with torch.no_grad():
        for name, param_a in model_a.named_parameters():
            param_b = dict(model_b.named_parameters())[name]
            result_param = dict(result.named_parameters())[name]

            # Random alpha per layer
            alpha = random.uniform(0.0, 1.0)
            result_param.copy_(alpha * param_a + (1 - alpha) * param_b)
    return result


def inject_noise_layer(model: nn.Module, layer_fraction: float = 0.3) -> nn.Module:
    """V3: Inject noise into random subset of layers."""
    result = copy.deepcopy(model)
    param_names = list(dict(result.named_parameters()).keys())
    num_to_noise = max(1, int(len(param_names) * layer_fraction))
    layers_to_noise = random.sample(param_names, num_to_noise)

    with torch.no_grad():
        for name in layers_to_noise:
            param = dict(result.named_parameters())[name]
            noise = torch.randn_like(param) * param.std() * 0.5
            param.add_(noise)
    return result


def get_temperature(gen: int, total_gens: int, config: EvoMergeConfig) -> float:
    """V3: Temperature schedule - hotter early, cooler late."""
    progress = gen / total_gens
    temp = config.initial_temperature * (1 - progress) + config.final_temperature * progress
    return temp


# ============================================================================
# BENCHMARKS (Same as before)
# ============================================================================

def extract_answer(text: str):
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(matches[-1]) if matches else None


def evaluate_gsm8k(model: nn.Module, tokenizer, device: str, num_samples: int, max_new_tokens: int) -> float:
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
    model.eval()
    dummy = torch.randint(0, 1000, (1, 64)).to(device)
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
    params = sum(p.numel() for p in model.parameters())
    return (params * 4 * 2) / (1024 * 1024)


def compute_fitness(ppl, acc, speed, memory, weights) -> float:
    ppl_score = 1.0 / max(1.0, ppl)
    spd_score = speed / 1200.0
    mem_score = 500.0 / max(1.0, memory)
    return weights['perplexity'] * ppl_score + weights['accuracy'] * acc + \
           weights['speed'] * spd_score + weights['memory'] * mem_score


def evaluate_model(model: nn.Module, tokenizer, device: str, config: EvoMergeConfig,
                   model_idx: int, gen: int) -> Tuple[float, Dict]:
    print(f"    Model {model_idx}/8 (gen {gen}):", flush=True)
    start = time.time()
    print(f"      Running GSM8K ({config.gsm8k_samples} samples)...", flush=True)
    acc = evaluate_gsm8k(model, tokenizer, device, config.gsm8k_samples, config.max_new_tokens)
    print(f"      Running Perplexity ({config.perplexity_samples} samples)...", flush=True)
    ppl = evaluate_perplexity(model, tokenizer, device, config.perplexity_samples)
    speed = evaluate_speed(model, device)
    memory = evaluate_memory(model)
    fitness = compute_fitness(ppl, acc, speed, memory, config.fitness_weights)
    elapsed = time.time() - start
    print(f"      DONE: PPL={ppl:.1f}, Acc={acc:.1%}, Spd={speed:.0f}, fit={fitness:.4f} ({elapsed:.1f}s)")
    return fitness, {'perplexity': ppl, 'accuracy': acc, 'speed': speed, 'memory': memory, 'fitness': fitness}


def compute_diversity(population: List[nn.Module]) -> float:
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
# V3: ENHANCED EVOLUTION LOOP
# ============================================================================

def run_evolution(base_models: List[nn.Module], tokenizer, config: EvoMergeConfig, output_dir: Path):
    device = config.device
    merger = MergeTechniques(config)

    print("\n" + "=" * 70)
    print("PHASE 2: EVOMERGE V3 - AGGRESSIVE DIVERSITY")
    print("=" * 70)
    print(f"V3 Enhancements:")
    print(f"  - Mutation sigma: {config.mutation_sigma} (5x stronger)")
    print(f"  - Mutation rate: {config.mutation_rate} (10x more weights)")
    print(f"  - Catastrophic mutation: sigma={config.catastrophic_sigma}, rate={config.catastrophic_rate}")
    print(f"  - Temperature schedule: {config.initial_temperature} -> {config.final_temperature}")
    print(f"  - Random injection every {config.random_inject_every} generations")
    print("=" * 70)

    start_time = time.time()

    # Initialize population
    print("\n[1/3] Creating initial population (8 binary combos)...")
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

    # Initial evaluation
    print("\n[2/3] Evaluating initial population...")
    fitness_scores = []
    for i, model in enumerate(population):
        f, m = evaluate_model(model, tokenizer, device, config, i + 1, 0)
        model._fitness = f
        model._metrics = m
        fitness_scores.append(f)

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

    # Evolution loop
    print("\n[3/3] Running V3 evolution (50 generations)...")

    for gen in range(1, config.generations + 1):
        gen_start = time.time()
        temperature = get_temperature(gen, config.generations, config)

        print(f"\n{'='*60}")
        print(f"GENERATION {gen}/{config.generations} (temp={temperature:.2f})")
        print(f"{'='*60}")

        # V3: Adaptive mutation based on temperature
        current_sigma = config.mutation_sigma * temperature
        current_rate = config.mutation_rate * temperature

        # Elite preservation: Top 2 -> 6 children with STRONGER mutations
        print(f"  Creating 6 elite children (sigma={current_sigma:.3f}, rate={current_rate:.2f})...")
        elite_children = []
        for elite in population[:2]:
            for mult in [0.5, 1.0, 2.0]:  # Varying mutation strengths
                child = mutate_model(elite, current_sigma * mult, current_rate, device)
                elite_children.append(child)

        # V3: Loser handling - MORE AGGRESSIVE
        print(f"  Creating 2 loser children with AGGRESSIVE mutations...")
        losers = population[-6:]

        # Loser child 1: Catastrophic mutation of random loser
        loser1 = catastrophic_mutate(
            random.choice(losers),
            config.catastrophic_sigma * temperature,
            config.catastrophic_rate,
            device
        )

        # Loser child 2: Random interpolation between two losers + noise
        loser2 = random_interpolate(
            random.choice(losers),
            random.choice(losers),
            device
        )
        loser2 = inject_noise_layer(loser2, layer_fraction=0.4)

        loser_children = [loser1, loser2]

        # V3: Random injection every N generations
        if gen % config.random_inject_every == 0:
            print(f"  [V3] Injecting fresh random model from base...")
            random_combo = random.randint(0, 7)
            fresh_model = merger.apply_combo(base_models, random_combo)
            fresh_model = inject_noise_layer(fresh_model, layer_fraction=0.5)
            # Replace worst elite child
            elite_children[-1] = fresh_model

        population = elite_children + loser_children

        # Evaluate
        print(f"  Evaluating 8 models...")
        fitness_scores = []
        for i, model in enumerate(population):
            f, m = evaluate_model(model, tokenizer, device, config, i + 1, gen)
            model._fitness = f
            model._metrics = m
            fitness_scores.append(f)

        idx = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in idx]
        fitness_scores = [fitness_scores[i] for i in idx]

        if fitness_scores[0] > champion_fitness:
            champion = copy.deepcopy(population[0])
            champion_fitness = fitness_scores[0]
            champion_metrics = population[0]._metrics
            print(f"\n  *** NEW CHAMPION: fitness={champion_fitness:.4f} ***")

        history.append(champion_fitness)

        # V3: More aggressive reseeding
        diversity = compute_diversity(population)
        if diversity < config.diversity_reseed_threshold:
            print(f"  [V3] Low diversity ({diversity:.3f}), aggressive reseed...")
            # Replace bottom 3 with fresh random combos + noise
            for i in range(3):
                combo = random.randint(0, 7)
                fresh = merger.apply_combo(base_models, combo)
                fresh = catastrophic_mutate(fresh, config.catastrophic_sigma, config.catastrophic_rate, device)
                population[-(i+1)] = fresh

        # Progress
        elapsed = time.time() - start_time
        gen_time = time.time() - gen_start
        eta = (elapsed / gen) * (config.generations - gen) / 3600
        improvement = (champion_fitness / initial_fitness - 1) * 100

        print(f"\n  Gen {gen} Summary:")
        print(f"    Best: {fitness_scores[0]:.4f}, Champion: {champion_fitness:.4f} (+{improvement:.1f}%)")
        print(f"    Diversity: {diversity:.3f}, Temperature: {temperature:.2f}")
        print(f"    Gen time: {gen_time/60:.1f}min, Elapsed: {elapsed/3600:.2f}h, ETA: {eta:.2f}h")

        if gen % 5 == 0:
            save_checkpoint_with_metrics(output_dir, gen, champion, champion_fitness, champion_metrics, start_time)

    # Final
    total_time = time.time() - start_time
    save_checkpoint_with_metrics(output_dir, config.generations, champion, champion_fitness, champion_metrics, start_time, is_final=True)

    print("\n" + "=" * 70)
    print("PHASE 2 V3 COMPLETE!")
    print("=" * 70)
    print(f"Initial Fitness: {initial_fitness:.4f}")
    print(f"Final Fitness:   {champion_fitness:.4f}")
    print(f"Improvement:     +{(champion_fitness/initial_fitness-1)*100:.1f}%")
    print(f"Total Time:      {total_time/3600:.2f} hours")
    print("=" * 70)

    return champion, {'initial': initial_fitness, 'final': champion_fitness, 'history': history, 'hours': total_time/3600}


def save_checkpoint_with_metrics(output_dir, gen, model, fitness, metrics, start_time, is_final=False):
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


def main():
    print("=" * 70)
    print("PHASE 2: EVOMERGE V3 - AGGRESSIVE DIVERSITY")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading tokenizer...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading Phase 1 models...")
    base_models = []
    for spec in ["reasoning", "memory", "speed"]:
        path = f"checkpoints/phase1/{spec}/epoch_10.safetensors"
        print(f"  {spec}: {path}")
        cfg = Phase1Config(specialization=spec)
        m = TRMTitansMAGModel(cfg)
        load_checkpoint(m, path, device=device)
        base_models.append(m.to(device).eval())

    config = EvoMergeConfig(device=device)
    output_dir = Path("checkpoints/phase2_v3")
    champion, metrics = run_evolution(base_models, tokenizer, config, output_dir)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 2 EvoMerge V2 - Benchmark-Based Overnight Evolution

EVOLUTION STRATEGY (8 models per generation):
- Benchmark all 8 models with actual tasks (GSM8K + perplexity)
- Top 2 winners: Each mutated into 3 variants = 6 offspring
- Bottom 6 losers: Grouped into 2 groups of 3, merged = 2 chaos offspring
- Next generation = 6 winner children + 2 chaos children = 8 models

Expected runtime: 10-20 hours for 50 generations
"""

import copy
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from cross_phase.utils.checkpoint_utils import load_checkpoint, save_checkpoint


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_phase1_model(checkpoint_path: str, specialization: str, device: str = "cuda"):
    """Load a Phase 1 model from checkpoint."""
    print(f"  Loading {specialization} model...")
    config = Phase1Config(specialization=specialization)
    model = TRMTitansMAGModel(config)
    load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()
    return model


def get_tokenizer():
    """Get GPT-2 tokenizer."""
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ============================================================================
# MERGE FUNCTIONS
# ============================================================================

def linear_merge(model_a: nn.Module, model_b: nn.Module, alpha: float = 0.5) -> nn.Module:
    """Simple linear interpolation merge: result = alpha*A + (1-alpha)*B"""
    result = copy.deepcopy(model_a)
    with torch.no_grad():
        for (name_a, param_a), (name_b, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            result_param = dict(result.named_parameters())[name_a]
            result_param.copy_(alpha * param_a + (1 - alpha) * param_b)
    return result


def slerp_merge(model_a: nn.Module, model_b: nn.Module, t: float = 0.5) -> nn.Module:
    """Spherical linear interpolation merge."""
    result = copy.deepcopy(model_a)
    with torch.no_grad():
        for (name_a, param_a), (name_b, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            result_param = dict(result.named_parameters())[name_a]

            flat_a = param_a.flatten().float()
            flat_b = param_b.flatten().float()

            norm_a = flat_a.norm()
            norm_b = flat_b.norm()

            if norm_a < 1e-8 or norm_b < 1e-8:
                result_param.copy_(t * param_a + (1 - t) * param_b)
                continue

            unit_a = flat_a / norm_a
            unit_b = flat_b / norm_b

            dot = torch.clamp(torch.dot(unit_a, unit_b), -1.0, 1.0)
            omega = torch.acos(dot)

            if omega.abs() < 1e-8:
                result_param.copy_(t * param_a + (1 - t) * param_b)
                continue

            sin_omega = torch.sin(omega)
            interp = (torch.sin((1 - t) * omega) / sin_omega) * flat_a + \
                     (torch.sin(t * omega) / sin_omega) * flat_b

            result_param.copy_(interp.view(param_a.shape).to(param_a.dtype))

    return result


def merge_three_models(models: List[nn.Module], technique: str = "linear") -> nn.Module:
    """Merge 3 models sequentially: (A+B) -> AB, then (AB+C) -> ABC"""
    merge_fn = linear_merge if technique == "linear" else slerp_merge

    alpha1 = random.uniform(0.25, 0.75)
    alpha2 = random.uniform(0.25, 0.75)

    merged_ab = merge_fn(models[0], models[1], alpha1)
    merged_abc = merge_fn(merged_ab, models[2], alpha2)

    return merged_abc


def mutate(model: nn.Module, noise_scale: float = 0.01) -> nn.Module:
    """Apply random weight perturbation."""
    result = copy.deepcopy(model)
    with torch.no_grad():
        for param in result.parameters():
            noise = torch.randn_like(param) * noise_scale
            param.add_(noise)
    return result


# ============================================================================
# BENCHMARK-BASED FITNESS (ACTUAL TASK EVALUATION)
# ============================================================================

def extract_answer(text: str):
    """Extract numeric answer from text."""
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return float(numbers[-1])
    return None


def benchmark_gsm8k(model: nn.Module, tokenizer, device: str, num_samples: int = 25) -> float:
    """Evaluate on GSM8K subset - returns accuracy."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main", split="test")
        samples = list(dataset)[:num_samples]
    except Exception as e:
        print(f"    GSM8K load error: {e}")
        return 0.0

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for sample in samples:
            question = sample["question"]
            gold_text = sample["answer"]
            gold_answer = extract_answer(gold_text)

            if gold_answer is None:
                continue

            prompt = f"Question: {question}\nAnswer: Let me solve this step by step.\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(device)

            try:
                outputs = model(input_ids=input_ids)
                logits = outputs["logits"]

                generated_ids = input_ids.clone()
                for _ in range(30):  # Generate 30 tokens
                    next_logits = logits[0, -1, :]
                    next_token = next_logits.argmax()
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                    outputs = model(input_ids=generated_ids)
                    logits = outputs["logits"]

                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                pred_answer = extract_answer(generated_text)

                if pred_answer is not None and abs(pred_answer - gold_answer) < 0.01:
                    correct += 1

            except Exception:
                pass

            total += 1

    return correct / total if total > 0 else 0.0


def benchmark_perplexity(model: nn.Module, tokenizer, device: str, num_samples: int = 20) -> float:
    """Calculate perplexity on validation samples - lower is better."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t for t in dataset["text"] if len(t.strip()) > 100][:num_samples]
    except Exception as e:
        print(f"    Perplexity load error: {e}")
        return 1000.0  # High perplexity = bad

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
        return 1000.0

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # Cap perplexity to avoid infinity
    return min(perplexity, 10000.0)


def compute_fitness(model: nn.Module, tokenizer, device: str, model_idx: int, gen: int) -> Tuple[float, Dict]:
    """
    Compute fitness using actual benchmarks.

    Returns: (fitness_score, metrics_dict)

    Fitness = 0.6 * gsm8k_accuracy + 0.4 * (1 / log(perplexity))
    """
    print(f"    Benchmarking model {model_idx + 1}/8 (gen {gen})...", end=" ", flush=True)
    start = time.time()

    # Run benchmarks
    gsm8k_acc = benchmark_gsm8k(model, tokenizer, device, num_samples=25)
    perplexity = benchmark_perplexity(model, tokenizer, device, num_samples=20)

    # Compute fitness
    # GSM8K accuracy: 0-1, higher is better
    # Perplexity: lower is better, use inverse log
    import math
    ppl_score = 1.0 / (1.0 + math.log(max(1.0, perplexity)))

    fitness = 0.6 * gsm8k_acc + 0.4 * ppl_score

    elapsed = time.time() - start
    print(f"GSM8K={gsm8k_acc:.1%}, PPL={perplexity:.1f}, fit={fitness:.4f} ({elapsed:.1f}s)")

    metrics = {
        "gsm8k_accuracy": gsm8k_acc,
        "perplexity": perplexity,
        "ppl_score": ppl_score,
        "fitness": fitness,
        "eval_time": elapsed
    }

    return fitness, metrics


# ============================================================================
# EVOLUTION ENGINE
# ============================================================================

def create_initial_population(base_models: List[nn.Module], size: int = 8) -> List[nn.Module]:
    """Create initial population of 8 models from 3 base models."""
    population = []

    # Add original 3 models
    for m in base_models:
        population.append(copy.deepcopy(m))

    # Fill remaining with merged variants
    techniques = ["linear", "slerp"]
    while len(population) < size:
        technique = random.choice(techniques)
        merged = merge_three_models(base_models, technique)
        population.append(merged)

    return population[:size]


def evolve_generation(
    population: List[nn.Module],
    fitness_scores: List[float],
    generation: int
) -> List[nn.Module]:
    """
    Evolve population using the specified strategy:
    - Top 2 winners: Each mutated into 3 variants = 6 offspring
    - Bottom 6 losers: Grouped into 2 groups of 3, merged = 2 chaos offspring

    Returns: New population of 8 models
    """
    # Sort by fitness (descending)
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)

    # Top 2 winners
    winner_indices = sorted_indices[:2]
    # Bottom 6 losers
    loser_indices = sorted_indices[2:]

    new_population = []

    # WINNERS: Each mutated into 3 variants = 6 offspring
    print(f"  Creating 6 winner offspring from top 2...")
    for winner_idx in winner_indices:
        winner = population[winner_idx]
        for mutation_scale in [0.005, 0.01, 0.02]:  # 3 different mutation strengths
            mutated = mutate(winner, noise_scale=mutation_scale)
            new_population.append(mutated)

    # LOSERS: Group into 2 groups of 3, merge each = 2 chaos offspring
    print(f"  Creating 2 chaos offspring from bottom 6...")
    random.shuffle(loser_indices)

    # Group 1: losers 0,1,2 -> merge
    group1 = [population[loser_indices[0]], population[loser_indices[1]], population[loser_indices[2]]]
    technique1 = random.choice(["linear", "slerp"])
    chaos1 = merge_three_models(group1, technique1)
    new_population.append(chaos1)

    # Group 2: losers 3,4,5 -> merge
    group2 = [population[loser_indices[3]], population[loser_indices[4]], population[loser_indices[5]]]
    technique2 = random.choice(["linear", "slerp"])
    chaos2 = merge_three_models(group2, technique2)
    new_population.append(chaos2)

    return new_population


def save_progress(
    output_dir: Path,
    generation: int,
    champion: nn.Module,
    champion_fitness: float,
    champion_metrics: Dict,
    all_metrics: List[Dict],
    start_time: float
):
    """Save checkpoint and progress."""
    elapsed = time.time() - start_time

    # Save champion model
    champion_path = output_dir / f"champion_gen{generation:03d}"
    save_checkpoint(
        champion,
        champion_path,
        metadata={
            "phase": 2,
            "type": "evomerge_v2_champion",
            "generation": generation,
            "fitness": champion_fitness,
            "gsm8k_accuracy": champion_metrics.get("gsm8k_accuracy", 0),
            "perplexity": champion_metrics.get("perplexity", 0),
            "timestamp": datetime.now().isoformat(),
        }
    )

    # Save progress JSON
    progress = {
        "current_generation": generation,
        "champion_fitness": champion_fitness,
        "champion_metrics": champion_metrics,
        "elapsed_hours": elapsed / 3600,
        "all_generation_metrics": all_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    progress_path = output_dir / "evolution_progress.json"
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2, default=str)

    print(f"  [Saved checkpoint: gen {generation}, fitness={champion_fitness:.4f}]")


def run_evolution(
    input_models: List[nn.Module],
    tokenizer,
    device: str,
    num_generations: int = 50,
    output_dir: Path = None
) -> Tuple[nn.Module, Dict]:
    """
    Run benchmark-based evolutionary optimization.

    Population: 8 models
    Selection: Top 2 -> 6 mutated offspring, Bottom 6 -> 2 merged chaos offspring
    """
    print(f"\n{'='*70}")
    print(f"STARTING EVOLUTION: {num_generations} generations, 8 models each")
    print(f"Estimated time: {num_generations * 8 * 2 / 60:.1f} hours")
    print(f"{'='*70}\n")

    start_time = time.time()
    all_gen_metrics = []

    # Initialize population
    print("Creating initial population of 8 models...")
    population = create_initial_population(input_models, size=8)

    # Evaluate initial population
    print("\nEvaluating initial population:")
    fitness_scores = []
    model_metrics = []
    for i, model in enumerate(population):
        fitness, metrics = compute_fitness(model, tokenizer, device, i, 0)
        fitness_scores.append(fitness)
        model_metrics.append(metrics)

    # Track best
    best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
    champion = copy.deepcopy(population[best_idx])
    champion_fitness = fitness_scores[best_idx]
    champion_metrics = model_metrics[best_idx]

    initial_fitness = champion_fitness

    print(f"\nInitial champion: fitness={champion_fitness:.4f}, GSM8K={champion_metrics['gsm8k_accuracy']:.1%}, PPL={champion_metrics['perplexity']:.1f}")

    all_gen_metrics.append({
        "generation": 0,
        "best_fitness": champion_fitness,
        "best_metrics": champion_metrics,
        "all_fitness": fitness_scores.copy()
    })

    # Save initial checkpoint
    if output_dir:
        save_progress(output_dir, 0, champion, champion_fitness, champion_metrics, all_gen_metrics, start_time)

    # Evolution loop
    for gen in range(1, num_generations + 1):
        print(f"\n{'='*50}")
        print(f"GENERATION {gen}/{num_generations}")
        print(f"{'='*50}")

        # Evolve
        population = evolve_generation(population, fitness_scores, gen)

        # Evaluate new population
        print(f"\nEvaluating generation {gen}:")
        fitness_scores = []
        model_metrics = []
        for i, model in enumerate(population):
            fitness, metrics = compute_fitness(model, tokenizer, device, i, gen)
            fitness_scores.append(fitness)
            model_metrics.append(metrics)

        # Update champion if improved
        best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        if fitness_scores[best_idx] > champion_fitness:
            champion = copy.deepcopy(population[best_idx])
            champion_fitness = fitness_scores[best_idx]
            champion_metrics = model_metrics[best_idx]
            print(f"\n*** NEW CHAMPION! fitness={champion_fitness:.4f} ***")

        # Progress
        elapsed = time.time() - start_time
        improvement = ((champion_fitness / initial_fitness) - 1) * 100 if initial_fitness > 0 else 0
        eta_hours = (elapsed / gen) * (num_generations - gen) / 3600

        print(f"\nGen {gen} Summary:")
        print(f"  Best this gen: {max(fitness_scores):.4f}")
        print(f"  Champion: {champion_fitness:.4f} (+{improvement:.1f}% from start)")
        print(f"  Elapsed: {elapsed/3600:.1f}h, ETA: {eta_hours:.1f}h")

        all_gen_metrics.append({
            "generation": gen,
            "best_fitness": champion_fitness,
            "best_metrics": champion_metrics,
            "all_fitness": fitness_scores.copy(),
            "elapsed_hours": elapsed / 3600
        })

        # Save progress every 5 generations
        if output_dir and gen % 5 == 0:
            save_progress(output_dir, gen, champion, champion_fitness, champion_metrics, all_gen_metrics, start_time)

    total_time = time.time() - start_time

    final_metrics = {
        "initial_fitness": initial_fitness,
        "final_fitness": champion_fitness,
        "improvement_pct": ((champion_fitness / initial_fitness) - 1) * 100 if initial_fitness > 0 else 0,
        "champion_gsm8k": champion_metrics.get("gsm8k_accuracy", 0),
        "champion_perplexity": champion_metrics.get("perplexity", 0),
        "generations": num_generations,
        "total_hours": total_time / 3600,
        "all_generation_metrics": all_gen_metrics
    }

    return champion, final_metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 2: EVOMERGE V2 - BENCHMARK-BASED OVERNIGHT RUN")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nEvolution Strategy:")
    print("  - 8 models per generation")
    print("  - Top 2 winners -> 3 mutations each = 6 offspring")
    print("  - Bottom 6 losers -> 2 groups of 3, merged = 2 chaos offspring")
    print("  - Fitness = actual GSM8K + perplexity benchmarks")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()

    # Define checkpoint paths
    reasoning_path = "checkpoints/phase1/reasoning/epoch_10.safetensors"
    memory_path = "checkpoints/phase1/memory/epoch_10.safetensors"
    speed_path = "checkpoints/phase1/speed/epoch_10.safetensors"

    # Load Phase 1 models
    print("\n" + "=" * 50)
    print("LOADING PHASE 1 MODELS")
    print("=" * 50)

    try:
        reasoning_model = load_phase1_model(reasoning_path, "reasoning", device)
        memory_model = load_phase1_model(memory_path, "memory", device)
        speed_model = load_phase1_model(speed_path, "speed", device)
    except Exception as e:
        print(f"ERROR loading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    models = [reasoning_model, memory_model, speed_model]
    print(f"Loaded 3 Phase 1 models successfully")

    # Create output directory
    output_dir = Path("checkpoints/phase2_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run Evolution
    try:
        champion, metrics = run_evolution(
            models,
            tokenizer,
            device,
            num_generations=50,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"\nERROR during evolution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save final champion
    print("\n" + "=" * 50)
    print("SAVING FINAL RESULTS")
    print("=" * 50)

    final_path = output_dir / "final_champion"
    save_checkpoint(
        champion,
        final_path,
        metadata={
            "phase": 2,
            "type": "evomerge_v2_final",
            "generations": 50,
            "fitness": metrics["final_fitness"],
            "gsm8k_accuracy": metrics["champion_gsm8k"],
            "perplexity": metrics["champion_perplexity"],
            "improvement_pct": metrics["improvement_pct"],
            "timestamp": datetime.now().isoformat(),
        }
    )
    print(f"Saved final champion to: {final_path}.safetensors")

    # Save final metrics
    metrics_path = output_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        # Remove non-serializable nested metrics for main file
        save_metrics = {k: v for k, v in metrics.items() if k != "all_generation_metrics"}
        json.dump(save_metrics, f, indent=2, default=str)
    print(f"Saved metrics to: {metrics_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 2 V2 COMPLETE!")
    print("=" * 70)
    print(f"Initial Fitness:     {metrics['initial_fitness']:.4f}")
    print(f"Final Fitness:       {metrics['final_fitness']:.4f}")
    print(f"Improvement:         {metrics['improvement_pct']:+.1f}%")
    print(f"Champion GSM8K:      {metrics['champion_gsm8k']:.1%}")
    print(f"Champion Perplexity: {metrics['champion_perplexity']:.1f}")
    print(f"Total Time:          {metrics['total_hours']:.1f} hours")
    print(f"Output Directory:    {output_dir}")
    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return champion


if __name__ == "__main__":
    main()

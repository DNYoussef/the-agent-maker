#!/usr/bin/env python3
"""
Phase 2 EvoMerge Overnight Script

Runs evolutionary optimization on the 3 Phase 1 models.
Merges models sequentially: (A+B) then (AB+C) for techniques that need 2 models.

Expected runtime: 1-2 hours for 50 generations
"""

import copy
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from cross_phase.utils.checkpoint_utils import load_checkpoint, save_checkpoint


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

            # Flatten for SLERP
            flat_a = param_a.flatten().float()
            flat_b = param_b.flatten().float()

            # Normalize
            norm_a = flat_a.norm()
            norm_b = flat_b.norm()

            if norm_a < 1e-8 or norm_b < 1e-8:
                # Fallback to linear
                result_param.copy_(t * param_a + (1 - t) * param_b)
                continue

            unit_a = flat_a / norm_a
            unit_b = flat_b / norm_b

            # Compute angle
            dot = torch.clamp(torch.dot(unit_a, unit_b), -1.0, 1.0)
            omega = torch.acos(dot)

            if omega.abs() < 1e-8:
                # Nearly parallel, use linear
                result_param.copy_(t * param_a + (1 - t) * param_b)
                continue

            # SLERP
            sin_omega = torch.sin(omega)
            interp = (torch.sin((1 - t) * omega) / sin_omega) * flat_a + \
                     (torch.sin(t * omega) / sin_omega) * flat_b

            result_param.copy_(interp.view(param_a.shape).to(param_a.dtype))

    return result


def merge_three_models(models: List[nn.Module], technique: str = "linear") -> nn.Module:
    """
    Merge 3 models sequentially: (A+B) -> AB, then (AB+C) -> ABC
    """
    merge_fn = linear_merge if technique == "linear" else slerp_merge

    # Random weights for diversity
    alpha1 = random.uniform(0.3, 0.7)
    alpha2 = random.uniform(0.3, 0.7)

    # First merge: models[0] + models[1]
    merged_ab = merge_fn(models[0], models[1], alpha1)

    # Second merge: merged_ab + models[2]
    merged_abc = merge_fn(merged_ab, models[2], alpha2)

    return merged_abc


def compute_fitness(model: nn.Module) -> float:
    """Quick fitness evaluation based on parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())

    # Compute parameter variance (lower = more stable)
    param_vars = []
    param_means = []
    for p in model.parameters():
        if p.numel() == 0:
            continue
        var_val = p.var().item()
        mean_val = p.abs().mean().item()
        # Skip NaN/Inf values
        if not (torch.isnan(torch.tensor(var_val)) or torch.isinf(torch.tensor(var_val))):
            param_vars.append(var_val)
        if not (torch.isnan(torch.tensor(mean_val)) or torch.isinf(torch.tensor(mean_val))):
            param_means.append(mean_val)

    if not param_vars or not param_means:
        return 0.1  # Default low fitness for invalid models

    avg_var = sum(param_vars) / len(param_vars)
    avg_mean = sum(param_means) / len(param_means)

    # Handle edge cases
    if avg_var < 0 or avg_mean < 0:
        return 0.1

    # Fitness: prefer moderate variance and reasonable magnitude
    # Lower variance = more stable, moderate magnitude = not collapsed
    variance_score = 1.0 / (1.0 + avg_var * 10)  # Lower variance is better
    magnitude_score = min(1.0, avg_mean * 10)     # Some magnitude is good

    fitness = 0.6 * variance_score + 0.4 * magnitude_score

    # Ensure valid fitness
    if torch.isnan(torch.tensor(fitness)) or torch.isinf(torch.tensor(fitness)):
        return 0.1

    return max(0.01, fitness)  # Ensure positive


def mutate(model: nn.Module, mutation_rate: float = 0.1, noise_scale: float = 0.01) -> nn.Module:
    """Apply random weight perturbation."""
    if random.random() > mutation_rate:
        return model

    result = copy.deepcopy(model)
    with torch.no_grad():
        for param in result.parameters():
            noise = torch.randn_like(param) * noise_scale
            param.add_(noise)
    return result


def run_evolution(
    input_models: List[nn.Module],
    num_generations: int = 50,
    population_size: int = 10,
    elite_count: int = 2
) -> tuple:
    """
    Run evolutionary optimization.

    Returns: (champion_model, metrics_dict)
    """
    print(f"\nStarting evolution: {num_generations} generations, population {population_size}")

    # Initialize population
    population = []

    # Add original models
    for m in input_models:
        population.append(copy.deepcopy(m))

    # Fill with merged variants
    techniques = ["linear", "slerp"]
    while len(population) < population_size:
        technique = random.choice(techniques)
        merged = merge_three_models(input_models, technique)
        population.append(merged)

    # Evaluate initial fitness
    for model in population:
        model._fitness = compute_fitness(model)

    population.sort(key=lambda m: m._fitness, reverse=True)
    initial_fitness = population[0]._fitness
    best_fitness = initial_fitness
    champion = copy.deepcopy(population[0])

    print(f"Initial best fitness: {initial_fitness:.4f}")

    fitness_history = [initial_fitness]

    # Evolution loop
    for gen in range(num_generations):
        # Tournament selection
        parents = []
        for _ in range(population_size):
            tournament = random.sample(population, min(3, len(population)))
            winner = max(tournament, key=lambda m: m._fitness)
            parents.append(winner)

        # Create offspring through merging
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            if random.random() < 0.7:  # Crossover rate
                technique = random.choice(techniques)
                alpha = random.uniform(0.3, 0.7)
                merge_fn = linear_merge if technique == "linear" else slerp_merge
                child = merge_fn(parents[i], parents[i+1], alpha)
            else:
                child = copy.deepcopy(parents[i])

            # Mutation
            child = mutate(child, mutation_rate=0.1)
            offspring.append(child)

        # Evaluate offspring
        for model in offspring:
            model._fitness = compute_fitness(model)

        # Elitism: keep top elite_count from current population
        elite = population[:elite_count]

        # New population
        population = elite + offspring
        population.sort(key=lambda m: m._fitness, reverse=True)
        population = population[:population_size]

        # Update champion
        if population[0]._fitness > best_fitness:
            best_fitness = population[0]._fitness
            champion = copy.deepcopy(population[0])

        fitness_history.append(best_fitness)

        # Progress
        if (gen + 1) % 5 == 0 or gen == 0:
            gain = (best_fitness / initial_fitness - 1) * 100
            print(f"  Gen {gen + 1:3d}/{num_generations}: fitness={best_fitness:.4f} (+{gain:+.1f}%)")

    final_gain = (best_fitness / initial_fitness - 1) if initial_fitness > 0 else 0

    metrics = {
        "initial_fitness": initial_fitness,
        "final_fitness": best_fitness,
        "fitness_gain": final_gain,
        "generations": num_generations,
        "population_size": population_size,
        "fitness_history": fitness_history,
    }

    return champion, metrics


def main():
    print("=" * 70)
    print("PHASE 2: EVOMERGE - OVERNIGHT RUN")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
    print(f"Loaded 3 models successfully")

    # Create output directory
    output_dir = Path("checkpoints/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run EvoMerge
    print("\n" + "=" * 50)
    print("STARTING EVOMERGE")
    print("=" * 50)
    print("Merge strategy: Sequential (A+B -> AB+C)")
    print("Techniques: linear, slerp")

    start_time = time.time()

    try:
        champion, metrics = run_evolution(
            models,
            num_generations=50,
            population_size=10,
            elite_count=2
        )
    except Exception as e:
        print(f"\nERROR during EvoMerge: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time

    # Save champion model
    print("\n" + "=" * 50)
    print("SAVING RESULTS")
    print("=" * 50)

    champion_path = output_dir / "champion_model"
    save_checkpoint(
        champion,
        champion_path,
        metadata={
            "phase": 2,
            "type": "evomerge_champion",
            "generations": 50,
            "fitness_gain": metrics.get("fitness_gain", 0),
            "timestamp": datetime.now().isoformat(),
        }
    )
    print(f"Saved champion to: {champion_path}.safetensors")

    # Save metrics
    metrics["duration_seconds"] = elapsed
    metrics["duration_hours"] = elapsed / 3600
    metrics["timestamp"] = datetime.now().isoformat()

    metrics_path = output_dir / "evomerge_metrics.json"
    with open(metrics_path, "w") as f:
        # Remove non-serializable items
        save_metrics = {k: v for k, v in metrics.items() if k != "fitness_history"}
        save_metrics["fitness_history_summary"] = {
            "start": metrics["fitness_history"][0] if metrics.get("fitness_history") else 0,
            "end": metrics["fitness_history"][-1] if metrics.get("fitness_history") else 0,
        }
        json.dump(save_metrics, f, indent=2, default=str)
    print(f"Saved metrics to: {metrics_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE!")
    print("=" * 70)
    print(f"Initial Fitness:  {metrics.get('initial_fitness', 0):.4f}")
    print(f"Final Fitness:    {metrics.get('final_fitness', 0):.4f}")
    print(f"Fitness Gain:     {metrics.get('fitness_gain', 0) * 100:+.1f}%")
    print(f"Duration:         {elapsed / 60:.1f} minutes")
    print(f"Champion saved:   {champion_path}.safetensors")
    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return champion


if __name__ == "__main__":
    main()

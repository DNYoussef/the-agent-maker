# Phase 1-4: Weights & Biases Integration Plan

**Version**: 2.0
**Date**: 2025-10-15
**Purpose**: Complete W&B tracking strategy for Phases 1-4

---

## Overview

This document specifies **exactly** what metrics, artifacts, and visualizations to log to Weights & Biases for Phases 1-4. The goal is to have complete observability into training, evolution, and compression.

### W&B Project Structure

```
agent-forge-v2/
├── phase1-cognate/
│   ├── model1-reasoning
│   ├── model2-memory
│   └── model3-speed
├── phase2-evomerge/
│   └── evolution-run-1
├── phase3-quietstar/
│   ├── step1-baking
│   └── step2-rl
└── phase4-bitnet/
    └── compression-run-1
```

---

## Phase 1: Cognate (3 Models)

### Run Configuration

```python
import wandb

# Initialize for each model
wandb.init(
    project="agent-forge-v2",
    name=f"phase1-cognate-model{model_id}",
    config={
        "phase": 1,
        "model_id": model_id,
        "architecture": "TRM-Titans-MAG",
        "target_params": "25M",
        "act_threshold": act_threshold,
        "ltm_slots": ltm_slots,
        "dataset_weights": dataset_weights,
        "optimizer": "Muon-Grokfast",
        "muon_lr": 1e-3,
        "grokfast_lambda": 0.3,
        "qk_clip": 30.0,
        "kl_coef": 0.0,
        "batch_size": batch_size,
        "num_epochs": 10,
        "curriculum_stages": 3
    },
    tags=["phase1", "cognate", f"model{model_id}", "pretraining"]
)
```

### Metrics to Log

#### Every Step (During Training)

```python
# Training metrics
wandb.log({
    "train/step": step,
    "train/epoch": epoch,
    "train/loss": loss.item(),
    "train/perplexity": torch.exp(loss).item(),
    "train/learning_rate": optimizer.param_groups[0]['lr'],
    "train/gradient_norm": grad_norm,

    # Muon × Grokfast specific
    "optimizer/muon_lr": muon_lr,
    "optimizer/grokfast_lambda": grokfast_lambda,
    "optimizer/slow_gradients_filtered": slow_grad_count,

    # ACT metrics
    "act/avg_halting_steps": avg_halting_steps,
    "act/max_halting_steps": max_halting_steps,
    "act/min_halting_steps": min_halting_steps,
    "act/halting_variance": halting_variance,

    # LTM metrics
    "ltm/memory_usage": memory_slots_used / total_ltm_slots,
    "ltm/write_operations": ltm_writes,
    "ltm/read_operations": ltm_reads,
    "ltm/avg_surprise_score": avg_surprise,

    # System metrics
    "system/gpu_memory_used_gb": torch.cuda.memory_allocated() / 1024**3,
    "system/gpu_memory_cached_gb": torch.cuda.memory_reserved() / 1024**3,
    "system/gpu_utilization": gpu_utilization_percent,
    "system/tokens_per_second": tokens_processed / elapsed_time,
})
```

#### Every Epoch

```python
# Validation metrics
wandb.log({
    "val/loss": val_loss,
    "val/perplexity": val_perplexity,
    "val/gsm8k_accuracy": gsm8k_acc,
    "val/arc_easy_accuracy": arc_acc,
    "val/piqa_accuracy": piqa_acc,

    # Curriculum stage tracking
    "curriculum/stage": current_stage,  # 1, 2, or 3
    "curriculum/dataset_mix": json.dumps(current_dataset_weights),

    # Time tracking
    "time/epoch_duration_minutes": epoch_time_minutes,
    "time/estimated_completion_hours": estimated_hours_remaining,
})
```

#### End of Training

```python
# Final metrics
wandb.log({
    "final/train_loss": final_train_loss,
    "final/val_loss": final_val_loss,
    "final/perplexity": final_perplexity,
    "final/gsm8k_accuracy": final_gsm8k_acc,
    "final/arc_accuracy": final_arc_acc,
    "final/piqa_accuracy": final_piqa_acc,
    "final/total_training_time_hours": total_time_hours,
    "final/model_size_mb": model_size_mb,
    "final/total_params": total_params,

    # Diversity metrics (for Phase 2)
    "diversity/avg_halting_steps": avg_halting,
    "diversity/ltm_usage": ltm_usage,
    "diversity/inference_time_ms": inference_time,
})

# Model diversity table (compare all 3 models)
diversity_table = wandb.Table(
    columns=["Model", "Halting Steps", "LTM Usage", "Inference Time (ms)", "GSM8K Acc"],
    data=[
        ["Model 1", 8.2, 0.35, 45, 0.12],
        ["Model 2", 9.5, 0.68, 52, 0.11],
        ["Model 3", 6.1, 0.21, 38, 0.13]
    ]
)
wandb.log({"diversity/model_comparison": diversity_table})
```

### Artifacts to Save

```python
# Save model checkpoint
artifact = wandb.Artifact(
    name=f"phase1-model{model_id}-checkpoint",
    type="model",
    description=f"Phase 1 Model {model_id} final checkpoint",
    metadata={
        "params": total_params,
        "size_mb": model_size_mb,
        "perplexity": final_perplexity,
        "gsm8k_acc": final_gsm8k_acc
    }
)
artifact.add_file(f"model{model_id}_final.pt")
wandb.log_artifact(artifact)

# Save config
artifact = wandb.Artifact(
    name=f"phase1-model{model_id}-config",
    type="config",
    description="Training configuration"
)
artifact.add_file(f"model{model_id}_config.yaml")
wandb.log_artifact(artifact)

# Save training curves
artifact = wandb.Artifact(
    name=f"phase1-model{model_id}-curves",
    type="plot",
    description="Training loss/accuracy curves"
)
# wandb automatically saves plots
```

### Visualizations

```python
# Loss curve
wandb.log({
    "charts/loss_curve": wandb.plot.line_series(
        xs=steps,
        ys=[train_losses, val_losses],
        keys=["Train Loss", "Val Loss"],
        title="Training vs Validation Loss",
        xname="Step"
    )
})

# ACT distribution
wandb.log({
    "charts/act_distribution": wandb.Histogram(halting_steps_all_samples)
})

# LTM heatmap
wandb.log({
    "charts/ltm_heatmap": wandb.Image(ltm_usage_heatmap)
})
```

---

## Phase 2: EvoMerge (Evolution)

### Run Configuration

```python
wandb.init(
    project="agent-forge-v2",
    name="phase2-evomerge",
    config={
        "phase": 2,
        "input_models": 3,
        "population_size": 8,
        "num_generations": 50,
        "mutation_rate": 0.01,
        "mutation_sigma": 0.01,
        "elite_count": 2,
        "fitness_weights": {
            "perplexity": 0.4,
            "accuracy": 0.3,
            "speed": 0.2,
            "memory": 0.1
        },
        "merge_techniques": ["Linear", "SLERP", "DARE", "TIES", "FrankenMerge", "DFS"],
        "early_stopping_patience": 5,
        "early_stopping_threshold": 0.001
    },
    tags=["phase2", "evomerge", "genetic-algorithm", "model-merging"]
)
```

### Metrics to Log

#### Every Generation

```python
# Population metrics
wandb.log({
    "evolution/generation": generation,

    # Fitness metrics
    "evolution/best_fitness": best_fitness,
    "evolution/avg_fitness": avg_fitness,
    "evolution/worst_fitness": worst_fitness,
    "evolution/fitness_std": fitness_std,
    "evolution/fitness_improvement": (best_fitness - initial_fitness) / initial_fitness,

    # Diversity metrics
    "evolution/diversity": population_diversity,  # Pairwise distance
    "evolution/unique_models": count_unique_models(population),

    # Combo usage
    "evolution/combo_000_count": combo_counts[0],
    "evolution/combo_001_count": combo_counts[1],
    "evolution/combo_010_count": combo_counts[2],
    "evolution/combo_011_count": combo_counts[3],
    "evolution/combo_100_count": combo_counts[4],
    "evolution/combo_101_count": combo_counts[5],
    "evolution/combo_110_count": combo_counts[6],
    "evolution/combo_111_count": combo_counts[7],

    # Elite metrics
    "evolution/elite1_fitness": elite1_fitness,
    "evolution/elite2_fitness": elite2_fitness,

    # Time tracking
    "time/generation_duration_seconds": generation_time,
    "time/estimated_completion_minutes": estimated_completion,
})
```

#### Every Model Evaluation (8 per generation)

```python
# Individual model metrics
wandb.log({
    f"models/model_{model_idx}/fitness": fitness,
    f"models/model_{model_idx}/perplexity": perplexity,
    f"models/model_{model_idx}/accuracy": accuracy,
    f"models/model_{model_idx}/inference_time": inference_time,
    f"models/model_{model_idx}/memory_usage": memory_usage,
    f"models/model_{model_idx}/combo_id": combo_id,  # 0-7
})
```

#### End of Evolution

```python
# Final metrics
wandb.log({
    "final/generations_completed": total_generations,
    "final/best_fitness": champion_fitness,
    "final/fitness_improvement": improvement_percent,
    "final/total_evolution_time_minutes": total_time_minutes,
    "final/champion_combo_id": champion_combo,

    # Combo effectiveness
    "final/best_combo": best_combo_id,
    "final/worst_combo": worst_combo_id,
    "final/combo_win_rate": combo_win_rates,  # Which combos produced champions

    # Convergence metrics
    "final/converged_at_generation": convergence_generation,
    "final/convergence_reason": convergence_reason,  # "max_gen" or "early_stop"
})

# Evolution history table
evolution_table = wandb.Table(
    columns=["Generation", "Best Fitness", "Avg Fitness", "Diversity", "Best Combo"],
    data=[[g, bf, af, d, bc] for g, bf, af, d, bc in evolution_history]
)
wandb.log({"evolution/history": evolution_table})

# Combo effectiveness table
combo_table = wandb.Table(
    columns=["Combo ID", "Binary", "Techniques", "Usage Count", "Avg Fitness", "Champions Produced"],
    data=[
        ["000", "000", "Linear + DARE + FrankenMerge", 12, 0.152, 2],
        ["001", "001", "Linear + DARE + DFS", 8, 0.148, 0],
        # ... (8 rows total)
    ]
)
wandb.log({"combos/effectiveness": combo_table})
```

### Artifacts to Save

```python
# Save champion model
artifact = wandb.Artifact(
    name="phase2-champion",
    type="model",
    description="Best evolved model after 50 generations",
    metadata={
        "fitness": champion_fitness,
        "improvement": improvement_percent,
        "combo_id": champion_combo,
        "generation": champion_generation
    }
)
artifact.add_file("champion_evolved.pt")
wandb.log_artifact(artifact)

# Save evolution log
artifact = wandb.Artifact(
    name="phase2-evolution-log",
    type="dataset",
    description="Complete evolution history"
)
artifact.add_file("evolution_log.json")
wandb.log_artifact(artifact)
```

### Visualizations

```python
# Fitness over generations
wandb.log({
    "charts/fitness_evolution": wandb.plot.line_series(
        xs=generations,
        ys=[best_fitness_per_gen, avg_fitness_per_gen, worst_fitness_per_gen],
        keys=["Best", "Average", "Worst"],
        title="Fitness Evolution",
        xname="Generation"
    )
})

# Diversity over generations
wandb.log({
    "charts/diversity_evolution": wandb.plot.line(
        diversity_per_generation_table,
        x="generation",
        y="diversity",
        title="Population Diversity Over Time"
    )
})

# Combo usage heatmap
combo_usage_heatmap = create_heatmap(combo_usage_matrix)  # 50 gens × 8 combos
wandb.log({
    "charts/combo_usage": wandb.Image(combo_usage_heatmap)
})
```

---

## Phase 3: Quiet-STaR (Two-Step Reasoning)

### Step 1: Prompt Baking

#### Run Configuration

```python
wandb.init(
    project="agent-forge-v2",
    name="phase3-step1-baking",
    config={
        "phase": 3,
        "step": 1,
        "method": "prompt-baking",
        "num_special_tokens": 12,
        "special_tokens": ["[thinking]", "[/endthinking]", "<step>", "<mece>", "<falsify>", "<expert>", "<orthogonal>", "<doubt>", "<bayesian>", "<multidomain>", "<correct>", "<uncertain>"],
        "num_strategies": 10,
        "training_examples": 25000,
        "examples_per_strategy": 2500,
        "optimizer": "Muon-Grokfast",
        "muon_lr": 1e-4,
        "grokfast_lambda": 0.2,
        "qk_clip": 30.0,
        "kl_coef": 0.0,
        "num_epochs": 5,
        "convergence_threshold": 0.85
    },
    tags=["phase3", "step1", "prompt-baking", "supervised"]
)
```

#### Metrics to Log

```python
# Every Step
wandb.log({
    "baking/step": step,
    "baking/epoch": epoch,
    "baking/loss": loss.item(),
    "baking/learning_rate": optimizer.param_groups[0]['lr'],

    # Strategy-specific accuracy
    "baking/accuracy_chain_of_thought": acc_cot,
    "baking/accuracy_mece": acc_mece,
    "baking/accuracy_falsification": acc_falsify,
    "baking/accuracy_expert": acc_expert,
    "baking/accuracy_orthogonal": acc_orthogonal,
    "baking/accuracy_doubt": acc_doubt,
    "baking/accuracy_bayesian": acc_bayesian,
    "baking/accuracy_multidomain": acc_multidomain,
    "baking/accuracy_correction": acc_correct,
    "baking/accuracy_uncertainty": acc_uncertain,
    "baking/accuracy_overall": acc_overall,

    # Token usage
    "baking/thinking_token_usage": thinking_token_usage,  # % of outputs using [thinking]
    "baking/strategy_token_usage": strategy_token_usage,  # % using strategy tokens

    # System metrics
    "system/gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3,
})

# Every Epoch
wandb.log({
    "baking/val_loss": val_loss,
    "baking/val_accuracy": val_accuracy,
    "baking/convergence_progress": val_accuracy / 0.85,  # % toward threshold
})

# End of Baking
wandb.log({
    "baking/final_accuracy": final_accuracy,
    "baking/converged": final_accuracy >= 0.85,
    "baking/total_epochs": total_epochs,
    "baking/training_time_hours": training_time_hours,
})

# Strategy accuracy table
strategy_table = wandb.Table(
    columns=["Strategy", "Accuracy", "Threshold", "Passed"],
    data=[
        ["Chain-of-Thought", 0.92, 0.90, "✅"],
        ["MECE", 0.87, 0.85, "✅"],
        ["Falsification", 0.81, 0.80, "✅"],
        ["Expert", 0.88, 0.85, "✅"],
        ["Orthogonal", 0.79, 0.75, "✅"],
        ["Self-Doubt", 0.91, 0.85, "✅"],
        ["Bayesian", 0.76, 0.75, "✅"],
        ["Multidomain", 0.84, 0.85, "❌"],  # Slightly below
        ["Self-Correction", 0.86, 0.85, "✅"],
        ["Uncertainty", 0.89, 0.85, "✅"]
    ]
)
wandb.log({"baking/strategy_accuracy_table": strategy_table})
```

### Step 2: Quiet-STaR (RL)

#### Run Configuration

```python
wandb.init(
    project="agent-forge-v2",
    name="phase3-step2-rl",
    config={
        "phase": 3,
        "step": 2,
        "method": "quiet-star-rl",
        "num_thoughts": 4,
        "thought_length": 12,
        "optimizer": "Muon-Grokfast",
        "muon_lr": 5e-4,
        "grokfast_lambda": 0.1,
        "qk_clip": 25.0,
        "kl_coef": 0.1,
        "num_episodes": 10000,
        "reward_type": "reinforce",
        "coherence_weights": {"semantic": 0.4, "syntactic": 0.3, "predictive": 0.3}
    },
    tags=["phase3", "step2", "quiet-star", "rl", "reinforce"]
)
```

#### Metrics to Log

```python
# Every Episode
wandb.log({
    "rl/episode": episode,
    "rl/reward": episode_reward,
    "rl/avg_reward": avg_reward_last_100,

    # Thought quality
    "rl/avg_coherence_score": avg_coherence,
    "rl/semantic_coherence": semantic_coherence,
    "rl/syntactic_coherence": syntactic_coherence,
    "rl/predictive_coherence": predictive_coherence,

    # Thought characteristics
    "rl/avg_thought_length": avg_thought_length,
    "rl/thought_diversity": thought_diversity,  # Pairwise distance

    # Anti-theater metrics
    "rl/divergence_from_direct": divergence,  # >0.3 good
    "rl/ablation_accuracy_drop": ablation_drop,  # >0.02 good

    # KL regularization
    "rl/kl_divergence": kl_div,  # Keep low (prevent drift)

    # Accuracy
    "rl/gsm8k_accuracy": gsm8k_acc,
    "rl/arc_accuracy": arc_acc,

    # Time
    "rl/inference_time_ms": inference_time,  # With thoughts
})

# Every 1000 Episodes (Anti-Theater Check)
if episode % 1000 == 0:
    theater_results = anti_theater_check(model, test_set)
    wandb.log({
        "anti_theater/divergence": theater_results["divergence"],
        "anti_theater/divergence_passed": theater_results["divergence"] > 0.3,
        "anti_theater/ablation_drop": theater_results["ablation_drop"],
        "anti_theater/ablation_passed": theater_results["ablation_drop"] > 0.02,
        "anti_theater/correlation": theater_results["correlation"],
        "anti_theater/correlation_passed": theater_results["correlation"] > 0.5,
        "anti_theater/all_passed": theater_results["all_passed"],
    })

# End of RL Training
wandb.log({
    "rl/final_reward": final_reward,
    "rl/final_coherence": final_coherence,
    "rl/final_gsm8k_accuracy": final_gsm8k_acc,
    "rl/improvement_over_baked": final_acc - baked_acc,
    "rl/total_training_time_hours": training_time_hours,
})
```

### Artifacts to Save

```python
# Baked model (after Step 1)
artifact = wandb.Artifact(
    name="phase3-baked-model",
    type="model",
    description="Model after prompt baking (Step 1)",
    metadata={"accuracy": final_baking_accuracy}
)
artifact.add_file("baked_model.pt")
wandb.log_artifact(artifact)

# Final reasoning model (after Step 2)
artifact = wandb.Artifact(
    name="phase3-reasoning-model",
    type="model",
    description="Model after Quiet-STaR RL (Step 2)",
    metadata={
        "gsm8k_acc": final_gsm8k_acc,
        "coherence": final_coherence,
        "inference_time_ms": final_inference_time
    }
)
artifact.add_file("reasoning_enhanced.pt")
wandb.log_artifact(artifact)

# Reasoning dataset
artifact = wandb.Artifact(
    name="phase3-reasoning-dataset",
    type="dataset",
    description="25K reasoning examples from frontier models"
)
artifact.add_file("reasoning_25k.json")
wandb.log_artifact(artifact)
```

### Visualizations

```python
# Baking convergence
wandb.log({
    "charts/baking_convergence": wandb.plot.line_series(
        xs=epochs,
        ys=[train_acc_per_epoch, val_acc_per_epoch],
        keys=["Train Accuracy", "Val Accuracy"],
        title="Prompt Baking Convergence"
    )
})

# RL reward curve
wandb.log({
    "charts/rl_reward": wandb.plot.line(
        reward_table,
        x="episode",
        y="reward",
        title="RL Training Reward"
    )
})

# Thought coherence distribution
wandb.log({
    "charts/coherence_distribution": wandb.Histogram(coherence_scores_all_episodes)
})

# Anti-theater trend
wandb.log({
    "charts/anti_theater_trend": wandb.plot.line_series(
        xs=checkpoints,
        ys=[divergence_over_time, ablation_over_time, correlation_over_time],
        keys=["Divergence", "Ablation Drop", "Correlation"],
        title="Anti-Theater Metrics Over Training"
    )
})
```

---

## Phase 4: BitNet (Compression)

### Run Configuration

```python
wandb.init(
    project="agent-forge-v2",
    name="phase4-bitnet-compression",
    config={
        "phase": 4,
        "method": "bitnet-1.58bit",
        "target_compression": 8.0,
        "sparsity_threshold": 0.1,
        "preserved_layers": ["embeddings", "lm_head"],
        "calibration_samples": 1000,
        "fine_tune_epochs": 2,
        "fine_tune_lr": 1e-5
    },
    tags=["phase4", "bitnet", "compression", "quantization"]
)
```

### Metrics to Log

#### During Calibration

```python
wandb.log({
    "calibration/sample": sample_idx,
    "calibration/layer_activations": layer_activation_stats,  # Per layer
    "calibration/scale_factors": scale_factors_per_layer,
})
```

#### During Compression

```python
# Per layer
wandb.log({
    "compression/layer": layer_idx,
    "compression/layer_name": layer_name,
    "compression/original_params": original_params,
    "compression/quantized_params": quantized_params,
    "compression/preserved_params": preserved_params,
    "compression/sparsity": sparsity,  # % of zeros
    "compression/compression_ratio": layer_compression_ratio,
})

# Overall progress
wandb.log({
    "compression/layers_compressed": layers_compressed,
    "compression/total_layers": total_layers,
    "compression/progress": layers_compressed / total_layers,
})
```

#### Post-Compression Evaluation

```python
wandb.log({
    "evaluation/accuracy_before": accuracy_before,
    "evaluation/accuracy_after": accuracy_after,
    "evaluation/accuracy_drop": accuracy_drop,
    "evaluation/accuracy_drop_percent": accuracy_drop_percent,

    "evaluation/perplexity_before": perplexity_before,
    "evaluation/perplexity_after": perplexity_after,
    "evaluation/perplexity_increase": perplexity_increase,

    "evaluation/inference_time_before_ms": inference_time_before,
    "evaluation/inference_time_after_ms": inference_time_after,
    "evaluation/inference_speedup": inference_speedup,

    "evaluation/model_size_before_mb": size_before,
    "evaluation/model_size_after_mb": size_after,
    "evaluation/compression_ratio": compression_ratio,

    "evaluation/sparsity_overall": overall_sparsity,
    "evaluation/zero_params": zero_params,
    "evaluation/nonzero_params": nonzero_params,
})
```

#### During Fine-Tuning (if needed)

```python
wandb.log({
    "finetune/epoch": epoch,
    "finetune/loss": loss,
    "finetune/accuracy": accuracy,
    "finetune/accuracy_recovery": accuracy - accuracy_after_compression,
})
```

#### Final Metrics

```python
wandb.log({
    "final/compression_ratio": final_compression,
    "final/accuracy_drop": final_accuracy_drop,
    "final/inference_speedup": final_speedup,
    "final/sparsity": final_sparsity,
    "final/total_time_hours": total_time_hours,
    "final/fine_tuning_needed": fine_tuning_needed,
})

# Compression breakdown table
compression_table = wandb.Table(
    columns=["Layer Type", "Params", "Quantized", "Preserved", "Compression"],
    data=[
        ["Attention", 15_000_000, 14_500_000, 500_000, 8.2],
        ["FFN", 8_000_000, 7_800_000, 200_000, 8.5],
        ["Embeddings", 1_500_000, 0, 1_500_000, 2.0],
        ["LM Head", 1_500_000, 0, 1_500_000, 2.0],
        ["Layer Norm", 24_672, 0, 24_672, 1.0]
    ]
)
wandb.log({"compression/breakdown": compression_table})
```

### Artifacts to Save

```python
# Compressed model
artifact = wandb.Artifact(
    name="phase4-compressed-model",
    type="model",
    description="BitNet 1.58-bit compressed model",
    metadata={
        "compression_ratio": compression_ratio,
        "accuracy_drop": accuracy_drop,
        "size_mb": compressed_size_mb
    }
)
artifact.add_file("compressed_1.58bit.pt")
wandb.log_artifact(artifact)

# Compression config
artifact = wandb.Artifact(
    name="phase4-compression-config",
    type="config",
    description="Quantization configuration and scale factors"
)
artifact.add_file("compression_config.json")
wandb.log_artifact(artifact)
```

### Visualizations

```python
# Compression ratio by layer
wandb.log({
    "charts/compression_by_layer": wandb.plot.bar(
        compression_by_layer_table,
        label="Layer",
        value="Compression Ratio",
        title="Compression Ratio by Layer"
    )
})

# Sparsity distribution
wandb.log({
    "charts/sparsity_distribution": wandb.Histogram(sparsity_per_layer)
})

# Accuracy recovery during fine-tuning
wandb.log({
    "charts/accuracy_recovery": wandb.plot.line(
        accuracy_recovery_table,
        x="epoch",
        y="accuracy",
        title="Accuracy Recovery During Fine-Tuning"
    )
})
```

---

## Cross-Phase Dashboards

### Dashboard 1: Phase Overview

```
+---------------------------+
| Phase 1: Cognate          |
| Status: ✅ Complete        |
| 3 models, 22.5 hrs        |
| Diversity: 0.45           |
+---------------------------+
| Phase 2: EvoMerge         |
| Status: ✅ Complete        |
| 50 gens, 90 min           |
| Improvement: 23.5%        |
+---------------------------+
| Phase 3: Quiet-STaR       |
| Status: 🔄 In Progress     |
| Step 2 (RL): 45%          |
| Accuracy: 0.17 → 0.18     |
+---------------------------+
| Phase 4: BitNet           |
| Status: ⏳ Pending         |
|                           |
+---------------------------+
```

### Dashboard 2: Model Performance Tracking

```
Metric Trends Across Phases:
- Perplexity: 15.4 (P1) → 13.2 (P2) → 12.8 (P3) → 13.5 (P4)
- GSM8K Acc:  0.12 (P1) → 0.15 (P2) → 0.18 (P3) → 0.17 (P4)
- ARC Acc:    0.28 (P1) → 0.31 (P2) → 0.34 (P3) → 0.32 (P4)
- Model Size: 1700 MB (P1) → 1700 MB (P2) → 1800 MB (P3) → 310 MB (P4)
- Inference:  45 ms (P1) → 42 ms (P2) → 185 ms (P3) → 71 ms (P4)
```

### Dashboard 3: Resource Usage

```
Resource Utilization:
- Total GPU Hours: 35.2 hrs
- Storage Used: 8.3 GB
- W&B Storage: 1.2 GB
- Peak VRAM: 5.8 GB
- API Cost (Phase 3): $127.50
```

---

## Summary: Key Metrics by Phase

| Phase | Run Name | Key Metrics | Critical Thresholds |
|-------|----------|-------------|---------------------|
| **Phase 1** | `phase1-cognate-model{1-3}` | perplexity, gsm8k_acc, act_halting, ltm_usage | diversity >0.3 |
| **Phase 2** | `phase2-evomerge` | best_fitness, diversity, improvement | improvement >20% |
| **Phase 3 Step 1** | `phase3-step1-baking` | accuracy, strategy_accuracy, token_usage | accuracy ≥0.85 |
| **Phase 3 Step 2** | `phase3-step2-rl` | reward, coherence, divergence, ablation | divergence >0.3, ablation >0.02 |
| **Phase 4** | `phase4-bitnet-compression` | compression_ratio, accuracy_drop, sparsity | compression >6x, drop <10% |

---

**Version**: 2.0
**Date**: 2025-10-15
**Status**: ✅ Complete - Ready for Implementation

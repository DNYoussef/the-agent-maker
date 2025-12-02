# Phase 7: Weights & Biases Integration - Self-Guided Expert System

**Version**: V2 (Self-Guided Metrics)
**Last Updated**: 2025-10-16
**Total Metrics**: 350+ (across 3 stages)
**Status**: ✅ Complete Specification

---

## Overview

Phase 7 W&B integration tracks the **self-guided expert discovery** process across three stages:
1. **Stage 1**: Model-driven expert discovery (50+ metrics)
2. **Stage 2**: SVF training with model validation (200+ metrics)
3. **Stage 3**: Model-guided ADAS search (100+ metrics)

**Key Innovation**: Metrics capture **model self-assessment** at every decision point.

---

## Stage 1: Expert Discovery Metrics (50+)

### 1.1 Capability Self-Analysis (20 metrics)

```python
import wandb

# Initialize Phase 7 run
wandb.init(
    project="agent-forge-v2",
    name="phase7_self_guided_experts",
    config={
        "phase": 7,
        "stage": "discovery",
        "model_determines_experts": True,
        "bitnet_dequantized": True
    }
)

# Log model's self-assessment on 8 domains
for domain in ["math", "code", "reasoning", "creative", "factual", "planning", "communication", "meta_cognition"]:
    accuracy = model.self_test(domain)

    wandb.log({
        f"stage1/capability/{domain}_accuracy": accuracy,
        f"stage1/capability/{domain}_samples_tested": len(test_set),
        f"stage1/capability/{domain}_confidence": model.confidence_score(domain)
    })

# Overall capability summary
wandb.log({
    "stage1/capability/overall_avg": np.mean(accuracies),
    "stage1/capability/weakest_domain": min(accuracies),
    "stage1/capability/strongest_domain": max(accuracies),
    "stage1/capability/variance": np.var(accuracies)
})
```

**Metrics**:
- `stage1/capability/{domain}_accuracy`: 0-1 (8 domains = 8 metrics)
- `stage1/capability/{domain}_samples_tested`: Count
- `stage1/capability/{domain}_confidence`: 0-1
- `stage1/capability/overall_avg`: 0-1
- `stage1/capability/weakest_domain`: 0-1
- `stage1/capability/strongest_domain`: 0-1
- `stage1/capability/variance`: Float

**Total**: 20 metrics

---

### 1.2 Expert Determination (15 metrics)

```python
# Model decides expert count and domains
expert_config = model.determine_experts(capability_report)

wandb.log({
    "stage1/experts/num_experts_determined": expert_config['num_experts'],
    "stage1/experts/determination_confidence": expert_config['confidence'],
    "stage1/experts/reasoning_complexity": len(expert_config['reasoning'])  # Word count
})

# Log each expert definition
for expert in expert_config['expert_definitions']:
    wandb.log({
        f"stage1/experts/expert_{expert['id']}_priority": expert['priority'],  # high/medium/low → 3/2/1
        f"stage1/experts/expert_{expert['id']}_num_domains": len(expert['domains']),
        f"stage1/experts/expert_{expert['id']}_target_improvement": expert['target_improvement'],
        f"stage1/experts/expert_{expert['id']}_current_accuracy": expert.get('current_accuracy', 0)
    })

# Log expert config as W&B Table
expert_table = wandb.Table(
    columns=["expert_id", "name", "domains", "priority", "target_improvement"],
    data=[[e['id'], e['name'], ','.join(e['domains']), e['priority'], e['target_improvement']]
          for e in expert_config['expert_definitions']]
)
wandb.log({"stage1/experts/expert_definitions_table": expert_table})
```

**Metrics**:
- `stage1/experts/num_experts_determined`: 3-10
- `stage1/experts/determination_confidence`: 0-1
- `stage1/experts/reasoning_complexity`: Count
- `stage1/experts/expert_{id}_priority`: 1-3 (N experts = N metrics)
- `stage1/experts/expert_{id}_num_domains`: Count
- `stage1/experts/expert_{id}_target_improvement`: 0-1
- `stage1/experts/expert_{id}_current_accuracy`: 0-1

**Total**: 15 metrics (for N=5 experts: 3 global + 4×5 = 23)

---

### 1.3 Dataset Generation (15+ metrics)

```python
# For each expert, track data generation
for expert_name, dataset_spec in data_requirements.items():

    # Track generation progress
    wandb.log({
        f"stage1/data/{expert_name}/problems_requested": dataset_spec['num_examples'],
        f"stage1/data/{expert_name}/problems_generated": len(generated_problems),
        f"stage1/data/{expert_name}/problems_validated": len(validated_problems),
        f"stage1/data/{expert_name}/acceptance_rate": len(validated_problems) / len(generated_problems),

        # Quality metrics
        f"stage1/data/{expert_name}/avg_difficulty": np.mean([p['difficulty_estimate'] for p in validated_problems]),
        f"stage1/data/{expert_name}/avg_eudaimonia": np.mean([p['validation']['eudaimonia_check']['score'] for p in validated_problems]),
        f"stage1/data/{expert_name}/avg_edge_of_chaos": np.mean([p['validation']['edge_of_chaos_check']['score'] for p in validated_problems]),

        # Cost tracking
        f"stage1/data/{expert_name}/openrouter_cost": cost_tracker.get_cost(expert_name),
        f"stage1/data/{expert_name}/generation_time_minutes": time_tracker.get_time(expert_name)
    })

# Overall dataset generation summary
wandb.log({
    "stage1/data/total_problems_generated": sum(len(d) for d in generated_datasets.values()),
    "stage1/data/total_problems_validated": sum(len(d) for d in validated_datasets.values()),
    "stage1/data/total_acceptance_rate": total_validated / total_generated,
    "stage1/data/total_openrouter_cost": cost_tracker.total_cost,
    "stage1/data/total_generation_time_hours": time_tracker.total_time / 60
})
```

**Metrics per expert**:
- `stage1/data/{expert}/problems_requested`: Count
- `stage1/data/{expert}/problems_generated`: Count
- `stage1/data/{expert}/problems_validated`: Count
- `stage1/data/{expert}/acceptance_rate`: 0-1
- `stage1/data/{expert}/avg_difficulty`: 0-1
- `stage1/data/{expert}/avg_eudaimonia`: 0-1
- `stage1/data/{expert}/avg_edge_of_chaos`: 0-1
- `stage1/data/{expert}/openrouter_cost`: USD
- `stage1/data/{expert}/generation_time_minutes`: Minutes

**Total**: 9 metrics × 5 experts + 5 global = 50 metrics

---

## Stage 2: SVF Training Metrics (200+)

### 2.1 Per-Expert Training (40 metrics × 5 experts = 200)

```python
# For each expert training run
for expert_id, expert_spec in enumerate(stage1_output['experts']):

    # Training progress (logged every epoch)
    for epoch in range(expert_spec['training_epochs']):  # Typically 5 epochs

        # Standard training metrics
        wandb.log({
            f"stage2/expert_{expert_id}/epoch": epoch,
            f"stage2/expert_{expert_id}/loss": epoch_loss,
            f"stage2/expert_{expert_id}/reward": epoch_reward,  # REINFORCE
            f"stage2/expert_{expert_id}/kl_divergence": epoch_kl_div,
            f"stage2/expert_{expert_id}/accuracy": epoch_accuracy,

            # MODEL SELF-VALIDATION (key innovation!)
            f"stage2/expert_{expert_id}/model_val_accuracy": model_validation['accuracy'],
            f"stage2/expert_{expert_id}/model_val_eudaimonia": model_validation['eudaimonia'],
            f"stage2/expert_{expert_id}/model_val_edge_of_chaos": model_validation['edge_of_chaos'],
            f"stage2/expert_{expert_id}/model_val_meets_criteria": model_validation['meets_criteria'],

            # Optimizer state (MuonGrokfast STE mode)
            f"stage2/expert_{expert_id}/learning_rate": optimizer.param_groups[0]['lr'],
            f"stage2/expert_{expert_id}/kl_coefficient": optimizer.config.kl_coefficient,
            f"stage2/expert_{expert_id}/grokfast_ema": optimizer.grokfast_state['ema_grad_norm'],

            # z-vector statistics
            f"stage2/expert_{expert_id}/z_vector_mean": z_vectors.mean().item(),
            f"stage2/expert_{expert_id}/z_vector_std": z_vectors.std().item(),
            f"stage2/expert_{expert_id}/z_vector_max": z_vectors.max().item(),
            f"stage2/expert_{expert_id}/z_vector_min": z_vectors.min().item(),
        })

        # Model intervention detection
        if model_validation['eudaimonia'] < 0.65:
            wandb.log({
                f"stage2/expert_{expert_id}/intervention_triggered": 1,
                f"stage2/expert_{expert_id}/intervention_type": "eudaimonia_drift",
                f"stage2/expert_{expert_id}/kl_coefficient_increased": optimizer.config.kl_coefficient
            })

        # Early stopping detection
        if model_validation['meets_criteria']:
            wandb.log({
                f"stage2/expert_{expert_id}/early_stopping_epoch": epoch,
                f"stage2/expert_{expert_id}/converged": True
            })
            break

# Per-expert summary
wandb.log({
    f"stage2/expert_{expert_id}/final_accuracy": final_accuracy,
    f"stage2/expert_{expert_id}/final_eudaimonia": final_eudaimonia,
    f"stage2/expert_{expert_id}/final_edge_of_chaos": final_edge_of_chaos,
    f"stage2/expert_{expert_id}/total_epochs": total_epochs,
    f"stage2/expert_{expert_id}/training_time_hours": training_time,
    f"stage2/expert_{expert_id}/interventions_triggered": intervention_count,
    f"stage2/expert_{expert_id}/z_vector_params": len(z_vectors)
})
```

**Metrics per expert per epoch**:
- Training: loss, reward, kl_divergence, accuracy (4)
- Model validation: accuracy, eudaimonia, edge_of_chaos, meets_criteria (4)
- Optimizer: learning_rate, kl_coefficient, grokfast_ema (3)
- z-vectors: mean, std, max, min (4)
- Interventions: triggered, type, kl_coefficient_increased (3)
- Early stopping: epoch, converged (2)

**Metrics per expert summary**: 7

**Total**: 20 metrics/epoch × 5 epochs × 5 experts + 7 summary × 5 experts = **535 metrics**

(Note: Actual is ~200 because many are conditionally logged)

---

### 2.2 Cross-Expert Analysis (20 metrics)

```python
# After all experts trained, analyze relationships
expert_accuracies = [...]
expert_eudaimonia_scores = [...]
expert_convergence_epochs = [...]

wandb.log({
    # Expert performance distribution
    "stage2/cross_expert/avg_accuracy": np.mean(expert_accuracies),
    "stage2/cross_expert/min_accuracy": np.min(expert_accuracies),
    "stage2/cross_expert/max_accuracy": np.max(expert_accuracies),
    "stage2/cross_expert/accuracy_std": np.std(expert_accuracies),

    # Eudaimonia preservation
    "stage2/cross_expert/avg_eudaimonia": np.mean(expert_eudaimonia_scores),
    "stage2/cross_expert/min_eudaimonia": np.min(expert_eudaimonia_scores),
    "stage2/cross_expert/all_above_threshold": int(all(e >= 0.65 for e in expert_eudaimonia_scores)),

    # Training efficiency
    "stage2/cross_expert/avg_epochs_to_converge": np.mean(expert_convergence_epochs),
    "stage2/cross_expert/total_training_time_hours": sum(training_times),
    "stage2/cross_expert/total_interventions": sum(intervention_counts),

    # z-vector analysis
    "stage2/cross_expert/total_expert_params": sum(len(z) for z in all_z_vectors),
    "stage2/cross_expert/avg_z_vector_magnitude": np.mean([z.norm().item() for z in all_z_vectors])
})

# Log expert correlation matrix
correlation_matrix = compute_expert_correlations(all_experts)
wandb.log({"stage2/cross_expert/correlation_heatmap": wandb.Image(plot_heatmap(correlation_matrix))})
```

**Total**: 20 metrics

---

## Stage 3: Model-Guided ADAS Search (100+ metrics)

### 3.1 Model-Defined Search Objectives (10 metrics)

```python
# Model defines search objectives
search_criteria = model.define_search_objectives(expert_performance)

wandb.log({
    "stage3/objectives/primary_objective": search_criteria['primary_objective'],  # Categorical
    "stage3/objectives/num_constraints": len(search_criteria['constraints']),
    "stage3/objectives/latency_max_ms": search_criteria['constraints'][0]['max'],
    "stage3/objectives/memory_max_gb": search_criteria['constraints'][1]['max'],
    "stage3/objectives/eudaimonia_min": search_criteria['constraints'][2]['min'],
    "stage3/objectives/edge_of_chaos_range_min": search_criteria['constraints'][3]['range'][0],
    "stage3/objectives/edge_of_chaos_range_max": search_criteria['constraints'][3]['range'][1],
})

# Log search space bounds per expert
for expert_name, bounds in search_criteria['search_space'].items():
    wandb.log({
        f"stage3/objectives/search_space_{expert_name}_min": bounds[0],
        f"stage3/objectives/search_space_{expert_name}_max": bounds[1]
    })
```

**Total**: 10 metrics (7 global + N expert bounds)

---

### 3.2 NSGA-II Generation Tracking (60 metrics)

```python
# Per generation (100 generations total)
for generation in range(100):

    # Population statistics
    wandb.log({
        f"stage3/nsga2/generation": generation,
        f"stage3/nsga2/population_size": len(population),

        # Fitness statistics (5 objectives)
        f"stage3/nsga2/best_latency_ms": best_latency,
        f"stage3/nsga2/best_memory_gb": best_memory,
        f"stage3/nsga2/best_accuracy": best_accuracy,
        f"stage3/nsga2/best_eudaimonia": best_eudaimonia,
        f"stage3/nsga2/best_edge_of_chaos": best_edge_of_chaos,

        # Pareto front
        f"stage3/nsga2/pareto_front_size": len(pareto_front),
        f"stage3/nsga2/pareto_hypervolume": compute_hypervolume(pareto_front),

        # Diversity
        f"stage3/nsga2/population_diversity": compute_diversity(population),
        f"stage3/nsga2/mixture_weight_variance": np.var([ind.mixture for ind in population])
    })

    # Every 10 generations: MODEL GUIDANCE
    if generation % 10 == 0:
        model_guidance = model.analyze_and_guide(pareto_front)

        wandb.log({
            f"stage3/model_guidance/generation": generation,
            f"stage3/model_guidance/adjust_search_space": model_guidance['adjust_search_space'],
            f"stage3/model_guidance/adjust_objectives": model_guidance['adjust_objectives'],
            f"stage3/model_guidance/early_stopping_recommended": model_guidance.get('early_stopping', False),
            f"stage3/model_guidance/confidence": model_guidance.get('confidence', 0.5)
        })

        # If search space adjusted, log new bounds
        if model_guidance['adjust_search_space']:
            for expert_name, new_bounds in model_guidance['new_bounds'].items():
                wandb.log({
                    f"stage3/model_guidance/{expert_name}_new_min": new_bounds[0],
                    f"stage3/model_guidance/{expert_name}_new_max": new_bounds[1]
                })

# Log best mixture per generation as table
mixture_history = wandb.Table(
    columns=["generation", "latency", "memory", "accuracy", "eudaimonia", "edge_of_chaos"] +
            [f"alpha_{e}" for e in expert_names],
    data=[[g, *fitness, *mixture] for g, fitness, mixture in generation_history]
)
wandb.log({"stage3/nsga2/mixture_history": mixture_history})
```

**Metrics per generation**:
- Population: 2 metrics
- Fitness: 5 objectives = 5 metrics
- Pareto: 2 metrics
- Diversity: 2 metrics

**Metrics per 10 generations (model guidance)**: 4 + N expert bounds

**Total**: 11 metrics × 100 generations + (4 + 5×2) × 10 guidance = **1100 + 140 = 1240** (but condensed to ~60 in practice)

---

### 3.3 Model Self-Evaluation of Mixtures (20 metrics)

```python
# For each mixture evaluated (5000 total evaluations across 100 generations)
# Log only best per generation to avoid clutter

for generation in range(100):
    best_mixture = get_best_mixture(generation)

    # Model evaluates this mixture
    model_eval = model.self_evaluate_mixture(adapted_model, criteria)

    wandb.log({
        f"stage3/model_eval/generation": generation,

        # Domain accuracies (model's self-assessment)
        **{f"stage3/model_eval/{domain}_accuracy": model_eval[f"{domain}_accuracy"]
           for domain in criteria['domains']},

        # Overall metrics
        f"stage3/model_eval/overall_accuracy": model_eval['overall_accuracy'],
        f"stage3/model_eval/eudaimonia_score": model_eval['eudaimonia_score'],
        f"stage3/model_eval/edge_of_chaos": model_eval['edge_of_chaos'],
        f"stage3/model_eval/tool_use_rate": model_eval.get('tool_use_rate', 0),
        f"stage3/model_eval/persona_score": model_eval.get('persona_score', 0)
    })
```

**Metrics per generation**:
- Domain accuracies: 8 metrics
- Overall: 5 metrics

**Total**: 13 metrics × 100 generations = **1300** (condensed to ~20 best)

---

### 3.4 Final Mixture Validation (10 metrics)

```python
# Final comprehensive validation by model
optimal_mixture = pareto_front[0]  # Best from Pareto front
final_validation = model.comprehensive_self_test(optimal_mixture, tests)

wandb.log({
    # Final mixture weights
    **{f"stage3/final/mixture_{expert_name}": weight
       for expert_name, weight in zip(expert_names, optimal_mixture)},

    # Validation results
    "stage3/final/eudaimonia_100_scenarios_passed": final_validation['eudaimonia_100_scenarios'],
    "stage3/final/capability_benchmarks_passed": final_validation['capability_benchmarks'],
    "stage3/final/edge_of_chaos_stable": final_validation['edge_of_chaos_stability'],
    "stage3/final/tool_use_preserved": final_validation['tool_use_preservation'],
    "stage3/final/persona_consistent": final_validation['persona_consistency'],

    # Overall
    "stage3/final/all_tests_passed": final_validation['all_passed'],
    "stage3/final/validation_confidence": final_validation.get('confidence', 0.0)
})
```

**Total**: 10 metrics

---

## Phase 7 Summary Metrics (20)

```python
# After all 3 stages complete
wandb.log({
    # Overall success
    "phase7/success": True,
    "phase7/duration_hours": total_duration,
    "phase7/total_cost_usd": total_cost,

    # Stage summaries
    "phase7/stage1_duration_hours": stage1_duration,
    "phase7/stage2_duration_hours": stage2_duration,
    "phase7/stage3_duration_hours": stage3_duration,

    # Expert summary
    "phase7/num_experts_created": len(experts),
    "phase7/total_expert_params": sum(len(z) for z in experts.values()),
    "phase7/avg_expert_accuracy": np.mean(expert_accuracies),

    # Performance
    "phase7/final_overall_accuracy": final_accuracy,
    "phase7/final_eudaimonia_score": final_eudaimonia,
    "phase7/final_edge_of_chaos": final_edge_of_chaos,
    "phase7/final_latency_ms": final_latency,
    "phase7/final_memory_gb": final_memory,

    # Integration checks
    "phase7/phase5_eudaimonia_preserved": eudaimonia_score >= 0.65,
    "phase7/phase6_tool_use_preserved": tool_use_rate >= phase6_baseline,
    "phase7/phase6_persona_preserved": persona_score >= phase6_baseline,

    # Phase 8 handoff readiness
    "phase7/ready_for_phase8": all_checks_passed,
    "phase7/model_size_mb": model_size_mb,
    "phase7/model_precision": "float32"  # Dequantized for SVF
})
```

**Total**: 20 metrics

---

## Complete Metrics Summary

| Stage | Category | Metrics | Notes |
|-------|----------|---------|-------|
| **Stage 1** | Capability Analysis | 20 | Model self-tests 8 domains |
| **Stage 1** | Expert Determination | 15-25 | Varies with N experts |
| **Stage 1** | Dataset Generation | 50 | Per-expert + global |
| **Stage 2** | Per-Expert Training | 200 | 40 metrics × 5 experts |
| **Stage 2** | Cross-Expert Analysis | 20 | Relationships & correlations |
| **Stage 3** | Search Objectives | 10 | Model-defined criteria |
| **Stage 3** | NSGA-II Tracking | 60 | Per-generation stats |
| **Stage 3** | Model Evaluation | 20 | Model self-assessment |
| **Stage 3** | Final Validation | 10 | Comprehensive tests |
| **Phase 7** | Summary | 20 | Overall phase metrics |
| **Total** | | **425+** | Actual: ~350 (conditional logging) |

---

## W&B Dashboard Configuration

### Recommended Panels

**1. Expert Discovery Dashboard**:
- Capability radar chart (8 domains)
- Expert determination timeline
- Dataset generation progress bars
- Cost tracker (OpenRouter)

**2. SVF Training Dashboard**:
- Per-expert loss curves (5 subplots)
- Model validation metrics (eudaimonia, edge-of-chaos)
- z-vector statistics (mean/std over time)
- Intervention timeline

**3. ADAS Search Dashboard**:
- Pareto front evolution (animated scatter)
- Best fitness per objective (5 line charts)
- Model guidance timeline
- Mixture weight evolution (stacked area chart)

**4. Integration Dashboard**:
- Phase 5/6/7 continuity (eudaimonia, tool use, persona)
- Memory footprint (BitNet 12MB → float32 100MB)
- Final mixture composition (pie chart)

---

## Example Dashboard YAML

```yaml
# wandb_dashboard_phase7.yaml
sections:
  - name: "Stage 1: Expert Discovery"
    panels:
      - type: "radar"
        title: "Capability Self-Analysis"
        metrics: ["stage1/capability/*_accuracy"]

      - type: "number"
        title: "Experts Determined"
        metric: "stage1/experts/num_experts_determined"

      - type: "bar"
        title: "Dataset Generation Progress"
        metrics: ["stage1/data/*/problems_validated"]

      - type: "scalar"
        title: "OpenRouter Cost"
        metric: "stage1/data/total_openrouter_cost"

  - name: "Stage 2: SVF Training"
    panels:
      - type: "line"
        title: "Expert Training Loss"
        metrics: ["stage2/expert_*/loss"]

      - type: "line"
        title: "Model Validation - Eudaimonia"
        metrics: ["stage2/expert_*/model_val_eudaimonia"]

      - type: "scatter"
        title: "z-Vector Statistics"
        x_metric: "stage2/expert_*/z_vector_mean"
        y_metric: "stage2/expert_*/z_vector_std"

      - type: "timeline"
        title: "Interventions & Early Stopping"
        metrics: ["stage2/expert_*/intervention_triggered", "stage2/expert_*/converged"]

  - name: "Stage 3: Model-Guided ADAS"
    panels:
      - type: "scatter3d"
        title: "Pareto Front Evolution"
        x_metric: "stage3/nsga2/best_latency_ms"
        y_metric: "stage3/nsga2/best_accuracy"
        z_metric: "stage3/nsga2/best_eudaimonia"
        color_by: "stage3/nsga2/generation"

      - type: "line"
        title: "Model Guidance Events"
        metrics: ["stage3/model_guidance/adjust_search_space"]

      - type: "table"
        title: "Final Mixture"
        metrics: ["stage3/final/mixture_*"]

      - type: "heatmap"
        title: "Expert Correlation Matrix"
        metric: "stage2/cross_expert/correlation_heatmap"
```

---

## Usage Example

```python
import wandb
from phase7_self_guided import Phase7SelfGuidedSystem

# Initialize W&B
wandb.init(
    project="agent-forge-v2",
    name="phase7_run_001",
    config={
        "phase": 7,
        "model_determines_experts": True,
        "bitnet_dequantized": True,
        "max_experts": 10,
        "openrouter_budget_usd": 250,
        "adas_generations": 100
    }
)

# Run Phase 7 with automatic W&B logging
phase7 = Phase7SelfGuidedSystem(
    model=phase6_model,
    config=config,
    wandb_enabled=True
)

result = phase7.run()

# Final summary
wandb.log({
    "phase7/success": result['success'],
    "phase7/final_accuracy": result['performance']['overall_accuracy'],
    "phase7/total_cost": result['total_cost_usd']
})

wandb.finish()
```

---

## Success Criteria (W&B Validation)

After Phase 7 completes, verify these metrics:

- ✅ `phase7/success == True`
- ✅ `phase7/num_experts_created >= 3` and `<= 10`
- ✅ `stage1/data/total_acceptance_rate >= 0.75`
- ✅ `stage2/cross_expert/all_above_threshold == 1` (eudaimonia ≥0.65)
- ✅ `stage3/final/all_tests_passed == True`
- ✅ `phase7/final_overall_accuracy >= 0.85`
- ✅ `phase7/phase5_eudaimonia_preserved == True`
- ✅ `phase7/phase6_tool_use_preserved == True`
- ✅ `phase7/total_cost_usd <= 250`

---

**Phase 7 W&B Integration**: ✅ Complete Specification
**Total Metrics**: 350+ (condensed from 1000+ with smart logging)
**Key Innovation**: Model self-assessment metrics at every decision point
**Dashboard**: 4 sections (Discovery, Training, Search, Integration)

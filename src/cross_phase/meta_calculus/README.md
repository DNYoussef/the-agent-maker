# Meta-Calculus Integration for Agent Forge V2

This module integrates insights from the **meta-calculus-toolkit** project to enhance Agent Forge V2's 8-phase pipeline for training small language models.

## Overview

Meta-calculus provides three key assets for neural network training:

| Asset | What It Does | Primary Benefit |
|-------|--------------|-----------------|
| **k(L) Formula** | Scale-dependent adaptation | Universal layer-wise tuning |
| **Bigeometric Transform** | Bounded gradient transformation | Prevents explosion without clipping |
| **MOO Infrastructure** | Multi-objective optimization | Pareto-optimal solutions |

## Installation

The module is self-contained within Agent Forge. Dependencies:

```bash
pip install torch numpy
pip install pymoo  # Optional, for MOO features
```

## Quick Start

### 1. Create Phase-Specific Optimizer

```python
from src.cross_phase.meta_calculus import create_phase_optimizer

# For Phase 1 (Cognate) pre-training
optimizer = create_phase_optimizer("phase1_cognate", model)

# For Phase 3 (Quiet-STaR) RL training
optimizer = create_phase_optimizer("phase3_quietstar", model)

# For Phase 5 (Curriculum) with BitNet STE
optimizer = create_phase_optimizer("phase5_curriculum", model)
```

### 2. Monitor Spectral Gaps

```python
from src.cross_phase.meta_calculus import quick_gap_check, SpectralGapMonitor

# Quick health check
health = quick_gap_check(model)
if health["needs_attention"]:
    print(f"Warning: {health['collapsed_layers']} layers have collapsed gaps")

# Detailed monitoring
monitor = SpectralGapMonitor()
gaps = monitor.compute_model_gaps(model)
for name, metrics in gaps.items():
    print(f"{name}: gap={metrics['gap']:.4f}, healthy={metrics['is_healthy']}")
```

### 3. Run Multi-Objective Optimization

```python
from src.cross_phase.meta_calculus import (
    ExpertDiscoveryProblem,
    MOORunner,
    MOOConfig,
    select_from_pareto
)

# Define problem for Phase 7 expert discovery
problem = ExpertDiscoveryProblem(expert_evaluator=my_evaluator)

# Run optimization
config = MOOConfig(n_generations=100, population_size=50)
result = MOORunner(config).optimize(problem)

# Select from Pareto front
preference = {"task_loss": 2.0, "expert_diversity": 1.5}
best_x, best_f = select_from_pareto(result, preference)
```

## Phase Integration Guide

### Phase 1: Cognate (Pre-training)

```python
from src.cross_phase.meta_calculus import MetaGrokfast

optimizer = MetaGrokfast.for_phase("phase1_cognate", model.parameters())
# Config: lr=1e-3, lambda=0.3 (gentle filtering), bigeometric=True
```

### Phase 2: EvoMerge (Evolutionary Merging)

```python
from src.cross_phase.meta_calculus import (
    EvoMergeProblem,
    MOORunner,
    compute_merge_diversity_change
)

# Multi-objective merge optimization
problem = EvoMergeProblem(
    n_layers=8,
    n_techniques=6,
    model_evaluator=my_merge_evaluator
)
result = MOORunner().optimize(problem)

# Validate merge preserved diversity
diversity = compute_merge_diversity_change(models_before, model_after)
assert diversity["satisfies_bound"], "Merge collapsed spectral gap!"
```

### Phase 3: Quiet-STaR (Reasoning Enhancement)

```python
from src.cross_phase.meta_calculus import (
    MetaGrokfast,
    compute_thought_diversity,
    thought_diversity_loss
)

# Optimizer with QK-clip for RL stability
optimizer = MetaGrokfast.for_phase("phase3_quietstar", model.parameters())

# Monitor thought diversity
thoughts = model.generate_parallel_thoughts(input)
diversity = compute_thought_diversity(thoughts)
if diversity["is_collapsed"]:
    print("Warning: Thought collapse detected!")

# Add diversity loss to training
loss = task_loss + 0.1 * thought_diversity_loss(thoughts, target_diversity=0.5)
```

### Phase 4: BitNet (Quantization)

```python
from src.cross_phase.meta_calculus import k_from_layer_index

# Layer-wise sparsity thresholds
for layer_idx in range(total_layers):
    k = k_from_layer_index(layer_idx, total_layers)
    # Higher k (early layers) = lower sparsity = preserve info
    # Lower k (later layers) = higher sparsity = compress more
    sparsity = 0.5 * (1 - k)
    apply_quantization(layer, sparsity)
```

### Phase 5: Curriculum Learning

```python
from src.cross_phase.meta_calculus import MetaGrokfast

# Optimizer with STE mode for BitNet fine-tuning
optimizer = MetaGrokfast.for_phase("phase5_curriculum", model.parameters())
# Config: lambda=2.0 (aggressive), ste_mode=True
```

### Phase 6: Tool & Persona Baking

```python
from src.cross_phase.meta_calculus import k_from_parameter_variance

# Adaptive half-baking strength
for name, param in model.named_parameters():
    k = k_from_parameter_variance(param)
    # High variance params -> stronger baking
    # Low variance params -> preserve original
    strength = 0.5 * (1 + (0.5 - k))
    half_bake(param, strength)
```

### Phase 7: Self-Guided Experts

```python
from src.cross_phase.meta_calculus import (
    ExpertDiscoveryProblem,
    MOORunner,
    compute_expert_diversity
)

# MOO for expert configuration
problem = ExpertDiscoveryProblem(expert_evaluator=eval_experts)
result = MOORunner().optimize(problem)

# Validate expert diversity
diversity = compute_expert_diversity([e.weight for e in experts])
assert diversity["gap"] > 0.1, "Experts too similar!"
```

### Phase 8: Final Compression

```python
from src.cross_phase.meta_calculus import (
    log_space_interpolation,
    compute_compression_gap_retention
)

# Log-space weight fitting for hypercompression
weights_compressed = log_space_interpolation(w_original, w_target, alpha=0.5)

# Validate compression preserved representations
retention = compute_compression_gap_retention(model_original, model_compressed)
assert retention["passes_threshold"], "Compression destroyed representations!"
```

## Key Formulas

### k(L) Formula (Verified)

```
k(L) = -0.0137 * log10(L) + 0.1593
```

- R^2 = 0.71, p = 0.008 (statistically significant)
- L can be: gradient magnitude, layer index, entropy, parameter variance
- Higher L -> lower k -> more aggressive behavior
- Lower L -> higher k -> more conservative behavior

### Bigeometric Gradient Transform

```
g_meta = g * |g|^(2k-1)
```

- k > 0.5: dampens large gradients (prevents explosion)
- k < 0.5: amplifies small gradients (escapes vanishing)
- k = 0.5: identity (classical gradient)

Key property: D_BG[x^n] = e^n (CONSTANT regardless of x)

### Spectral Gap Preservation

```
gap(P_mix) >= min(gaps)
```

Composed/mixed operators preserve at least the minimum spectral gap of components.

## Configuration Reference

### MetaGrokfast Phase Configs

| Phase | lr | lambda | bigeometric | Special |
|-------|-----|--------|-------------|---------|
| phase1_cognate | 1e-3 | 0.3 | Yes | Gentle pretraining |
| phase3_quietstar | 5e-4 | 0.1 | Yes | QK-clip for RL |
| phase5_curriculum | 1e-3 | 2.0 | Yes | STE mode |
| phase6_baking | 1e-4 | 0.2 | Yes | Fine-tuning |
| phase7_experts | 5e-4 | 0.15 | Yes | Expert training |
| phase8_compression | 1e-4 | 1.5 | Yes | Post-compression |

### MOO Objectives

**EvoMerge (Phase 2):**
- perplexity (minimize)
- spectral_gap (maximize)
- weight_norm (minimize)
- invariance_score (maximize)
- fragility (minimize)

**Expert Discovery (Phase 7):**
- task_loss (minimize)
- expert_diversity (maximize)
- routing_entropy (maximize)
- compute_cost (minimize)
- robustness (maximize)

## Benchmarking

To compare with/without meta-calculus:

```python
from src.cross_phase.meta_calculus import MetaGrokfast

# With meta-calculus
optimizer_meta = MetaGrokfast.for_phase("phase1_cognate", model.parameters())

# Without (standard Adam)
optimizer_standard = torch.optim.Adam(model.parameters(), lr=1e-3)

# Compare training stability and final performance
```

## Files

```
meta_calculus/
    __init__.py          # Public API and convenience functions
    k_formula.py         # k(L) formula implementation
    bigeometric.py       # Bigeometric transforms
    moo_bridge.py        # Multi-objective optimization
    meta_grokfast.py     # Enhanced Grokfast optimizer
    spectral_gap.py      # Spectral gap monitoring
    README.md            # This file
```

## References

- Meta-Calculus Toolkit: Internal project with verified k(L) formula
- Grokfast: "Grokking at the Edge of Numerical Stability"
- Muon: Newton-Schulz orthogonalization for gradient diversity
- NSGA-II: Non-dominated Sorting Genetic Algorithm II
- Bigeometric Calculus: Non-Newtonian calculus for scale-invariance

# Meta-Calculus Integration for Agent Forge V2

**Version**: 1.0.0
**Date**: 2025-12-09
**Location**: `src/cross_phase/meta_calculus/`

---

## Overview

Meta-calculus enhancements have been integrated from the meta-calculus-toolkit project to improve training stability, optimization, and model quality across all 8 phases of Agent Forge V2.

### Core Enhancements

| Component | What It Does | Key Benefit |
|-----------|--------------|-------------|
| **k(L) Formula** | Scale-dependent adaptation | Universal layer-wise tuning |
| **Bigeometric Transform** | Gradient stabilization | Prevents explosion without clipping |
| **MOO Bridge** | Multi-objective optimization | Pareto-optimal solutions |
| **MetaGrokfast** | Enhanced Grokfast optimizer | Phase-specific configs |
| **Spectral Gap** | Diversity monitoring | Prevents representation collapse |

---

## Quick Start

### Option 1: Auto-Integration (Recommended)

```python
from cross_phase.meta_calculus import auto_integrate_phase

# Automatically get optimizer + monitors for any phase
components = auto_integrate_phase("phase1", model)

optimizer = components["optimizer"]      # MetaGrokfast
integration = components["integration"]  # Logging/monitoring
monitors = components["monitors"]        # Spectral gap, etc.

# Training loop
for step, batch in enumerate(dataloader):
    loss = train_step(model, batch, optimizer)
    integration.log_step(loss.item(), step)

# Final metrics
metrics = integration.get_metrics()
```

### Option 2: Manual Optimizer Creation

```python
from cross_phase.meta_calculus import MetaGrokfast

# Phase-specific optimizer
optimizer = MetaGrokfast.for_phase("phase1_cognate", model.parameters())

# Or with custom config
from cross_phase.meta_calculus import MetaGrokfastConfig
config = MetaGrokfastConfig(
    lr=1e-3,
    grokfast_lambda=0.5,
    use_bigeometric=True
)
optimizer = MetaGrokfast(model.parameters(), config=config)
```

### Option 3: Individual Components

```python
# k(L) Formula
from cross_phase.meta_calculus import compute_k, k_from_layer_index
k = compute_k(gradient_norm)  # Scale-dependent k
k = k_from_layer_index(layer_idx, total_layers)  # Layer-wise k

# Spectral Gap Monitoring
from cross_phase.meta_calculus import SpectralGapMonitor, quick_gap_check
health = quick_gap_check(model)  # Quick health check
monitor = SpectralGapMonitor()
gaps = monitor.compute_model_gaps(model)  # Detailed analysis

# MOO (Phase 2, 7)
from cross_phase.meta_calculus import run_moo_optimization
result = run_moo_optimization("phase7", expert_evaluator)
```

---

## Phase Integration Details

### Phase 1: Cognate (Pre-training)

**Enhancement**: MetaGrokfast with gentle filtering

```python
from cross_phase.meta_calculus import auto_integrate_phase

components = auto_integrate_phase("phase1", model)
optimizer = components["optimizer"]
# Config: lr=1e-3, lambda=0.3, bigeometric=True
```

**Key Benefits**:
- Bigeometric gradient transform prevents explosion
- Gentle EMA filtering (lambda=0.3) for stable pretraining
- Spectral gap monitoring for representation health

---

### Phase 2: EvoMerge (Evolutionary Merging)

**Enhancement**: Multi-objective merge optimization

```python
from cross_phase.meta_calculus import (
    EvoMergeProblem,
    MOORunner,
    compute_merge_diversity_change
)

# Define MOO problem with 5 objectives
problem = EvoMergeProblem(
    n_layers=8,
    n_techniques=6,
    model_evaluator=my_merge_evaluator
)

# Run optimization
result = MOORunner().optimize(problem)
pareto_front = result["pareto_front"]

# Validate merge preserved diversity
diversity = compute_merge_diversity_change(models_before, model_after)
assert diversity["satisfies_bound"], "Merge collapsed spectral gap!"
```

**Objectives**:
1. Perplexity (minimize)
2. Spectral gap (maximize)
3. Weight norm (minimize)
4. Invariance score (maximize)
5. Fragility (minimize)
6. Description length (minimize) - two-part MDL model bits

**MDL objective (added 2026-06-11, plan P2)**: `description_length` carries
the model-bits half of a two-part MDL code (`description_length_bits` in
`moo_bridge.py`); perplexity/task-loss is already the data half, so the two
are never combined into one number inside the objective. The exchange rate
between bits and perplexity is therefore a *selection-time* regime choice:
it lives in the `preference` weights passed to `select_from_pareto`, which
now logs every selection (`pareto-select: ...`) so the choice is auditable.
`weight_norm` remains as an objective but `description_length` supersedes it
for size-vs-quality selection. Evaluators may supply their own measured
`description_length`; otherwise a proxy is computed from the merge recipe
(EvoMerge) or `n_experts * svf_rank * z_dim` (Expert Discovery). Disable
with `include_mdl=False` for fixed-architecture sweeps where model bits are
constant.

---

### Phase 3: Quiet-STaR (Reasoning Enhancement)

**Enhancement**: Thought diversity monitoring + RL stability

```python
from cross_phase.meta_calculus import (
    auto_integrate_phase,
    compute_thought_diversity,
    thought_diversity_loss
)

components = auto_integrate_phase("phase3", model)
optimizer = components["optimizer"]
# Config: lr=5e-4, lambda=0.1, qk_clip=True

# Monitor thought diversity
thoughts = model.generate_parallel_thoughts(input)
diversity = compute_thought_diversity(thoughts)
if diversity["is_collapsed"]:
    print("Warning: Thought collapse detected!")

# Add diversity loss to training
loss = task_loss + 0.1 * thought_diversity_loss(thoughts, target_diversity=0.5)
```

**Key Benefits**:
- QK-clip for RL training stability
- Spectral gap monitoring prevents thought collapse
- Diversity loss maintains parallel thought variety

---

### Phase 4: BitNet (Quantization)

**Enhancement**: Layer-wise k(L) for sparsity thresholds

```python
from cross_phase.meta_calculus import k_from_layer_index

# Layer-wise sparsity based on k(L)
for layer_idx in range(total_layers):
    k = k_from_layer_index(layer_idx, total_layers)
    # Higher k (early layers) = lower sparsity = preserve info
    # Lower k (later layers) = higher sparsity = compress more
    sparsity = 0.5 * (1 - k)
    apply_quantization(layer, sparsity)
```

---

### Phase 5: Curriculum Learning

**Enhancement**: MetaGrokfast with STE mode

```python
from cross_phase.meta_calculus import auto_integrate_phase

components = auto_integrate_phase("phase5", model)
optimizer = components["optimizer"]
# Config: lr=1e-3, lambda=2.0, ste_mode=True
```

**Key Benefits**:
- Aggressive Grokfast filtering (lambda=2.0) for BitNet STE
- STE-compatible gradient flow
- Accelerates grokking during curriculum

---

### Phase 6: Tool & Persona Baking

**Enhancement**: Adaptive half-baking strength

```python
from cross_phase.meta_calculus import k_from_parameter_variance

# Adaptive baking strength based on parameter importance
for name, param in model.named_parameters():
    k = k_from_parameter_variance(param)
    # High variance params -> stronger baking (lower k)
    # Low variance params -> preserve original (higher k)
    strength = 0.5 * (1 + (0.5 - k))
    half_bake(param, strength)
```

---

### Phase 7: Self-Guided Experts

**Enhancement**: MOO expert discovery

```python
from cross_phase.meta_calculus import (
    ExpertDiscoveryProblem,
    MOORunner,
    compute_expert_diversity
)

# Define MOO problem with 5 objectives
problem = ExpertDiscoveryProblem(expert_evaluator=eval_experts)

# Run optimization (100 generations, 50 population)
result = MOORunner().optimize(problem)

# Select from Pareto front
preference = {"task_loss": 2.0, "expert_diversity": 1.5}
best_x, best_f = select_from_pareto(result, preference)

# Validate expert diversity
diversity = compute_expert_diversity([e.weight for e in experts])
assert diversity["gap"] > 0.1, "Experts too similar!"
```

**Objectives**:
1. Task loss (minimize)
2. Expert diversity (maximize)
3. Routing entropy (maximize)
4. Compute cost (minimize)
5. Robustness (maximize)

---

### Phase 8: Final Compression

**Enhancement**: Log-space transforms + gap retention

```python
from cross_phase.meta_calculus import (
    auto_integrate_phase,
    log_space_interpolation,
    compute_compression_gap_retention
)

components = auto_integrate_phase("phase8", model)
optimizer = components["optimizer"]
# Config: lr=1e-4, lambda=1.5

# Log-space weight fitting for hypercompression
weights_compressed = log_space_interpolation(w_original, w_target, alpha=0.5)

# Validate compression preserved representations
retention = compute_compression_gap_retention(model_original, model_compressed)
assert retention["passes_threshold"], "Compression destroyed representations!"
```

---

## Key Formulas

### k(L) Formula (Verified)

```
k(L) = -0.0137 * log10(L) + 0.1593
```

- **R^2 = 0.71, p = 0.008** (statistically significant)
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

**Key property**: D_BG[x^n] = e^n (CONSTANT regardless of x)

### Spectral Gap Preservation

```
gap(P_mix) >= min(gaps)
```

Composed/mixed operators preserve at least the minimum spectral gap.

---

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

---

## File Structure

```
src/cross_phase/meta_calculus/
    __init__.py           # Public API
    k_formula.py          # k(L) = -0.0137*log10(L) + 0.1593
    bigeometric.py        # D_BG transforms
    moo_bridge.py         # PyMOO integration
    meta_grokfast.py      # Enhanced optimizer
    spectral_gap.py       # Diversity monitoring
    phase_integration.py  # Auto-triggering
    README.md             # This file
```

---

## Verification

Run verification tests:

```bash
# k(L) reference table
python -m src.cross_phase.meta_calculus.k_formula

# Bigeometric verification
python -m src.cross_phase.meta_calculus.bigeometric

# Optimizer demo
python -m src.cross_phase.meta_calculus.meta_grokfast

# Spectral gap demo
python -m src.cross_phase.meta_calculus.spectral_gap

# MOO demo (requires pymoo)
python -m src.cross_phase.meta_calculus.moo_bridge

# Integration status
python -m src.cross_phase.meta_calculus.phase_integration
```

---

## Dependencies

**Required**:
- torch >= 2.0
- numpy

**Optional**:
- pymoo >= 0.6 (for MOO features - already installed)

---

## References

- Meta-Calculus Toolkit: Internal project with verified k(L) formula
- Grokfast: "Grokking at the Edge of Numerical Stability"
- Muon: Newton-Schulz orthogonalization
- NSGA-II: Non-dominated Sorting Genetic Algorithm II
- Bigeometric Calculus: Non-Newtonian calculus for scale-invariance

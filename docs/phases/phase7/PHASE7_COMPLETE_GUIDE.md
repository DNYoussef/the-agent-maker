# Phase 7: Transformer² + ADAS - Complete Implementation Guide

**Version**: 2.0 (Agent Forge V2)
**Phase Name**: Self-Adaptive Expert System
**Duration**: ~2-3 days on GTX 1660
**Status**: ✅ Ready for Implementation

---

## Overview

### What Phase 7 Accomplishes

Phase 7 creates a **self-adaptive model** that dynamically adjusts behavior for different domains:

1. **Train SVF Experts** (Transformer²): Learn 4 expert vectors (math, code, reasoning, vision)
2. **Search Expert Mixtures** (ADAS): Find optimal combinations for edge deployment
3. **Validate Performance**: Ensure <100ms latency, <2GB VRAM, >85% accuracy

### Key Innovation

**Transformer² SVF** (Singular Value Fine-tuning):
- Learns 1-D z vectors that scale singular values of weight matrices
- **32× fewer parameters** than LoRA (4096 vs 131,072 for a 4096×4096 matrix)
- **High compositionality**: z vectors can be linearly mixed
- **RL-compatible**: Stable REINFORCE training with small datasets

**ADAS Architecture Search**:
- Multi-objective optimization (NSGA-II) over expert mixtures
- Search space: α = [α₁, α₂, α₃, α₄] ∈ [0, 2]⁴
- Fitness: Minimize (latency, memory), Maximize (accuracy, throughput)

---

## Input/Output

### Input from Phase 6
```json
{
  "model": "baked_model_with_tools_personas",
  "metrics": {
    "tool_success_rate": 0.94,
    "persona_consistency": 0.89
  }
}
```

### Output to Phase 8
```json
{
  "success": true,
  "experts": {
    "math": "z_vectors_math.pt",
    "code": "z_vectors_code.pt",
    "reasoning": "z_vectors_reasoning.pt",
    "vision": "z_vectors_vision.pt"
  },
  "best_mixture": [1.2, 0.8, 1.5, 0.9],
  "adapted_model": "svf_adapted_model.pt",
  "metrics": {
    "latency_ms": 87.3,
    "accuracy": 0.87,
    "vram_gb": 1.8,
    "throughput_tps": 52.1
  }
}
```

---

## MuonGrokfast Integration (Fallback-Only)

### Why Fallback-Only for Phase 7?

SVF z vectors are **1-D parameters** (shape: `[r]` where r = min(m,n)). MuonGrokfast's parameter routing logic automatically routes 1-D params to the fallback optimizer:

```python
# In MuonGrokfast.step():
for param in params:
    if param.ndim >= 2 and config.enable_muon:
        # Muon path: Newton-Schulz orthogonalization
        update = self._muon_step(param, grad, group)
    else:
        # Fallback path: AdamW/Lion (1-D params like z vectors)
        self.fallback_optimizer.step()
```

Since z vectors have `ndim=1`, they **always use fallback path**. Phase 7 explicitly sets `enable_muon=False` to clarify this behavior.

### Configuration

```python
from cross_phase import MuGrokConfig

# Phase 7 configuration for SVF training
config = MuGrokConfig.from_phase(7)

# Equivalent manual configuration:
config = MuGrokConfig(
    # Muon settings (disabled for 1-D z vectors)
    enable_muon=False,

    # Fallback optimizer (AdamW)
    fallback_type="adamw",
    fallback_lr=2e-3,          # Per Transformer² paper
    fallback_betas=(0.9, 0.999),
    fallback_weight_decay=0.01,

    # Grokfast settings (RL gradient filtering)
    enable_grokfast=True,
    grokfast_alpha=0.98,       # EMA decay
    grokfast_lambda=0.05,      # Amplification factor (light)

    # QK-Clip (disabled for z vectors)
    use_qk_clip=False,

    # KL regularization
    kl_coefficient=0.2,

    # Phase identifier
    phase_name="phase7_transformer2"
)
```

### Training Flow

```
Input Task Batch
    ↓
SVF Model Forward (W' = U(Σ ⊗ diag(z))V^T)
    ↓
Generate Predictions
    ↓
REINFORCE Reward (correct/incorrect)
    ↓
Policy Gradient Loss + KL Regularization
    ↓
Backward (∂Loss/∂z for each z vector)
    ↓
MuonGrokfast Optimizer Step:
    ├─ Grokfast Prefilter (λ=0.05)
    │   └─ EMA-filtered gradients: g̃ = g + 0.05·μ
    ├─ Parameter Routing
    │   └─ z vectors (1-D) → Fallback path (AdamW)
    ├─ AdamW Update (lr=2e-3)
    │   └─ Adaptive learning rates per parameter
    └─ NO Muon (z vectors don't need orthogonalization)
```

---

## Performance Targets

| Metric | Target | Validation |
|--------|--------|------------|
| **SVF Expert Accuracy** | >85% per domain | On domain-specific eval set |
| **ADAS Convergence** | <100 generations | Pareto front stabilizes |
| **Inference Latency** | <100ms | Batch size 1, seq length 512, GTX 1660 |
| **Memory Usage** | <2GB VRAM | Peak usage during inference |
| **Throughput** | >50 tokens/sec | On GTX 1660 |
| **Overall Accuracy** | >85% | Across all domains |

---

## W&B Metrics Integration

```python
import wandb

# Initialize W&B
wandb.init(
    project="agent-forge-v2",
    name="phase7_transformer2_adas",
    config={
        "phase": 7,
        "num_experts": 4,
        "adas_generations": 100,
        "adas_population": 50
    }
)

# Log SVF training metrics (per expert)
wandb.log({
    f"svf/{domain}/loss": loss,
    f"svf/{domain}/reward": reward,
    f"svf/{domain}/kl_div": kl_div,
    f"svf/{domain}/accuracy": accuracy
})

# Log ADAS search metrics
wandb.log({
    "adas/generation": generation,
    "adas/best_latency_ms": best_latency,
    "adas/best_accuracy": best_accuracy,
    "adas/best_memory_gb": best_memory,
    "adas/best_throughput_tps": best_throughput,
    "adas/pareto_front_size": len(pareto_front)
})

# Log final mixture
wandb.log({
    "mixture/math": best_mixture[0],
    "mixture/code": best_mixture[1],
    "mixture/reasoning": best_mixture[2],
    "mixture/vision": best_mixture[3]
})

# Log final performance
wandb.log({
    "final/latency_ms": final_latency,
    "final/accuracy": final_accuracy,
    "final/vram_gb": final_vram,
    "final/throughput_tps": final_throughput
})
```

---

## Troubleshooting

### Issue: SVF training unstable (loss oscillates)

**Cause**: REINFORCE has high variance

**Solution**:
- Increase `grokfast_lambda` to 0.1 (more aggressive gradient filtering)
- Reduce `fallback_lr` to 1e-3 (smaller updates)
- Increase `kl_coefficient` to 0.3 (stronger regularization)

### Issue: ADAS search doesn't converge

**Cause**: Fitness landscape is too noisy

**Solution**:
- Increase `population_size` to 100
- Increase `num_generations` to 200
- Use averaged fitness over multiple evaluations

### Issue: Adapted model worse than base

**Cause**: Expert mixtures may be negative or too large

**Solution**:
- Constrain mixture weights: α_i ∈ [0, 1]
- Normalize mixture weights: sum(α) = 1
- Check KL divergence (should be <1.0)

### Issue: Out of memory during ADAS search

**Cause**: Evaluating too many solutions in parallel

**Solution**:
- Evaluate solutions sequentially (not in batches)
- Use smaller eval datasets (100 samples per domain)
- Clear CUDA cache between evaluations

---

## Implementation Guide

### Step 1: Prepare Datasets

```python
# Download and prepare domain-specific datasets
from datasets import load_dataset

datasets = {
    'math': load_dataset('gsm8k', 'main', split='train[:1000]'),
    'code': load_dataset('mbpp', split='train[:1000]'),
    'reasoning': load_dataset('ai2_arc', 'ARC-Easy', split='train[:1000]'),
    'vision': load_dataset('textvqa', split='train[:1000]')
}
```

### Step 2: Train SVF Experts

See [LOGICAL_UNDERSTANDING.md](LOGICAL_UNDERSTANDING.md) for complete implementation details.

### Step 3: Run ADAS Search

Use NSGA-II to find optimal expert mixture (see LOGICAL_UNDERSTANDING.md).

### Step 4: Apply Best Mixture

Compose experts with discovered mixture weights and save adapted model.

---

## Success Criteria

- ✅ **SVF Training**: K=4 experts trained (math, code, reasoning, vision) with >85% accuracy
- ✅ **ADAS Search**: Converges within 100 generations, Pareto front found
- ✅ **Edge Performance**:
  - Inference latency <100ms on GTX 1660
  - Memory usage <2GB VRAM
  - Throughput >50 tokens/second
  - Task accuracy >85% across domains
- ✅ **Composability**: Linear mixture of experts achieves better performance than any single expert
- ✅ **W&B Metrics**: 650+ metrics tracked including per-expert performance, ADAS fitness evolution

---

## Research Papers

1. **Transformer²**: "Transformer-Squared: Self-Adaptive LLMs" (2501.06252v3.pdf)
   - SVF parameterization
   - Two-pass inference
   - REINFORCE + KL training

2. **ADAS**: "Automated Design of Agentic Systems"
   - Architecture search
   - Multi-objective optimization
   - Sandboxed evaluation

---

## Next Steps

After Phase 7 completes:

1. **Validate Expert Mixtures**: Test on held-out domains
2. **Profile Edge Performance**: Benchmark on actual edge hardware
3. **Proceed to Phase 8**: Final compression (SeedLM + VPTQ + Hyper)

---

**Phase 7 Status**: ✅ Ready for Implementation
**Estimated Duration**: 2-3 days on GTX 1660
**Dependencies**: Phase 6 model, domain datasets, PyMOO library
**Next Phase**: [Phase 8: Final Compression](../phase8/PHASE8_COMPLETE_GUIDE.md)

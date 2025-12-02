# Phase 7 Documentation Complete: Self-Guided Expert System

**Status**: ✅ **Complete V2 Specification - Ready for Implementation**
**Version**: V2 (Self-Guided Architecture)
**Last Updated**: 2025-10-16
**Timeline**: 3-4 days
**Cost**: $150-250

---

## Overview

Complete redesign and documentation of **Phase 7 (Expert Specialization)** from "Manual 4-expert ADAS" to **self-guided 3-stage expert discovery system** where the model itself determines its specialization strategy.

**Fully documented** with:
- Model-driven expert discovery (N=3-10, model determines)
- OpenRouter data generation ($150-250)
- Transformer² SVF training with model validation
- Model-guided ADAS search (NSGA-II)
- Full Phase 5/6 integration
- BitNet dequantization strategy
- W&B integration (350+ metrics)
- Success criteria & risk analysis

---

## What Was Documented

### Three-Stage Self-Guided System

Complete redesign to model-driven architecture:

1. **Stage 1: Expert Discovery** (Model-Driven, 8 hours)
   - Model self-analyzes capabilities across 8 domains
   - Model determines expert count N (3-10, not hardcoded)
   - Model generates dataset specifications per expert
   - OpenRouter generates 300-500 problems/expert
   - **Model validates** each problem (eudaimonia, edge-of-chaos, quality)

2. **Stage 2: SVF Training** (Transformer², 36 hours)
   - Train N model-determined experts using Singular Value Fine-tuning
   - z-vectors (1-D params, 32× smaller than LoRA)
   - REINFORCE + KL regularization (MuonGrokfast fallback-only)
   - **Model validates** training every epoch (eudaimonia, accuracy, edge-of-chaos)
   - Model intervention on drift (auto-adjust KL coefficient)

3. **Stage 3: Model-Guided ADAS** (NSGA-II Search, 42 hours)
   - **Model defines** search objectives and constraints
   - NSGA-II evolutionary search (100 generations, 50 population)
   - **Model evaluates** each mixture (self-assessment across domains)
   - **Model guides** search every 10 generations (adjust bounds, objectives)
   - **Model validates** final mixture (comprehensive self-test)

---

## Key Innovations

### 1. Model Self-Determination (Stage 1)

**V1**: Hardcoded 4 experts (math, code, reasoning, vision)
**V2**: Model decides N experts and domains based on self-analysis

**Example Model Decision**:
```json
{
  "num_experts": 5,
  "expert_definitions": [
    {
      "id": 1,
      "name": "analytical",
      "domains": ["math", "reasoning"],
      "rationale": "Both weak (0.72, 0.65), synergistic improvement",
      "target_improvement": 0.15
    },
    {
      "id": 2,
      "name": "creative",
      "domains": ["creative"],
      "rationale": "Weakest domain (0.58), needs dedicated focus",
      "target_improvement": 0.17
    },
    // ... 3 more experts
  ],
  "reasoning": "I have 3 weak domains needing aggressive improvement.
                5 experts balances specialization with training efficiency."
}
```

**Metrics**: Model analyzes 8 domains, determines optimal N, generates justification

---

### 2. Model-Generated Training Data (Stage 1)

**V1**: Human-curated datasets (GSM8K, HumanEval, etc.)
**V2**: Model generates specifications → OpenRouter creates data → Model validates

**Process**:
1. Model specifies what it needs to learn:
   ```json
   {
     "topics": ["multi-step arithmetic", "algebraic equations", "logical inference"],
     "difficulty_target": "75% correctness (edge-of-chaos)",
     "num_examples": 450,
     "format": "free_response_with_reasoning",
     "special_requirements": ["vary surface features", "mix problem types"]
   }
   ```

2. OpenRouter (GPT-4/Claude-3.5) generates 500 problems per spec

3. **Model validates EACH problem**:
   - Difficulty check: 60-90% confidence (solvable but challenging)
   - Eudaimonia check: Score ≥0.65 (Phase 5 alignment preserved)
   - Edge-of-chaos check: In learning zone (70-80%)
   - Quality check: Clear, correct, non-repetitive

4. Accept 75%+, augment with variants if needed

**Result**: 300-500 validated problems per expert, perfectly calibrated to model's edge-of-chaos

**Cost**: ~$30 per expert dataset = $150 total (5 experts)

---

### 3. Transformer² SVF (Singular Value Fine-tuning)

**Why SVF vs LoRA**:
- **LoRA**: (m+n)×r parameters per weight (e.g., 131,072 for 4096×4096, r=16)
- **SVF**: r parameters per weight (e.g., 4,096 for same matrix, **32× fewer**)

**How SVF Works**:
```python
# Standard weight: W ∈ ℝ^(m×n)
# SVF decomposes: W = U Σ V^T (singular value decomposition)
# Expert learns z ∈ ℝ^r that scales singular values:
W_expert = U (Σ ⊗ diag(z)) V^T

# z-vectors are 1-D → MuonGrokfast uses fallback-only (AdamW)
# Grokfast filters noisy REINFORCE gradients
# KL regularization prevents drift from eudaimonia
```

**Training**:
- REINFORCE (policy gradient): Reward +1 correct, -1 incorrect
- KL penalty: Keep adapted model close to base (preserve alignment)
- **Model validation every epoch**: Eudaimonia, accuracy, edge-of-chaos
- **Model intervention**: If eudaimonia <0.65, increase KL coefficient
- **Early stopping**: Model decides when converged

**Result**: N high-quality experts, each specialized to model-determined domain

---

### 4. Model-Guided ADAS Search (Stage 3)

**V1**: Metric-based NSGA-II search
**V2**: Model-guided evolutionary search

**Process**:
1. **Model defines objectives**:
   ```python
   {
     "primary_objective": "maximize_overall_accuracy",
     "constraints": [
       {"type": "latency", "max": 100, "unit": "ms"},
       {"type": "eudaimonia_score", "min": 0.65},
       {"type": "edge_of_chaos", "range": [0.70, 0.80]}
     ],
     "search_space": {
       "analytical": [0.0, 3.0],    # High weight (weak domain)
       "creative": [0.0, 2.5],      # High weight (weak domain)
       "code": [0.0, 1.5],          # Restrict (already strong)
       "general": [0.5, 1.5]        # Require minimum (base capability)
     }
   }
   ```

2. **NSGA-II search** (100 generations, 50 population):
   - For each mixture: Compose experts, apply to model
   - **Model self-evaluates** mixture:
     - Test on all domains
     - Check eudaimonia preservation
     - Check edge-of-chaos stability
     - Check tool use / persona preservation (Phase 6)

3. **Model guidance every 10 generations**:
   ```python
   model_analysis_prompt = f"""
   Current best mixtures: {pareto_front_summary}
   Am I converging toward my goals?
   Should I adjust search space or objectives?
   """

   model_guidance = {
     "adjust_search_space": True,
     "new_bounds": {
       "analytical": [1.5, 3.0],  # Narrow (all good mixtures >1.5)
       ...
     },
     "early_stopping": False
   }
   ```

4. **Model validates final mixture**:
   - Comprehensive self-test (100 scenarios)
   - All validation criteria must pass
   - Model can reject and choose 2nd best from Pareto front

**Result**: Optimal mixture α* that **model itself** validated as best

---

### 5. BitNet Integration Strategy

**Critical Issue**: Phase 7 receives BitNet 1.58-bit model from Phase 6

**Problem**: SVF requires full-precision weights (SVD decomposition)

**Solution**: **Dequantize for Phase 7**

**Pipeline**:
```
Phase 4: Quantize to BitNet (100MB → 12MB, 8.2×)
Phase 5-6: Stay BitNet (12MB)
  ├─ MuonGrokfast STE mode (works with quantized)
  └─ Prompt Baking via LoRA (works with quantized)

Phase 7: Dequantize to float32 (12MB → 100MB)
  ├─ SVF training requires full precision
  ├─ Expert quality > compression during training
  └─ Phase 8 will re-compress anyway

Phase 8: Final compression (100MB → 0.4MB, 280×)
```

**Justification**:
- Phase 8 compresses to 0.4MB regardless of Phase 7 input
- SVF expert quality requires full precision
- Temporary decompression for 3-4 days acceptable
- No research risk (proven technique)

**Memory**: 100MB during Phase 7 (vs 12MB if stayed BitNet)

---

## Files Created

### Core Documentation (7 files, 70,000+ words)

1. **[PHASE7_SELF_GUIDED_SYSTEM.md](./PHASE7_SELF_GUIDED_SYSTEM.md)** (21,000 words)
   - Complete 3-stage self-guided architecture
   - Model-driven expert discovery system
   - Model validation at all stages
   - Phase 5/6 integration
   - Model-guided ADAS search
   - Risk analysis & success criteria

2. **[PHASE7_DATA_GENERATION_GUIDE.md](./PHASE7_DATA_GENERATION_GUIDE.md)** (11,000 words)
   - OpenRouter integration ($150-250)
   - Model-generated dataset specifications
   - Model self-validation of training data
   - Eudaimonia & edge-of-chaos checks
   - Quality assurance & cost optimization
   - Example code & workflows

3. **[LOGICAL_UNDERSTANDING.md](./LOGICAL_UNDERSTANDING.md)** (13,000 words) - UPDATED
   - Self-guided overview
   - Three-stage architecture explained
   - SVF training with model validation
   - Model-guided ADAS search
   - Complete integration specs
   - Expected inputs/outputs

4. **[PHASE7_BITNET_INTEGRATION.md](./PHASE7_BITNET_INTEGRATION.md)** (5,000 words)
   - BitNet compatibility analysis
   - Three possible approaches evaluated
   - **Recommendation**: Dequantize for Phase 7
   - Memory analysis & justification
   - Documentation update checklist

5. **[PHASE7_WANDB_INTEGRATION.md](./PHASE7_WANDB_INTEGRATION.md)** (8,000 words)
   - 350+ metrics across 3 stages
   - Model self-assessment metrics
   - Stage 1: 50+ metrics (capability, experts, data)
   - Stage 2: 200+ metrics (per-expert training)
   - Stage 3: 100+ metrics (ADAS search, model guidance)
   - Dashboard YAML configuration
   - Success criteria validation

6. **[PHASE7_COMPLETE_GUIDE.md](./PHASE7_COMPLETE_GUIDE.md)** (UPDATED)
   - Implementation guide with self-guided system
   - MuonGrokfast fallback-only configuration
   - Performance targets & troubleshooting

7. **[PHASE7_DOCUMENTATION_COMPLETE.md](./PHASE7_DOCUMENTATION_COMPLETE.md)** (this file)
   - Standalone complete documentation
   - All innovations, formulas, examples
   - File manifest & success criteria

### Integration Documentation (1 file, 7,000 words)

8. **[docs/integration/BITNET_COMPATIBILITY_ALL_PHASES.md](../../docs/integration/BITNET_COMPATIBILITY_ALL_PHASES.md)** (7,000 words)
   - BitNet compatibility across Phases 4-8
   - STE (Straight-Through Estimator) explained
   - MuonGrokfast STE mode (Phase 5)
   - Prompt Baking on BitNet (Phase 5-6)
   - SVF dequantization requirement (Phase 7)
   - Complete compatibility table

---

## Success Criteria

### Stage 1: Expert Discovery
- ✅ Model generates capability report across ≥8 domains
- ✅ Model determines expert count N (3 ≤ N ≤ 10) with justification
- ✅ Model clusters domains into expert groups
- ✅ Model generates N datasets (300-500 problems each)
- ✅ Validation acceptance rate ≥75% per dataset
- ✅ All datasets pass eudaimonia check (≥0.65)
- ✅ Total cost <$250 (target: $150)

### Stage 2: SVF Training
- ✅ All N experts trained to target accuracy (model-determined)
- ✅ Eudaimonia maintained (≥0.65) throughout training
- ✅ Edge-of-chaos preserved (0.70-0.80)
- ✅ Model validation checkpoint every epoch
- ✅ Model intervention on drift (KL coefficient adjustment)
- ✅ Each expert achieves >85% on specialization domain

### Stage 3: Model-Guided ADAS
- ✅ Model defines search objectives and constraints
- ✅ NSGA-II converges within 100 generations
- ✅ Model guidance provided every 10 generations
- ✅ Final mixture passes comprehensive model validation:
  - Overall accuracy improvement >10%
  - Eudaimonia score ≥0.65
  - Edge-of-chaos: 0.70-0.80
  - Tool use preserved (≥Phase 6 baseline)
  - Persona preserved (≥Phase 6 baseline)
  - Latency <100ms (GTX 1660)
  - Memory <2GB VRAM

### Integration Validation
- ✅ Phase 5 eudaimonia integrated at all stages
- ✅ Phase 6 tool/persona capabilities preserved
- ✅ Phase 7 output compatible with Phase 8 compression
- ✅ BitNet dequantization accuracy ≥99%

---

## Integration Points

### From Phase 6

**Input**:
```json
{
  "model": "phase6_optimized_model.pt",
  "quantization": "bitnet_1.58",
  "size_mb": 12,
  "metrics": {
    "swe_bench_score": 0.613,
    "aggregate_benchmark": 0.794,
    "tool_use_rate": 0.94,
    "persona_consistency": 0.89
  },
  "baked_capabilities": {
    "eudaimonia_aligned": true,
    "tool_prompts_baked": 5,
    "persona_prompts_baked": 10
  }
}
```

**Phase 6 → Phase 7 Handoff**:
1. Validate Phase 6 model meets success criteria
2. **Dequantize BitNet** to float32 (12MB → 100MB)
3. Verify dequantization accuracy ≥99%
4. Begin Stage 1 (Expert Discovery)

### To Phase 8

**Output**:
```json
{
  "model": "phase7_adapted_model.pt",
  "quantization": "float32",
  "size_mb": 100,
  "expert_library": {
    "analytical": "z_vectors_analytical.pt",
    "creative": "z_vectors_creative.pt",
    "planning": "z_vectors_planning.pt",
    "code": "z_vectors_code.pt",
    "general": "z_vectors_general.pt"
  },
  "optimal_mixture": [1.8, 1.3, 1.0, 0.9, 1.1],
  "num_experts_determined_by_model": 5,
  "performance": {
    "latency_ms": 87.3,
    "vram_gb": 1.8,
    "overall_accuracy": 0.854,
    "eudaimonia_score": 0.71,
    "edge_of_chaos": 0.76,
    "tool_use_rate": 0.93,
    "persona_consistency": 0.88
  },
  "compression_notes": {
    "z_vectors_are_1d": true,
    "total_expert_params": 20480,
    "base_model_params": 25000000,
    "adaptation_overhead_percent": 0.082,
    "note": "Dequantized for SVF, ready for Phase 8 re-compression"
  }
}
```

**Phase 7 → Phase 8 Handoff**:
1. Validate all Phase 7 success criteria
2. Export base model (100MB, float32)
3. Export expert library (80KB, float32)
4. Export optimal mixture weights
5. Phase 8 compresses all (100MB → 0.4MB, 280×)

---

## Timeline & Cost

### Timeline: 3-4 Days (GTX 1660)

**Day 1: Stage 1 (Expert Discovery)** - 8 hours
- Capability self-analysis: 2 hours
- Expert determination: 1 hour
- Data generation (OpenRouter): 4 hours
- Validation: 1 hour

**Day 2-3: Stage 2 (SVF Training)** - 36 hours
- 5 experts × 5 epochs = 25 training runs
- ~1.5 hours per expert (with validation)
- Sequential: 36 hours (1.5 days)
- Parallel (5× GPU): 7.5 hours

**Day 3-4: Stage 3 (ADAS Search)** - 42 hours
- 100 generations × 50 population = 5000 evals
- ~30 sec per eval (model-intensive)
- Model guidance: 10 × 5 min = 50 min
- Total: ~42 hours (1.75 days)

**Total**: 3-4 days on single GTX 1660

### Cost: $150-250

**Data Generation (OpenRouter)**: $150
- 5 experts × 500 problems = 2500 problems
- $0.06 per problem (GPT-4 generation + validation)

**Model Self-Evaluation (Optional)**: $100
- ADAS: 5000 evals, ~100 need OpenRouter
- $1 per call (complex prompts)
- Can be done locally for free

**Total**: $150-250 (target: $150 local-only)

---

## Research Papers Integrated

1. **Transformer²**: "Transformer-Squared: Self-Adaptive LLMs" (2501.06252v3.pdf)
   - SVF parameterization (singular value fine-tuning)
   - REINFORCE + KL training
   - Expert composition via linear mixing

2. **ADAS**: "Automated Design of Agentic Systems"
   - Multi-objective optimization (NSGA-II)
   - Architecture search
   - Sandboxed evaluation
   - **Extended**: Model-guided search (V2 innovation)

3. **Prompt Baking** (Phase 5/6 reference): arXiv:2409.13697v1
   - Eudaimonia alignment system
   - Edge-of-chaos learning
   - Half-baking stacking mechanics

4. **BitNet** (Phase 4 reference):
   - 1.58-bit quantization
   - Straight-Through Estimator (STE)
   - MuonGrokfast STE mode compatibility

---

## Risk Analysis & Mitigation

### Risk 1: Model Determines Too Many Experts (30%)

**Impact**: Training time × N, cost × N

**Mitigation**:
- Cap N at 10 experts
- If model requests >10, prompt to merge similar domains
- Fallback: Use top 5 most impactful only

### Risk 2: Model-Generated Datasets Are Poor Quality (25%)

**Impact**: Experts won't train effectively

**Mitigation**:
- Model validates each problem (4 checks)
- Use multiple frontier models (GPT-4 + Claude-3.5)
- Human spot-check 10% of dataset
- Fallback: Use human-curated datasets (GSM8K, etc.)

### Risk 3: Model Guidance Misleads ADAS Search (15%)

**Impact**: Suboptimal mixture found

**Mitigation**:
- Model guidance is advisory, not mandatory
- Keep NSGA-II objective function as final arbiter
- Track correlation between model recommendations and fitness
- Fallback: Disable guidance, use pure NSGA-II

### Risk 4: Eudaimonia Drift During SVF Training (30%)

**Impact**: Violates Phase 5 alignment

**Mitigation**:
- Check eudaimonia every epoch (not just at end)
- Auto-increase KL coefficient if drift detected
- Hard stop training if eudaimonia <0.60
- Fallback: Restart training with higher KL from beginning

### Risk 5: Dequantization Loses Accuracy (10%)

**Impact**: Model performance degrades

**Mitigation**:
- Test dequantization on Phase 4 output (target: ≥99% recovery)
- If <99%, use Phase 3 output (pre-quantization) for Phase 7
- Verify with comprehensive benchmark suite

---

## V1 vs V2 Comparison

| Aspect | V1 (Manual) | V2 (Self-Guided) |
|--------|-------------|------------------|
| **Expert Count** | Hardcoded (N=4) | **Model determines** (N=3-10) |
| **Domains** | Predefined (math, code, reasoning, vision) | **Model discovers** via self-analysis |
| **Data** | Human-curated datasets | **Model generates** via OpenRouter |
| **ADAS** | Metric-based search | **Model-guided** optimization |
| **Validation** | Accuracy thresholds | **Model validates** alignment preservation |
| **Integration** | Isolated | **Full** Phase 5/6 integration |
| **BitNet** | Not addressed | **Complete** dequantization strategy |
| **W&B** | ~100 metrics | **350+ metrics** (self-assessment) |
| **Timeline** | 2-3 days | **3-4 days** (model decision overhead) |
| **Cost** | $0 (local datasets) | **$150-250** (OpenRouter generation) |

---

## Next Phase

**Phase 8**: Final Compression (SeedLM + VPTQ + Hypercompression)

Phase 8 receives:
- Base model (100MB, float32, dequantized)
- Expert library (5 × 4K params = 20K total, float32)
- Optimal mixture weights

Phase 8 compresses:
- SeedLM: 100MB → 25MB (4×)
- VPTQ: 25MB → 2.5MB (10×)
- Hypercompression: 2.5MB → 0.36MB (7×)
- **Total**: 280× compression from 100MB

**Critical**: Phase 8 must validate quality preservation with comprehensive benchmark testing.

---

## Key Takeaways

1. **Model Self-Guidance**: All major decisions (expert count, domains, datasets, search) made by model itself
2. **Data Quality**: Model validates every training example against eudaimonia, edge-of-chaos, quality
3. **Training Validation**: Model checks itself every epoch, intervenes on drift
4. **Search Guidance**: Model analyzes Pareto front, adjusts search space, validates final mixture
5. **BitNet Strategy**: Dequantize for Phase 7 SVF quality, re-compress in Phase 8
6. **Integration**: Full Phase 5 eudaimonia + Phase 6 tool/persona preservation throughout
7. **Timeline**: 3-4 days, $150-250, on consumer hardware (GTX 1660)

---

**Phase 7 Status**: ✅ Complete V2 Specification
**Documentation**: 70,000+ words, 8 files, all systems integrated
**Key Innovation**: Model-driven expert discovery, training, and validation
**Next**: Phase 8 verification (compression + quality testing)

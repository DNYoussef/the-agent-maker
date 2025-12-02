# Phase 1-4: Complete Summary & Documentation Index

**Version**: 2.0
**Date**: 2025-10-15
**Purpose**: Master index for all Phase 1-4 documentation

---

## Executive Summary

This session completed a **comprehensive review and update** of Phases 1-4 for Agent Forge V2, with critical focus on **model-size-agnostic architecture**. All documentation is now aligned with the reality that **we don't know the final model size** until runtime.

### Key Deliverables

1. ✅ **Comprehensive Plan** - Complete Phase 1-4 integration specification
2. ✅ **Premortem Analysis** - All failure modes, risks, and mitigations identified
3. ✅ **W&B Integration** - Complete observability strategy
4. ✅ **Dataset Specification** - 16 datasets fully documented
5. ✅ **GraphViz Updates** - Visual flows for Phases 1-3
6. ✅ **Phase 3 Data Generator** - Updated with correct token structure

---

## Documentation Structure

### Core Planning Documents

#### 1. [PHASE1-4_COMPREHENSIVE_PLAN_V2.md](PHASE1-4_COMPREHENSIVE_PLAN_V2.md)
**Purpose**: Master plan for Phases 1-4 with model-size-agnostic strategies

**Contents**:
- Phase 1: Cognate (3 models, 16 datasets, 3-stage curriculum)
- Phase 2: EvoMerge (50 generations, 8 binary combos, genetic algorithm)
- Phase 3: Quiet-STaR (two-step: baking → RL, 12 special tokens)
- Phase 4: BitNet (1.58-bit quantization, 6-12x compression depending on size)
- Cross-phase integration (data flow, storage, VRAM, W&B)
- Model size detection and adaptive strategies

**Key Insights**:
- All phases must query model size at runtime
- Compression ratios vary: 6x (tiny) to 12x (large)
- Batch sizes adaptive to model size and VRAM
- Storage requirements: 2.2 GB minimal, 13 GB with intermediates

#### 2. [PHASE1-4_PREMORTEM_V2.md](PHASE1-4_PREMORTEM_V2.md)
**Purpose**: Identify all failure modes and mitigation strategies

**Contents**:
- **Phase 1 Risks**: Dataset download (🔴), diversity (🟡), OOM (🟡), convergence (🟡)
- **Phase 2 Risks**: Premature convergence (🟡), degenerate merges (🟡), slow fitness (🟢)
- **Phase 3 Risks**: Data generation failure (🔴), baking convergence (🟡), theater (🔴)
- **Phase 4 Risks**: Low compression (🟢), accuracy drop (🔴)
- **Infrastructure Risks**: W&B (🟢), storage (🔴), GPU (🟡)

**Risk Summary**:
- 5 CRITICAL risks (🔴) - project blockers
- 7 HIGH risks (🟡) - major setbacks
- 3 MEDIUM risks (🟢) - minor issues

**Key Mitigations**:
- Pre-download all datasets before Phase 1
- Enforce diversity metrics during Phase 1 training
- Run anti-theater checks every 1000 steps in Phase 3
- Staged quantization with accuracy checks in Phase 4
- Pre-flight storage and VRAM checks

#### 3. [PHASE1-4_WANDB_INTEGRATION.md](PHASE1-4_WANDB_INTEGRATION.md)
**Purpose**: Complete observability strategy for all phases

**Contents**:
- W&B project structure
- Metrics to log (per step, per epoch, final)
- Artifacts to save (models, configs, datasets)
- Visualizations (loss curves, distributions, tables)
- Cross-phase dashboards

**Key Metrics by Phase**:
- **Phase 1**: perplexity, gsm8k_acc, act_halting, ltm_usage, diversity
- **Phase 2**: best_fitness, diversity, improvement, combo_usage
- **Phase 3**: accuracy (baking), reward (RL), coherence, divergence, ablation
- **Phase 4**: compression_ratio, accuracy_drop, sparsity, inference_speedup

**Critical Tracking**:
- Diversity metrics (Phase 1, 2) - early detection of convergence
- Anti-theater metrics (Phase 3) - detect empty reasoning
- Accuracy drop (Phase 4) - trigger fallbacks if >10%

---

### Phase-Specific Documents

#### Phase 1: Cognate

**Primary**: [phases/phase1/LOGICAL_UNDERSTANDING.md](../phases/phase1/LOGICAL_UNDERSTANDING.md)
- TRM × Titans-MAG architecture
- 3 model specializations (reasoning, memory, speed)
- Updated with 16-dataset specification

**Dataset Spec**: [PHASE1_DATASET_SPECIFICATION.md](PHASE1_DATASET_SPECIFICATION.md)
- Complete details on all 16 datasets
- HuggingFace IDs, sample counts, processing
- 3-stage curriculum (Foundation → Reasoning → Advanced)
- Download scripts and PyTorch Dataset code
- Storage: ~1.35 GB (or ~50 GB with OpenWebText)

**Integration**: [PHASE1_DATASET_INTEGRATION_SUMMARY.md](PHASE1_DATASET_INTEGRATION_SUMMARY.md)
- V1 vs V2 comparison (5 datasets → 16 datasets)
- Dataset categories and statistics
- Training curriculum details

**GraphViz**: [phases/phase1/graphviz/phase-flow-v2.dot](../phases/phase1/graphviz/phase-flow-v2.dot)
- Visual representation of TRM × Titans-MAG architecture
- Shows 8 layers (not 12), MAG gate, LTM, ACT head
- Muon × Grokfast Phase 1 config

#### Phase 2: EvoMerge

**Primary**: [phases/phase2/LOGICAL_UNDERSTANDING.md](../phases/phase2/LOGICAL_UNDERSTANDING.md)
- 6 merge techniques (3 pairs)
- Binary combination strategy (8 initial models)
- 50 generations genetic algorithm
- Elite preservation + loser merging

**GraphViz**: [phases/phase2/graphviz/phase-flow-v2.dot](../phases/phase2/graphviz/phase-flow-v2.dot)
- Shows binary pairing strategy
- Note: No optimizer used (pure merging)

#### Phase 3: Quiet-STaR

**Primary**: [phases/phase3/LOGICAL_UNDERSTANDING.md](../phases/phase3/LOGICAL_UNDERSTANDING.md)
- Two-step process: Prompt Baking (Step 1) → Quiet-STaR (Step 2)
- 12 special tokens (2 outer + 10 inner)
- Anti-theater validation

**Data Generator**: [phases/phase3/phase3_data_generator.py](../phases/phase3/phase3_data_generator.py)
- **Updated v2.0.0** with correct token structure
- 5 frontier models × 10 strategies × 500 examples = 25,000 examples
- OpenRouter API integration with batch generation
- Cost: ~$100-200, Time: ~4 hours

**Data Gen Update**: [PHASE3_DATA_GENERATOR_UPDATE_V2.md](PHASE3_DATA_GENERATOR_UPDATE_V2.md)
- Documents v1.0.0 → v2.0.0 changes
- Token structure: `[thinking]`/`[/endthinking]` + 10 strategies
- Added Self-Correction and Uncertainty strategies

**GraphViz**: [phases/phase3/graphviz/phase-flow-v2.dot](../phases/phase3/graphviz/phase-flow-v2.dot)
- Shows two-step process
- Token structure diagram
- Two Muon × Grokfast configs (supervised vs RL)

#### Phase 4: BitNet

**Primary**: [phases/phase4/LOGICAL_UNDERSTANDING.md](../phases/phase4/LOGICAL_UNDERSTANDING.md)
- 1.58-bit ternary quantization {-1, 0, +1}
- 8x compression target (adaptive by model size)
- Straight-Through Estimator (STE)
- Fine-tuning with Grokfast

**Complete Guide**: [phases/phase4/PHASE4_COMPLETE_GUIDE.md](../phases/phase4/PHASE4_COMPLETE_GUIDE.md)
- Detailed implementation
- Calibration strategy
- Sparsity optimization
- Selective precision preservation

---

## Key Technical Decisions

### 1. Model Size Agnostic Architecture

**Problem**: We don't know the final model size until Phase 3 completes.

**Solution**: All phases detect model size at runtime and adapt:

```python
def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 ** 2)
    return {
        "params": total_params,
        "size_mb": size_mb,
        "size_category": categorize_size(total_params)  # tiny/small/medium/large
    }

# Adaptive strategies
if size_category == "tiny":
    batch_size = 32
    target_compression = 6.0
    sparsity_threshold = 0.05
elif size_category == "small":
    batch_size = 16
    target_compression = 8.0
    sparsity_threshold = 0.1
# ... etc
```

**Impact**:
- **Phase 1**: Adaptive batch size (32 → 8) based on VRAM
- **Phase 2**: Adaptive fitness weights and memory management
- **Phase 3**: Adaptive thought count (8 → 4) and length (20 → 10)
- **Phase 4**: Adaptive compression targets (6x → 12x)

### 2. Phase 3 Token Structure (CORRECTED)

**Previous Understanding** (incorrect):
- 16 special tokens (8 strategies × 2 tokens each)
- Outer wrapper: `<think>` and `</think>`

**Corrected Understanding**:
- **12 special tokens total**:
  - **2 outer**: `[thinking]` and `[/endthinking]`
  - **10 inner**: `<step>`, `<mece>`, `<falsify>`, `<expert>`, `<orthogonal>`, `<doubt>`, `<bayesian>`, `<multidomain>`, `<correct>`, `<uncertain>`

**Token Usage**:
```
[thinking]
  <step>Break down the problem...</step>
  <mece>Categories: A, B, C</mece>
  <doubt>Could I be wrong?</doubt>
[/endthinking]
```

**Impact**:
- Model size increase: ~12K params (negligible for 25M model)
- Vocabulary: 50,257 → 50,269 tokens
- Data generation: 25,000 examples (10 strategies × 2,500 each)

### 3. Phase 3 Two-Step Process

**Step 1: Prompt Baking** (Supervised Learning)
- Train on 25,000 reasoning examples
- Supervised cross-entropy loss
- Convergence threshold: 85% accuracy
- Muon × Grokfast config: lr=1e-4, lambda=0.2, kl=0.0

**Step 2: Quiet-STaR** (Reinforcement Learning)
- Generate 4-8 parallel thoughts
- Score coherence (semantic + syntactic + predictive)
- REINFORCE reward: accuracy improvement
- Muon × Grokfast config: lr=5e-4, lambda=0.1, kl=0.1 (prevent drift)

**Why Two Steps?**
- **Without baking**: RL starts from scratch, high variance, slow
- **With baking**: RL starts with reasoning patterns already embedded, stable and fast

### 4. Phase 4 Compression Strategy

**Size-Adaptive Targets**:
- **Tiny models** (<50M): 6x compression (embeddings are 15-20% of params)
- **Small models** (<500M): 8x compression (embeddings are 10-15%)
- **Medium models** (<2B): 10x compression (embeddings are 5-10%)
- **Large models** (>2B): 12x compression (embeddings are 2-5%)

**Preservation Strategy**:
- Quantize to 1.58-bit: Attention, FFN layers (85-95% of params)
- Keep in FP16: Embeddings, LM head, layer norms (5-15% of params)

**Sparsity Targets**:
- Tiny: 25% (conservative)
- Small: 35% (standard)
- Medium: 40% (aggressive)
- Large: 45% (very aggressive)

---

## Success Metrics (Phases 1-4)

### Phase 1: Cognate
- ✅ 3 models created (~25M params each)
- ✅ Diversity >0.3 (halting, memory, speed)
- ✅ GSM8K accuracy >10%
- ✅ Training time <30 hours total
- ✅ Storage <10 GB

### Phase 2: EvoMerge
- ✅ Fitness improvement ≥20% (target 23.5%)
- ✅ Evolution time <2 hours
- ✅ Diversity maintained (>0.3)
- ✅ All 8 combos used

### Phase 3: Quiet-STaR
- ✅ Baking convergence ≥85%
- ✅ Reasoning accuracy +3-5%
- ✅ Anti-theater tests pass
- ✅ Inference time <200ms with thoughts

### Phase 4: BitNet
- ✅ Compression ≥6x (size-dependent)
- ✅ Accuracy drop <10%
- ✅ Inference speedup ≥2x
- ✅ Sparsity >30%

---

## Resource Requirements

### Storage

**Per Phase**:
- Phase 1: 10 GB (3 models + datasets + checkpoints)
- Phase 2: 5 GB (8 models + logs)
- Phase 3: 8 GB (model + reasoning data + checkpoints)
- Phase 4: 3 GB (compressed model + calibration data)

**Cumulative** (with intermediates): ~13 GB
**Minimal** (delete intermediates): ~2.2 GB

### VRAM (GTX 1660 Target: 6GB)

**Peak Requirements**:
- Phase 1: 5.5 GB (1 model training)
- Phase 2: 16 GB (8 models) → **Fallback to CPU if needed**
- Phase 3 Step 1: 5.5 GB (baking)
- Phase 3 Step 2: 6.5 GB (RL with thoughts)
- Phase 4: 3.5 GB (compression)

**Mitigations**:
- Gradient checkpointing
- Mixed precision (FP16)
- Gradient accumulation
- CPU offloading for Phase 2

### Time

**Estimated Timeline**:
- Phase 1: 20-30 hours (3 models × 7-10 hours)
- Phase 2: 1-2 hours (50 generations)
- Phase 3: 12-16 hours (Step 1: 6hrs, Step 2: 8hrs)
- Phase 4: 3-5 hours (calibration + compression + fine-tuning)

**Total**: ~40-50 hours on GTX 1660

### Cost

**API Costs**:
- Phase 3 data generation: $100-200 (OpenRouter)

**Compute Costs**:
- Local GPU: $0 (own hardware)
- Cloud GPU (optional): ~$0.50/hour × 50 hours = $25

**Total**: ~$100-225

---

## Integration Points

### Phase 1 → Phase 2
```python
phase1_output = {
    "models": ["model1.pt", "model2.pt", "model3.pt"],
    "metrics": {"model1": {...}, "model2": {...}, "model3": {...}},
    "diversity": {"halting": [8.2, 9.5, 6.1], ...}
}

phase2_input = load_phase1_models(phase1_output["models"])
validate_diversity(phase2_input)  # Assert diversity >0.3
```

### Phase 2 → Phase 3
```python
phase2_output = {
    "model": "champion_evolved.pt",
    "fitness": 0.185,
    "improvement": 0.235
}

phase3_input = load_model(phase2_output["model"])
add_special_tokens(phase3_input, num_tokens=12)  # Adds ~12K params
```

### Phase 3 → Phase 4
```python
phase3_output = {
    "model": "reasoning_enhanced.pt",
    "params": 25_024_672,  # Base + 12K tokens
    "metrics": {"gsm8k_acc": 0.18, ...}
}

phase4_input = load_model(phase3_output["model"])
model_size = get_model_size(phase4_input)  # Detect size at runtime
target_compression = get_compression_target(model_size["size_category"])
```

---

## Critical Risks & Mitigations

### Top 5 Risks (by severity × probability)

1. **Phase 3 Data Generation Fails** (🔴 CRITICAL, 25% probability)
   - **Mitigation**: Pre-generate data, retry logic, cost monitoring
   - **Fallback**: Local model generation (Llama 3, Mixtral)

2. **Phase 1 Models Don't Diversify** (🟡 HIGH, 40% probability)
   - **Mitigation**: Enforce diversity metrics, aggressive config differences
   - **Detection**: Validate after each epoch

3. **Phase 3 Theater (Empty Reasoning)** (🔴 CRITICAL, 35% probability)
   - **Mitigation**: Anti-theater checks every 1000 steps, strict rewards
   - **Detection**: Divergence <0.3, ablation <0.02, correlation <0.5

4. **Phase 4 Accuracy Drop >10%** (🔴 CRITICAL, 20% probability)
   - **Mitigation**: Staged quantization, conservative fallback, extended fine-tuning
   - **Detection**: Evaluate after each layer quantization

5. **Storage Full During Training** (🔴 CRITICAL, 25% probability)
   - **Mitigation**: Pre-flight checks, monitor during training, delete old checkpoints
   - **Detection**: Check disk usage every 1000 steps

---

## Next Steps

### Immediate (Before Implementation)
1. ✅ Review all documentation (this session)
2. ⏳ Test dataset download script
3. ⏳ Verify GPU environment (CUDA, PyTorch, VRAM)
4. ⏳ Set up W&B project structure
5. ⏳ Implement model size detection utilities
6. ⏳ Implement diversity validation functions
7. ⏳ Implement anti-theater validation functions

### Phase 1 Preparation
1. Download all 16 datasets (~1.35 GB)
2. Verify dataset integrity
3. Test Phase 1 training loop (1 epoch on 100 samples)
4. Validate diversity metrics work
5. Set up Phase 1 W&B logging

### Phase 3 Preparation
1. Pre-generate reasoning data (25,000 examples, $100-200)
2. Validate data quality (token structure, strategy distribution)
3. Test OpenRouter API with small batch
4. Implement data generator locally

### Infrastructure
1. Set up W&B dashboards
2. Implement storage monitoring
3. Implement VRAM monitoring
4. Set up checkpointing strategy
5. Create runbooks for each failure mode

---

## File Index

### Core Documents (docs/)
- `PHASE1-4_COMPREHENSIVE_PLAN_V2.md` - Master plan
- `PHASE1-4_PREMORTEM_V2.md` - Risk analysis
- `PHASE1-4_WANDB_INTEGRATION.md` - Observability
- `PHASE1-4_COMPLETE_SUMMARY.md` - This file
- `PHASE1_DATASET_SPECIFICATION.md` - 16 datasets
- `PHASE1_DATASET_INTEGRATION_SUMMARY.md` - Dataset integration
- `PHASE3_DATA_GENERATOR_UPDATE_V2.md` - Data generator v2.0.0

### Phase Documentation (phases/phaseN/)
- `phases/phase1/LOGICAL_UNDERSTANDING.md`
- `phases/phase1/graphviz/phase-flow-v2.dot`
- `phases/phase2/LOGICAL_UNDERSTANDING.md`
- `phases/phase2/graphviz/phase-flow-v2.dot`
- `phases/phase3/LOGICAL_UNDERSTANDING.md`
- `phases/phase3/graphviz/phase-flow-v2.dot`
- `phases/phase3/phase3_data_generator.py` (v2.0.0)
- `phases/phase4/LOGICAL_UNDERSTANDING.md`
- `phases/phase4/PHASE4_COMPLETE_GUIDE.md`

### GraphViz Visualizations
- Phase 1: TRM × Titans-MAG architecture
- Phase 2: Binary pairing strategy
- Phase 3: Two-step process with token structure

---

## Version History

### v2.0 (2025-10-15) - This Session
- ✅ Integrated 16 datasets into Phase 1
- ✅ Updated Phase 3 with correct token structure (12 tokens)
- ✅ Updated Phase 3 data generator (v1.0.0 → v2.0.0)
- ✅ Created model-size-agnostic architecture
- ✅ Comprehensive premortem analysis
- ✅ Complete W&B integration plan
- ✅ Updated all GraphViz flows

### v1.0 (Previous)
- Initial Phase 1-4 documentation
- 5 datasets (not 16)
- Incorrect Phase 3 token structure
- No model-size-agnostic strategies

---

## Conclusion

Phases 1-4 are now **fully specified** with:
- ✅ Complete integration plan
- ✅ All risks identified and mitigated
- ✅ Full observability via W&B
- ✅ Model-size-agnostic architecture
- ✅ 16-dataset training curriculum
- ✅ Corrected Phase 3 token structure
- ✅ Updated data generator (v2.0.0)

**Status**: ✅ **Ready for Implementation**

**Next**: Begin Phase 1 implementation with dataset download and environment setup.

---

**Version**: 2.0
**Date**: 2025-10-15
**Author**: Claude (Anthropic)
**Project**: Agent Forge V2 - Local-First Rebuild

# Phase 5 V2 - Complete Implementation Summary

**Version**: 2.0 (Curriculum Learning System)
**Date**: 2025-01-16
**Status**: Specification Complete - Ready for Implementation

---

## Executive Summary

Phase 5 has been **completely redesigned** from a simple "BitNet + Grokfast training" system to a **sophisticated curriculum learning pipeline** inspired by three research papers:

1. **"Intelligence at the Edge of Chaos"** - 75% correctness threshold for optimal learning
2. **"Unexpected Benefits of Self-Modeling"** - Temperature range self-prediction training
3. **"Dreaming Is All You Need"** - Memory consolidation through high-temp replay

---

## What Changed from V1

### V1 (Old System)
- Simple Grokfast-accelerated training
- Fixed dataset
- Claim: "50x speedup" (unvalidated)
- Single training loop

### V2 (New System - Documented)
- **7-stage curriculum learning** pipeline
- **Adaptive dataset** that shrinks as concepts are mastered
- **Frontier model integration** for question generation, variants, hints
- **Self-modeling** across temperature ranges
- **Dream consolidation** after each level
- **Modular specialization** (coding/research/writing agents)

---

## 7-Stage Pipeline Architecture

```
INPUT: Phase 4 BitNet 1.58-bit quantized model
    ↓
1. ASSESSMENT - Find Edge of Chaos
    ├─ Frontier models generate 1-100 difficulty scale
    ├─ Test student → Find 75% correctness threshold
    └─ Output: Baseline level (e.g., 40)
    ↓
2. CURRICULUM GENERATION
    ├─ Rescale: Baseline→1, 100→10
    ├─ Frontier models create 500Q × 10 levels
    └─ Output: ~2,000 questions per level
    ↓
3. TRAINING LOOP (per level, iterative)
    ├─ Question → TRM reasoning → Code tool → Validate
    ├─ SUCCESS: Create variant, remove after 3× success
    ├─ FAILURE: Root cause → Generate hint → Re-shuffle
    └─ Dataset shrinks = comprehension proof
    ↓
4. PROMPT BAKING
    ├─ Bake eudaimonia moral compass (4 rules)
    ├─ Bake ethical OODA loop (3 parts)
    └─ Bake identity/purpose
    ↓
5. SELF-MODELING
    ├─ Generate at temp ranges (expanding per level)
    ├─ Mask generated text
    ├─ Predict own text at midpoint temps
    └─ Train until grokking about itself
    ↓
6. SLEEP & DREAM
    └─ Memory consolidation (high-temp replay)
    ↓
7. LEVEL PROGRESSION
    └─ Repeat stages 3-6 for levels 2-10
    ↓
OUTPUT: Specialized model for Phase 6
```

---

## Key Innovations

### 1. Dataset Shrinkage as Progress Metric

**Traditional training**: Fixed dataset, measure loss/accuracy
**V2 training**: Dataset shrinks from 2,000 → 0-50 questions

**Why this works**:
- Mastered concepts removed after 3× consecutive successes
- Variants prevent memorization (test *understanding*, not recall)
- Visible progress: 95%+ mastery = level complete

---

### 2. Frontier Model Orchestration

**Three roles for frontier models** (GPT-4, Claude-3.5, Gemini, etc.):

**Role 1: Question Generation**
```
Task: Create 500 coding questions across levels 1-10
Output: ~2,000 questions per level (4 models × 500)
Cost: $50-100 per level generation
```

**Role 2: Variant Creation**
```
Task: Tweak question (change nouns/numbers, keep concept)
Output: 1 variant per successful question
Cost: $0.01 per variant × 20,000 variants = $200
```

**Role 3: Hint Generation (Root Cause Analysis)**
```
Task: Analyze failed code → Find flawed thinking → Generate targeted hint
Output: 1-5 hints per failed question
Cost: $0.05 per hint × 8,000 hints = $400
```

**Total OpenRouter cost estimate**: $600-800 per specialty

---

### 3. Temperature Range Self-Modeling

**Level 1 training**:
- Generate at temps: [0.0-0.2], [0.2-0.4], ..., [1.8-2.0] (10 ranges)
- Mask 20% of generated tokens
- Model predicts masked tokens, knowing it generated them
- Reward correct predictions, punish errors

**Level 10 training**:
- Temperature ranges expand: [0.9-2.0], [1.0-2.1], ..., [2.7-3.8]
- Model learns to distinguish outputs across wider spectrum
- Develops self-awareness of capabilities

**Formula**:
```python
def temperature_ranges(level):
    width = 0.2 + (level - 1) * 0.1
    num_ranges = 10 + level - 1
    base_start = 0.0 + (level - 1) * 0.1

    return [(base_start + i*0.1, base_start + i*0.1 + width)
            for i in range(num_ranges)]
```

---

### 4. Dream Consolidation

**After each level**:
1. Sample training outputs (successful solutions, failed attempts)
2. Generate "dreams" at temp 1.5 (creative replay)
3. Train on dreams for 1 epoch
4. Strengthens memory, prevents catastrophic forgetting

**Inspired by**: "Dreaming Is All You Need" (PDF in phase5 folder)

---

## Mathematical Formalizations

### Curriculum Difficulty Mapping

```
new_difficulty(orig) = 1 + (orig - B) × 9 / (100 - B)

where B = baseline level from assessment

Example (B=40):
  40 → 1  (baseline = level 1)
  70 → 5  (midpoint)
  100 → 10 (hardest)
```

### Dataset Size Dynamics

```
D(t) = D_0 × (1 - M(t))

M(t) = 1 / (1 + e^(-k(t - t_0)))  (sigmoid mastery curve)

where:
  D_0 = 2,000 (initial)
  k = 0.1 (learning rate)
  t_0 = 15 (half-mastery epoch)

Predicted shrinkage:
  Epoch 0:  D = 2,000 (0% mastery)
  Epoch 10: D = 1,640 (18% mastery)
  Epoch 20: D = 540 (73% mastery)
  Epoch 30: D = 100 (95% mastery)
  Epoch 40: D = 20 (99% mastery)
```

### Self-Modeling Loss

```
L_self = -∑ log P(token | own_generation, temp=t_mid)

Combined with consistency loss:
L_total = L_self + λ × KL(P(·|t1), P(·|t2))

Ensures similar temps produce similar outputs
```

---

## Implementation Roadmap (16 Weeks)

### Weeks 1-2: Infrastructure
- [ ] OpenRouter API client for frontier models
- [ ] Coding sandbox environment (Docker containers)
- [ ] Question database schema + CRUD operations
- [ ] Validation harness for code execution

### Weeks 3-4: Assessment System
- [ ] Frontier model prompts for difficulty scales
- [ ] Assessment loop (test model on 2,000 questions)
- [ ] Edge-of-chaos detection (75% threshold finder)
- [ ] Curriculum rescaling logic

### Weeks 5-6: Training Loop Core
- [ ] Question sampling from dataset
- [ ] TRM recursive thinking integration
- [ ] Tool use (code generation + execution)
- [ ] Validation pipeline

### Weeks 7-8: Variant & Hint Systems
- [ ] Variant generation API (frontier models)
- [ ] Variant quality validation
- [ ] Root cause analysis prompts
- [ ] Hint management system (1-5 levels)

### Weeks 9-10: Dataset Mechanics
- [ ] Question state machine (active/hinted/mastered)
- [ ] Success counter + consecutive tracking
- [ ] Removal logic (3× consecutive successes)
- [ ] Dataset shrinkage monitoring

### Weeks 11-12: Self-Modeling
- [ ] Temperature range calculator
- [ ] Multi-temp generation pipeline
- [ ] Token masking (15-30% mask rate)
- [ ] Self-prediction loss function
- [ ] Grokking detection (95%+ self-accuracy)

### Weeks 13-14: Dream Consolidation
- [ ] Dream data sampling
- [ ] High-temp replay generation (1.5)
- [ ] Consolidation training loop
- [ ] Catastrophic forgetting metrics

### Weeks 15-16: Integration & Testing
- [ ] Level progression loop (1-10)
- [ ] Prompt baking integration (eudaimonia/OODA/identity)
- [ ] Hard wall detection
- [ ] End-to-end testing
- [ ] W&B logging (7,200+ metrics)

---

## Phase 4 Integration Challenges

**Input**: BitNet 1.58-bit quantized model (ternary weights {-1, 0, +1})

**Challenge 1: Tool Use Precision**
- Problem: Code generation requires high precision (syntax errors fatal)
- Solution: MuonGrokfast STE mode (full-precision gradients)
- Mitigation: Validation harness catches errors, frontier hints guide fixes

**Challenge 2: Recursive Thinking Depth**
- Problem: Quantized attention may limit reasoning depth
- Solution: TRM architecture designed for low-precision from Phase 1
- Mitigation: Hints scaffold thinking when model struggles

**Challenge 3: Self-Modeling Stability**
- Problem: Predicting own quantized outputs is noisy
- Solution: Aggressive Grokfast filtering (λ=2.0)
- Mitigation: Train longer per level, use consistency loss

**See full analysis**: [PHASE5_INTEGRATION_PHASE4.md](./PHASE5_INTEGRATION_PHASE4.md)

---

## Pre-Mortem Risk Analysis

### High-Risk Issues

**Risk 1: Frontier Model Costs Explode**
- Estimate: $600-800 per specialty
- Mitigation: Cache generated questions, reuse across training runs
- Contingency: Use smaller frontier models (GPT-3.5 vs GPT-4) for variants/hints

**Risk 2: Dataset Never Shrinks (No Mastery)**
- Symptom: Still 1,500+ questions after 40 epochs
- Cause: Curriculum too hard, hints ineffective
- Mitigation: Adjust baseline, improve hint quality, skip level if stuck

**Risk 3: Self-Modeling Doesn't Grok**
- Symptom: Self-prediction accuracy stuck at 60-70%
- Cause: Quantization noise, temperature ranges too wide
- Mitigation: Narrow temp ranges, use full-precision for self-modeling stage

**Risk 4: Training Time Too Long (Weeks, Not Days)**
- Estimate: 120-240 hours (5-10 days)
- Cause: 10 levels × ~24hrs each
- Mitigation: Parallelize levels (train multiple in separate processes)
- Contingency: Reduce to 5 levels instead of 10

**Risk 5: Dream Consolidation Fails (Catastrophic Forgetting)**
- Symptom: Level 10 model forgets level 1 concepts
- Mitigation: Longer dream phases, cross-level validation
- Contingency: Interleave all levels (don't do sequentially)

---

## Success Criteria

**Phase 5 is successful if**:

### Quantitative Metrics
- ✅ Assessment finds edge of chaos (73-77% baseline accuracy)
- ✅ Dataset shrinks to <5% by level completion (95%+ mastery)
- ✅ Self-modeling achieves >95% self-prediction accuracy
- ✅ Tool use (coding) success rate >90% by level 10
- ✅ No catastrophic forgetting (level 1 concepts still work at level 10)

### Qualitative Outcomes
- ✅ Model generates working code for specialized domain
- ✅ Code quality improves across levels (fewer bugs, cleaner structure)
- ✅ Model explains reasoning clearly (recursive thoughts useful)
- ✅ Moral compass prevents unethical outputs (eudaimonia test)
- ✅ Identity consistent across training (knows it's "CodeForge")

### Performance Targets
- ✅ Total training time: 120-240 hours (acceptable on consumer GPU)
- ✅ Frontier model costs: <$1,000 per specialty
- ✅ Final model size: Still 1.58-bit (no precision drift)
- ✅ Inference speed: Maintained from Phase 4 (2-4× speedup vs full-precision)

---

## Modularity: Different Specializations

**Same Phase 5 system, different parameters:**

### Coding Agent (Documented Above)
- Domain: Software development
- Tool: Python/JS sandbox
- Validation: Does code run? Pass tests?
- Output: Model that writes functional programs

### Research Agent
- Domain: Academic research, paper synthesis
- Tool: Web search + PDF retrieval
- Validation: Citation accuracy, coherence
- Output: Model that writes literature reviews

### Writing Agent
- Domain: Creative/technical writing
- Tool: Style analyzer, grammar checker
- Validation: Readability, adherence to style guide
- Output: Model that writes polished prose

### Math Agent
- Domain: Mathematical problem-solving
- Tool: Symbolic math engine (SymPy)
- Validation: Proof correctness, numerical accuracy
- Output: Model that solves calculus/algebra

**Implementation**: Change prompts, tool environment, validation—everything else identical

---

## Documentation Status

### ✅ Completed Documents

1. **PHASE5_LOGICAL_UNDERSTANDING_V2.md** (5,200 words)
   - Complete system overview
   - All 7 stages explained
   - Mathematical formulas
   - Integration notes

2. **PHASE5_CURRICULUM_SYSTEM.md** (6,800 words)
   - Question lifecycle state machine
   - Variant generation specifications
   - Hint management (root cause analysis)
   - Dataset shrinkage mechanics
   - Implementation classes + database schema

3. **PHASE5_V2_IMPLEMENTATION_SUMMARY.md** (This file)
   - Executive summary
   - Roadmap (16 weeks)
   - Risk analysis
   - Success criteria

### 📋 Pending Documents (Outlines Below)

**High Priority**:
- PHASE5_FORMULAS.md - All mathematical models consolidated
- PHASE5_INTEGRATION_PHASE4.md - BitNet-specific training considerations
- PHASE5_PREMORTEM_V2.md - Expanded risk analysis with mitigation strategies

**Medium Priority**:
- PHASE5_FRONTIER_INTEGRATION.md - OpenRouter API specs, cost optimization
- PHASE5_TOOL_USE_SPEC.md - Coding environment architecture, sandboxing
- PHASE5_SELF_MODELING.md - Full temperature range training specification
- PHASE5_EUDAIMONIA_SYSTEM.md - Moral compass prompts, OODA loop details

**Lower Priority** (can reference existing V1 docs):
- PHASE5_IMPLEMENTATION_GUIDE.md - Detailed week-by-week implementation
- PHASE5_RESEARCH_INTEGRATION.md - Paper citations, claims validation

### 🎨 GraphViz Diagrams (5 Planned)

1. **phase5-master-curriculum-flow.dot** - Full 7-stage pipeline
2. **phase5-question-lifecycle.dot** - State machine diagram
3. **phase5-self-modeling-cycle.dot** - Temperature range training flow
4. **phase5-level-progression.dot** - Levels 1-10 with expanding temp ranges
5. **phase5-tool-validation.dot** - Code generation → execution → validation

---

## Quick-Start Pseudo-Implementation

```python
# Phase 5 Main Loop (Simplified)
def phase5_specialized_training(model, specialty="coding"):
    # Stage 1: Assessment
    baseline_level = assess_edge_of_chaos(model, frontier_models)

    # Stage 2: Generate Curriculum
    curriculum = generate_curriculum(
        baseline=baseline_level,
        levels=10,
        frontier_models=frontier_models
    )

    # Stages 3-7: Level Loop
    for level in range(1, 11):
        # Stage 3: Train on curriculum
        model = train_level(model, curriculum[level], coding_env)

        # Stage 4: Bake moral compass
        model = bake_prompts(model, [eudaimonia, ooda, identity])

        # Stage 5: Self-modeling
        temp_ranges = calculate_temp_ranges(level)
        model = self_modeling_training(model, temp_ranges)

        # Stage 6: Dream
        model = dream_consolidation(model, curriculum[level])

        # Check completion
        stats = curriculum.get_stats(level)
        if stats['mastery'] > 0.95:
            print(f"Level {level} complete!")
        else:
            print(f"Hard wall at level {level}, stopping")
            break

    return model  # Specialized agent
```

---

## Next Steps

### For Reviewers
1. **Read**: PHASE5_LOGICAL_UNDERSTANDING_V2.md for conceptual overview
2. **Read**: PHASE5_CURRICULUM_SYSTEM.md for implementation details
3. **Provide**: Answers to clarification questions (see below)

### For Implementers
1. **Start**: Week 1-2 infrastructure (OpenRouter API, sandbox, database)
2. **Reference**: 16-week roadmap above
3. **Track**: Use W&B for all 7,200+ metrics (defined in V1 docs, still valid)

### For Users
1. **Specify**: Eudaimonia 4 rules (current placeholders)
2. **Specify**: OODA 3 parts (current placeholders)
3. **Choose**: Specialty (coding/research/writing/math)
4. **Budget**: ~$1,000 for frontier model API costs

---

## Open Questions (Need User Input)

### Critical Questions

**Q1: Eudaimonia 4 Rules**
- What are the specific 4 moral principles to bake?
- Example format: "1. Do no harm, 2. Respect autonomy, 3. ..."

**Q2: OODA Loop 3 Parts**
- What are the 3 parts of the ethical decision loop?
- Current guess: Observe consequences, Orient to principles, Decide transparently

**Q3: Frontier Model Selection**
- Which frontier models to use? GPT-4, Claude-3.5, Gemini, Llama-3?
- Budget constraints? ($600-800 baseline, can optimize)

**Q4: Dream Duration**
- How long per dream phase? 1 epoch (current), or longer?
- Trade-off: Longer = better consolidation, but slower training

**Q5: Hard Wall Threshold**
- What accuracy defines "can't progress"? 50%? 60%?
- Should we skip levels or stop entirely?

### Nice-to-Have Clarifications

**Q6: Temperature Range Width**
- Current formula: width = 0.2 + (level-1) × 0.1
- Too aggressive? Too conservative?

**Q7: Consecutive Successes**
- 3× consecutive = mastered (current)
- Should this be higher (4×, 5×) for safety?

**Q8: Hint Limit**
- Max 5 hints per question (current)
- Should we give up sooner (3 hints) or later (10 hints)?

---

## File Manifest (Phase 5 Folder)

### Completed
```
phases/phase5/
├── PHASE5_LOGICAL_UNDERSTANDING_V2.md  (5.2K words) ✅
├── PHASE5_CURRICULUM_SYSTEM.md         (6.8K words) ✅
├── PHASE5_V2_IMPLEMENTATION_SUMMARY.md (This file) ✅
├── INTELLIGENCE AT THE EDGE OF CHAOS.pdf            ✅
├── Unexpected Benefits of Self-Modeling in Neural Systems.pdf  ✅
├── DREAMING IS ALL YOU NEED.pdf                     ✅
└── graphviz/
    └── phase-flow.dot (V1 version, will be replaced)
```

### To Be Created
```
phases/phase5/
├── PHASE5_FORMULAS.md                   ⏳
├── PHASE5_INTEGRATION_PHASE4.md         ⏳
├── PHASE5_PREMORTEM_V2.md               ⏳
├── PHASE5_FRONTIER_INTEGRATION.md       ⏳
├── PHASE5_TOOL_USE_SPEC.md              ⏳
├── PHASE5_SELF_MODELING.md              ⏳
├── PHASE5_EUDAIMONIA_SYSTEM.md          ⏳
└── graphviz/
    ├── phase5-master-curriculum-flow.dot     ⏳
    ├── phase5-question-lifecycle.dot         ⏳
    ├── phase5-self-modeling-cycle.dot        ⏳
    ├── phase5-level-progression.dot          ⏳
    └── phase5-tool-validation.dot            ⏳
```

---

## W&B Metrics (Preliminary List)

**Per Level** (10 levels × 720 metrics = 7,200):
```python
metrics_per_level = {
    # Dataset dynamics
    "dataset_size": int,
    "mastered_count": int,
    "mastery_percent": float,

    # Training performance
    "epoch": int,
    "success_rate": float,
    "avg_attempts_per_question": float,

    # Hints
    "total_hints": int,
    "avg_hints_per_question": float,
    "hint_effectiveness": float,

    # Variants
    "variants_generated": int,
    "variant_success_rate": float,

    # Self-modeling
    "self_prediction_accuracy": float,
    "temp_ranges_trained": int,
    "grokking_achieved": bool,

    # Tool use
    "code_executions": int,
    "successful_programs": int,
    "validation_accuracy": float,

    # Dream
    "dream_samples": int,
    "consolidation_loss": float,
    "forgetting_rate": float,
}
```

**Global Metrics**:
- Total training time
- Frontier model API costs
- Final model size (bits)
- Inference speed (tokens/sec)
- Overall specialization accuracy

---

## Conclusion

Phase 5 V2 is a **fundamentally different system** from V1. Instead of a simple training loop, it's a **curriculum learning pipeline** that:

1. Finds the model's learning edge (75% threshold)
2. Generates personalized curriculum
3. Trains with adaptive difficulty + scaffolding (hints)
4. Uses frontier models as "teachers" (variants, hints)
5. Self-models across temperature ranges
6. Dreams to consolidate memory
7. Repeats across 10 levels

**Documentation Status**: 3/20 files complete (15%)
**Next Priority**: Create formulas doc + Phase 4 integration doc + GraphViz diagrams

**Ready for**: Implementation start (Weeks 1-2 infrastructure)

---

**Related Documents**:
- [PHASE5_LOGICAL_UNDERSTANDING_V2.md](./PHASE5_LOGICAL_UNDERSTANDING_V2.md) - Start here
- [PHASE5_CURRICULUM_SYSTEM.md](./PHASE5_CURRICULUM_SYSTEM.md) - Implementation details
- [PHASE4_COMPLETE_GUIDE.md](../phase4/PHASE4_COMPLETE_GUIDE.md) - Input from Phase 4
- [PHASE6_LOGICAL_UNDERSTANDING.md](../phase6/LOGICAL_UNDERSTANDING.md) - Output to Phase 6

**Questions?** See "Open Questions" section above or contact project lead.

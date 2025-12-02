# Phase 5: Specialized Curriculum Training - Logical Understanding

**Version**: 2.0 (Complete Redesign)
**Phase Purpose**: Curriculum-based specialization training with self-modeling and dream consolidation

---

## 🚨 IMPORTANT: Phase 5 Has Been Completely Redesigned

**This is NOT the old "BitNet + Grokfast training" system.**

Phase 5 V2 is a **7-stage curriculum learning pipeline** that creates specialized AI agents through:
- Edge-of-chaos assessment
- Adaptive curriculum generation
- Tool use with code validation
- Frontier model orchestration (variants, hints)
- Self-modeling across temperature ranges
- Dream consolidation

---

## Quick Links

### **Start Here** 👇
📘 **[PHASE5_LOGICAL_UNDERSTANDING_V2.md](./PHASE5_LOGICAL_UNDERSTANDING_V2.md)**
- Complete conceptual overview
- All 7 stages explained
- Mathematical formulas
- Research paper integration

### Implementation Details
📗 **[PHASE5_CURRICULUM_SYSTEM.md](./PHASE5_CURRICULUM_SYSTEM.md)**
- Question lifecycle state machine
- Variant generation
- Hint management (root cause analysis)
- Dataset shrinkage mechanics
- Database schema

### Architecture Integration ⭐ NEW
🔧 **[PHASE5_TRM_TITANS_INTEGRATION.md](./PHASE5_TRM_TITANS_INTEGRATION.md)**
- **CRITICAL**: Phase 1 TRM × Titans-MAG architecture reuse
- MuGrokfast optimizer STE mode for BitNet
- Tool use integration with recursive thinking
- All Phase 1 stability fixes preserved
- Complete training flow per level

### Executive Summary
📙 **[PHASE5_V2_IMPLEMENTATION_SUMMARY.md](./PHASE5_V2_IMPLEMENTATION_SUMMARY.md)**
- System overview
- 16-week roadmap
- Risk analysis
- Success criteria
- W&B metrics (7,200+)

### Visual Documentation
🎨 **[graphviz/phase5-master-curriculum-flow.dot](./graphviz/phase5-master-curriculum-flow.dot)**
- Complete system flowchart
- All 7 stages visualized
- Formulas annotated

---

## V1 vs V2: What Changed?

| Aspect | V1 (Old) | V2 (New - Documented) |
|--------|----------|------------------------|
| **Approach** | Simple training loop | 7-stage curriculum pipeline |
| **Dataset** | Fixed | Adaptive (shrinks as concepts mastered) |
| **Frontier Models** | None | Orchestrates GPT-4/Claude for questions, variants, hints |
| **Training** | Standard backprop | Recursive thinking + tool use + validation |
| **Self-Awareness** | None | Temperature range self-modeling |
| **Memory** | Standard | Dream consolidation (prevents forgetting) |
| **Specialization** | Generic | Modular (coding/research/writing) |
| **Duration** | Unknown | 120-240 hrs (5-10 days) |
| **Cost** | Compute only | + $600-800 OpenRouter API |

---

## The 7-Stage Pipeline

```
Phase 4 BitNet Model
    ↓
1. ASSESSMENT
   └─ Find edge of chaos (75% accuracy threshold)
    ↓
2. CURRICULUM GENERATION
   └─ Frontier models create 20,000 questions (10 levels)
    ↓
3. TRAINING LOOP (per level)
   ├─ Question → Recursive thinking → Code tool → Validate
   ├─ Success: Create variant, remove after 3× consecutive
   └─ Failure: Root cause → Generate hint → Re-shuffle
    ↓
4. PROMPT BAKING
   ├─ Eudaimonia moral compass (4 rules)
   ├─ OODA loop (3 parts)
   └─ Identity/purpose
    ↓
5. SELF-MODELING
   ├─ Generate at temperature ranges (expanding per level)
   └─ Predict own outputs (grokking about itself)
    ↓
6. SLEEP & DREAM
   └─ Memory consolidation (high-temp replay)
    ↓
7. LEVEL PROGRESSION
   └─ Repeat 3-6 for levels 2-10
    ↓
Phase 6 Specialized Agent
```

---

## Key Innovations

### 1. **Dataset Shrinkage = Progress Metric**
- Start: 2,000 questions/level
- End: 0-50 questions/level (95%+ mastery)
- Variants prevent memorization
- Removal requires 3× consecutive successes

### 2. **Frontier Model Orchestration**
- **Question Gen**: GPT-4, Claude, Gemini create 500Q × 10 levels
- **Variants**: Tweak questions (change nouns/numbers, keep concept)
- **Hints**: Root cause analysis → Targeted guidance

### 3. **Self-Modeling Across Temperatures**
- Level 1: Train at temps 0.0-2.0 (10 ranges)
- Level 10: Train at temps 0.9-3.8 (19 ranges, wider)
- Model predicts its own outputs → Meta-cognition

### 4. **Dream Consolidation**
- After each level: Generate dreams at temp 1.5
- Train on dreams → Strengthen memory
- Prevents catastrophic forgetting

---

## Research Papers

Located in `phases/phase5/` folder:

1. **"Intelligence at the Edge of Chaos"** (PDF)
   - 75% correctness = optimal learning zone
   - Assessment stage finds this threshold

2. **"Unexpected Benefits of Self-Modeling in Neural Systems"** (PDF)
   - Self-prediction improves internal representations
   - Stage 5 implements this

3. **"Dreaming Is All You Need"** (PDF)
   - High-temp replay consolidates memory
   - Stage 6 prevents forgetting

---

## Expected Outputs

```python
{
    "success": True,
    "model_path": "./phase5_specialized_output",
    "specialization": "coding",  # or "research", "writing"

    "metrics": {
        "levels_completed": 10,
        "total_training_time_hours": 156,

        "curriculum_stats": {
            "initial_questions": 20000,
            "final_questions_level_10": 47,  # 97.6% mastered
            "variants_generated": 18500,
            "avg_hints_per_failed_question": 2.3
        },

        "self_modeling": {
            "self_prediction_accuracy": 0.96,
            "temperature_ranges_trained": 145,
            "grokking_achieved": True
        },

        "tool_use": {
            "code_executions": 45230,
            "successful_programs": 41870,  # 92.6%
            "validation_accuracy": 0.98
        }
    }
}
```

---

## Success Criteria

- ✅ Assessment finds 73-77% baseline accuracy (edge of chaos)
- ✅ Dataset shrinks to <5% per level (95%+ mastery)
- ✅ Self-modeling achieves >95% self-prediction accuracy
- ✅ Tool use (coding) success rate >90% by level 10
- ✅ No catastrophic forgetting across levels
- ✅ Moral compass prevents unethical outputs
- ✅ Identity stable across training

---

## Implementation Timeline

**16 Weeks Total**:
- Weeks 1-2: Infrastructure (OpenRouter API, sandbox, DB)
- Weeks 3-4: Assessment system
- Weeks 5-6: Training loop core
- Weeks 7-8: Variant & hint systems
- Weeks 9-10: Dataset mechanics
- Weeks 11-12: Self-modeling
- Weeks 13-14: Dream consolidation
- Weeks 15-16: Integration & testing

---

## Modular Specializations

**Same system, different parameters:**

| Specialty | Tool | Validation | Output |
|-----------|------|------------|--------|
| **Coding** | Python/JS sandbox | Code runs, passes tests | Functional programmer |
| **Research** | Web search, PDFs | Citation accuracy | Literature reviewer |
| **Writing** | Style analyzer | Readability, coherence | Content writer |
| **Math** | SymPy | Proof correctness | Problem solver |

---

## Critical Questions (Need User Input)

1. **Eudaimonia 4 Rules**: What are the specific moral principles?
2. **OODA 3 Parts**: Define the 3-part ethical decision loop
3. **Frontier Models**: GPT-4, Claude-3.5, Gemini? Budget?
4. **Dream Duration**: How long per consolidation phase?
5. **Hard Wall**: What accuracy threshold means "can't progress"?

---

## File Manifest

### ✅ Completed
- `PHASE5_LOGICAL_UNDERSTANDING_V2.md` (5,200 words) - **START HERE**
- `PHASE5_CURRICULUM_SYSTEM.md` (6,800 words) - Implementation details
- `PHASE5_V2_IMPLEMENTATION_SUMMARY.md` - Executive summary + roadmap
- `PHASE5_TRM_TITANS_INTEGRATION.md` (3,800 words) - **⭐ Phase 1 architecture reuse**
- `graphviz/phase5-master-curriculum-flow.dot` - Master flowchart
- Research PDFs: Edge of Chaos, Self-Modeling, Dreaming

### 📋 To Be Created
- `PHASE5_FORMULAS.md` - Mathematical models consolidated
- `PHASE5_INTEGRATION_PHASE4.md` - BitNet-specific considerations
- `PHASE5_PREMORTEM_V2.md` - Expanded risk analysis
- `PHASE5_FRONTIER_INTEGRATION.md` - OpenRouter API specs
- `PHASE5_TOOL_USE_SPEC.md` - Coding environment architecture
- `PHASE5_SELF_MODELING.md` - Temperature training details
- `PHASE5_EUDAIMONIA_SYSTEM.md` - Moral compass prompts
- Additional GraphViz diagrams (4 more planned)

---

## Old V1 Documentation (Archived)

**V1 System** (Simple Grokfast training):
- `PHASE5_COMPLETE_GUIDE_V1_OLD.md` - Old implementation guide
- `graphviz/phase-flow.dot` - Old simple flowchart
- `README.md` - Old overview (will be updated)

**Note**: V1 was never fully implemented. V2 is a complete redesign based on clarified requirements.

---

## Next Steps

1. **Read**: [PHASE5_LOGICAL_UNDERSTANDING_V2.md](./PHASE5_LOGICAL_UNDERSTANDING_V2.md)
2. **Review**: [PHASE5_CURRICULUM_SYSTEM.md](./PHASE5_CURRICULUM_SYSTEM.md)
3. **Check**: [PHASE5_V2_IMPLEMENTATION_SUMMARY.md](./PHASE5_V2_IMPLEMENTATION_SUMMARY.md)
4. **Answer**: Critical questions (eudaimonia rules, OODA parts, etc.)
5. **Implement**: Follow 16-week roadmap

---

**Phase Input**: [Phase 4: BitNet Compression](../phase4/LOGICAL_UNDERSTANDING.md)
**Phase Output**: [Phase 6: Tool & Persona Baking](../phase6/LOGICAL_UNDERSTANDING.md)

**Questions?** See implementation summary or contact project lead.

---

**Last Updated**: 2025-01-16
**Version**: 2.0 (V2 Curriculum Learning System)
**Status**: Specification Complete - Ready for Implementation

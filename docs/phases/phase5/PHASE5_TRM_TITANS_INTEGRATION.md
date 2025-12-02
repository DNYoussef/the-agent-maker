# Phase 5: TRM × Titans-MAG Training Integration

**Date**: 2025-10-16
**Purpose**: Integrate Phase 1's TRM × Titans-MAG architecture into Phase 5's curriculum learning system
**Status**: Specification Complete

---

## Executive Summary

Phase 5 will **reuse the complete TRM × Titans-MAG training system from Phase 1**, adapting it for curriculum-based specialization training with the BitNet 1.58-bit quantized model.

**Key Integration Points**:
1. **Architecture**: Use same TRM recursive loop + Titans time-based memory
2. **Optimizer**: MuGrokfast in STE mode (for BitNet compatibility)
3. **Training Loop**: Adapt Phase 1 trainer for curriculum + tool use
4. **Memory Systems**: Both LTM (time-based) and TRM (recursive) work together

---

## Architecture Reuse from Phase 1

### What We're Bringing to Phase 5

**Complete TRM × Titans-MAG System**:
- ✅ 8-layer Titans-MAG backbone
- ✅ Long-Term Memory (LTM) with exponential decay
- ✅ MAG Gate (blend current + memory)
- ✅ TRM recursive loop (3 iterations, 2 micro-steps)
- ✅ ACT Head (Adaptive Computation Time)
- ✅ MuGrokfast optimizer (Grokfast × Muon)
- ✅ All critical fixes (detached memory states, dimension-aware orthogonalization)

**Why This Works for Phase 5**:
1. **Recursive Thinking**: TRM's multi-pass reasoning perfect for complex coding problems
2. **Time-Based Memory**: Titans LTM retains context across long sequences
3. **Tool Use**: TRM can plan → execute → validate in loop
4. **BitNet Compatible**: MuGrokfast has STE mode for 1.58-bit quantization

---

## Phase 1 vs Phase 5: Architecture Comparison

| Component | Phase 1 (Cognate) | Phase 5 (Curriculum) | Changes Needed |
|-----------|-------------------|----------------------|----------------|
| **Model Architecture** | TRM × Titans-MAG (26.97M params) | Same architecture, but BitNet quantized | None - architecture identical |
| **Parameter Precision** | Full precision (FP32/FP16) | 1.58-bit (from Phase 4) | Optimizer uses STE mode |
| **Training Objective** | General language modeling | Specialized curriculum (coding/research/writing) | Task-specific loss functions |
| **Optimizer** | MuGrokfast (Phase 1 preset) | MuGrokfast (Phase 5 preset, STE enabled) | Enable STE mode, increase λ |
| **Dataset** | 16 HuggingFace datasets, 3-stage curriculum | Adaptive curriculum (shrinks from 20K→0), frontier-generated | New dataset pipeline |
| **Training Loop** | Standard backprop on language modeling | Recursive thinking → Tool use → Validation → Variant/Hint | Major extension |
| **Memory Management** | LTM detached, features detached | Same detach strategy | No changes |
| **Prompt Baking** | None | Eudaimonia + OODA + Identity (3×5min per level) | Add baking system |
| **Self-Modeling** | None | Temperature range prediction (10-19 ranges/level) | New training stage |
| **Dream Consolidation** | None | High-temp replay (1 epoch/level) | New training stage |

---

## Detailed Integration Plan

### 1. Model Architecture (From Phase 1)

**Use Existing Phase 1 Model**:
```python
# Phase 5 imports Phase 1 model directly
from phase1_cognate.model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config, TitansMAGConfig, TRMConfig, ACTConfig

# Load Phase 4 BitNet model
phase4_model = load_bitnet_model("./phase4_output/compressed_model.pt")

# Extract architecture config (same as Phase 1, but quantized)
phase5_config = Phase1Config(
    specialization="coding",  # New: specialization type
    titans_config=TitansMAGConfig(
        d_model=320,
        n_layers=8,
        n_heads=5,
        # ... same as Phase 1
    ),
    trm_config=TRMConfig(
        T_max=3,  # 3 recursive iterations
        micro_steps=2,  # 2 refinement steps per iteration
        deep_supervision=False,  # Keep disabled (from Phase 1 fix)
        detach_between_steps=True,  # CRITICAL: Memory efficiency
    ),
    act_config=ACTConfig(
        halt_threshold=0.5,  # Same as Phase 1
        ema_decay=0.98,
    ),
    # Phase 5 specific
    curriculum_enabled=True,
    tool_use_enabled=True,
    self_modeling_enabled=True,
)

# Create model (reuses Phase 1 architecture)
model = TRMTitansMAGModel(phase5_config)

# Load Phase 4 weights
model.load_state_dict(phase4_model.state_dict())
```

**No Architecture Changes Needed** ✅
- Same TRM recursive loop (3 iterations)
- Same Titans-MAG backbone (8 layers, LTM, MAG gate)
- Same ACT head for adaptive halting
- All Phase 1 fixes preserved (detached memory, dimension-aware Muon)

---

### 2. MuGrokfast Optimizer (STE Mode for BitNet)

**Phase 1 Optimizer** (Full Precision):
```python
from cross_phase.mugrokfast import create_optimizer_from_phase

optimizer = create_optimizer_from_phase(model, phase_num=1)
# Uses: muon_lr=1e-3, grokfast_lambda=0.3, muon_ste_mode=False
```

**Phase 5 Optimizer** (BitNet STE Mode):
```python
from cross_phase.mugrokfast import create_optimizer_from_phase

optimizer = create_optimizer_from_phase(model, phase_num=5)
# Uses: muon_lr=5e-4, grokfast_lambda=2.0, muon_ste_mode=True, qk_clip=25.0
```

**What Changes**:
- ✅ **STE Mode Enabled**: `muon_ste_mode=True`
  - Forward pass: 1.58-bit quantized weights
  - Backward pass: Full-precision gradients
  - Critical for training BitNet models
- ✅ **Aggressive Filtering**: `grokfast_lambda=2.0` (vs 0.3 in Phase 1)
  - Stronger EMA gradient filtering to handle quantization noise
- ✅ **Lower Learning Rate**: `muon_lr=5e-4` (vs 1e-3 in Phase 1)
  - More conservative updates for quantized weights
- ✅ **QK-Clip Enabled**: `qk_clip_threshold=25.0`
  - Prevents attention instability during tool use training

**Implementation** (already in mugrokfast/presets.py):
```python
def get_phase_5_config() -> MuGrokConfig:
    """
    Phase 5: Curriculum learning with BitNet
    - STE mode for quantized training
    - Aggressive filtering for noise
    - QK-clip for tool use stability
    """
    return MuGrokConfig(
        muon_lr=5e-4,
        fallback_lr=1e-4,
        grokfast_alpha=0.98,
        grokfast_lambda=2.0,  # Aggressive filtering
        qk_clip_threshold=25.0,
        muon_ste_mode=True,  # BitNet compatibility
        momentum=0.95,
        nesterov=True,
        ns_steps=5
    )
```

---

### 3. Training Loop Integration

**Phase 1 Training Loop** (Simple):
```python
# Phase 1: Standard language modeling
for batch in dataloader:
    output = model(input_ids, labels=labels)
    loss = output["loss"]  # CE + ACT + MAG gate

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Phase 5 Training Loop** (Extended for Curriculum + Tool Use):
```python
# Phase 5: Recursive thinking → Tool use → Validation → Feedback
for question in curriculum[current_level]:
    # 1. RECURSIVE THINKING (TRM system)
    # Model uses TRM's 3 iterations to plan approach
    output = model(
        input_ids=question.prompt_tokens,
        labels=None,  # No labels yet - generating
        return_all_steps=True  # Get all recursion steps
    )

    # Extract thought process from TRM recursion
    thoughts = output["y_history"]  # [y0, y1, y2, y3]
    final_thoughts = thoughts[-1]  # y3 = final answer state

    # 2. CODE GENERATION (using TRM's final state)
    code_tokens = model.generate_code(
        thoughts=final_thoughts,
        tool_instructions=coding_env.get_instructions(),
        max_tokens=512
    )

    # 3. TOOL USE (execute in sandbox)
    execution_result = coding_env.execute(
        code=tokenizer.decode(code_tokens),
        timeout=5.0
    )

    # 4. VALIDATION
    validation = validate_code(execution_result, question.test_cases)
    success = validation.passed

    # 5. FEEDBACK LOOP
    if success:
        # SUCCESS PATH: Create variant
        variant = frontier_model.create_variant(question)
        curriculum.replace_question(question.id, variant)
        curriculum.increment_success_count(question.concept_id)

        # Remove after 3 consecutive successes
        if curriculum.get_success_count(question.concept_id) >= 3:
            curriculum.remove_question(question.id)

    else:
        # FAILURE PATH: Generate hint
        hint = frontier_model.generate_hint(
            question=question,
            thoughts=thoughts,
            code=code_tokens,
            error=execution_result.error
        )
        curriculum.add_hint(question.id, hint)
        curriculum.reset_success_count(question.concept_id)

    # 6. TRAIN ON OUTCOME (only if we have labels)
    if success:
        # Train on successful execution
        loss = compute_success_loss(
            model_output=output,
            correct_code=code_tokens,
            execution_result=execution_result
        )
    else:
        # Train on corrected version (with hints)
        corrected_output = model(
            input_ids=question.with_hints(),
            labels=question.correct_answer_tokens
        )
        loss = corrected_output["loss"]

    # 7. BACKPROP (same as Phase 1)
    loss.backward()
    optimizer.step()  # MuGrokfast with STE mode
    optimizer.zero_grad()
```

**Key Differences from Phase 1**:
1. **TRM Used for Planning**: Recursive thinking generates approach before coding
2. **Tool Use Integration**: TRM's final state feeds into code generator
3. **External Validation**: Code execution provides ground truth
4. **Curriculum Feedback**: Success/failure updates dataset
5. **Same Optimizer Step**: MuGrokfast handles backprop (with STE for BitNet)

---

### 4. Memory System Integration

**Phase 1 Memory Systems** (Both Work in Phase 5):

#### Long-Term Memory (LTM) - Time-Based
```python
# Titans-MAG backbone
class LongTermMemory(nn.Module):
    def forward(self, x):
        # Process sequence sequentially (time-based)
        for t in range(seq_len):
            self.memory_state = (
                decay * self.memory_state +
                (1 - decay) * x[:, t:t+1, :]
            ).detach()  # ✅ CRITICAL FIX from Phase 1
            m_list.append(self.memory_state)

        return self.w_up(torch.cat(m_list, dim=1))
```

**Usage in Phase 5**: Retains coding context across long sequences
- Example: Multi-step problems with setup → implementation → testing
- LTM remembers earlier parts while processing later parts

#### TRM Recursive Memory - Iteration-Based
```python
# TRM wrapper
class TRMWrapper(nn.Module):
    def forward(self, features):
        # Initialize
        z = self.z0_proj(features)
        y = self.y0_proj(z)
        y_history = [y]

        # Recursive loop (3 iterations)
        for t in range(T_max=3):
            z_refined = self.refiner(z, features, y, n_steps=2)
            y_new = self.updater(y, z_refined)

            y_history.append(y_new)

            # ✅ CRITICAL FIX from Phase 1: Detach
            y = y_new.detach()
            z = z_refined.detach()

        return y_history  # [y0, y1, y2, y3]
```

**Usage in Phase 5**: Multi-pass reasoning for complex problems
- Iteration 0 (y0): Initial understanding
- Iteration 1 (y1): Plan approach
- Iteration 2 (y2): Refine plan with edge cases
- Iteration 3 (y3): Final solution strategy

**Both Systems Work Together** ✅:
- LTM: Tracks context across time (within a problem)
- TRM: Refines reasoning across iterations (within thought process)
- No interference (fixed in Phase 1 with detached states)

---

### 5. Prompt Baking Integration

**Phase 5 Addition** (Not in Phase 1):

After each curriculum level, bake 3 prompts into weights:

```python
from prompt_baking import bake_prompt, PromptBakingConfig

# After level N training completes:
config = PromptBakingConfig(
    lora_r=16,
    num_epochs=3,
    learning_rate=1e-4
)

# 1. Eudaimonia moral compass (5 min)
model = bake_prompt(model, eudaimonia_prompt, config)

# 2. OODA ethical loop (5 min)
model = bake_prompt(model, ooda_prompt, config)

# 3. Identity & purpose (5 min)
model = bake_prompt(model, identity_prompt, config)

# Total: 15 minutes per level
# Total Phase 5: 10 levels × 15 min = 150 minutes (2.5 hours)
```

**Integration with TRM Architecture**:
- Prompt baking uses LoRA adapters on TRM's refiner/updater
- Baked prompts influence thought process during recursion
- Example: Eudaimonia rules integrated into y1 → y2 refinement step

---

### 6. Self-Modeling Integration

**Phase 5 Addition** (Not in Phase 1):

Train model to predict its own outputs across temperature ranges:

```python
# After level N training + baking
def self_modeling_stage(model, level):
    # Generate temperature ranges (expanding per level)
    temp_ranges = calculate_temperature_ranges(level)
    # Level 1: 10 ranges, Level 10: 19 ranges

    for temp_range in temp_ranges:
        # 1. GENERATION PHASE
        # Use TRM to generate at midpoint temperature
        self_generated = []
        for question in sample_questions(n=100):
            output = model.generate(
                question,
                temperature=temp_range.midpoint,
                include_thoughts=True,  # TRM y_history
                use_trm=True  # Enable recursive thinking
            )
            self_generated.append({
                'text': output.generated_text,
                'thoughts': output.y_history,  # [y0, y1, y2, y3]
                'temp': temp_range.midpoint
            })

        # 2. MASKING PHASE
        for sample in self_generated:
            # Mask 20% of tokens
            masked, targets = mask_tokens(sample.text, rate=0.2)

            # 3. PREDICTION PHASE
            # Model predicts own text (self-aware)
            predictions = model(
                masked,
                temperature=temp_range.midpoint,
                context_tag="self_generated_at_{}".format(temp_range.midpoint),
                use_trm=True  # Recursive prediction
            )

            # 4. SELF-MODELING LOSS
            loss = cross_entropy(predictions.logits, targets)

            # 5. BACKPROP (MuGrokfast STE mode)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Return self-modeled model
    return model
```

**TRM Integration**:
- TRM's recursive thinking used during generation
- TRM's y_history captured as "thought process"
- During prediction, TRM helps reconstruct own reasoning
- Self-modeling improves TRM's internal coherence

---

### 7. Dream Consolidation Integration

**Phase 5 Addition** (Not in Phase 1):

After each level, consolidate memory via high-temp replay:

```python
# After level N self-modeling
def dream_consolidation(model, level_training_data):
    # 1. Sample training experiences
    dream_prompts = sample(level_training_data, n=1000)

    # 2. Generate "dreams" at high temp (creative replay)
    dreams = []
    for prompt in dream_prompts:
        # TRM generates at temp 1.5 (more creative)
        output = model.generate(
            prompt,
            temperature=1.5,
            use_trm=True,  # Recursive dreaming
            max_tokens=512
        )
        dreams.append(output)

    # 3. Train on dreams (consolidation)
    for epoch in range(1):  # 1 epoch
        for dream_batch in batch(dreams, batch_size=16):
            # TRM processes dream (all recursion steps)
            output = model(dream_batch.input_ids, labels=dream_batch.labels)
            loss = output["loss"]  # CE + ACT + MAG gate

            # MuGrokfast STE mode backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # 4. Strengthen memory connections
    # LTM state gets reinforced with dream patterns
    # TRM recursion pathways strengthened

    return model
```

**TRM + LTM Benefits**:
- **LTM**: Retains patterns from dreams (exponential decay strengthened)
- **TRM**: Recursive "reflection" on dreams improves reasoning paths
- **Combined**: Prevents catastrophic forgetting across levels

---

## Complete Phase 5 Training Flow

### Per-Level Pipeline

```python
def train_phase5_level(model, level, curriculum, optimizer):
    """
    Complete training for one curriculum level using Phase 1 architecture
    """
    print(f"=== LEVEL {level} ===")

    # STAGE 3: CURRICULUM TRAINING (12-24 hours)
    # Uses TRM × Titans-MAG for recursive thinking + tool use
    for epoch in range(30):  # Until dataset shrinks to ~0
        for question in curriculum[level]:
            # TRM recursive thinking → Code gen → Validate → Feedback
            success = train_one_question(model, question, coding_env)

            # Update curriculum (variants/hints/removal)
            curriculum.update(question, success)

        # Check convergence
        if len(curriculum[level]) < 50:  # 97.5% mastered
            break

    # STAGE 4: PROMPT BAKING (15 minutes)
    # Bake moral compass + identity into TRM weights
    model = bake_prompt(model, eudaimonia_prompt)  # 5 min
    model = bake_prompt(model, ooda_prompt)        # 5 min
    model = bake_prompt(model, identity_prompt)    # 5 min

    # STAGE 5: SELF-MODELING (4-8 hours)
    # TRM predicts own outputs across temperature ranges
    temp_ranges = calculate_temperature_ranges(level)
    model = self_modeling_stage(model, temp_ranges)

    # STAGE 6: DREAM CONSOLIDATION (1-2 hours)
    # High-temp replay strengthens TRM + LTM memory
    model = dream_consolidation(model, curriculum[level])

    print(f"Level {level} complete. Dataset shrunk to {len(curriculum[level])} questions")

    return model

# FULL PHASE 5 (10 levels, 120-240 hours)
for level in range(1, 11):
    model = train_phase5_level(model, level, curriculum, optimizer)
```

---

## Technical Specifications

### Model Configuration (Phase 5)

```python
{
    # Architecture (same as Phase 1)
    "titans_config": {
        "d_model": 320,
        "n_layers": 8,
        "n_heads": 5,
        "d_ff": 1280,
        "vocab_size": 32768,
        "max_seq_len": 2048,
        "sw_window": 1024,
        "d_mem": 160,  # LTM compression
        "memory_decay": 0.99,
        "mag_hidden": 160
    },

    "trm_config": {
        "T_max": 3,  # 3 recursive iterations
        "micro_steps": 2,  # 2 refinement steps per iteration
        "deep_supervision": False,  # Disabled (Phase 1 fix)
        "step_weights": [0.33, 0.5, 0.75, 1.0],  # y0 + 3 recursion steps
        "detach_between_steps": True  # CRITICAL: Memory efficiency
    },

    "act_config": {
        "halt_threshold": 0.5,
        "ema_decay": 0.98,
        "entropy_reg": 0.001,
        "act_loss_weight": 0.01
    },

    # Phase 5 specific
    "specialization": "coding",  # or "research", "writing"
    "curriculum_enabled": True,
    "tool_use_enabled": True,
    "self_modeling_enabled": True,
    "dream_consolidation_enabled": True,

    # Optimizer (MuGrokfast Phase 5 preset)
    "optimizer": {
        "type": "MuGrokfast",
        "muon_lr": 5e-4,
        "fallback_lr": 1e-4,
        "grokfast_lambda": 2.0,  # Aggressive filtering
        "qk_clip_threshold": 25.0,
        "muon_ste_mode": True,  # BitNet STE mode
        "momentum": 0.95,
        "nesterov": True,
        "ns_steps": 5
    },

    # Training
    "batch_size": 16,
    "gradient_clip": 1.0,
    "mixed_precision": False,
    "gradient_checkpointing": True
}
```

### Performance Expectations

| Metric | Phase 1 (Cognate) | Phase 5 (Curriculum) | Notes |
|--------|-------------------|----------------------|-------|
| **VRAM Usage** | ~5GB | ~5GB | Same architecture, BitNet weights slightly smaller |
| **Training Speed** | ~1h/epoch | ~12-24h/level | Longer due to tool use + validation + feedback |
| **Model Quality** | General language | Specialized (90%+ tool success) | Task-specific mastery |
| **Memory Stability** | Stable (detached LTM) | Stable (same fixes) | No changes needed |
| **Optimizer Stability** | Stable (Muon + Grokfast) | Stable (STE mode for BitNet) | Tested in Phase 1 |

---

## Critical Fixes Preserved from Phase 1

**All Phase 1 fixes are preserved in Phase 5**:

### 1. LTM Memory State Detach ✅
```python
# Titans-MAG LTM (Phase 1 fix applied)
self.memory_state = (
    decay * self.memory_state + (1-decay) * x[:, t:t+1, :]
).detach()  # ← CRITICAL: Prevents "backward through graph twice" error
```

### 2. TRM Features Detach ✅
```python
# TRM wrapper (Phase 1 fix applied)
features_first = features
features_detached = features.detach()

for t in range(T_max):
    feat_to_use = features_first if t == 0 else features_detached
    z_refined = self.refiner(z, feat_to_use, y)
```

### 3. MuGrokfast Dimension-Aware Orthogonalization ✅
```python
# MuGrokfast optimizer (Phase 1 fix applied)
if G.shape[0] > G.shape[1]:  # Tall matrix (embeddings)
    col_norms = G.norm(dim=0, keepdim=True)
    G = G / col_norms  # ← Prevents low-rank collapse
```

### 4. Step Weights Configuration ✅
```python
# TRMConfig (Phase 1 fix applied)
step_weights = [0.33, 0.5, 0.75, 1.0]  # y0 + 3 recursion steps
```

**No regressions** - All stability fixes from Phase 1 remain active in Phase 5.

---

## File Structure (Phase 5)

```
src/phase5_curriculum/
├── model/
│   ├── __init__.py
│   └── phase5_config.py          # Extends Phase1Config with curriculum settings
│
├── training/
│   ├── curriculum_trainer.py     # Extends Phase1 trainer with tool use
│   ├── tool_environment.py       # Coding sandbox
│   ├── validation.py             # Code validation
│   ├── variant_generator.py      # Frontier model variant creation
│   ├── hint_generator.py         # Root cause analysis → Hints
│   ├── self_modeling.py          # Temperature range self-prediction
│   └── dream_consolidation.py    # High-temp memory replay
│
├── curriculum/
│   ├── curriculum_manager.py     # Dataset shrinkage, variant/hint tracking
│   ├── question_lifecycle.py     # State machine (active→variant→mastered)
│   └── frontier_api.py           # OpenRouter integration
│
└── train_phase5.py               # Main CLI (similar to Phase 1)
```

**Reused from Phase 1**:
- `src/phase1_cognate/model/` - Complete TRM × Titans-MAG architecture
- `src/cross_phase/mugrokfast/` - MuGrokfast optimizer with STE mode
- `src/cross_phase/prompt_baking/` - Prompt baking system

---

## Success Criteria

### Architecture Integration
- ✅ Phase 1 TRM × Titans-MAG model loads successfully
- ✅ Phase 4 BitNet weights compatible with architecture
- ✅ MuGrokfast STE mode enables quantized training
- ✅ All Phase 1 memory fixes preserved (no graph errors)

### Training Stability
- ✅ LTM memory state detached (no backward errors)
- ✅ TRM recursion stable across curriculum levels
- ✅ MuGrokfast handles embedding layers correctly
- ✅ Tool use execution doesn't corrupt gradients

### Performance
- ✅ VRAM usage ≤6GB (same as Phase 1)
- ✅ Training converges within 12-24 hours/level
- ✅ Dataset shrinks to <5% (proves mastery)
- ✅ Tool use success rate >90% by level 10

### Quality
- ✅ Self-modeling achieves >95% self-prediction accuracy
- ✅ Dream consolidation prevents catastrophic forgetting
- ✅ Moral compass + identity baked successfully
- ✅ Model specializes in target domain

---

## Risk Mitigation

### Potential Issues from Phase 1 Integration

| Risk | Mitigation | Status |
|------|------------|--------|
| **BitNet quantization breaks TRM recursion** | Use MuGrokfast STE mode (full-precision gradients) | ✅ Addressed |
| **Tool use corrupts LTM memory state** | Keep LTM detach, add validation checkpointing | ✅ Addressed |
| **Curriculum feedback causes graph issues** | Use same detach strategy from Phase 1 | ✅ Addressed |
| **Self-modeling breaks with quantized weights** | Generate at inference mode, train with STE | ✅ Addressed |
| **Dream consolidation OOM errors** | Use gradient checkpointing (same as Phase 1) | ✅ Addressed |

---

## Implementation Timeline

### Weeks 1-2: Phase 1 Integration
- Import Phase 1 model architecture
- Load Phase 4 BitNet weights
- Configure MuGrokfast Phase 5 preset (STE mode)
- Test forward/backward pass with quantized weights
- **Deliverable**: Phase 1 architecture runs with BitNet weights

### Weeks 3-4: Tool Use Integration
- Build coding sandbox environment
- Extend Phase 1 trainer with tool use loop
- Implement validation harness
- Test TRM → Code generation → Execution
- **Deliverable**: Tool use works with TRM recursion

### Weeks 5-6: Curriculum System
- Implement variant generator (frontier models)
- Implement hint generator (root cause analysis)
- Build curriculum manager (shrinkage mechanics)
- Test variant/hint feedback loop
- **Deliverable**: Curriculum adapts to model performance

### Weeks 7-8: Prompt Baking
- Integrate prompt baking system
- Test baking with TRM architecture
- Verify moral compass persistence
- **Deliverable**: Eudaimonia + OODA + Identity baked successfully

### Weeks 9-10: Self-Modeling
- Implement temperature range generation
- Build self-prediction training loop
- Test with TRM recursive prediction
- **Deliverable**: Model predicts own outputs >95% accuracy

### Weeks 11-12: Dream Consolidation
- Implement high-temp dream generation
- Build consolidation training loop
- Test memory strengthening
- **Deliverable**: No catastrophic forgetting across levels

### Weeks 13-14: Integration Testing
- Run full pipeline (levels 1-3)
- Monitor VRAM, training time, quality
- Fix any issues
- **Deliverable**: End-to-end pipeline working

### Weeks 15-16: Full Training
- Train all 10 levels (or until hard wall)
- Monitor dataset shrinkage
- Validate final model
- **Deliverable**: Specialized agent ready for Phase 6

**Total**: 16 weeks (same as original Phase 5 timeline)

---

## Conclusion

Phase 5 will **fully reuse the TRM × Titans-MAG training system from Phase 1**, extending it with curriculum learning, tool use, self-modeling, and dream consolidation.

**Key Advantages**:
1. **Proven Architecture**: Phase 1 tested and validated (26.97M params, all tests passing)
2. **Stability Fixes**: All critical fixes preserved (detached memory, dimension-aware Muon)
3. **BitNet Compatible**: MuGrokfast STE mode enables quantized training
4. **Modular Extension**: Tool use + curriculum built on top of Phase 1 base

**No Regressions**:
- Same memory management (detached LTM, detached TRM features)
- Same optimizer (MuGrokfast with Phase 5 preset)
- Same VRAM usage (~5GB)
- Same training stability

**New Capabilities**:
- Recursive thinking → Tool use → Validation
- Adaptive curriculum (shrinks from 20K→0)
- Self-modeling across temperatures
- Dream consolidation
- Moral compass baking

**Ready for Implementation** ✅

---

**Phase Input**: [Phase 4: BitNet Compression](../phase4/LOGICAL_UNDERSTANDING.md)
**Phase Output**: [Phase 6: Tool & Persona Baking](../phase6/LOGICAL_UNDERSTANDING.md)

**Related Documentation**:
- [Phase 1: TRM × Titans-MAG Architecture](../phase1/ARCHITECTURE_VALIDATION.md)
- [Phase 1: Implementation Status](../phase1/PHASE1_IMPLEMENTATION_STATUS.md)
- [Phase 5: Logical Understanding V2](./PHASE5_LOGICAL_UNDERSTANDING_V2.md)
- [Phase 5: Curriculum System](./PHASE5_CURRICULUM_SYSTEM.md)

---

**Last Updated**: 2025-10-16
**Version**: 1.0
**Status**: Specification Complete - Ready for Implementation

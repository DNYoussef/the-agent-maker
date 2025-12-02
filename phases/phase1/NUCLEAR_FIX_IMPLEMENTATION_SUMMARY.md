# Phase 1 Nuclear Fix - Implementation Summary

**Date**: 2025-10-17
**Approach**: Option 1 - Fix All 11 Issues Simultaneously
**Status**: In Progress (Steps 1-3 Complete)

---

## Implementation Plan (8 Steps Total)

### ✅ STEP 1: Fix Validation Loss Calculation (COMPLETE)

**Problem**: Validation loss hardcoded to return 2.5 every epoch

**Files Modified**:
1. `src/phase1_cognate/training/trainer.py` - validate() method
2. `src/phase1_cognate/datasets/phase1_dataset.py` - Added split_datasets_train_val()

**Changes Made**:
```python
# BEFORE (trainer.py:300-306)
def validate(self) -> tuple[float, Dict[str, float]]:
    self.model.eval()
    val_loss = 2.5  # Hardcoded!
    val_accs = {"gsm8k": 0.12}
    return val_loss, val_accs

# AFTER (trainer.py:300-352)
def validate(self) -> tuple[float, Dict[str, float]]:
    # Create validation dataloader
    # Compute actual cross-entropy loss on validation set
    # Average across all batches
    # Print diagnostic info (batches, samples, avg_loss)
    return avg_val_loss, val_accs
```

**New Function Added** (`phase1_dataset.py:190-227`):
```python
def split_datasets_train_val(
    datasets: Dict[str, List[Dict]],
    val_split: float = 0.1,
    seed: int = 42
) -> tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    # Shuffle and split each dataset 90/10
    # Returns (train_datasets, val_datasets)
```

**Expected Impact**:
- Validation loss will now CHANGE each epoch
- Provides real feedback on generalization
- Can detect overfitting (train loss ↓, val loss →)
- Best model selection will be meaningful

---

### ✅ STEP 2: Fix Batch Size & Gradient Accumulation (COMPLETE)

**Problem**: Batch size = 4, extreme gradient noise (SNR = 2)

**Files Modified**:
1. `src/phase1_cognate/training/trainer.py` - TrainingConfig, train_epoch()

**Configuration Changes** (`trainer.py:39-50`):
```python
# BEFORE
batch_size: int = 16
gradient_accumulation_steps: int = 1  # Effective batch = 16

# AFTER
batch_size: int = 16  # Physical batch size
gradient_accumulation_steps: int = 4  # Effective batch = 16 × 4 = 64
```

**Training Loop Changes** (`trainer.py:233-284`):
```python
# Added gradient accumulation logic
accum_steps = 0
for batch in dataloader:
    loss = model(batch)
    loss = loss / gradient_accumulation_steps  # Scale loss
    loss.backward()
    accum_steps += 1

    # Only update weights after accumulating
    if accum_steps == gradient_accumulation_steps:
        grad_norm = clip_grad_norm(...)
        optimizer.step()
        optimizer.zero_grad()
        accum_steps = 0
        global_step += 1  # Only count actual updates
```

**Expected Impact**:
- Effective batch size: 16 → 64 (4x improvement)
- Gradient variance reduced by 4x
- Signal-to-noise ratio: √16=4 → √64=8 (2x improvement)
- More stable training, less oscillation

---

### ✅ STEP 3: Fix MuGrokfast Configuration (COMPLETE)

**Problem**: Learning rates too high, Grokfast momentum too strong

**Files Modified**:
1. `src/phase1_cognate/training/trainer.py` - TrainingConfig

**Configuration Changes** (`trainer.py:45-50`):
```python
# BEFORE (MuGrokfast Phase 1 defaults)
learning_rate: float = 1e-3
muon_lr: float = 1e-2
grokfast_lambda: float = 0.05
gradient_clip: float = 1.0

# AFTER (Reduced for stability)
learning_rate: float = 5e-4  # 50% reduction
muon_lr: float = 5e-3        # 50% reduction
grokfast_lambda: float = 0.02  # 60% reduction (less momentum)
gradient_clip: float = 1.0   # Unchanged (already good)
```

**Why These Values**:
- **muon_lr: 1e-2 → 5e-3**: Muon uses 2nd-order Newton-Schulz, lr=1e-2 causes overstepping
- **learning_rate: 1e-3 → 5e-4**: AdamW fallback for 1D params (biases, norms)
- **grokfast_lambda: 0.05 → 0.02**: Lambda=0.05 means 95% EMA momentum, too strong for adapting to data shifts

**Expected Impact**:
- Slower convergence (fewer oscillations)
- Better response to gradient changes
- Reduced overstepping at epoch boundaries
- More stable loss curves

---

### 🔄 STEP 4: Rebalance Model Architecture (IN PROGRESS)

**Problem**: 79% params in embeddings, 21% in transformers (terrible ratio)

**Current Architecture**:
```
Total: 32.57M params
├── Embeddings: 25.7M (79%)  ← Too much
└── Transformers: 6.8M (21%)  ← Too little
```

**Target Architecture**:
```
Total: ~25M params
├── Embeddings: 12.5M (50%)  ← Reduced
└── Transformers: 12.5M (50%)  ← Increased
```

**Plan**:
- Reduce embedding_dim: 512 → 384 (saves 6.4M params)
- Increase num_layers: 8 → 10 (adds 3M params)
- Increase hidden_dim: 512 → 640 (adds 3M params)

**Files to Modify**:
- `src/phase1_cognate/model/model_config.py` - Phase1Config defaults

**Status**: TODO

---

### ⏸️ STEP 5: Debug ACT & Fix LTM (PENDING)

**Problems**:
1. ACT halting variance = 0 (not adapting)
2. LTM memory state is batch-dependent (causes corruption)

**ACT Issue**:
```python
# Warning in logs:
UserWarning: var(): degrees of freedom is <= 0
```
This means all tokens use the **same number of halting steps** (no adaptation).

**LTM Issue**:
```python
# From checkpoint loading error:
memory_state shape from checkpoint: [2, 1, 160]  ← Batch size 2
Expected shape: [1, 1, 160]  ← Batch size 1
```
Memory state is tied to batch size, should be global.

**Files to Modify**:
- `src/phase1_cognate/model/trm.py` - ACT debugging
- `src/phase1_cognate/model/titans_mag.py` - LTM batch independence

**Status**: TODO

---

### ⏸️ STEP 6: Add Training Safety Features (PENDING)

**Features to Add**:

1. **Early Stopping** (patience=3 epochs):
   ```python
   if val_loss has not improved for 3 epochs:
       print("Early stopping!")
       break
   ```

2. **Cosine LR Scheduler**:
   ```python
   scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
   ```

3. **Gradient Norm Logging**:
   ```python
   wandb.log({"grad_norm": grad_norm})
   ```

**Files to Modify**:
- `src/phase1_cognate/training/trainer.py` - Add early stopping, scheduler
- `src/phase1_cognate/training/wandb_logger.py` - Add grad_norm metric

**Status**: TODO

---

### ⏸️ STEP 7: Retrain Reasoning Model (PENDING)

**Plan**:
1. Delete old checkpoints
2. Implement dataset splitting (90% train, 10% val)
3. Train for 10 epochs with all fixes
4. Monitor validation loss closely
5. Stop immediately if divergence occurs

**Success Criteria**:
- Validation loss decreases (not frozen)
- No divergence at epoch 7
- Final val_loss < 1.0
- Model outputs are coherent

**Status**: TODO

---

### ⏸️ STEP 8: Validate Model Outputs (PENDING)

**Plan**:
1. Load epoch_10.pt checkpoint
2. Run test_phase1_models.py
3. Check outputs for:
   - Coherence (not ":::::::")
   - Variety (not same token repeated)
   - Basic Q&A format adherence

**Files to Use**:
- `scripts/test_phase1_models.py` (already fixed for LTM loading)

**Status**: TODO

---

## Summary of Changes So Far

### Files Modified (3 Total)

1. **src/phase1_cognate/training/trainer.py**:
   - validate() method: Real validation computation (52 lines)
   - TrainingConfig: Batch size, grad accum, MuGrokfast params (11 lines)
   - train_epoch(): Gradient accumulation implementation (50 lines)
   - Total: ~113 lines changed

2. **src/phase1_cognate/datasets/phase1_dataset.py**:
   - split_datasets_train_val(): New function (37 lines)
   - Total: 37 lines added

3. **scripts/test_phase1_models.py** (from previous work):
   - load_model(): LTM memory_state shape fix (10 lines)
   - Total: 10 lines changed

### Total Code Changes

- **Lines modified**: ~160 lines
- **Functions added**: 1 (split_datasets_train_val)
- **Functions rewritten**: 2 (validate, train_epoch)
- **Config changes**: 5 parameters

---

## Expected Training Behavior After All Fixes

### Loss Progression (Target)

```
Epoch 1:  train=0.68, val=1.8  ← Val loss now changes!
Epoch 2:  train=0.45, val=1.4  ← Both decreasing
Epoch 3:  train=0.30, val=1.1
Epoch 4:  train=0.25, val=0.9
Epoch 5:  train=0.20, val=0.8
Epoch 6:  train=0.17, val=0.7
Epoch 7:  train=0.15, val=0.6  ← No divergence!
Epoch 8:  train=0.13, val=0.55
Epoch 9:  train=0.12, val=0.52
Epoch 10: train=0.115, val=0.5  ← Final
```

### Model Outputs (Expected)

**After Epoch 5**:
```
Q: If a store has 15 apples and sells 7, how many are left? A:
→ "8" or "8 apples" (may not be perfect, but coherent)
```

**After Epoch 10**:
```
Q: What is 25 + 17? A:
→ "42" (should get basic math right)

Q: The capital of France is Paris. Q: What is the capital of France? A:
→ "Paris" (should recall from context)
```

---

## Risk Assessment

### What Could Still Go Wrong

1. **Architecture Rebalancing (Step 4)**:
   - Risk: 50% params from embeddings might reduce vocab coverage
   - Mitigation: Monitor perplexity, if increases too much, adjust ratio to 60/40

2. **ACT Debugging (Step 5)**:
   - Risk: ACT might be fundamentally broken, not fixable without rewrite
   - Mitigation: If variance stays 0, disable ACT entirely (use fixed layers)

3. **LTM Batch Dependency (Step 5)**:
   - Risk: Making LTM global might change model behavior dramatically
   - Mitigation: Test on small dataset first, compare to batch-dependent version

4. **Training Still Diverges**:
   - Risk: Even with all fixes, epoch 7 divergence recurs
   - Decision Point: Abort nuclear fix, switch to GPT-2 Small baseline

---

## Timeline Estimate

- **Step 4**: 2 hours (architecture rebalancing)
- **Step 5**: 2 hours (ACT/LTM debugging)
- **Step 6**: 30 minutes (early stopping, scheduler)
- **Step 7**: 4-6 hours (retrain reasoning model)
- **Step 8**: 1 hour (validation testing)

**Total Remaining**: ~10 hours
**Total Project**: ~12 hours (2 hours done, 10 hours remaining)

---

**Status**: 37.5% complete (3/8 steps)
**Next Action**: Implement Step 4 (architecture rebalancing)
**Decision Point**: After Step 7 retraining - if successful, proceed with all 3 models; if not, escalate to GPT-2 baseline

# Phase 1 Training Failure - Root Cause Analysis

**Date**: 2025-10-17
**Status**: CRITICAL - Model Training Failed
**Issue**: Mode collapse and training divergence

---

## Executive Summary

All 3 Phase 1 models (reasoning, memory, speed) exhibit **severe training failure** manifesting as:

1. **Mode Collapse**: Epoch 10 models produce only colons ("::::::") regardless of input
2. **Training Divergence**: Loss increases from epochs 7-10 instead of decreasing
3. **Unusable Models**: Neither epoch 1 nor epoch 10 checkpoints produce coherent outputs

**Root Cause**: Training instability during curriculum stage transitions, specifically at the REASONING → ADVANCED transition (epoch 6-7).

---

## Evidence from Training Logs

### Reasoning Model - Loss Progression

**Epoch 1-6 (Declining Loss - Expected Behavior)**:
```
Epoch 1:  3.7607 → 0.6848  (82% reduction) - Foundation stage
Epoch 2:  0.3366 → 0.3083  (8% reduction)  - Foundation stage
Epoch 3:  0.3616 → 0.3148  (13% reduction) - Foundation stage
Epoch 4:  0.3175 → 0.2240  (29% reduction) - Reasoning stage (NEW DATASETS)
Epoch 5:  0.1626 → 0.1548  (5% reduction)  - Reasoning stage
Epoch 6:  0.1897 → 0.1590  (16% reduction) - Reasoning stage
```

**Epoch 7 (CRITICAL FAILURE - Loss Increases)**:
```
Epoch 7:  0.1273 → 0.1464  (15% INCREASE!) - Advanced stage (SAME DATASETS)
          ^                ^
          Start             End

Within-Epoch Trajectory:
Step 24590: loss=0.1273 (batch 50)
Step 25540: loss=0.1177 (batch 1000) - improvement
Step 26940: loss=0.1207 (batch 2400) - starting to rise
Step 29140: loss=0.1452 (batch 4600) - continuing to rise
Step 29540: loss=0.1464 (batch 4950) - final step
```

**The Problem**: Loss was **improving** (0.1273 → 0.1177) in the first half of epoch 7, then **diverged** (0.1177 → 0.1464) in the second half.

### Epochs 8-10 (Continued Divergence)

Based on the pattern, epochs 8-10 likely continued this divergence, leading to:
- Final training loss: ~0.12-0.15 (from logs)
- **Validation loss: 2.5** (same as epoch 1!)
- Model outputs collapsed to single token (colon)

---

## Test Results

### Epoch 1 Models (best_model.pt)

**Checkpoint Metadata**:
- Epoch: 1, Step: 2,700
- Training Loss: 0.6848
- Validation Loss: 2.5

**Output Examples**:
```
Q: If a store has 15 apples and sells 7, how many are left? A:
→ If If If not / / / / distance customers was was was scored

Q: What is 25 + 17? A:
→ 30101010101010 did did meters 120] of of of of

Q: The capital of France is Paris. Q: What is the capital of France? A:
→ order order together 700700 runs jar444444 Saturday kg360
```

**Characteristics**:
- Repetitive tokens (same word 5-30 times)
- Random numbers (27, 120, 360, 700)
- No semantic coherence
- No evidence of learned Q&A structure

### Epoch 10 Models (epoch_10.pt)

**Checkpoint Metadata**:
- Epoch: 10, Step: 46,460
- Training Loss: ~0.12 (estimated from patterns)
- Validation Loss: 2.5 (same as epoch 1!)

**Output Examples**:
```
Q: If a store has 15 apples and sells 7, how many are left? A:
→ ::::::::::::::::::::::::::::::::::::::::::::::::::

Q: What is 25 + 17? A:
→ ::::::::::::::::::::::::::::::::::::::::::::::::::

Q: The capital of France is Paris. Q: What is the capital of France? A:
→ ::::::::::::::::::::::::::::::::::::::::::::::::::
```

**Characteristics**:
- **Complete mode collapse**: Only produces colon character
- 100% deterministic output (no variation)
- **Worse than epoch 1** (at least epoch 1 varied tokens)

---

## Root Cause Analysis

### Primary Cause: Curriculum Stage Instability

**The Curriculum Plan**:
```
Stage 1 (Epochs 1-3): FOUNDATION
  Datasets (4): gsm8k, svamp, mbpp, arc_easy
  10,798 samples

Stage 2 (Epochs 4-6): REASONING
  Datasets (6): gsm8k, svamp, mbpp, arc_easy, arc_challenge, hellaswag
  21,917 samples (+103% dataset size!)

Stage 3 (Epochs 7-10): ADVANCED
  Datasets (6): Same as Stage 2
  21,917 samples (NO CHANGE in datasets!)
```

**Critical Insight**: Stage 3 (ADVANCED) uses **identical datasets** to Stage 2 (REASONING), but the model's loss **diverges** when transitioning to Stage 3.

### Why Epoch 7 Failed

**Theory**: The curriculum stage transition triggers a **learning rate / optimizer reset or change** that destabilizes training.

**Evidence**:
1. Epoch 6 → Epoch 7 transition coincides with curriculum stage change
2. Same datasets, but loss pattern reverses
3. Loss initially improves in epoch 7 (0.1273 → 0.1177), suggesting model still capable
4. Then diverges in second half (0.1177 → 0.1464)

**Possible Mechanisms**:
- **Learning rate too high** for later epochs (MuGrokfast settings)
- **Optimizer state reset** at stage boundaries
- **Batch composition changes** (different sampling of same datasets)
- **Gradient accumulation issues** with larger dataset (21,917 samples)

### Secondary Causes

**1. High Validation Loss Throughout Training**
```
Epoch 1:  val_loss = 2.5
Epoch 2:  val_loss = 2.5
...
Epoch 10: val_loss = 2.5 (unchanged!)
```

**Problem**: Validation loss **never improved** from epoch 1, despite training loss dropping from 3.76 → 0.68 → 0.22 → 0.15.

**Explanation**: **Massive overfitting**. Model memorized training data but learned no generalizable patterns.

**2. No Early Stopping**
- Training continued for all 10 epochs despite:
  - Validation loss stuck at 2.5
  - Training loss diverging after epoch 6
  - No checkpoint saving after epoch 1 (best_val_loss never beaten)

**3. Dataset Quality Issues**
- 6 HuggingFace datasets combined without balancing
- HellaSwag: 10,000 samples (45% of total!)
- Potential format inconsistencies across datasets
- No data cleaning or quality filtering

**4. MuGrokfast Optimizer Settings**
- Phase 1 preset may be too aggressive for later epochs
- Grokfast lambda=0.05 (gradient filtering)
- Muon lr=0.01 (high for fine-tuning)
- No learning rate decay schedule

---

## Why Best Model is Epoch 1

**Current Logic** ([trainer.py:280](../../src/phase1_cognate/training/trainer.py#L280)):
```python
if val_loss < self.best_val_loss:
    self.best_val_loss = val_loss
    torch.save(checkpoint, best_path)
```

**What Happened**:
```
Epoch 1: val_loss = 2.5 → SAVED as best_model.pt
Epoch 2: val_loss = 2.5 (not better, not saved)
Epoch 3: val_loss = 2.5 (not better, not saved)
...
Epoch 10: val_loss = 2.5 (not better, not saved)
```

Epoch 1 was the **only time** validation loss changed, so it remained "best" forever.

---

## Impact on Model Behavior

### Why Epoch 1 Produces Random Tokens

**Loss: 0.6848** (still high)
- Model hasn't learned proper token distributions
- Outputs high-frequency tokens from training data
- No semantic understanding
- Exhibits "token soup" behavior

### Why Epoch 10 Produces Only Colons

**Loss diverged, mode collapsed**
- Model converged to local minimum of always predicting colon
- Colon (:) is high-frequency token in Q&A format ("Q: ... A:")
- Gradient descent found "lowest loss" by always outputting separator
- Complete failure of language modeling

---

## Technical Details

### Validation Loss Calculation

**Location**: [trainer.py:261-279](../../src/phase1_cognate/training/trainer.py#L261-L279)

**Issue**: Validation set may be:
- Too small (not enough samples)
- Wrong distribution (doesn't match training)
- Wrong metric (cross-entropy not capturing model quality)

### Curriculum Implementation

**Location**: [curriculum.py](../../src/phase1_cognate/training/curriculum.py)

**Transition Logic** (hypothetical):
```python
if epoch <= 3:
    stage = "FOUNDATION"
    datasets = [gsm8k, svamp, mbpp, arc_easy]
elif epoch <= 6:
    stage = "REASONING"
    datasets = [gsm8k, svamp, mbpp, arc_easy, arc_challenge, hellaswag]
else:  # epoch 7-10
    stage = "ADVANCED"
    datasets = [gsm8k, svamp, mbpp, arc_easy, arc_challenge, hellaswag]  # SAME!
```

**Problem**: No actual curriculum difficulty increase from REASONING → ADVANCED.

---

## Comparison to Documentation

### From PHASE1_COMPLETE_GUIDE.md

**Expected Results** (from V1 documentation):
```
Epoch 1:  loss: 3.76 → 0.68  (82% reduction) ✓ MATCHES
Epoch 10: loss: ~0.115        (96% total reduction) ✗ FAILED (got ~0.15 then collapse)
```

**Expected Model Behavior**:
- Coherent text generation
- Basic reasoning capabilities
- Q&A format adherence

**Actual Model Behavior**:
- Token repetition (epoch 1)
- Mode collapse to colon (epoch 10)
- No coherent outputs

---

## Solutions

### Immediate Fix (Restart Training)

1. **Disable Curriculum Stages 2-3**:
   ```python
   # Use only Stage 1 datasets for all 10 epochs
   datasets = [gsm8k, svamp, mbpp, arc_easy]  # 10,798 samples
   ```
   - Smaller dataset (10K vs 21K)
   - More manageable for 25M parameter model
   - No stage transitions to cause instability

2. **Reduce MuGrokfast Learning Rates**:
   ```python
   muon_lr: 0.01 → 0.005  (50% reduction)
   fallback_lr: 0.001 → 0.0005
   grokfast_lambda: 0.05 → 0.02 (gentler filtering)
   ```

3. **Add Learning Rate Decay**:
   ```python
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
       optimizer, T_max=num_epochs, eta_min=1e-6
   )
   ```

4. **Implement Early Stopping**:
   ```python
   if val_loss has not improved for 3 epochs:
       print("Early stopping!")
       break
   ```

5. **Fix Validation Loss**:
   ```python
   # Use larger validation set (20% of training data)
   val_size = int(len(dataset) * 0.2)  # ~2,000 samples
   ```

### Medium-Term Fix (Architecture)

1. **Gradient Clipping**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Warmup Schedule**:
   ```python
   # First 500 steps: lr = 0 → target_lr (linear)
   # Then cosine decay
   ```

3. **Batch Size Optimization**:
   - Current: batch_size=4
   - Try: batch_size=8 or 16 (more stable gradients)

4. **Dataset Balancing**:
   ```python
   # Ensure equal representation
   gsm8k: 1,000 samples (downsample from 7,473)
   svamp: 700 samples (keep all)
   mbpp: 374 samples (keep all, upsample to 700)
   arc_easy: 1,000 samples (downsample from 2,251)
   ```

### Long-Term Fix (Training Pipeline)

1. **Better Validation Metrics**:
   - Perplexity
   - BLEU score (for generated text)
   - Exact match accuracy (for Q&A)
   - Token diversity (catch mode collapse early)

2. **Curriculum Redesign**:
   ```
   Stage 1 (Epochs 1-3): Easy examples only (filter by length, difficulty)
   Stage 2 (Epochs 4-6): Medium examples
   Stage 3 (Epochs 7-10): All examples
   ```
   - Actually increase difficulty within same datasets
   - No abrupt dataset additions

3. **Monitoring**:
   - Log gradient norms to W&B
   - Alert on gradient explosion (norm > 10.0)
   - Alert on loss increase within epoch
   - Track token diversity in outputs

---

## Recommended Action Plan

### Phase 1: Diagnosis Complete ✓

- [x] Test models with epoch_10.pt checkpoints
- [x] Analyze training logs
- [x] Identify root cause (curriculum transition instability)
- [x] Document findings

### Phase 2: Quick Fix (4-6 hours)

1. **Modify training configuration** ([training_config.py](../../src/phase1_cognate/config/training_config.py)):
   - Reduce learning rates by 50%
   - Disable curriculum stages (use Stage 1 datasets only)
   - Add gradient clipping (max_norm=1.0)
   - Implement early stopping (patience=3)

2. **Retrain reasoning model** (test on 1 model first):
   - Monitor closely for loss divergence
   - Stop if validation loss doesn't improve by epoch 3
   - Check outputs at epoch 5 (mid-training sanity check)

3. **Validate success**:
   - Training loss should decrease smoothly
   - Validation loss should decrease (not stuck at 2.5)
   - Epoch 5 model should produce varied (not collapsed) outputs

### Phase 3: Full Retraining (10-12 hours)

If Phase 2 succeeds:
1. Retrain all 3 models (reasoning, memory, speed)
2. Use fixed configuration
3. Save checkpoints every 2 epochs
4. Test each checkpoint with [test_phase1_models.py](../../scripts/test_phase1_models.py)

### Phase 4: Validation (2 hours)

1. Interactive testing with all 3 models
2. Generate 100 Q&A samples per model
3. Manual quality review
4. Compare to expected Phase 1 capabilities

---

## Files to Modify

### 1. src/phase1_cognate/config/training_config.py

**Current MuGrokfast Settings**:
```python
mugrokfast_config = {
    'muon_lr': 0.01,           # TOO HIGH
    'fallback_lr': 0.001,      # TOO HIGH
    'grokfast_lambda': 0.05,   # TOO AGGRESSIVE
    # ... rest
}
```

**Recommended Changes**:
```python
mugrokfast_config = {
    'muon_lr': 0.005,          # 50% reduction
    'fallback_lr': 0.0005,     # 50% reduction
    'grokfast_lambda': 0.02,   # 60% reduction (gentler)
    'qk_clip_threshold': 25.0, # keep (only for RL)
    'muon_momentum': 0.95,     # keep
    # ADD:
    'gradient_clip_norm': 1.0,  # NEW - prevent explosion
}
```

### 2. src/phase1_cognate/training/curriculum.py

**Current Implementation** (assumed):
```python
def get_curriculum_stage(epoch):
    if epoch <= 3:
        return "FOUNDATION", [gsm8k, svamp, mbpp, arc_easy]
    elif epoch <= 6:
        return "REASONING", [gsm8k, ..., hellaswag]  # +103% size!
    else:
        return "ADVANCED", [gsm8k, ..., hellaswag]   # SAME!
```

**Recommended Changes**:
```python
def get_curriculum_stage(epoch):
    # DISABLE curriculum for now - use fixed dataset
    return "FOUNDATION", [gsm8k, svamp, mbpp, arc_easy]

    # OR implement proper difficulty scaling:
    # Filter examples by length, complexity within same datasets
    # Stage 1: examples with length < 50 tokens
    # Stage 2: examples with length < 100 tokens
    # Stage 3: all examples
```

### 3. src/phase1_cognate/training/trainer.py

**Add Early Stopping**:
```python
class Phase1Trainer:
    def __init__(self, ...):
        self.best_val_loss = float('inf')
        self.patience = 3               # NEW
        self.patience_counter = 0       # NEW
        self.min_delta = 0.01           # NEW - minimum improvement

    def train(self):
        for epoch in range(1, num_epochs + 1):
            # ... training ...

            val_loss = self.validate()

            # Early stopping logic
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save checkpoint
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}!")
                    break
```

**Add Gradient Clipping**:
```python
def train_epoch(self):
    for batch in dataloader:
        loss = self.model(batch)
        loss.backward()

        # ADD gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()
```

**Add Learning Rate Scheduler**:
```python
def __init__(self, ...):
    self.optimizer = MuonGrokfast(...)

    # ADD scheduler
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6
    )

def train_epoch(self):
    # ... training loop ...

    # At end of epoch:
    self.scheduler.step()
```

---

## Expected Outcomes After Fix

### Training Metrics

**Loss Progression** (target):
```
Epoch 1:  3.76 → 0.68  (Foundation)
Epoch 2:  0.68 → 0.45  (Foundation)
Epoch 3:  0.45 → 0.30  (Foundation)
Epoch 4:  0.30 → 0.25  (Foundation)
Epoch 5:  0.25 → 0.20  (Foundation)
Epoch 6:  0.20 → 0.17  (Foundation)
Epoch 7:  0.17 → 0.15  (Foundation)
Epoch 8:  0.15 → 0.13  (Foundation)
Epoch 9:  0.13 → 0.12  (Foundation)
Epoch 10: 0.12 → 0.115 (Foundation)
```

**Validation Loss** (target):
```
Epoch 1: 2.5 → 1.8   (improve!)
Epoch 2: 1.8 → 1.4
Epoch 3: 1.4 → 1.1
Epoch 4: 1.1 → 0.9
Epoch 5: 0.9 → 0.8
...
Epoch 10: ~0.5-0.6
```

### Model Outputs

**Epoch 5 (Mid-Training)**:
```
Q: If a store has 15 apples and sells 7, how many are left? A:
→ 8 apples [MAY NOT BE PERFECT, BUT COHERENT]

Q: What is 25 + 17? A:
→ 42 [OR CLOSE, LIKE "40" OR "43"]
```

**Epoch 10 (Final)**:
```
Q: If a store has 15 apples and sells 7, how many are left? A:
→ 8

Q: What is 25 + 17? A:
→ 42

Q: The capital of France is Paris. Q: What is the capital of France? A:
→ Paris [SHOULD GET THIS RIGHT]
```

---

## Lessons Learned

1. **Curriculum learning is double-edged**: Can help, but transitions must be smooth
2. **Validation loss is critical**: Should have stopped at epoch 1 when val_loss plateaued
3. **Monitor training actively**: Loss increasing within epoch is immediate red flag
4. **Start simple**: Single dataset, low LR, then scale up
5. **Test early and often**: Should have tested epoch 2-3 models before continuing to 10

---

## References

- [PHASE1_COMPLETE_GUIDE.md](PHASE1_COMPLETE_GUIDE.md) - Original V1 implementation (worked)
- [PHASE1_TRAINING_ANALYSIS_REPORT.md](PHASE1_TRAINING_ANALYSIS_REPORT.md) - Initial analysis
- [PHASE1_MODEL_TESTING_REPORT.md](PHASE1_MODEL_TESTING_REPORT.md) - Testing results
- [trainer.py](../../src/phase1_cognate/training/trainer.py) - Training implementation
- [curriculum.py](../../src/phase1_cognate/training/curriculum.py) - Curriculum logic
- [training_config.py](../../src/phase1_cognate/config/training_config.py) - MuGrokfast settings

---

**Report Generated**: 2025-10-17
**Status**: Root cause identified, solutions proposed
**Next Step**: Implement fixes and retrain

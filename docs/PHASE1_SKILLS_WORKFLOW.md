# Phase 1 Training - Skills-Based Workflow Guide

**Date**: 2025-10-17
**Status**: Production-ready workflow using ml-training-debugger and ml-expert skills

---

## Overview

Phase 1 (Cognate) training has encountered specific failures (loss divergence, mode collapse, ACT variance=0). Rather than manually debugging and fixing these issues, you now have two powerful specialist skills that work together:

1. **ml-training-debugger**: Diagnoses training failures with >80% confidence
2. **ml-expert**: Implements fixes with production-quality code

This workflow shows you how to use these skills effectively for Phase 1 and future ML projects.

---

## The Skills-Based Approach

### Traditional Approach (Manual)
```
User reports issue → Claude analyzes → Claude proposes fix → Claude implements → Test
```
**Problems**: Inconsistent analysis depth, no structured methodology, fixes may be incomplete

### Skills-Based Approach (Specialist Agents)
```
User reports issue → ml-training-debugger (systematic diagnosis) → ml-expert (production implementation) → Validated fix
```
**Benefits**:
- Systematic analysis using proven methodologies
- Evidence-based diagnosis with confidence scores
- Production-quality implementations with tests
- Reproducible process

---

## Quick Start: Fix Phase 1 Training

### Step 1: Invoke ML Training Debugger

**Your command**:
```
Please use the ml-training-debugger skill to diagnose why my Phase 1 training failed.

The model was training fine through epoch 6, then at epoch 7 the loss started increasing instead of decreasing. By epoch 10, the model only outputs colons (::::) regardless of input.

Training logs are in phases/phase1/TRAINING_FAILURE_DIAGNOSIS.md and NUCLEAR_FIX_IMPLEMENTATION_SUMMARY.md
```

**What happens**:
1. ml-training-debugger skill activates
2. Spawns ML debugging specialist agent
3. Agent gathers artifacts (logs, code, config)
4. Agent performs systematic analysis (4 phases, 5 minutes)
5. Agent returns structured diagnosis with confidence scores

**Example diagnosis output**:
```json
{
  "root_causes": [
    {
      "rank": 1,
      "issue": "Learning rate too high for Muon optimizer",
      "confidence": 0.95,
      "evidence": [
        "grad_norm spiked to 45.2 at step 24590",
        "loss increased 15% at epoch 7 (was decreasing)",
        "muon_lr=1e-2 is 2x recommended 5e-3"
      ],
      "fix": "Reduce muon_lr from 1e-2 to 5e-3"
    },
    {
      "rank": 2,
      "issue": "ACT variance = 0 (all tokens use same halting steps)",
      "confidence": 0.98,
      "evidence": [
        "Warning: var(): degrees of freedom is <= 0",
        "ACT loss computation shows zero diversity"
      ],
      "fix": "Add diversity regularization to ACT loss"
    }
  ],
  "quick_wins": [
    "Reduce learning rates by 50%",
    "Add diversity loss to ACT head",
    "Enable early stopping"
  ]
}
```

### Step 2: Invoke ML Expert to Implement Fixes

**Your command**:
```
Please use the ml-expert skill to implement the top 2 fixes from the diagnosis:

1. Reduce muon_lr from 1e-2 to 5e-3 in trainer.py
2. Add diversity regularization to ACT head compute_act_loss() method

Make sure all tests pass after changes.
```

**What happens**:
1. ml-expert skill activates
2. Spawns ML implementation specialist agent
3. Agent shows implementation plan (you approve)
4. Agent implements fixes with tests
5. Agent validates all tests pass

**Example implementation output**:
```json
{
  "code_changes": [
    {
      "file": "src/phase1_cognate/training/trainer.py",
      "line": 47,
      "change": "muon_lr: float = 1e-2  →  muon_lr: float = 5e-3",
      "description": "Reduced Muon learning rate by 50%"
    },
    {
      "file": "src/phase1_cognate/model/act_head.py",
      "line": 120,
      "change": "Added diversity regularization term",
      "loc_added": 8,
      "description": "Computes token variance and penalizes low diversity"
    }
  ],
  "tests": "48/48 passing (added 2 new tests)",
  "verification": {
    "trainer_lr": "✓ LR reduced to 5e-3",
    "act_diversity": "✓ Variance > 0.05 in tests",
    "no_breaking_changes": "✓ All existing tests pass"
  }
}
```

### Step 3: Retrain and Validate

**Your command**:
```bash
# Quick test (CPU, 1 epoch)
python src/phase1_cognate/train_phase1.py --model reasoning --test

# If test passes, full GPU training
python src/phase1_cognate/train_phase1.py --model reasoning --epochs 10
```

**Expected result**: Training progresses smoothly, no divergence at epoch 7, model outputs are coherent.

---

## Detailed Workflow Examples

### Example 1: Diagnosing Loss Divergence

**Scenario**: Training loss suddenly increases at epoch 7

**Step 1: Invoke debugger**
```
/ml-training-debugger

I'm seeing loss divergence at epoch 7. Here are the symptoms:
- Epochs 1-6: Loss decreased from 3.76 to 0.16
- Epoch 7: Loss INCREASED from 0.16 to 0.19
- Within epoch 7: grad_norm spiked to 45.2 at step 24590
- Final model outputs only colons

Logs: phases/phase1/TRAINING_FAILURE_DIAGNOSIS.md
Code: src/phase1_cognate/training/trainer.py
Config: src/phase1_cognate/model/model_config.py
```

**Debugger analysis**:
```markdown
## Diagnosis Complete (Confidence: 95%)

### Root Cause
Learning rate too high for Muon optimizer during curriculum stage transition.

### Evidence
1. **Gradient explosion detected**
   - grad_norm: 2.0 (normal) → 45.2 (spike) at step 24590
   - Spike occurred early in epoch 7 (curriculum transition)

2. **Loss trajectory analysis**
   - Epochs 1-6: Smooth descent (3.76 → 0.16)
   - Epoch 7: Divergence started (0.16 → 0.19)
   - Within-epoch: Was improving (0.1273 → 0.1177), then exploded (→ 0.1464)

3. **Hyperparameter analysis**
   - muon_lr = 1e-2 (configured)
   - Recommended for Phase 1: 5e-3 (from MuGrokfast docs)
   - Current LR is 2x too high

4. **Curriculum transition**
   - Epoch 7 is when "Advanced" stage begins
   - New dataset distribution may have higher gradient magnitudes
   - LR that worked for Foundation stage is too aggressive for Advanced

### Recommended Fixes (Prioritized)

**Fix 1 (Critical): Reduce Muon Learning Rate**
- File: `src/phase1_cognate/training/trainer.py:47`
- Change: `muon_lr: float = 1e-2` → `muon_lr: float = 5e-3`
- Impact: HIGH (directly addresses gradient explosion)
- Effort: LOW (config change)
- Risk: LOW (well-tested value)

**Fix 2 (High): Add LR Scheduler**
- Already implemented! Cosine annealing with warmup.
- Verify it's enabled: `use_lr_scheduler: bool = True`

**Fix 3 (Medium): Reduce Grokfast Lambda**
- File: `src/phase1_cognate/training/trainer.py:48`
- Change: `grokfast_lambda: float = 0.05` → `0.02`
- Impact: MEDIUM (less momentum, better adaptation)
- Already done per Nuclear Fix doc!

### Expected Outcome
After reducing muon_lr to 5e-3:
- Gradient norm should stay < 10.0 throughout training
- Loss should decrease smoothly through epoch 7-10
- No divergence at curriculum transitions
```

**Step 2: Implement fix**
```
/ml-expert

Implement Fix 1 from the diagnosis: Reduce muon_lr to 5e-3 in trainer.py
```

**Expert implementation**:
```markdown
## Implementation Plan

File: `src/phase1_cognate/training/trainer.py`
Line: 47
Change: `muon_lr: float = 1e-2` → `muon_lr: float = 5e-3`

Testing:
- Verify config loads correctly
- Check optimizer initialization uses new LR
- Run quick train test (1 epoch, CPU)

Proceed? [User approves]

## Implementation Complete

✅ Modified trainer.py:47
✅ All tests passing (48/48)
✅ Quick test run successful (grad_norm stayed < 3.0)

Ready to retrain!
```

---

### Example 2: Fixing ACT Variance = 0

**Scenario**: Warning "var(): degrees of freedom is <= 0" during training

**Step 1: Invoke debugger**
```
/ml-training-debugger

Getting this warning during training:
"UserWarning: var(): degrees of freedom is <= 0"

It appears in the ACT loss computation. All tokens seem to use the same number of halting steps (no adaptation).

Code: src/phase1_cognate/model/act_head.py
```

**Debugger analysis**:
```markdown
## Diagnosis Complete (Confidence: 98%)

### Root Cause
ACT head learns uniform halt probabilities (no token-level diversity).

### Evidence
1. **Warning source**: Line 123 in act_head.py, `torch.var()` call
2. **Behavior**: All tokens output same halt probability → variance = 0
3. **Missing regularization**: No diversity loss to encourage variance

### Recommended Fix

**Add Diversity Regularization to ACT Loss**

File: `src/phase1_cognate/model/act_head.py:120`
Method: `compute_act_loss()`

Add after entropy loss:
```python
# Diversity regularization (encourage variance across tokens)
q_mean = q.mean(dim=1, keepdim=True)  # [batch, 1, 1]
q_variance = ((q - q_mean) ** 2).mean()  # Scalar

target_variance = 0.1  # Target variance
diversity_loss = torch.clamp(target_variance - q_variance, min=0.0)

return loss_bce + loss_entropy + 0.01 * diversity_loss
```

Expected: Variance will increase from 0 → 0.05-0.15, warning eliminated.
```

**Step 2: Implement fix**
```
/ml-expert

Implement the ACT diversity regularization fix from the diagnosis.

Add the diversity loss term to compute_act_loss() method in act_head.py.
Include a test to verify variance > 0.
```

**Expert implementation**:
```markdown
## Implementation Plan

1. Modify `src/phase1_cognate/model/act_head.py:120`
   - Add diversity regularization after entropy loss
   - Target variance: 0.1
   - Loss weight: 0.01

2. Update docstring to document diversity term

3. Add test: `test_act_diversity()`
   - Create ACT head
   - Forward pass with varied inputs
   - Assert variance > 0.05

Proceed? [User approves]

## Implementation Complete

### Code Changes

File: `src/phase1_cognate/model/act_head.py`

```python
# Lines 120-129 (MODIFIED)
def compute_act_loss(...):
    # ... existing BCE and entropy loss ...

    # ✅ ADDED: Diversity regularization
    q_mean = q.mean(dim=1, keepdim=True)
    q_variance = ((q - q_mean) ** 2).mean()
    target_variance = 0.1
    diversity_loss = torch.clamp(target_variance - q_variance, min=0.0)

    return loss_bce + loss_entropy + 0.01 * diversity_loss
```

### Tests Added

File: `tests/test_act_head.py`

```python
def test_act_diversity():
    """Verify ACT head produces diverse halt probabilities"""
    config = ACTConfig()
    act_head = ACTHead(d_model=256, config=config)

    # Create varied latent states
    z = torch.randn(4, 64, 256)  # [batch=4, seq=64, d_model=256]

    # Forward pass
    q = act_head(z)  # [4, 64, 1]

    # Check variance across sequence
    variance = q.var(dim=1).mean().item()
    assert variance > 0.05, f"ACT variance too low: {variance}"
    print(f"✓ ACT variance: {variance:.4f}")
```

### Verification

✅ All tests passing: 49/49 (added 1 new test)
✅ ACT diversity test: variance = 0.087 (> 0.05 threshold)
✅ Quick training run: No variance warning!

**Fix deployed successfully.**
```

---

### Example 3: Architecture Imbalance

**Scenario**: 79% of parameters in embeddings, model underperforms

**Step 1: Invoke debugger**
```
/ml-training-debugger

Model has 32M parameters but seems to have low capacity. Training loss decreases but outputs are nonsense.

Suspect parameter distribution issue.

Code: src/phase1_cognate/model/full_model.py
Config: src/phase1_cognate/model/model_config.py
```

**Debugger analysis**:
```markdown
## Diagnosis Complete (Confidence: 90%)

### Root Cause
Architecture severely imbalanced: 79% params in embeddings, only 21% in transformers.

### Evidence
1. **Parameter distribution**:
   ```
   Total: 32.57M
   ├── Embeddings: 25.7M (79%)  ← TOO MUCH
   └── Transformers: 6.8M (21%)  ← TOO LITTLE
   ```

2. **Healthy architecture** should be:
   ```
   Total: 25M target
   ├── Embeddings: 12.5M (50%)
   └── Transformers: 12.5M (50%)
   ```

3. **Current config** (model_config.py:22-29):
   ```python
   d_model: int = 320  # Too small!
   n_layers: int = 8   # Too few!
   vocab_size: int = 50257  # Fixed (GPT-2)
   ```

4. **Problem**: vocab_size (50257) × d_model (320) = 16M in embeddings alone.
   Leaves only 9M for transformers (8 layers).

### Recommended Fix

**Rebalance Architecture**

Target: 25M params, 50/50 distribution

**Option 1: Reduce vocabulary** (easier)
- Use smaller tokenizer (vocab_size: 50257 → 32000)
- Keeps d_model=320, n_layers=8
- Embeddings: 10.2M, Transformers: 14.8M (better!)

**Option 2: Adjust d_model and n_layers** (recommended)
- vocab_size: 50257 (keep GPT-2 tokenizer)
- d_model: 320 → 256 (saves 7M in embeddings)
- n_layers: 8 → 10 (adds 3M in transformers)
- Result: 13M embeddings, 12M transformers (balanced!)

Recommend Option 2.
```

**Step 2: Implement fix**
```
/ml-expert

Implement the architecture rebalancing (Option 2):
- d_model: 320 → 256
- n_layers: 8 → 10

Update model_config.py and verify parameter count is ~25M with 50/50 split.
```

**Expert implementation**:
```markdown
## Implementation Plan

1. Modify `src/phase1_cognate/model/model_config.py:22-24`
   ```python
   d_model: int = 320  →  d_model: int = 256
   n_layers: int = 8   →  n_layers: int = 10
   ```

2. Update dependent values:
   - n_heads: 5 → 4 (d_model / 64)
   - d_ff: 1280 → 1024 (4x d_model)

3. Verify parameter count:
   - Expected: ~25M total
   - Expected split: ~50% embeddings, ~50% transformers

4. Run tests to ensure no breaking changes

Proceed? [User approves]

## Implementation Complete

### Changes Applied

File: `src/phase1_cognate/model/model_config.py:21-27`

```python
# BEFORE
d_model: int = 320
n_layers: int = 8
n_heads: int = 5
d_ff: int = 1280

# AFTER
d_model: int = 256
n_layers: int = 10
n_heads: int = 4  # d_model / 64
d_ff: int = 1024  # 4x d_model
```

### Verification

**Parameter Count**:
```
Total: 25,623,681 params
├── Embeddings: 13,39M (52.3%)  ✓ Near 50% target
└── Transformers: 10.66M (41.6%)  ✓ Much better!
```

**Tests**: 48/48 passing
**Forward Pass**: ✓ Correct shapes
**Training Test**: ✓ 100 steps successful

**Architecture successfully rebalanced!**
```

---

## Workflow Best Practices

### 1. Always Start with Diagnosis

Don't jump to implementing fixes. Let ml-training-debugger analyze systematically first.

**✅ GOOD**:
```
User: Training failed → ml-training-debugger → ml-expert → Fix
```

**❌ BAD**:
```
User: Training failed → Claude guesses fix → Implement → Might not work
```

### 2. Provide Artifacts to Debugger

The more evidence you give, the better the diagnosis:
- Training logs (stdout/stderr)
- Loss curves (CSV or description)
- Error messages and tracebacks
- Model code
- Config files

**Example**:
```
/ml-training-debugger

Training failed at epoch 7.

Artifacts:
- Logs: phases/phase1/TRAINING_FAILURE_DIAGNOSIS.md
- Code: src/phase1_cognate/training/trainer.py
- Code: src/phase1_cognate/model/full_model.py
- Config: src/phase1_cognate/model/model_config.py

Symptoms:
- Loss was 0.16 at end of epoch 6
- Loss increased to 0.19 at end of epoch 7
- grad_norm spiked to 45.2 during epoch 7
- Final model outputs only colons
```

### 3. Review Implementation Plans

ml-expert will show you its plan before coding (plan mode). Review and approve:

**Example plan**:
```markdown
## Implementation Plan

I will:
1. Reduce muon_lr from 1e-2 to 5e-3 in trainer.py:47
2. Add diversity loss to act_head.py:120
3. Run all 48 tests to verify no breaking changes
4. Run quick training test (1 epoch, CPU)

Estimated time: 5 minutes

Proceed?
```

**Your response**: "Yes, proceed" or "Actually, just do fix #1 first"

### 4. Verify Fixes Work

After implementation, test before full retraining:

```bash
# Quick test (1 epoch, CPU, 2 minutes)
python src/phase1_cognate/train_phase1.py --model reasoning --test

# If successful, full GPU training
python src/phase1_cognate/train_phase1.py --model reasoning --epochs 10
```

### 5. Iterate if Needed

If first fix doesn't fully resolve, go back to debugger:

```
/ml-training-debugger

Applied Fix #1 (reduced muon_lr to 5e-3). Training is better but still diverges at epoch 9.

New symptoms:
- Epochs 1-8: Loss decreasing normally (3.76 → 0.10)
- Epoch 9: Loss increased from 0.10 to 0.12
- No gradient explosion this time (grad_norm stayed < 5.0)

What else could be causing this?
```

Debugger will analyze with new context and propose additional fixes.

---

## Skill Invocation Syntax

### Explicit Skill Invocation (Recommended)
```
/ml-training-debugger
[Describe issue and provide artifacts]

/ml-expert
[Specify implementation requirements]
```

### Natural Language (Also Works)
```
Please use the ml-training-debugger skill to analyze why my training diverged at epoch 7.

Then use the ml-expert skill to implement the top 2 fixes.
```

Claude Code will recognize the skill references and invoke appropriately.

---

## Expected Timelines

| Step | Duration | Notes |
|------|----------|-------|
| ml-training-debugger analysis | 2-5 min | Depends on artifact complexity |
| ml-expert implementation | 3-10 min | Depends on fix complexity |
| Quick test (CPU) | 1-2 min | Verify fix doesn't break anything |
| Full retrain (GPU) | 4-6 hours | Full 10 epochs |
| **Total (diagnosis → fix → test)** | **5-15 min** | Before committing to full retrain |

The skills save massive time by:
- Systematic diagnosis (no manual log analysis)
- Production-quality fixes (with tests)
- Automated validation (test suite runs)

---

## Troubleshooting Skill Usage

### Skill Not Activating

**Problem**: You invoked /ml-training-debugger but skill didn't activate

**Solutions**:
1. Check skill is installed: `ls ~/.claude/skills/ml-training-debugger/`
2. Try explicit command: `/ml-training-debugger` (with leading slash)
3. Try natural language: "Use the ml-training-debugger skill to..."
4. Restart Claude Code session

### Debugger Can't Find Artifacts

**Problem**: Debugger says "Cannot locate training logs"

**Solutions**:
1. Provide absolute or relative paths from project root
2. Copy-paste relevant log sections directly in your message
3. Describe symptoms even if artifacts unavailable (lower confidence diagnosis)

### Implementation Breaks Tests

**Problem**: ml-expert implemented fix but tests are failing

**Solutions**:
1. ml-expert will automatically debug and refix
2. If stuck, revert changes and try alternative fix
3. Provide more context about test failures to ml-expert

---

## Summary

You now have two powerful skills for ML development:

1. **ml-training-debugger**: Your systematic ML diagnostician
   - Evidence-based root cause analysis
   - Confidence-scored diagnoses
   - Prioritized fix recommendations

2. **ml-expert**: Your production ML engineer
   - Clean, tested implementations
   - Best practices enforcement
   - Comprehensive documentation

**Workflow**: Problem → Diagnose → Implement → Test → Done

**Time Saved**: 2-4 hours per debugging cycle (estimated)

**Quality**: Higher consistency, better test coverage, production-ready code

---

**Ready to use these skills on Phase 1!**

Next: Try the workflow on the current training failure.

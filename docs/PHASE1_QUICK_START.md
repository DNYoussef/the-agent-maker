# Phase 1 (Cognate) - Quick Start Guide

**Status**: ✅ **ALL FIXES COMPLETE** - Ready to train!

---

## What's Been Fixed

✅ **All 6/6 critical fixes complete**:
1. **Validation loss** - Real computation (not hardcoded 2.5)
2. **Batch size** - Effective batch 64 with gradient accumulation
3. **MuGrokfast LRs** - Reduced 50% for stability
4. **Architecture** - 52% embeddings, 25.6M params (well-balanced)
5. **ACT variance** - Added diversity loss to fix variance=0
6. **Training safety** - Early stopping, LR scheduler, gradient clipping

**Models verified**: All 3 models (reasoning, memory, speed) create successfully with 25.6M params each.

---

## Quick Test (1-2 minutes)

Verify the model can train without errors:

```bash
cd "c:\Users\17175\Desktop\the agent maker"

# Quick test (CPU, 1 epoch, 2 datasets, batch_size=2)
python src/phase1_cognate/train_phase1.py --model reasoning --test --epochs 1
```

**Expected output**:
```
*** TEST MODE: CPU, 1 epoch, 2 datasets ***

======================================================================
STEP 1: DOWNLOAD DATASETS
======================================================================
...

======================================================================
PHASE 1 TRAINING: REASONING
======================================================================
...
Epoch 1/1
...
✓ REASONING model training complete!
```

**If this succeeds** → Ready for full GPU training!
**If this fails** → Share the error message for debugging.

---

## Full Training (4-6 hours on GPU)

### Step 1: Clear Old Checkpoints (Optional)
```bash
# Remove failed training checkpoints
rm -rf "checkpoints/phase1/reasoning"
```

### Step 2: Run Full Training
```bash
cd "c:\Users\17175\Desktop\the agent maker"

# Full training (GPU, 10 epochs, all foundation datasets)
python src/phase1_cognate/train_phase1.py \
    --model reasoning \
    --epochs 10 \
    --batch-size 16 \
    --wandb-mode offline \
    --datasets gsm8k svamp mbpp arc_easy piqa wikitext
```

### Step 3: Monitor Training

Watch for these **success indicators**:

#### ✅ Validation Loss Decreases
```
Epoch 1: val_loss = 1.8  (NOT 2.5!)
Epoch 2: val_loss = 1.4  (decreasing)
Epoch 3: val_loss = 1.1  (decreasing)
...
```

#### ✅ No Divergence at Epoch 7
```
Epoch 6: loss = 0.17 → 0.15  (decreasing)
Epoch 7: loss = 0.15 → 0.13  (STILL decreasing, not increasing!)
Epoch 8: loss = 0.13 → 0.12  (smooth)
```

#### ✅ ACT Diversity Working
```
(Check W&B logs or console)
ACT variance: 0.05-0.15  (NOT 0!)
Halting steps vary across tokens
```

### Step 4: Early Stopping (If Needed)

Training will **automatically stop** if validation loss doesn't improve for 3 epochs:
```
Epoch 8: No improvement for 3 epoch(s)
⚠️  Early stopping triggered (patience=3)
   Best val loss: 0.55 at epoch 5
```

This saves GPU time if training plateaus.

---

## Test Model Outputs (After Training)

### Quick Output Test

```bash
cd "c:\Users\17175\Desktop\the agent maker"

python -c "
import torch
import sys
sys.path.insert(0, 'src')

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config

# Load model
config = Phase1Config(specialization='reasoning')
model = TRMTitansMAGModel(config)

# Load checkpoint
ckpt = torch.load('checkpoints/phase1/reasoning/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print('Model loaded successfully!')
print(f'Epoch: {ckpt[\"epoch\"]}, Val loss: {ckpt[\"best_val_loss\"]:.4f}')

# Test forward pass
test_input = torch.randint(0, 50257, (1, 32))
with torch.no_grad():
    output = model(test_input)
    print(f'Output shape: {output[\"logits\"].shape}')
    print(f'Halting steps: {output[\"halting_steps\"].item():.1f}')

print('✅ Model works!')
"
```

**Expected**:
```
Model loaded successfully!
Epoch: 7, Val loss: 0.5523
Output shape: torch.Size([1, 32, 50257])
Halting steps: 2.3
✅ Model works!
```

### Generate Text (Detailed Test)

Create `test_outputs.py`:
```python
import torch
import sys
sys.path.insert(0, 'src')

from phase1_cognate.model.full_model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config
from transformers import GPT2Tokenizer

# Load
config = Phase1Config(specialization='reasoning')
model = TRMTitansMAGModel(config)
ckpt = torch.load('checkpoints/phase1/reasoning/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Test prompts
prompts = [
    "If a store has 15 apples and sells 7, how many are left? A:",
    "What is 25 + 17? A:",
    "The capital of France is Paris. Q: What is the capital of France? A:"
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, max_length=64, truncation=True)

    with torch.no_grad():
        output = model(inputs['input_ids'])
        logits = output['logits'][0, -1, :]  # Last token logits

        # Top-5 predictions
        top5 = torch.topk(logits, 5)

    print(f"\nQ: {prompt}")
    print("Top 5 predictions:")
    for i, (idx, score) in enumerate(zip(top5.indices, top5.values)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. '{token}' (score: {score:.2f})")
    print()
```

Run: `python test_outputs.py`

**Success indicators**:
- ✅ Diverse tokens (not all colons)
- ✅ Relevant tokens for math questions (numbers, arithmetic words)
- ✅ No single token dominates all predictions
- ✅ Halting steps vary (not all same)

**Failure indicators**:
- ❌ Only colons in top 5 (::::)
- ❌ Random gibberish repeated
- ❌ Same token for all prompts

---

## Training Configuration Summary

The model will train with these settings (all in `trainer.py`):

### Optimizer (MuGrokfast)
```python
learning_rate: 5e-4        # AdamW fallback (1D params)
muon_lr: 5e-3              # Muon optimizer (2D params)
grokfast_lambda: 0.02      # Gradient filtering
gradient_clip: 1.0         # Max gradient norm
```

### Batch & Accumulation
```python
batch_size: 16             # Physical batch
gradient_accumulation: 4   # Effective batch = 64
```

### Learning Rate Schedule
```python
Epoch 1: Linear warmup (0.1 → 1.0 of base LR)
Epoch 2-10: Cosine decay (base LR → 1e-6)
```

### Early Stopping
```python
patience: 3 epochs
min_delta: 0.01
```

### Curriculum (Default)
```python
Foundation datasets: gsm8k, svamp, mbpp, arc_easy, piqa, wikitext
All epochs: Same datasets (no stage transitions)
```

---

## Troubleshooting

### GPU Out of Memory
**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--batch-size 8` (or `4`)
2. Enable gradient checkpointing in `model_config.py`:
   ```python
   gradient_checkpointing: bool = True
   ```
3. Use CPU (slower): `--device cpu`

### Training Diverges (Loss Increases)
**Symptom**: Loss increases instead of decreasing after epoch 5-7

**Solutions**:
1. Reduce learning rates further in `trainer.py:46-48`:
   ```python
   muon_lr: float = 2e-3  # Was 5e-3
   learning_rate: float = 2e-4  # Was 5e-4
   ```
2. Disable curriculum: `--no-curriculum` (if implemented)
3. Check gradient norms in W&B (should be < 10.0)

### Validation Loss Plateaus
**Symptom**: Val loss stuck at ~2.0 from epoch 2 onwards

**Analysis**: Model may be overfitting or datasets too difficult.

**Solutions**:
1. Early stopping will trigger automatically (saves time)
2. Try smaller dataset: `--datasets gsm8k piqa` (just 2 easy ones)
3. Increase dropout in `model_config.py:44`: `dropout: 0.2`

### ACT Variance Still 0
**Symptom**: Warning `var(): degrees of freedom is <= 0`

**Solutions**:
1. Increase diversity loss weight in `act_head.py:129`:
   ```python
   return loss_bce + loss_entropy + 0.05 * diversity_loss  # Was 0.01
   ```
2. If persists, disable ACT (use fixed recursion):
   ```python
   # In model_config.py
   trm_config.T_max = 3  # Fixed 3 steps
   # Remove ACT loss from full_model.py
   ```

---

## Next Steps After Success

### 1. Train Memory Model (4-6 hours)
```bash
python src/phase1_cognate/train_phase1.py \
    --model memory \
    --epochs 10 \
    --batch-size 16
```

### 2. Train Speed Model (4-6 hours)
```bash
python src/phase1_cognate/train_phase1.py \
    --model speed \
    --epochs 10 \
    --batch-size 16
```

### 3. Phase 1 → Phase 2 Handoff

Once all 3 models trained successfully:

```bash
# Verify handoff readiness
python -c "
import torch
from pathlib import Path

models = ['reasoning', 'memory', 'speed']
for name in models:
    ckpt_path = Path(f'checkpoints/phase1/{name}/best_model.pt')
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f'✅ {name}: Epoch {ckpt[\"epoch\"]}, Val loss {ckpt[\"best_val_loss\"]:.4f}')
    else:
        print(f'❌ {name}: Checkpoint not found')
"
```

**Expected**:
```
✅ reasoning: Epoch 7, Val loss 0.5523
✅ memory: Epoch 8, Val loss 0.4891
✅ speed: Epoch 6, Val loss 0.6120
```

Then proceed to Phase 2 (EvoMerge)!

---

## Total Timeline

| Task | Time | Status |
|------|------|--------|
| Quick test (CPU) | 1-2 min | ⏳ Ready |
| Reasoning model | 4-6 hours | ⏳ Ready |
| Memory model | 4-6 hours | ⏳ After reasoning |
| Speed model | 4-6 hours | ⏳ After memory |
| **TOTAL** | **12-18 hours** | **All fixes complete** |

If early stopping triggers consistently at epoch 5-7, total time may be **8-12 hours** (saves GPU time).

---

## Files Modified (All Fixes Complete)

1. ✅ `src/phase1_cognate/training/trainer.py` - Validation, batch size, early stopping, LR scheduler
2. ✅ `src/phase1_cognate/model/act_head.py` - ACT diversity loss
3. ✅ `src/phase1_cognate/model/titans_mag.py` - LTM batch independence (was already fixed)

**No further code changes needed!**

---

## Support

If you encounter issues:

1. **Check logs**: Look for error messages in console output
2. **Check W&B**: `wandb offline` logs saved to `wandb/` directory
3. **Check checkpoints**: `ls checkpoints/phase1/reasoning/`
4. **Test model loading**: Run quick output test above

Share:
- Error messages
- Last 20 lines of console output
- W&B metrics (loss curve, grad_norm)

---

**Ready to train!** 🚀

Start with: `python src/phase1_cognate/train_phase1.py --model reasoning --test --epochs 1`

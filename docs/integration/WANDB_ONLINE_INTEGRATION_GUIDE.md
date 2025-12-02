# W&B Online Integration Guide

**Created**: 2025-10-17
**Status**: Implementation Ready
**Purpose**: Enable real-time metric tracking on wandb.ai dashboard

---

## Overview

This guide shows how to integrate Weights & Biases (W&B) online mode with Phase 1 training to view real-time metrics on the wandb.ai dashboard.

### Benefits of Online Mode

- **Real-time Visualization**: See loss, accuracy, and GPU metrics update live
- **Model Versioning**: Automatic checkpoint storage and versioning
- **Collaboration**: Share training runs with team members
- **Experiment Comparison**: Compare multiple training runs side-by-side
- **Gradient Tracking**: Monitor gradient flow with `wandb.watch()`

---

## Quick Setup (5 minutes)

### Step 1: Get API Key

1. Visit [wandb.ai/authorize](https://wandb.ai/authorize)
2. Copy your API key (starts with a long alphanumeric string)

### Step 2: Authenticate

**Option A: Command Line** (Recommended)
```bash
python -m wandb login
# Paste your API key when prompted
```

**Option B: Environment Variable**
```bash
# Windows
set WANDB_API_KEY=<your_api_key>

# Linux/Mac
export WANDB_API_KEY=<your_api_key>
```

**Option C: Python Script**
```python
import wandb
wandb.login(key="<your_api_key>")
```

### Step 3: Verify Authentication

```bash
python -m wandb status
# Should show: "Logged in as <your_username>"
```

---

## Implementation Changes

### 1. Update Training Script

**File**: `scripts/train_phase1_cached.py`

**Change Line 21-23** from:
```python
# Configure W&B for offline mode (NO authentication required)
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_DIR'] = str(Path.cwd() / 'wandb')
os.environ['WANDB_SILENT'] = 'true'
```

**To**:
```python
# Configure W&B for online mode (dashboard at wandb.ai)
os.environ['WANDB_MODE'] = 'online'  # Changed from 'offline'
os.environ['WANDB_DIR'] = str(Path.cwd() / 'wandb')
os.environ['WANDB_PROJECT'] = 'agent-forge-v2'  # Project name
# REMOVE: os.environ['WANDB_SILENT'] = 'true'  # Allow W&B output
```

**Change Line 156** from:
```python
wandb_mode="offline",
```

**To**:
```python
wandb_mode="online",  # Enable dashboard syncing
```

### 2. Update W&B Logger (Add Gradient Tracking)

**File**: `src/phase1_cognate/training/wandb_logger.py`

**Add to `__init__` method** (after line 47):
```python
def __init__(
    self,
    config: Dict[str, Any],
    model_name: str,
    mode: str = "offline"
):
    self.model_name = model_name

    # Initialize W&B
    wandb.init(
        project=config.get("wandb_project", "agent-forge-v2"),
        name=f"phase1-cognate-{model_name}",
        config=config,
        mode=mode,
        tags=["phase1", "cognate", model_name, "pretraining"]
    )

    print(f"W&B initialized: {wandb.run.name}")

    # NEW: Store run reference for model tracking
    self.run = wandb.run
```

**Add new method** (after `log_step`):
```python
def watch_model(self, model: nn.Module, log_freq: int = 100):
    """
    Track model gradients and parameters

    Args:
        model: PyTorch model to track
        log_freq: How often to log gradients (every N steps)
    """
    self.run.watch(
        model,
        log="all",  # Log gradients and parameters
        log_freq=log_freq
    )
    print(f"W&B watching model (log_freq={log_freq})")
```

### 3. Update Trainer to Use wandb.watch()

**File**: `src/phase1_cognate/training/trainer.py`

**Add after line 108** (after W&B logger init):
```python
# W&B logger
self.logger = Phase1WandBLogger(
    config=config.model_config.to_dict(),
    model_name=config.model_config.specialization,
    mode=config.wandb_mode
)

# NEW: Track model gradients with W&B
self.logger.watch_model(self.model, log_freq=100)
```

---

## Testing the Integration

### Test 1: Verify Authentication

```bash
cd "c:\Users\17175\Desktop\the agent maker"
python -c "import wandb; wandb.login(); print('Authentication successful!')"
```

**Expected Output**:
```
wandb: Logged in as <your_username>
Authentication successful!
```

### Test 2: Quick Training Test (5 minutes)

```bash
# Create a minimal test script
python -c "
import os
os.environ['WANDB_MODE'] = 'online'
os.environ['WANDB_PROJECT'] = 'agent-forge-v2-test'

import wandb
run = wandb.init(project='agent-forge-v2-test', name='quick-test')

# Log some dummy metrics
for i in range(10):
    run.log({'loss': 10.0 / (i + 1), 'step': i})

run.finish()
print('Test complete! Check wandb.ai for results.')
"
```

**Expected**:
- Visit [wandb.ai/home](https://wandb.ai/home)
- See project "agent-forge-v2-test"
- See run "quick-test" with 10 logged steps

### Test 3: Full Training with Online Mode

```bash
# After authentication, run training with online mode
python scripts/train_phase1_cached.py
```

**Dashboard Location**: [wandb.ai/home](https://wandb.ai/home) → Project: "agent-forge-v2"

---

## What You'll See on the Dashboard

### Run Overview Page
- **Run Name**: `phase1-cognate-reasoning` (or memory/speed)
- **Tags**: phase1, cognate, reasoning, pretraining
- **Status**: Running (live updates)

### Metrics Tab (Real-time Charts)

**Training Metrics** (updated every ~100 steps):
- `train/loss` - Training loss over time
- `train/perplexity` - Model perplexity
- `train/learning_rate` - Learning rate schedule
- `train/grad_norm` - Gradient norm (for stability)

**ACT Metrics**:
- `act/halting_steps_mean` - Average ACT halting steps
- `act/halting_steps_std` - ACT variance
- `act/ponder_cost` - Computational cost

**LTM Metrics**:
- `ltm/usage` - Long-term memory usage (0-1)

**GPU Metrics**:
- `gpu/memory_gb` - GPU memory usage
- `gpu/utilization` - GPU utilization %

**Epoch Metrics** (every epoch):
- `val/loss` - Validation loss
- `val/accuracy` - Validation accuracy
- `curriculum/stage` - Current curriculum stage (1-3)

### Model Tab
- **Architecture**: TRM × Titans-MAG details
- **Parameter Count**: 32,571,041 params
- **Gradients**: Histogram of gradient distributions (if `wandb.watch()` enabled)
- **Weights**: Parameter value distributions over time

### System Tab
- **GPU**: NVIDIA GeForce RTX 2060 SUPER (8GB)
- **CPU**: Utilization %
- **Disk**: I/O metrics
- **Network**: Upload/download speed

### Files Tab
- **Checkpoints**: Saved model files (if uploaded)
- **Code**: Training script snapshot
- **Config**: Full hyperparameter config JSON

---

## Switching from Offline to Online (Mid-Training)

If training is already running in offline mode, you can sync the logs later:

### Option 1: Sync After Training Completes

```bash
# Sync all offline runs
python -m wandb sync wandb/
```

### Option 2: Stop and Restart with Online Mode

```bash
# 1. Stop current training (Ctrl+C or kill process)
taskkill //F //IM python.exe

# 2. Authenticate
python -m wandb login

# 3. Update environment variables in train_phase1_cached.py
# Change WANDB_MODE from 'offline' to 'online'

# 4. Restart training (will resume from last checkpoint)
python scripts/train_phase1_cached.py
```

The trainer's checkpoint resume system will automatically continue from the last saved epoch!

---

## Troubleshooting

### Issue: "wandb: ERROR Error uploading"

**Solution**: Check internet connection, verify API key
```bash
python -m wandb status
python -m wandb login --relogin
```

### Issue: "wandb: ERROR Run failed with error code 401"

**Solution**: API key expired or invalid
```bash
# Get new API key from wandb.ai/authorize
python -m wandb login --relogin
```

### Issue: Dashboard shows "offline run"

**Solution**: Environment variable still set to offline
```python
# Verify in Python:
import os
print(os.environ.get('WANDB_MODE'))  # Should be 'online' not 'offline'
```

### Issue: No gradient histograms appearing

**Solution**: Add `wandb.watch()` call in trainer
```python
# In trainer.py __init__:
self.logger.watch_model(self.model, log_freq=100)
```

---

## Performance Impact

**Offline Mode**:
- **Overhead**: ~1-2% (local logging only)
- **Storage**: ~50MB per training run (local wandb/ directory)

**Online Mode**:
- **Overhead**: ~3-5% (network uploads)
- **Bandwidth**: ~10-50 MB/hour (depending on log frequency)
- **Storage**: Unlimited cloud storage on W&B servers

**Recommendation**: Use online mode for important runs, offline for debugging/testing.

---

## Next Steps

1. ✅ Authenticate with `python -m wandb login`
2. ✅ Update `train_phase1_cached.py` (change 'offline' → 'online')
3. ✅ Add `watch_model()` to wandb_logger.py
4. ✅ Test with quick 5-minute test run
5. ✅ Start full training with online mode
6. ✅ Open [wandb.ai/home](https://wandb.ai/home) to view live metrics

---

## Additional Resources

- **W&B Quickstart**: https://docs.wandb.ai/quickstart/
- **PyTorch Integration**: https://docs.wandb.ai/tutorials/pytorch/
- **Dashboard Guide**: https://docs.wandb.ai/guides/track/
- **API Reference**: https://docs.wandb.ai/ref/python/

---

**Ready to implement!** Follow the steps above to enable real-time dashboard tracking for Phase 1 training.

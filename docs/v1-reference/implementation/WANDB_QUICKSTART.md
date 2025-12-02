# 🚀 Weights & Biases Quick Start - 3 Steps to Get Started

## Before You Run the Pipeline

You need to set up your Weights & Biases account and authenticate. This is a **ONE-TIME SETUP**.

---

## Step 1: Create a Weights & Biases Account

1. Go to https://wandb.ai
2. Click "Sign Up" (free account available)
3. Create account using:
   - Email
   - GitHub
   - Google
   - Or other options

**That's it!** Your account is ready.

---

## Step 2: Get Your API Key

1. After logging in, go to https://wandb.ai/settings
2. Scroll down to "API Keys" section
3. Click "Reveal" to see your API key
4. Copy the key (looks like: `a1b2c3d4e5f6...`)

**Keep this key secure!** Treat it like a password.

---

## Step 3: Authenticate on Your Machine

Open your terminal and run **ONE** of these methods:

### Method A: Interactive Login (Easiest)
```bash
wandb login
```
This will:
- Open your browser
- Ask you to paste your API key
- Save it permanently on your machine

### Method B: Environment Variable
```bash
# Linux/Mac
export WANDB_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:WANDB_API_KEY="your_api_key_here"

# Windows (CMD)
set WANDB_API_KEY=your_api_key_here
```

### Method C: Add to Your Shell Config (Permanent)
```bash
# Linux/Mac - Add to ~/.bashrc or ~/.zshrc
echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

---

## ✅ You're Done! Now Run the Pipeline

```python
from agent_forge.core.unified_pipeline import UnifiedPipeline

# All phases will automatically log to wandb
pipeline = UnifiedPipeline()
result = await pipeline.run_complete_pipeline(
    base_models=["model1", "model2", "model3"],
    output_dir="./output",
)
```

---

## 📊 View Your Results

1. Go to https://wandb.ai
2. Click on "Projects" in the left sidebar
3. Find your project: **"agent-forge-pipeline"**
4. See all 8 phases with their metrics!

---

## 🔧 Alternative: Run Without Internet (Offline Mode)

If you don't have internet or want to test locally first:

```bash
# Run in offline mode
export WANDB_MODE=offline

# Run your pipeline
python run_pipeline.py

# Later, when you have internet, sync:
wandb sync ./wandb/offline-run-*
```

---

## 🛑 Disable Wandb Completely (Optional)

If you don't want to use wandb at all:

```bash
export WANDB_MODE=disabled
```

Or in code:
```python
config = CognateConfig(wandb_enabled=False)
```

---

## 🎯 Quick Verification

Test if wandb is working:

```python
import wandb

# This should work without errors
wandb.init(project="test-project", name="test-run")
wandb.log({"test_metric": 1.0})
wandb.finish()

print("✅ Wandb is working!")
```

---

## 📚 What Happens When You Run?

When you run the Agent Forge pipeline with wandb:

1. **Phase 1 (Cognate)**: Logs 3 model creation with 37 metrics
2. **Phase 2 (EvoMerge)**: Logs 50 generations with 370 metrics
3. **Phase 3 (Quiet-STaR)**: Logs reasoning enhancement with 17 metrics
4. **Phase 4 (BitNet)**: Logs 8× compression with 19 metrics
5. **Phase 5 (Forge)**: Logs 50K training steps with 7,208 metrics
6. **Phase 6 (Baking)**: Logs tool/persona training with 25 metrics
7. **Phase 7 (ADAS)**: Logs architecture search with 100 metrics
8. **Phase 8 (Final)**: Logs 280× compression with 25 metrics

**Total: ~7,800 metrics tracked automatically!**

---

## ⚡ Pro Tips

### Tip 1: Name Your Runs
```python
pipeline_id = f"experiment-{experiment_name}-{timestamp}"
```

### Tip 2: Use Tags for Organization
```python
config = CognateConfig(
    wandb_tags=["baseline", "grokfast", "batch-size-32"],
)
```

### Tip 3: Check Your Dashboard Regularly
- Compare different experiments
- Track model improvements
- Share results with team

---

## 🆘 Common Issues

### "Error: wandb not installed"
```bash
pip install wandb==0.18.3
```

### "Authentication failed"
```bash
wandb login --relogin
```

### "Network error"
```bash
export WANDB_MODE=offline  # Run without internet
```

### "Too much data being logged"
Wandb is working! You can:
- Let it run (it compresses automatically)
- Use offline mode
- Reduce logging frequency in code

---

## 🎉 That's It!

You're ready to track your Agent Forge pipeline with Weights & Biases!

**Next Steps**:
- Run your first pipeline
- Check the wandb dashboard
- See the full [WANDB_SETUP_GUIDE.md](WANDB_SETUP_GUIDE.md) for advanced features

**Happy Tracking!** 🚀

---

**Quick Links**:
- Wandb Dashboard: https://wandb.ai
- Full Setup Guide: [WANDB_SETUP_GUIDE.md](WANDB_SETUP_GUIDE.md)
- Implementation Details: [WANDB_100_PERCENT_COMPLETE.md](WANDB_100_PERCENT_COMPLETE.md)

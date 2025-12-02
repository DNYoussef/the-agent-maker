# Phase 1 Cognate - Quick Start Guide

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (6GB+ VRAM recommended)
- 16GB+ system RAM
- 50GB disk space

## Installation

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets wandb

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Quick Test (5 minutes)

Run the test suite to verify everything works:

```bash
cd "c:\Users\17175\Desktop\the agent maker"
python src/phase1_cognate/test_training.py
```

Expected output:
```
======================================================================
ALL TESTS PASSED - OK
======================================================================
```

## Download Datasets (1-2 hours)

Download all 16 HuggingFace datasets:

```bash
python src/phase1_cognate/datasets/dataset_downloader.py
```

This will download:
- Math: GSM8K, SVAMP, ASDiv
- Code: MBPP, CodeXGLUE
- Science: ARC-Easy, ARC-Challenge
- Reasoning: StrategyQA, DROP, HotpotQA
- QA: BoolQ
- Commonsense: PIQA, HellaSwag
- Language: WikiText

## Train Models

### Option 1: Train All 3 Models (Recommended)

```bash
# Train all 3 specialized models sequentially
python src/phase1_cognate/train_phase1.py --all

# Expected time: ~30 hours GPU time
# Will train: reasoning → memory → speed
```

### Option 2: Train Individual Models

```bash
# Reasoning model (thinks longer, ACT=0.95)
python src/phase1_cognate/train_phase1.py --specialization reasoning

# Memory model (large LTM, 8192 slots)
python src/phase1_cognate/train_phase1.py --specialization memory

# Speed model (halts quickly, ACT=0.99)
python src/phase1_cognate/train_phase1.py --specialization speed
```

### Option 3: Quick Test Mode (5 minutes)

Train for just 1 epoch to verify the pipeline:

```bash
python src/phase1_cognate/train_phase1.py --test
```

## Command-Line Arguments

```
python src/phase1_cognate/train_phase1.py --help

Options:
  --specialization {reasoning,memory,speed}
                        Which model to train (default: reasoning)
  --all                 Train all 3 models sequentially
  --test                Quick test mode (1 epoch, synthetic data)
  --batch-size INT      Batch size (default: 16)
  --epochs INT          Number of epochs (default: 10)
  --device {cuda,cpu}   Device to use (default: cuda)
  --wandb-mode {online,offline,disabled}
                        W&B mode (default: offline)
```

## Monitor Training

### Weights & Biases (W&B)

1. **Login to W&B** (optional, for cloud syncing):
   ```bash
   wandb login
   ```

2. **View metrics locally**:
   ```bash
   # W&B runs in offline mode by default
   # Metrics saved to: wandb/latest-run/
   ```

3. **Key metrics to watch**:
   - `loss`: Should decrease from ~10.5 to ~2.0
   - `perplexity`: Should decrease from ~36,000 to ~7
   - `avg_halting_steps`: ACT behavior (varies by model)
   - `memory_usage`: LTM utilization

### Console Output

Training progress is printed every 50 steps:
```
Epoch 1/10
Curriculum Stage: FOUNDATION
Datasets (6): gsm8k, svamp, mbpp, arc_easy, piqa, wikitext

  Step 50, Batch 50: loss=8.2451
  Step 100, Batch 100: loss=6.5123
  ...
```

## Expected Training Results

### Timeline
- **Per epoch**: ~1 hour (GTX 1660, batch_size=16)
- **Per model**: ~10 hours (10 epochs)
- **All 3 models**: ~30 hours

### GPU Usage
- **VRAM**: ~5GB peak (out of 6GB)
- **Utilization**: ~90%
- **Temperature**: Monitor to stay under 80°C

### Loss Curves
- **Initial loss**: ~10.5
- **After epoch 1**: ~6-8
- **After epoch 5**: ~3-4
- **Final loss**: ~1.5-2.5

### Model Quality (Rough Estimates)
- **Math (GSM8K)**: 30-40% accuracy
- **Code (MBPP)**: 20-30% pass@1
- **Reasoning (ARC)**: 40-50% accuracy

## Checkpoints

Models are saved automatically:

```
checkpoints/phase1/
├── reasoning_epoch_10.pt       # Final checkpoint
├── reasoning_best.pt           # Best validation loss
├── memory_epoch_10.pt
├── memory_best.pt
├── speed_epoch_10.pt
└── speed_best.pt
```

Load a checkpoint:
```python
import torch
from phase1_cognate.model import TRMTitansMAGModel
from phase1_cognate.model.model_config import Phase1Config

config = Phase1Config(specialization='reasoning')
model = TRMTitansMAGModel(config)

checkpoint = torch.load('checkpoints/phase1/reasoning_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_phase1.py --batch-size 8

# Or use CPU (slow)
python train_phase1.py --device cpu
```

### Datasets not found
```bash
# Download datasets first
python src/phase1_cognate/datasets/dataset_downloader.py

# Or check HuggingFace cache
ls ~/.cache/huggingface/datasets/
```

### Loss not decreasing
- Check learning rate (should be 1e-3)
- Verify dataset quality
- Try longer training (15-20 epochs)
- Check for NaN gradients in logs

### W&B errors
```bash
# Run offline mode
python train_phase1.py --wandb-mode offline

# Or disable W&B
python train_phase1.py --wandb-mode disabled
```

## Next Steps

After training completes:

1. **Evaluate models**:
   ```bash
   python src/phase1_cognate/evaluate.py --checkpoint checkpoints/phase1/reasoning_best.pt
   ```

2. **Compare models**:
   ```bash
   python src/phase1_cognate/compare_models.py
   ```

3. **Prepare for Phase 2**:
   - Phase 2 (EvoMerge) will merge these 3 models using evolutionary optimization
   - Ensure all checkpoints are saved
   - Verify model diversity (ACT, LTM behaviors differ)

## Support

- **Documentation**: [phases/phase1/PHASE1_COMPLETE_GUIDE.md](PHASE1_COMPLETE_GUIDE.md)
- **Architecture**: [phases/phase1/TRM_TITANS_ARCHITECTURE.md](TRM_TITANS_ARCHITECTURE.md)
- **Implementation Status**: [phases/phase1/PHASE1_IMPLEMENTATION_STATUS.md](PHASE1_IMPLEMENTATION_STATUS.md)

---

**Ready to train?** Run: `python src/phase1_cognate/train_phase1.py --all`

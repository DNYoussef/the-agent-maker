# Phase 4 Implementation Guide & Troubleshooting

**Version**: 1.0
**Target Audience**: Developers implementing Phase 4
**Difficulty**: Intermediate

---

## Table of Contents

- [Getting Started](#getting-started)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Common Issues & Solutions](#common-issues--solutions)
- [Performance Optimization](#performance-optimization)
- [Debugging Guide](#debugging-guide)
- [FAQ](#faq)

---

## Getting Started

### Prerequisites

**Required**:
- Python 3.10+
- PyTorch 2.0+
- transformers library
- Phase 3 output (HuggingFace model format)
- 6GB+ GPU VRAM (or CPU with 16GB+ RAM)

**Optional**:
- Weights & Biases account (for logging)
- GPU with CUDA support (2-4x faster)

### Installation

```bash
# Install Phase 4 dependencies
cd "the agent maker"
pip install -e .

# Install testing dependencies
pip install pytest pytest-cov

# Verify installation
python -c "from src.phase4_bitnet import Phase4Controller; print('✅ Phase 4 ready')"
```

---

## Step-by-Step Implementation

### Step 1: Prepare Phase 3 Output

**Requirements**:
- Model files in HuggingFace format
- Tokenizer files
- Model must be ≥100M parameters

**Directory Structure**:
```
phase3_output/
├── pytorch_model.bin  # or model.safetensors
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

**Validation**:
```python
from transformers import AutoModel, AutoTokenizer

# Test loading
model = AutoModel.from_pretrained("phase3_output/")
tokenizer = AutoTokenizer.from_pretrained("phase3_output/")

print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
```

---

### Step 2: Configure Phase 4

#### Basic Configuration

```python
from src.phase4_bitnet import Phase4Config

config = Phase4Config(
    # Paths
    model_path="phase3_output/",
    output_path="phase4_output/",

    # Compression
    target_compression_ratio=8.0,  # Will auto-adapt
    sparsity_threshold=0.1,        # 10% threshold

    # Calibration
    calibration_samples=1000,
    calibration_dataset="openwebtext",

    # Fine-tuning
    enable_fine_tuning=True,
    fine_tune_epochs=2,

    # System
    device="auto",  # Auto-detect GPU/CPU
    wandb_enabled=True,
)
```

#### Size-Adaptive Configuration

```python
# Load model to get size
from transformers import AutoModel

model = AutoModel.from_pretrained("phase3_output/")
num_params = sum(p.numel() for p in model.parameters())

# Adapt config
config.adapt_to_model_size(num_params)

print(f"Model size: {num_params:,} params")
print(f"Category: {config.get_size_category(num_params)}")
print(f"Target compression: {config.target_compression_ratio}x")
print(f"Sparsity threshold: {config.sparsity_threshold}")
```

**Output Example**:
```
Model size: 25,000,000 params
Category: small
Target compression: 8.0x
Sparsity threshold: 0.1
```

---

### Step 3: Execute Compression

#### Basic Execution

```python
from src.phase4_bitnet import Phase4Controller

# Create controller
controller = Phase4Controller(config)

# Execute
results = controller.execute(
    phase3_output_path="phase3_output/",
    wandb_logger=None  # Optional
)

# Check results
if results['success']:
    print("✅ Phase 4 Complete")
    print(f"Compression: {results['metrics']['compression_ratio']:.2f}x")
    print(f"Sparsity: {results['metrics']['sparsity_ratio']:.1%}")
else:
    print(f"❌ Error: {results['error']}")
```

#### With W&B Logging

```python
from src.cross_phase.monitoring.wandb_integration import WandBIntegration

# Initialize W&B (offline mode)
wandb_logger = WandBIntegration(mode="offline")
wandb_logger.init_phase_run(
    phase_name="phase4",
    config=config.to_dict(),
    session_id="phase4_run_001"
)

# Execute with logging
results = controller.execute(
    phase3_output_path="phase3_output/",
    wandb_logger=wandb_logger
)

# Finish W&B run
wandb_logger.finish()
```

---

### Step 4: Validate Outputs

#### Check Dual Model Output

```python
import torch
from pathlib import Path

output_dir = Path("phase4_output/")

# Check quantized model (12MB)
quantized_path = output_dir / "bitnet_quantized_model.pt"
assert quantized_path.exists(), "Quantized model missing"

quantized = torch.load(quantized_path)
print(f"Quantized size: {quantized_path.stat().st_size / 1024**2:.1f} MB")

# Check dequantized FP16 (50MB, PRIMARY)
dequant_path = output_dir / "bitnet_dequantized_fp16.pt"
assert dequant_path.exists(), "Dequantized model missing"

dequant = torch.load(dequant_path)
print(f"Dequantized size: {dequant_path.stat().st_size / 1024**2:.1f} MB")

# Check metadata
from src.phase4_bitnet.utils import load_compression_metadata

metadata = load_compression_metadata(output_dir)
print(f"Compression ratio: {metadata['metrics']['compression_ratio']:.2f}x")
```

#### Validate Gradient Flow

```python
from src.phase4_bitnet.utils import test_gradient_flow
from transformers import AutoModel

# Load dequantized model
dequant_state = torch.load("phase4_output/bitnet_dequantized_fp16.pt")

# Create model and load
model = AutoModel.from_pretrained("phase3_output/")
model.load_state_dict(dequant_state, strict=False)

# Test gradient flow
passed, error = test_gradient_flow(model, device='cpu')

if passed:
    print("✅ Gradient flow PASSED - Ready for Phase 5")
else:
    print(f"❌ Gradient flow FAILED: {error}")
```

---

## Common Issues & Solutions

### Issue 1: Low Compression Ratio

**Symptom**:
```
Compression ratio 5.2x below target 8.0x
```

**Causes**:
- Model too small (< 50M params)
- Many preserved layers (embeddings, LM head)
- Low sparsity threshold

**Solutions**:

```python
# Solution 1: Increase sparsity threshold
config.sparsity_threshold = 0.15  # More aggressive

# Solution 2: Accept lower target for small models
if num_params < 50_000_000:
    config.target_compression_ratio = 6.0

# Solution 3: Check preserved layer ratio
stats = compressed_model.get_compression_stats()
preserved_ratio = stats['layers_preserved'] / (stats['layers_quantized'] + stats['layers_preserved'])
print(f"Preserved layers: {preserved_ratio:.1%}")

# If >20%, model is small or has many special layers
```

---

### Issue 2: High Accuracy Loss

**Symptom**:
```
Post-compression perplexity: 18.5 (pre: 12.4)
Degradation: 49.2% - CRITICAL
```

**Causes**:
- Sparsity threshold too high
- Insufficient calibration samples
- Fine-tuning disabled or failing

**Solutions**:

```python
# Solution 1: Reduce sparsity
config.sparsity_threshold = 0.05

# Solution 2: More calibration samples
config.calibration_samples = 2000

# Solution 3: Stronger fine-tuning
config.enable_fine_tuning = True
config.fine_tune_epochs = 5
config.grokfast_lambda = 3.0  # Stronger recovery

# Solution 4: Use better calibration dataset
config.calibration_dataset = "c4"  # Larger, more diverse
```

---

### Issue 3: Out of Memory (OOM)

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Causes**:
- Model too large for GPU
- Calibration batch size too large
- Sequence length too long

**Solutions**:

```python
# Solution 1: Reduce calibration batch size
config.calibration_batch_size = 2  # From 4

# Solution 2: Reduce sequence length
config.calibration_sequence_length = 256  # From 512

# Solution 3: Use CPU (slower but works)
config.device = "cpu"

# Solution 4: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 5: Process layers in chunks
# (Requires modifying quantizer - see advanced section)
```

---

### Issue 4: Calibration Dataset Loading Fails

**Symptom**:
```
ValueError: OpenWebText dataset not available
Falling back to wikitext
```

**Causes**:
- No internet connection
- HuggingFace datasets not installed
- Dataset download timeout

**Solutions**:

```python
# Solution 1: Use local dataset
from src.phase4_bitnet.calibration import CalibrationDataset

dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")
dataset.set_custom_samples([
    "Your custom text here...",
    "More samples...",
    # Add 1000+ samples
])

# Solution 2: Use WikiText (smaller, more reliable)
config.calibration_dataset = "wikitext"

# Solution 3: Use synthetic samples
# CalibrationDataset automatically falls back to synthetic if all fail
```

---

### Issue 5: Fine-Tuning Divergence

**Symptom**:
```
Fine-tuning loss increasing instead of decreasing
Epoch 1: loss=2.5
Epoch 2: loss=3.2  # Worse!
```

**Causes**:
- Learning rate too high
- Grokfast lambda too aggressive
- Model already at local minimum

**Solutions**:

```python
# Solution 1: Lower learning rate
config.fine_tune_lr = 5e-6  # From 1e-5

# Solution 2: Increase warmup
config.warmup_steps = 100  # From 50

# Solution 3: Reduce Grokfast amplification
config.grokfast_lambda = 1.0  # From 2.0

# Solution 4: Shorter fine-tuning
config.fine_tune_epochs = 1

# Solution 5: Disable fine-tuning if quality acceptable
if perplexity_degradation < 0.05:  # <5% drop
    config.enable_fine_tuning = False
```

---

## Performance Optimization

### GPU Optimization

```python
# Use mixed precision
config.mixed_precision = True

# Pin memory for faster data transfer
config.num_workers = 4  # DataLoader workers

# Use larger batch sizes if memory allows
config.calibration_batch_size = 8  # From 4

# Compile model (PyTorch 2.0+)
model = torch.compile(model)
```

### CPU Optimization

```python
# Use all cores
import torch
torch.set_num_threads(torch.get_num_threads())

# Reduce batch size
config.calibration_batch_size = 1

# Reduce sequence length
config.calibration_sequence_length = 128

# Disable fine-tuning for speed
config.enable_fine_tuning = False
```

### Speed vs Quality Tradeoffs

| Priority | Calibration Samples | Fine-Tune Epochs | Expected Time | Quality |
|----------|-------------------|------------------|---------------|---------|
| **Speed** | 100 | 0 | 5 min | ~85% |
| **Balanced** | 1000 | 2 | 20 min | ~90% |
| **Quality** | 2000 | 5 | 60 min | ~95% |

---

## Debugging Guide

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('phase4_bitnet')

# Add to your code
logger.debug("Compression stats: %s", stats)
```

### Inspect Intermediate State

```python
# After compression
stats = compressed_model.get_compression_stats()
print("="*50)
print("Compression Statistics")
print("="*50)
print(f"Original size: {stats['original_size_mb']:.1f} MB")
print(f"Compressed size: {stats['quantized_size_mb']:.1f} MB")
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"Sparsity: {stats['sparsity_ratio']:.1%}")
print(f"Layers quantized: {stats['layers_quantized']}")
print(f"Layers preserved: {stats['layers_preserved']}")
print("="*50)

# Inspect quantized weights
quantized_dict = compressed_model.get_quantized_state_dict()
for name, param in list(quantized_dict.items())[:5]:
    print(f"{name}: dtype={param.dtype}, unique={torch.unique(param).tolist()}")
```

### Test Components Individually

```python
# Test quantizer
from src.phase4_bitnet import BitNetQuantizer

quantizer = BitNetQuantizer(config)
test_tensor = torch.randn(10, 10)

quantized, scale = quantizer.quantize_tensor(test_tensor)
print(f"Quantized values: {torch.unique(quantized)}")
print(f"Scale factor: {scale.mean():.4f}")

# Test calibration
from src.phase4_bitnet.calibration import create_calibration_dataloader

dataloader = create_calibration_dataloader(tokenizer, config)
batch = next(iter(dataloader))
print(f"Batch shape: {batch['input_ids'].shape}")

# Test gradient flow
from src.phase4_bitnet.utils import test_gradient_flow

passed, error = test_gradient_flow(model, device='cpu')
print(f"Gradient flow: {'PASS' if passed else 'FAIL'}")
```

---

## FAQ

### Q: What's the difference between quantized and dequantized output?

**A**:
- **Quantized** (12MB): int8 ternary weights, used for inference testing
- **Dequantized FP16** (50MB): **PRIMARY** output for Phase 5, enables training

Phase 5-7 require gradient backpropagation for training. Quantized weights break gradients, so we output dequantized FP16 for training while keeping quantized for validation.

### Q: How do I choose the right sparsity threshold?

**A**: Use size-adaptive thresholds (automatic):
- Tiny models: 0.05 (conservative)
- Small models: 0.10 (balanced)
- Medium models: 0.15 (aggressive)
- Large models: 0.20 (very aggressive)

Manual tuning:
```python
# Start with auto
config.adapt_to_model_size(num_params)

# Adjust based on results
if compression_ratio < 6.0:
    config.sparsity_threshold *= 1.5  # More aggressive
elif accuracy_loss > 0.10:
    config.sparsity_threshold *= 0.7  # More conservative
```

### Q: Should I always fine-tune?

**A**: Fine-tune if:
- Accuracy drop > 5% (automatic threshold)
- Perplexity degradation > 10%
- Model used for critical tasks

Skip fine-tuning if:
- Accuracy drop < 3%
- Time-constrained
- Model will be fine-tuned in Phase 5 anyway

### Q: How long does Phase 4 take?

**A**: Typical times (25M param model):

| Component | CPU | GPU (GTX 1660) |
|-----------|-----|----------------|
| Loading | 30s | 10s |
| Calibration | 5min | 1min |
| Compression | 2min | 30s |
| Fine-tuning (2 epochs) | 20min | 5min |
| Saving | 1min | 30s |
| **Total** | **~28min** | **~7min** |

### Q: Can I use Phase 4 standalone?

**A**: Yes! Phase 4 can compress any HuggingFace transformer:

```python
from src.phase4_bitnet import Phase4Controller, Phase4Config

# Compress any model
config = Phase4Config(
    model_path="path/to/any/huggingface/model/",
    output_path="compressed_output/",
)

controller = Phase4Controller(config)
results = controller.execute("path/to/any/huggingface/model/")
```

### Q: What if gradient flow test fails?

**A**: This is critical for Phase 5. Debug:

1. **Check dequantization accuracy**:
```python
# Should be ≥99.5%
stats = compressed_model.get_compression_stats()
# Add dequant accuracy metric
```

2. **Verify model loadable**:
```python
dequant_state = torch.load("phase4_output/bitnet_dequantized_fp16.pt")
model.load_state_dict(dequant_state, strict=False)
```

3. **Test manually**:
```python
model.train()
x = torch.randn(1, 512).to(device)
output = model(x)
loss = output.mean()
loss.backward()  # Should not error

# Check gradients exist
has_grads = any(p.grad is not None for p in model.parameters())
assert has_grads
```

---

## Advanced Topics

### Custom Quantization Strategy

```python
from src.phase4_bitnet import BitNetQuantizer

class CustomQuantizer(BitNetQuantizer):
    def quantize_tensor(self, tensor, threshold=None):
        # Custom quantization logic
        # Example: asymmetric quantization

        # Your custom code here
        pass
```

### Distributed Compression

```python
# For very large models (>10B params)
# Compress layers in parallel across GPUs

import torch.distributed as dist

# Initialize distributed
dist.init_process_group(backend='nccl')

# Compress different layers on different GPUs
# (Implementation left as exercise)
```

### Compression Quality Profiling

```python
# Profile quality loss per layer
from src.phase4_bitnet import BitNetQuantizer

quantizer = BitNetQuantizer(config)

layer_quality = {}
for name, param in model.named_parameters():
    if quantizer.should_quantize_layer(name):
        # Quantize
        quantized, scale = quantizer.quantize_tensor(param.data)

        # Dequantize
        dequantized = quantizer.dequantize_tensor(quantized, scale)

        # Calculate error
        error = ((param.data - dequantized) ** 2).mean().item()
        layer_quality[name] = error

# Find worst layers
worst_layers = sorted(layer_quality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Worst 10 layers:")
for name, error in worst_layers:
    print(f"  {name}: MSE={error:.6f}")
```

---

## Checklist

### Pre-Execution ✓
- [ ] Phase 3 output exists and loads
- [ ] GPU memory sufficient (6GB+) or using CPU
- [ ] Configuration validated
- [ ] Output directory writable
- [ ] W&B credentials (if logging)

### Post-Execution ✓
- [ ] Both models saved (quantized + dequantized)
- [ ] Compression ratio ≥ 6.0x
- [ ] Accuracy loss ≤ 10%
- [ ] Gradient flow test passed
- [ ] Metadata file exists
- [ ] Ready for Phase 5

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
**Status**: ✅ Complete

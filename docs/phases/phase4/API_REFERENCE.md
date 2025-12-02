# Phase 4 API Reference

**Version**: 1.0
**Module**: `src.phase4_bitnet`
**Status**: ✅ Production Ready

---

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Quantizer](#quantizer)
- [Compressed Model](#compressed-model)
- [Calibration](#calibration)
- [Fine-Tuning](#fine-tuning)
- [Phase Controller](#phase-controller)
- [Utilities](#utilities)
- [Examples](#examples)

---

## Overview

The Phase 4 API provides BitNet 1.58-bit compression for transformer models. The API is designed to be simple and modular, with clear separation between quantization, compression, calibration, and fine-tuning.

### Quick Start

```python
from src.phase4_bitnet import Phase4Controller, Phase4Config

# Configure
config = Phase4Config(
    model_path="phase3_output/",
    output_path="phase4_output/",
    target_compression_ratio=8.0
)

# Execute Phase 4
controller = Phase4Controller(config)
results = controller.execute(
    phase3_output_path="phase3_output/",
    wandb_logger=None  # Optional
)

# Check results
print(f"Compression: {results['metrics']['compression_ratio']:.2f}x")
print(f"Output: {results['output_paths']['primary_output']}")
```

---

## Configuration

### `Phase4Config`

**Location**: `src.phase4_bitnet.config`

Dataclass for Phase 4 configuration with size-adaptive compression targets.

#### Constructor

```python
Phase4Config(
    # Paths
    model_path: str = "",
    output_path: str = "phase4_bitnet_output",

    # Hardware
    device: str = "auto",  # "auto", "cuda", "cpu"

    # BitNet quantization
    quantization_bits: float = 1.58,
    preserve_embedding_precision: bool = True,
    preserve_output_precision: bool = True,
    sparsity_threshold: float = 0.1,

    # Calibration
    calibration_samples: int = 1000,
    calibration_dataset: str = "openwebtext",
    calibration_batch_size: int = 4,
    calibration_sequence_length: int = 512,

    # Fine-tuning
    enable_fine_tuning: bool = True,
    fine_tune_epochs: int = 2,
    fine_tune_lr: float = 1e-5,
    warmup_steps: int = 50,

    # Grokfast
    enable_grokfast: bool = True,
    grokfast_ema_alpha: float = 0.98,
    grokfast_lambda: float = 2.0,

    # Compression targets
    target_compression_ratio: float = 8.0,
    max_accuracy_drop: float = 0.10,
    fine_tune_threshold: float = 0.05,

    # System
    mixed_precision: bool = True,
    seed: int = 42,
    num_workers: int = 4,

    # W&B
    wandb_enabled: bool = True,
    wandb_project: str = "agent-forge-v2",

    # Output options
    save_quantized: bool = True,
    save_dequantized_fp16: bool = True,
)
```

#### Methods

##### `adapt_to_model_size(num_params: int)`

Automatically adapt compression targets based on model size.

**Parameters**:
- `num_params` (int): Number of model parameters

**Example**:
```python
config = Phase4Config()
config.adapt_to_model_size(25_000_000)  # 25M params

print(config.target_compression_ratio)  # 8.0 (small model)
print(config.sparsity_threshold)        # 0.10
```

##### `get_size_category(num_params: int) -> str`

Determine model size category.

**Returns**: `"tiny"`, `"small"`, `"medium"`, or `"large"`

**Example**:
```python
config = Phase4Config()
category = config.get_size_category(25_000_000)
print(category)  # "small"
```

##### `to_dict() -> Dict`

Convert configuration to dictionary.

**Returns**: Dictionary representation

##### `from_dict(config_dict: Dict) -> Phase4Config`

Create configuration from dictionary.

**Parameters**:
- `config_dict` (Dict): Configuration dictionary

**Returns**: Phase4Config instance

---

## Quantizer

### `BitNetQuantizer`

**Location**: `src.phase4_bitnet.quantizer`

Core BitNet 1.58-bit ternary quantizer.

#### Constructor

```python
BitNetQuantizer(config: Phase4Config)
```

**Parameters**:
- `config` (Phase4Config): Configuration instance

#### Methods

##### `quantize_tensor(tensor: Tensor, threshold: Optional[float] = None) -> Tuple[Tensor, Tensor]`

Quantize tensor to ternary {-1, 0, +1}.

**Parameters**:
- `tensor` (torch.Tensor): Weight tensor to quantize
- `threshold` (float, optional): Sparsity threshold (uses config if None)

**Returns**:
- `quantized` (torch.Tensor): Quantized int8 tensor
- `scale` (torch.Tensor): Per-channel scale factors

**Example**:
```python
quantizer = BitNetQuantizer(config)
weights = torch.randn(128, 256)

quantized, scale = quantizer.quantize_tensor(weights)

print(quantized.dtype)  # torch.int8
print(quantized.unique())  # tensor([-1, 0, 1])
print(scale.shape)  # torch.Size([128, 1])
```

##### `dequantize_tensor(quantized: Tensor, scale: Tensor) -> Tensor`

Dequantize tensor back to FP32.

**Parameters**:
- `quantized` (torch.Tensor): Quantized int8 tensor
- `scale` (torch.Tensor): Scale factor tensor

**Returns**:
- `dequantized` (torch.Tensor): Dequantized FP32 tensor

**Example**:
```python
dequantized = quantizer.dequantize_tensor(quantized, scale)
print(dequantized.dtype)  # torch.float32
```

##### `quantize_model(model: nn.Module) -> Tuple[Dict, Dict]`

Quantize entire model.

**Parameters**:
- `model` (nn.Module): PyTorch model

**Returns**:
- `quantized_state_dict` (Dict[str, Tensor]): Quantized parameters
- `scale_factors` (Dict[str, Tensor]): Scale factors

**Example**:
```python
quantizer = BitNetQuantizer(config)
quantized_dict, scales = quantizer.quantize_model(model)

print(len(quantized_dict))  # Number of parameters
print(quantized_dict['linear1.weight'].dtype)  # torch.int8
```

##### `dequantize_model(quantized_state_dict: Dict, scale_factors: Dict) -> Dict`

Dequantize model to FP16.

**Parameters**:
- `quantized_state_dict` (Dict): Quantized state dict
- `scale_factors` (Dict): Scale factors

**Returns**:
- `dequantized_state_dict` (Dict): FP16 state dict

##### `get_stats() -> Dict`

Get quantization statistics.

**Returns**: Dictionary with:
- `layers_quantized` (int)
- `layers_preserved` (int)
- `total_params` (int)
- `quantized_params` (int)
- `sparsity_ratio` (float)

---

## Compressed Model

### `CompressedModel`

**Location**: `src.phase4_bitnet.compressed_model`

STE-enabled wrapper for compressed models.

#### Constructor

```python
CompressedModel(
    base_model: nn.Module,
    quantizer: BitNetQuantizer,
    config: Phase4Config
)
```

**Parameters**:
- `base_model` (nn.Module): Original model
- `quantizer` (BitNetQuantizer): Quantizer instance
- `config` (Phase4Config): Configuration

#### Methods

##### `compress()`

Compress the model using BitNet quantization.

**Example**:
```python
compressed = CompressedModel(model, quantizer, config)
compressed.compress()

print(compressed.is_compressed)  # True
```

##### `forward(*args, **kwargs)`

Forward pass with STE.

**Note**: Forward uses quantized weights, backward uses full-precision gradients.

##### `get_quantized_state_dict() -> Dict`

Get quantized (int8) state dict.

**Returns**: Quantized state dictionary

**Example**:
```python
quantized_dict = compressed.get_quantized_state_dict()
torch.save(quantized_dict, "quantized_model.pt")
```

##### `get_dequantized_state_dict() -> Dict`

Get dequantized FP16 state dict (PRIMARY for Phase 5).

**Returns**: Dequantized FP16 state dictionary

**Example**:
```python
dequant_dict = compressed.get_dequantized_state_dict()
torch.save(dequant_dict, "dequantized_fp16.pt")
```

##### `get_scale_factors() -> Dict`

Get scale factors.

**Returns**: Scale factor dictionary

##### `get_compression_stats() -> Dict`

Get compression statistics.

**Returns**: Dictionary with:
- `is_compressed` (bool)
- `original_size_mb` (float)
- `quantized_size_mb` (float)
- `compression_ratio` (float)
- `layers_quantized` (int)
- `layers_preserved` (int)
- `sparsity_ratio` (float)

---

## Calibration

### `CalibrationDataset`

**Location**: `src.phase4_bitnet.calibration`

Dataset for calibration samples.

#### Constructor

```python
CalibrationDataset(
    tokenizer: PreTrainedTokenizer,
    config: Phase4Config,
    dataset_name: Optional[str] = None
)
```

**Parameters**:
- `tokenizer` (PreTrainedTokenizer): HuggingFace tokenizer
- `config` (Phase4Config): Configuration
- `dataset_name` (str, optional): Dataset ("openwebtext", "c4", "wikitext", "custom")

#### Methods

##### `set_custom_samples(samples: List[str])`

Set custom calibration samples.

**Parameters**:
- `samples` (List[str]): Text samples

**Example**:
```python
dataset = CalibrationDataset(tokenizer, config, dataset_name="custom")
dataset.set_custom_samples([
    "Sample text 1...",
    "Sample text 2...",
])
```

### `create_calibration_dataloader`

Create calibration dataloader.

```python
create_calibration_dataloader(
    tokenizer: PreTrainedTokenizer,
    config: Phase4Config,
    dataset_name: Optional[str] = None
) -> DataLoader
```

**Returns**: DataLoader instance

**Example**:
```python
from src.phase4_bitnet.calibration import create_calibration_dataloader

dataloader = create_calibration_dataloader(tokenizer, config)

for batch in dataloader:
    print(batch['input_ids'].shape)
    break
```

### `collect_activation_statistics`

Collect activation statistics from model.

```python
collect_activation_statistics(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]
```

**Returns**: Dictionary mapping layer names to statistics

**Example**:
```python
stats = collect_activation_statistics(model, dataloader, device='cpu')

for layer_name, layer_stats in stats.items():
    print(f"{layer_name}: mean={layer_stats['mean']:.4f}")
```

---

## Fine-Tuning

### `FineTuner`

**Location**: `src.phase4_bitnet.fine_tuner`

MuGrokfast-based fine-tuning for compressed models.

#### Constructor

```python
FineTuner(
    model: CompressedModel,
    config: Phase4Config,
    device: str = "cuda"
)
```

**Parameters**:
- `model` (CompressedModel): Compressed model
- `config` (Phase4Config): Configuration
- `device` (str): Device for training

#### Methods

##### `fine_tune(train_dataloader, eval_dataloader=None, log_callback=None) -> Dict`

Fine-tune compressed model.

**Parameters**:
- `train_dataloader` (DataLoader): Training data
- `eval_dataloader` (DataLoader, optional): Evaluation data
- `log_callback` (Callable, optional): Logging callback

**Returns**: Dictionary with:
- `epochs_completed` (int)
- `final_loss` (float)
- `best_perplexity` (float)
- `training_history` (List[Dict])

**Example**:
```python
tuner = FineTuner(compressed_model, config, device='cuda')

results = tuner.fine_tune(
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    log_callback=lambda metrics: wandb.log(metrics)
)

print(f"Final loss: {results['final_loss']:.4f}")
```

##### `should_fine_tune(pre_perplexity: float, post_perplexity: float) -> bool`

Check if fine-tuning is needed.

**Parameters**:
- `pre_perplexity` (float): Pre-compression perplexity
- `post_perplexity` (float): Post-compression perplexity

**Returns**: True if fine-tuning recommended

**Example**:
```python
should_tune = tuner.should_fine_tune(
    pre_perplexity=12.0,
    post_perplexity=13.5  # 12.5% degradation
)
print(should_tune)  # True (> 5% threshold)
```

##### `get_training_summary() -> Dict`

Get training summary.

**Returns**: Summary dictionary

---

## Phase Controller

### `Phase4Controller`

**Location**: `src.phase4_bitnet.phase_controller`

Main orchestrator for Phase 4 pipeline.

#### Constructor

```python
Phase4Controller(config: Phase4Config)
```

**Parameters**:
- `config` (Phase4Config): Configuration

#### Methods

##### `execute(phase3_output_path: str, wandb_logger=None) -> Dict`

Execute Phase 4 compression pipeline.

**Parameters**:
- `phase3_output_path` (str): Path to Phase 3 output
- `wandb_logger` (WandBIntegration, optional): W&B logger

**Returns**: Dictionary with:
- `success` (bool)
- `phase` (str): "phase4_bitnet"
- `output_paths` (Dict): Paths to outputs
- `pre_compression` (Dict): Pre-compression metrics
- `post_compression` (Dict): Post-compression metrics
- `fine_tuning` (Dict): Fine-tuning results (if performed)
- `gradient_flow_test` (Dict): Gradient flow validation
- `timing` (Dict): Execution timing
- `metrics` (Dict): All metrics

**Example**:
```python
from src.phase4_bitnet import Phase4Controller, Phase4Config

config = Phase4Config(
    model_path="phase3_output/",
    output_path="phase4_output/",
)

controller = Phase4Controller(config)

results = controller.execute(
    phase3_output_path="phase3_output/",
    wandb_logger=wandb_integration
)

if results['success']:
    print(f"✅ Compression: {results['metrics']['compression_ratio']:.2f}x")
    print(f"Primary output: {results['output_paths']['primary_output']}")
else:
    print(f"❌ Error: {results['error']}")
```

---

## Utilities

### Model Size Functions

#### `calculate_model_size_mb(model: nn.Module) -> float`

Calculate model size in megabytes.

**Parameters**:
- `model` (nn.Module): PyTorch model

**Returns**: Size in MB

**Example**:
```python
from src.phase4_bitnet.utils import calculate_model_size_mb

size = calculate_model_size_mb(model)
print(f"Model size: {size:.1f} MB")
```

#### `count_parameters(model: nn.Module) -> Dict[str, int]`

Count model parameters.

**Returns**: Dictionary with:
- `total` (int)
- `trainable` (int)
- `frozen` (int)

#### `calculate_sparsity_ratio(model: nn.Module) -> float`

Calculate fraction of zero weights.

**Returns**: Sparsity ratio (0.0 to 1.0)

### Compression Functions

#### `calculate_compression_ratio(original_size_mb: float, compressed_size_mb: float) -> float`

Calculate compression ratio.

**Returns**: Compression ratio (e.g., 8.2 for 8.2x)

#### `estimate_inference_speedup(original_size_mb: float, compressed_size_mb: float) -> float`

Estimate inference speedup from compression.

**Returns**: Estimated speedup (e.g., 2.5 for 2.5x faster)

### Validation Functions

#### `test_gradient_flow(model: nn.Module, device: str = "cuda") -> Tuple[bool, Optional[str]]`

Test if model supports gradient backpropagation.

**Returns**:
- `success` (bool): True if gradients flow
- `error` (str, optional): Error message if failed

**Example**:
```python
from src.phase4_bitnet.utils import test_gradient_flow

passed, error = test_gradient_flow(model, device='cpu')

if passed:
    print("✅ Gradient flow test PASSED")
else:
    print(f"❌ Gradient flow test FAILED: {error}")
```

#### `validate_compression_quality(pre_perplexity: float, post_perplexity: float, max_accuracy_drop: float) -> Tuple[bool, float]`

Validate compression quality.

**Returns**:
- `is_valid` (bool): True if quality acceptable
- `degradation_ratio` (float): Degradation ratio

### Metadata Functions

#### `save_compression_metadata(output_dir: Path, metadata: Dict)`

Save compression metadata to JSON.

#### `load_compression_metadata(output_dir: Path) -> Dict`

Load compression metadata from JSON.

---

## Examples

### Basic Compression

```python
from src.phase4_bitnet import (
    Phase4Config,
    BitNetQuantizer,
    CompressedModel
)

# Configure
config = Phase4Config(
    sparsity_threshold=0.1,
    target_compression_ratio=8.0
)

# Create quantizer
quantizer = BitNetQuantizer(config)

# Compress model
compressed_model = CompressedModel(model, quantizer, config)
compressed_model.compress()

# Get stats
stats = compressed_model.get_compression_stats()
print(f"Compression: {stats['compression_ratio']:.2f}x")
print(f"Sparsity: {stats['sparsity_ratio']:.1%}")

# Save dual outputs
quantized_dict = compressed_model.get_quantized_state_dict()
dequantized_dict = compressed_model.get_dequantized_state_dict()

torch.save(quantized_dict, "quantized_12mb.pt")
torch.save(dequantized_dict, "dequantized_fp16_50mb.pt")
```

### Full Pipeline with Fine-Tuning

```python
from src.phase4_bitnet import (
    Phase4Controller,
    Phase4Config
)
from src.cross_phase.monitoring.wandb_integration import WandBIntegration

# Configure
config = Phase4Config(
    model_path="phase3_output/",
    output_path="phase4_output/",
    enable_fine_tuning=True,
    fine_tune_epochs=2,
    wandb_enabled=True,
)

# Setup W&B
wandb_logger = WandBIntegration(mode="offline")

# Execute Phase 4
controller = Phase4Controller(config)
results = controller.execute(
    phase3_output_path="phase3_output/",
    wandb_logger=wandb_logger
)

# Check results
if results['success']:
    print("✅ Phase 4 Complete")
    print(f"Compression: {results['metrics']['compression_ratio']:.2f}x")
    print(f"Gradient flow: {results['gradient_flow_test']['passed']}")
    print(f"Primary output: {results['output_paths']['primary_output']}")
else:
    print(f"❌ Phase 4 Failed: {results['error']}")
```

### Custom Calibration

```python
from src.phase4_bitnet.calibration import CalibrationDataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create custom dataset
dataset = CalibrationDataset(
    tokenizer,
    config,
    dataset_name="custom"
)

# Set custom samples
dataset.set_custom_samples([
    "This is a calibration sample for quantization.",
    "Another sample with different content.",
    # ... more samples
])

# Create dataloader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4)

# Use for calibration
for batch in dataloader:
    model(input_ids=batch['input_ids'])
```

---

## Error Handling

### Common Exceptions

```python
# Invalid quantization bits
try:
    config = Phase4Config(quantization_bits=8.0)
except ValueError as e:
    print(e)  # "Phase 4 uses BitNet 1.58-bit quantization only"

# Invalid accuracy drop
try:
    config = Phase4Config(max_accuracy_drop=1.5)
except ValueError as e:
    print(e)  # "max_accuracy_drop must be between 0.0 and 0.5"

# Compression before state access
compressed = CompressedModel(model, quantizer, config)
try:
    quantized_dict = compressed.get_quantized_state_dict()
except RuntimeError as e:
    print(e)  # "Model not compressed yet"
```

---

## Best Practices

1. **Always use `adapt_to_model_size()`** after loading a model
2. **Save both outputs**: quantized (inference) + dequantized (training)
3. **Use `primary_output`** for Phase 5 handoff
4. **Validate gradient flow** before training
5. **Enable W&B logging** for debugging
6. **Use calibration** for better quantization quality

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
**Status**: ✅ Production Ready

# Phase 2 EvoMerge - Quick Start Guide

**Validated**: 2025-12-02 (All functionality tested and working)

This guide shows you how to use the Phase 2 EvoMerge implementations that just passed comprehensive functionality audit.

---

## Installation

### Required Dependencies
```bash
pip install torch numpy optuna datasets
```

### Optional (Recommended for Optimal Performance)
```bash
pip install cmaes  # Improves CMA-ES optimization speed
```

---

## 1. CMA-ES Optimizer for Parameter Space Merging

**File**: `src/phase2_evomerge/evolution/cma_es.py`

### Basic Usage

```python
from phase2_evomerge.evolution.cma_es import CMAESConfig, CMAESOptimizer

# Configure optimizer
config = CMAESConfig(
    population_size=50,      # Number of candidates per generation
    sigma=0.3,              # Initial step size (exploration)
    max_generations=100,    # Maximum iterations
    patience=50,            # Early stopping patience
    seed=42                 # Reproducibility
)

# Create optimizer
optimizer = CMAESOptimizer(config)

# Define objective function (higher is better)
def fitness_function(merge_coefficients):
    # merge_coefficients: numpy array of shape [n_models]
    # Returns: float (fitness score, higher is better)

    # Your model merging and evaluation logic here
    merged_model = merge_models(models, merge_coefficients)
    accuracy = evaluate_model(merged_model)
    return accuracy

# Optimize merge coefficients for 3 models
best_coeffs, best_fitness = optimizer.optimize(
    objective_fn=fitness_function,
    n_dimensions=3,         # Number of models to merge
    n_trials=500,          # Total optimization budget
    verbose=True           # Show progress
)

print(f"Best coefficients: {best_coeffs}")  # e.g., [0.4, 0.3, 0.3]
print(f"Best fitness: {best_fitness:.4f}")
```

### Advanced: Direct PS Merge with CMA-ES

```python
from phase2_evomerge.evolution.cma_es import ps_merge_with_cmaes

# Load your models
models = [model1, model2, model3]

# Define fitness function that evaluates a merged model
def evaluate(model):
    return model_accuracy(model)  # Your evaluation logic

# Run PS merge with CMA-ES optimization
merged_model, best_coeffs, best_fitness = ps_merge_with_cmaes(
    models=models,
    fitness_fn=evaluate,
    config=config,
    verbose=True
)
```

**Output**: Optimized merged model with coefficients that maximize fitness.

---

## 2. Benchmark Evaluation for Real Task Fitness

**File**: `src/phase2_evomerge/fitness/benchmarks.py`

### GSM8K Evaluation

```python
from phase2_evomerge.fitness.benchmarks import (
    BenchmarkConfig,
    evaluate_gsm8k
)
from transformers import AutoTokenizer

# Configure evaluation
config = BenchmarkConfig(
    benchmark_name="gsm8k",
    max_samples=100,        # Limit samples for fast eval (None = all)
    batch_size=8,
    max_length=512,
    temperature=0.0,        # Greedy decoding (deterministic)
    device="cuda"           # or "cpu"
)

# Load your model and tokenizer
model = load_your_model()
tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# Evaluate on GSM8K
results = evaluate_gsm8k(model, tokenizer, config)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Correct: {results['correct']}/{results['total']}")
```

### Quick Fitness Function for Evolution

```python
from phase2_evomerge.fitness.benchmarks import evaluate_benchmark

def fitness_fn(model):
    """Fast fitness evaluation for evolution."""
    config = BenchmarkConfig(
        benchmark_name="gsm8k",
        max_samples=50,  # Fast eval with subset
        device="cuda"
    )
    accuracy = evaluate_benchmark(model, tokenizer, "gsm8k", config)
    return accuracy
```

### Answer Extraction (Utility)

```python
from phase2_evomerge.fitness.benchmarks import extract_numeric_answer

# Extract numbers from various formats
text1 = "The answer is #### 42"
text2 = "Result equals 123.5"
text3 = "The value is -10"

answer1 = extract_numeric_answer(text1)  # 42.0
answer2 = extract_numeric_answer(text2)  # 123.5
answer3 = extract_numeric_answer(text3)  # -10.0
```

---

## 3. DFS Paper-Accurate Merge

**File**: `src/phase2_evomerge/merge/dfs_paper_accurate.py`

### Basic DFS Merge

```python
from phase2_evomerge.merge.dfs_paper_accurate import (
    DFSConfig,
    DFSPaperAccurate
)

# Configure DFS
config = DFSConfig(
    init_strategy="uniform",    # How to initialize indicators
    scale_init="ones",          # How to initialize scaling matrix
    min_layers_per_model=1      # Minimum layers to select from each model
)

# Create DFS merger
dfs = DFSPaperAccurate(config)

# Merge 3 models
models = [model1, model2, model3]
merged_model = dfs.merge(models)

# Access merge parameters
print(f"Indicator array: {dfs.indicator_array}")  # Binary selection
print(f"Scaling matrix: {dfs.scaling_matrix}")    # Layer-wise weights
```

### Advanced: Optimize Indicators and Scaling

```python
# Define fitness function
def fitness_fn(model):
    return evaluate_model_performance(model)

# Optimize DFS parameters
best_indicators, best_scaling = dfs.optimize_indicators_and_scaling(
    models=models,
    fitness_fn=fitness_fn,
    n_iterations=100  # Optimization budget
)

# Merge with optimized parameters
optimized_merged = dfs.merge(
    models=models,
    indicator_array=best_indicators,
    scaling_matrix=best_scaling
)
```

### Custom Indicators (Manual Control)

```python
import numpy as np

# Create custom indicator array
# For 3 models with 4 layers each: T = 3 * 4 = 12
M = 3  # number of models
r = 4  # layers per model
T = M * r

# Example: Select all layers from model 0, half from model 1, none from model 2
custom_indicators = np.array([
    1, 1, 1, 1,  # Model 0: all layers selected
    1, 0, 1, 0,  # Model 1: alternating layers
    0, 0, 0, 0   # Model 2: no layers selected
], dtype=np.float32)

# Custom scaling matrix (3x3)
custom_scaling = np.array([
    [1.0, 0.5, 0.0],  # Model 0 weights
    [0.5, 1.0, 0.0],  # Model 1 weights
    [0.0, 0.0, 0.0]   # Model 2 weights (excluded)
], dtype=np.float32)

# Merge with custom parameters
merged = dfs.merge(
    models=models,
    indicator_array=custom_indicators,
    scaling_matrix=custom_scaling
)
```

---

## 4. Hybrid PS+DFS Merge (Best Performance)

**File**: `src/phase2_evomerge/merge/hybrid_ps_dfs.py`

### Complete Hybrid Pipeline

```python
from phase2_evomerge.merge.hybrid_ps_dfs import (
    HybridConfig,
    HybridPSDFS,
    hybrid_merge
)

# Configure hybrid merge
config = HybridConfig(
    # PS phase settings
    ps_candidates_multiplier=3,  # Create 3x base models (e.g., 3 base -> 9 candidates)
    ps_generations=50,           # CMA-ES generations for PS

    # DFS phase settings
    dfs_optimization_iterations=100,  # Iterations for optimizing indicators

    # General
    device="cuda"
)

# Load base models
base_models = [model1, model2, model3]

# Define fitness function
def fitness_fn(model):
    return evaluate_model_on_benchmark(model)

# Run hybrid merge
champion_model, metrics = hybrid_merge(
    base_models=base_models,
    fitness_fn=fitness_fn,
    config=config,
    verbose=True
)

# Check results
print(f"Baseline fitness: {metrics['baseline_fitness']:.4f}")
print(f"Champion fitness: {metrics['champion_fitness']:.4f}")
print(f"Improvement: {metrics['fitness_improvement'] * 100:.2f}%")
print(f"PS candidates: {metrics['n_ps_candidates']}")
```

### Using HybridPSDFS Class

```python
# Create hybrid merger
hybrid = HybridPSDFS(config)

# Run merge
champion, metrics = hybrid.merge(
    base_models=base_models,
    fitness_fn=fitness_fn,
    verbose=True
)

# Access intermediate results
ps_candidates = hybrid.get_ps_candidates()  # PS phase candidate models
final_champion = hybrid.get_champion()       # Final merged model

print(f"PS phase created {len(ps_candidates)} candidates")
print(f"PS fitness scores: {metrics['ps_candidates_fitness']}")
```

---

## Complete Example: End-to-End Phase 2 Pipeline

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Phase 2 components
from phase2_evomerge.evolution.cma_es import CMAESConfig
from phase2_evomerge.fitness.benchmarks import BenchmarkConfig, evaluate_benchmark
from phase2_evomerge.merge.hybrid_ps_dfs import HybridConfig, hybrid_merge

# 1. Load base models (from Phase 1)
print("Loading base models...")
model1 = AutoModelForCausalLM.from_pretrained("path/to/model1")
model2 = AutoModelForCausalLM.from_pretrained("path/to/model2")
model3 = AutoModelForCausalLM.from_pretrained("path/to/model3")
base_models = [model1, model2, model3]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")

# 2. Configure benchmark evaluation
benchmark_config = BenchmarkConfig(
    benchmark_name="gsm8k",
    max_samples=100,  # Use subset for faster evolution
    batch_size=8,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 3. Define fitness function using real benchmarks
def fitness_function(model):
    """Evaluate model on GSM8K benchmark."""
    accuracy = evaluate_benchmark(
        model=model,
        tokenizer=tokenizer,
        benchmark_name="gsm8k",
        config=benchmark_config
    )
    return accuracy

# 4. Configure hybrid merge
hybrid_config = HybridConfig(
    ps_candidates_multiplier=3,      # Create 9 PS candidates from 3 base
    ps_generations=50,               # 50 CMA-ES generations
    dfs_optimization_iterations=100, # 100 DFS optimization iterations
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 5. Run hybrid PS+DFS merge
print("Starting hybrid PS+DFS merge...")
champion_model, metrics = hybrid_merge(
    base_models=base_models,
    fitness_fn=fitness_function,
    config=hybrid_config,
    verbose=True
)

# 6. Report results
print("\n" + "="*70)
print("PHASE 2 EVOMERGE COMPLETE")
print("="*70)
print(f"Base models: {len(base_models)}")
print(f"PS candidates created: {metrics['n_ps_candidates']}")
print(f"Baseline fitness: {metrics['baseline_fitness']:.4f}")
print(f"Champion fitness: {metrics['champion_fitness']:.4f}")
print(f"Fitness improvement: {metrics['fitness_improvement'] * 100:.2f}%")
print(f"Target improvement: 23.5% (from paper)")
print("="*70)

# 7. Save champion model
champion_model.save_pretrained("outputs/phase2_champion")
print(f"Champion model saved to: outputs/phase2_champion")
```

---

## Performance Tips

### 1. Fast Prototyping (Lightweight)
```python
# For quick testing
config = HybridConfig(
    ps_candidates_multiplier=2,  # Fewer candidates
    ps_generations=10,           # Fewer CMA-ES generations
    dfs_optimization_iterations=20  # Fewer DFS iterations
)

benchmark_config = BenchmarkConfig(
    max_samples=20  # Small subset for fast eval
)
```

### 2. Production Settings (Thorough)
```python
# For best results (slower)
config = HybridConfig(
    ps_candidates_multiplier=3,
    ps_generations=100,
    dfs_optimization_iterations=200
)

benchmark_config = BenchmarkConfig(
    max_samples=None  # Use full benchmark
)
```

### 3. GPU Acceleration
```python
# Enable GPU for model evaluation
benchmark_config = BenchmarkConfig(
    device="cuda",
    batch_size=16  # Increase batch size for GPU
)
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'cmaes'`
**Solution**: Install optional dependency for faster CMA-ES:
```bash
pip install cmaes
```
**Note**: Code will work without it (falls back to random sampling) but optimization may be slower.

### Issue: `ModuleNotFoundError: No module named 'datasets'`
**Solution**: Install Hugging Face datasets:
```bash
pip install datasets
```

### Issue: GSM8K dataset not found
**Solution**: Dataset will auto-download on first use. If offline, run once with internet:
```python
from datasets import load_dataset
dataset = load_dataset("gsm8k", "main", split="test")
```

### Issue: Out of memory during evaluation
**Solution**: Reduce batch size or max_samples:
```python
config = BenchmarkConfig(
    batch_size=4,      # Smaller batches
    max_samples=50     # Fewer samples
)
```

---

## Next Steps

1. **Phase 1 Integration**: Connect Phase 2 to Phase 1 Cognate models
2. **Phase 3 Preparation**: Champion model ready for Quiet-STaR reasoning
3. **Full Pipeline**: Integrate into complete Agent Forge V2 workflow

---

## References

- **Paper**: Evolutionary Optimization of Model Merging Recipes (Sakana AI, arXiv:2403.13187v1)
- **Audit Report**: `docs/PHASE2_FUNCTIONALITY_AUDIT_COMPLETE.md`
- **Test Suite**: `tests/phase2_evomerge/functionality_audit_phase2.py`

**Last Updated**: 2025-12-02
**Status**: ✅ All functionality validated and working

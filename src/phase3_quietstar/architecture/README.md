# Phase 3 Quiet-STaR Architecture Module

Modular implementation of Quiet-STaR (Self-Taught Reasoner) architecture for thought generation and reasoning enhancement.

## Quick Start

```python
# Import the complete model
from architecture import QuietSTaRModel

# Or import individual components
from architecture.thought_generator import ThoughtGenerator
from architecture.coherence_scorer import CoherenceScorer
from architecture.mixing_head import MixingHead
from architecture.thought_injector import ThoughtInjector
```

## Module Structure

```
architecture/
├── __init__.py              # Re-exports all components
├── dataclasses.py          # ThoughtOutput, CoherenceScores
├── thought_generator.py    # ThoughtGenerator class
├── coherence_scorer.py     # CoherenceScorer class
├── mixing_head.py          # MixingHead class
├── thought_injector.py     # ThoughtInjector class
└── quiet_star_model.py     # QuietSTaRModel wrapper
```

## Components Overview

### 1. Data Classes (`dataclasses.py`)

**ThoughtOutput**: Results from thought generation
- `thoughts`: Tensor (batch, num_thoughts, thought_len)
- `thought_ids`: Generated token IDs
- `log_probs`: Log probabilities
- `attention_weights`: Optional attention weights

**CoherenceScores**: Multi-dimensional thought quality scores
- `semantic`: Embedding similarity (40% weight)
- `syntactic`: Grammar validity (30% weight)
- `predictive`: Prediction utility (30% weight)
- `composite`: Weighted average

### 2. ThoughtGenerator (`thought_generator.py`)

Generates 4-8 parallel thought continuations at each token position.

```python
generator = ThoughtGenerator(
    base_model=model,
    num_thoughts=4,           # Number of parallel thoughts
    max_length=20,            # Max tokens per thought
    min_length=10,            # Min tokens per thought
    temperature=1.0,          # Sampling temperature
    top_p=0.9                # Nucleus sampling threshold
)

output = generator(input_ids, position, hidden_states)
```

**Key Features**:
- Nucleus (top-p) sampling for diversity
- Adaptive thought length (10-20 tokens)
- Parallel thought generation
- Log probability tracking

### 3. CoherenceScorer (`coherence_scorer.py`)

Scores thought quality across 3 dimensions:

```python
scorer = CoherenceScorer(
    hidden_size=768,
    weights={                 # Optional custom weights
        "semantic": 0.4,      # Embedding similarity
        "syntactic": 0.3,     # Grammar validity
        "predictive": 0.3     # Prediction utility
    }
)

scores = scorer(base_hidden, thought_hiddens, next_token_logits)
```

**Scoring Dimensions**:
- **Semantic** (40%): Cosine similarity between base and thought embeddings
- **Syntactic** (30%): Grammar validity via learned MLP
- **Predictive** (30%): How much thought helps next-token prediction

### 4. MixingHead (`mixing_head.py`)

Attention-based integration of thoughts with base representation.

```python
mixer = MixingHead(
    hidden_size=768,
    num_heads=8,              # Multi-head attention
    dropout=0.1               # Dropout rate
)

mixed = mixer(base_hidden, thought_hiddens, coherence_scores)
```

**Architecture**:
- 8-head multi-head attention
- Coherence-weighted attention (scores as bias)
- Gating mechanism (blend base + thoughts)
- Residual connection + LayerNorm

### 5. ThoughtInjector (`thought_injector.py`)

Identifies difficult token positions for thought injection.

```python
injector = ThoughtInjector(
    threshold=0.6,            # Difficulty threshold
    min_interval=3            # Min tokens between injections
)

should_inject = injector(logits, attention_weights, loss, position)
```

**Difficulty Metrics**:
- **Entropy** (40%): Prediction uncertainty
- **Attention Dispersion** (30%): Spread of attention
- **Loss** (30%): Prediction error

### 6. QuietSTaRModel (`quiet_star_model.py`)

Complete Quiet-STaR model integrating all components.

```python
model = QuietSTaRModel(
    base_model=base_model,
    hidden_size=768,
    num_thoughts=4,
    max_thought_length=20,
    injection_threshold=0.6,
    coherence_weights=None    # Optional custom weights
)

outputs = model(input_ids, labels=labels, use_thoughts=True)
```

**Workflow**:
1. Base model forward pass
2. For each token position:
   - Check if thought injection needed (ThoughtInjector)
   - Generate thoughts (ThoughtGenerator)
   - Score coherence (CoherenceScorer)
   - Mix with base hidden (MixingHead)
   - Update hidden states
3. Compute final logits and loss

**Returns**:
- `logits`: Final predictions
- `loss`: Cross-entropy loss (if labels provided)
- `thought_positions`: Positions where thoughts injected
- `avg_coherence`: Average coherence score
- `num_thoughts_used`: Total thoughts generated

## Usage Examples

### Basic Usage

```python
from architecture import QuietSTaRModel
import torch

# Create model
model = QuietSTaRModel(
    base_model=your_base_model,
    hidden_size=768,
    num_thoughts=4,
    max_thought_length=20
)

# Forward pass
input_ids = torch.randint(0, 50000, (2, 128))  # (batch=2, seq=128)
outputs = model(input_ids, use_thoughts=True)

print(f"Logits shape: {outputs['logits'].shape}")
print(f"Thoughts used: {outputs['num_thoughts_used']}")
print(f"Avg coherence: {outputs['avg_coherence']:.3f}")
print(f"Thought positions: {outputs['thought_positions']}")
```

### Training Example

```python
import torch.nn as nn

# Create model with labels
criterion = nn.CrossEntropyLoss()

input_ids = torch.randint(0, 50000, (2, 128))
labels = torch.randint(0, 50000, (2, 128))

# Forward pass with loss
outputs = model(input_ids, labels=labels, use_thoughts=True)

loss = outputs['loss']
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
```

### Custom Coherence Weights

```python
# Emphasize semantic coherence over syntactic
custom_weights = {
    "semantic": 0.6,      # 60% weight
    "syntactic": 0.2,     # 20% weight
    "predictive": 0.2     # 20% weight
}

model = QuietSTaRModel(
    base_model=base_model,
    hidden_size=768,
    coherence_weights=custom_weights
)
```

### Inference (No Thoughts)

```python
# Skip thought generation for faster inference
outputs = model(input_ids, use_thoughts=False)

# Just get base model predictions
logits = outputs['logits']
```

## Component Testing

```python
# Test individual components
from architecture.thought_generator import ThoughtGenerator
from architecture.coherence_scorer import CoherenceScorer

# Test thought generation
generator = ThoughtGenerator(base_model, num_thoughts=4)
thought_output = generator(input_ids, position=10)

print(f"Thoughts shape: {thought_output.thoughts.shape}")
print(f"Number of thoughts: {len(thought_output.thought_ids)}")

# Test coherence scoring
scorer = CoherenceScorer(hidden_size=768)
scores = scorer(base_hidden, thought_hiddens)

print(f"Semantic scores: {scores.semantic}")
print(f"Composite scores: {scores.composite}")
```

## Performance Characteristics

### Memory Usage
- **Base Model**: Original memory footprint
- **Thought Generation**: +4x thoughts × 20 tokens (minimal)
- **Coherence Scoring**: +3 scoring heads (lightweight)
- **Mixing Head**: +8 attention heads (moderate)

### Computational Cost
- **Without Thoughts**: Same as base model
- **With Thoughts**: ~2-3x slower (depending on injection frequency)
- **Injection Frequency**: Controlled by `threshold` (0.6 default)

### Typical Injection Rates
- **Easy Text**: 5-10% of tokens
- **Medium Text**: 15-25% of tokens
- **Hard Text**: 30-50% of tokens

## Configuration Guidelines

### For Speed (Fewer Thoughts)
```python
model = QuietSTaRModel(
    base_model=base_model,
    hidden_size=768,
    num_thoughts=2,              # Fewer thoughts
    max_thought_length=10,       # Shorter thoughts
    injection_threshold=0.7      # Higher threshold (fewer injections)
)
```

### For Quality (More Thoughts)
```python
model = QuietSTaRModel(
    base_model=base_model,
    hidden_size=768,
    num_thoughts=8,              # More thoughts
    max_thought_length=30,       # Longer thoughts
    injection_threshold=0.5      # Lower threshold (more injections)
)
```

### Balanced (Default)
```python
model = QuietSTaRModel(
    base_model=base_model,
    hidden_size=768,
    num_thoughts=4,              # Moderate thoughts
    max_thought_length=20,       # Moderate length
    injection_threshold=0.6      # Moderate threshold
)
```

## NASA POT10 Compliance

✅ **All files comply with NASA Power of Ten (POT10) rules**:
- All files < 200 lines (largest: 159 lines)
- All functions < 60 lines (largest: ~50 lines)
- Clear separation of concerns
- Single responsibility per module

## Backward Compatibility

**Old imports still work** (via `__init__.py`):
```python
# Old style (still supported)
from architecture import QuietSTaRModel

# New style (recommended)
from architecture.quiet_star_model import QuietSTaRModel
```

## Testing

Run the verification script to test all imports:
```bash
python test_architecture_refactor.py
```

Expected output:
```
✓ All backward compatibility imports successful
✓ All direct module imports successful
✓ Backward compatibility imports reference same classes
✓ All tests passed! Refactoring successful.
```

## Migration Guide

### From Monolithic `architecture.py`

**Before** (626 lines in one file):
```python
from architecture import QuietSTaRModel
```

**After** (7 focused modules):
```python
# Same import works!
from architecture import QuietSTaRModel

# Or be explicit about which component
from architecture.quiet_star_model import QuietSTaRModel
```

**No code changes required** - just reorganized files.

## Documentation

- **Full Refactoring Summary**: See `REFACTORING_SUMMARY.md`
- **Component Details**: See individual `.py` file docstrings
- **Phase 3 Guide**: See `phases/phase3/PHASE3_COMPLETE_GUIDE.md`

## License

Part of Agent Forge V2 - Phase 3 Quiet-STaR Implementation

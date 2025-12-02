# Phase 2: Day 6 Implementation Plan - Fitness Evaluation System

**Date**: 2025-10-17
**Focus**: Fitness Evaluation & Metrics
**Duration**: 1 day (estimated 6-8 hours)

---

## Objectives

Create a production-ready fitness evaluation system that:
1. Calculates perplexity, accuracy, speed, and memory metrics
2. Computes composite fitness score with configurable weights
3. Supports caching for performance
4. Batch-processes models efficiently
5. Integrates with existing merge techniques
6. Provides comprehensive test coverage (≥90%)

---

## V1 Reference (Fitness Formula)

```python
composite_fitness = (
    0.4 * (1 / perplexity) +        # Lower perplexity = better (40%)
    0.3 * accuracy +                 # Higher accuracy = better (30%)
    0.2 * (1 / inference_time) +    # Faster = better (20%)
    0.1 * (1 / memory_usage)        # Less memory = better (10%)
)
```

**Key Insight**: Fitness is **composite** - balances quality, speed, and efficiency.

---

## Architecture Design

### File Structure

```
src/phase2_evomerge/
├── fitness/
│   ├── __init__.py (API)
│   ├── perplexity.py (Perplexity calculation)
│   ├── accuracy.py (Accuracy measurement)
│   ├── speed.py (Inference speed benchmark)
│   ├── memory.py (VRAM usage tracking)
│   ├── composite.py (Composite fitness scoring)
│   └── cache.py (Fitness caching system)

tests/unit/
└── test_fitness_evaluation.py (Comprehensive tests)
```

### API Design

```python
from src.phase2_evomerge.fitness import FitnessEvaluator

# Initialize evaluator
evaluator = FitnessEvaluator(
    validation_dataset=val_data,
    fitness_weights={'perplexity': 0.4, 'accuracy': 0.3, 'speed': 0.2, 'memory': 0.1},
    cache_enabled=True,
    device='cuda'
)

# Evaluate single model
fitness = evaluator.evaluate(model)
# Returns: {
#     'composite': 0.185,
#     'components': {
#         'perplexity': 15.2, 'perplexity_score': 0.0658,
#         'accuracy': 0.48,
#         'speed': 1250, 'speed_score': 0.65,
#         'memory': 520, 'memory_score': 0.82
#     }
# }

# Batch evaluate (for population)
fitness_scores = evaluator.evaluate_batch([model1, model2, model3, ...])
# Returns: List[float] - composite scores
```

---

## Implementation Tasks (Day 6)

### Task 1: Perplexity Calculation (2 hours)

**File**: `src/phase2_evomerge/fitness/perplexity.py`

**Requirements**:
- Calculate perplexity on validation dataset
- Batch processing (avoid OOM)
- Support for mixed precision (faster evaluation)
- Handle edge cases (inf perplexity, NaN values)

**Function Signature**:
```python
def calculate_perplexity(
    model: nn.Module,
    validation_dataset: DataLoader,
    device: str = 'cuda',
    mixed_precision: bool = True,
    max_batches: Optional[int] = None
) -> float:
    """
    Calculate perplexity on validation dataset.

    Args:
        model: Model to evaluate
        validation_dataset: DataLoader with validation data
        device: Device to use ('cuda' or 'cpu')
        mixed_precision: Use torch.amp for faster evaluation
        max_batches: Limit evaluation to N batches (for speed)

    Returns:
        Perplexity value (float, lower is better)

    Raises:
        ValueError: If perplexity is NaN or Inf
    """
```

**Implementation Steps**:
1. Set model to eval mode
2. Iterate through validation batches
3. Compute cross-entropy loss
4. Calculate exp(avg_loss) = perplexity
5. Handle edge cases (NaN, Inf)
6. Support early stopping (max_batches)

**NASA POT10**: ≤60 LOC per function
- Main function: ~45 LOC
- Helper (batch processing): ~35 LOC

---

### Task 2: Accuracy Measurement (1 hour)

**File**: `src/phase2_evomerge/fitness/accuracy.py`

**Requirements**:
- Calculate accuracy on test set
- Support multi-class classification
- Batch processing
- Handle different tasks (next-token prediction, Q&A, etc.)

**Function Signature**:
```python
def calculate_accuracy(
    model: nn.Module,
    test_dataset: DataLoader,
    task_type: str = 'next_token',
    device: str = 'cuda',
    max_batches: Optional[int] = None
) -> float:
    """
    Calculate accuracy on test dataset.

    Args:
        model: Model to evaluate
        test_dataset: DataLoader with test data
        task_type: 'next_token', 'classification', or 'qa'
        device: Device to use
        max_batches: Limit evaluation to N batches

    Returns:
        Accuracy (float, 0.0-1.0, higher is better)
    """
```

**NASA POT10**: ≤60 LOC per function
- Main function: ~40 LOC

---

### Task 3: Speed Benchmark (1.5 hours)

**File**: `src/phase2_evomerge/fitness/speed.py`

**Requirements**:
- Measure inference latency (tokens/second)
- Warmup period (avoid cold start)
- CUDA synchronization (accurate timing)
- Batch size normalization

**Function Signature**:
```python
def benchmark_speed(
    model: nn.Module,
    benchmark_batch: torch.Tensor,
    device: str = 'cuda',
    num_warmup: int = 10,
    num_iterations: int = 100
) -> float:
    """
    Benchmark inference speed (tokens/second).

    Args:
        model: Model to benchmark
        benchmark_batch: Representative batch (batch_size, seq_len)
        device: Device to use
        num_warmup: Warmup iterations (avoid cold start)
        num_iterations: Measurement iterations

    Returns:
        Tokens per second (float, higher is better)
    """
```

**Implementation Steps**:
1. Warmup: Run inference N times
2. CUDA sync before timing
3. Run inference M times
4. CUDA sync after timing
5. Calculate tokens/second
6. Normalize by expected throughput

**NASA POT10**: ≤60 LOC per function
- Main function: ~50 LOC

---

### Task 4: Memory Measurement (1 hour)

**File**: `src/phase2_evomerge/fitness/memory.py`

**Requirements**:
- Measure peak VRAM usage
- Clear cache before measurement
- Reset peak stats
- Return MB usage

**Function Signature**:
```python
def measure_memory_usage(
    model: nn.Module,
    benchmark_batch: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Measure peak VRAM usage during inference (MB).

    Args:
        model: Model to measure
        benchmark_batch: Representative batch
        device: Device to use (must be 'cuda')

    Returns:
        Peak memory in MB (float, lower is better)

    Raises:
        RuntimeError: If device is not 'cuda'
    """
```

**Implementation Steps**:
1. Clear CUDA cache
2. Reset peak memory stats
3. Run forward pass
4. Get peak memory allocated
5. Convert bytes → MB

**NASA POT10**: ≤60 LOC per function
- Main function: ~35 LOC

---

### Task 5: Composite Fitness Scoring (1 hour)

**File**: `src/phase2_evomerge/fitness/composite.py`

**Requirements**:
- Combine 4 metrics into single score
- Configurable weights
- Normalize components to [0, 1]
- Handle edge cases (division by zero)

**Function Signature**:
```python
def compute_composite_fitness(
    perplexity: float,
    accuracy: float,
    speed: float,
    memory: float,
    weights: Dict[str, float] = None,
    expected_values: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Compute composite fitness score from components.

    Args:
        perplexity: Perplexity value (lower is better)
        accuracy: Accuracy value (0.0-1.0, higher is better)
        speed: Tokens/second (higher is better)
        memory: Peak memory MB (lower is better)
        weights: Fitness weights {'perplexity': 0.4, ...}
        expected_values: Normalization baselines

    Returns:
        Dictionary with composite score and components:
        {
            'composite': 0.185,
            'components': {
                'perplexity': 15.2,
                'perplexity_score': 0.0658,
                'accuracy': 0.48,
                'speed': 1250,
                'speed_score': 0.65,
                'memory': 520,
                'memory_score': 0.82
            }
        }
    """
```

**Default Weights** (from V1):
- perplexity: 0.4 (40%)
- accuracy: 0.3 (30%)
- speed: 0.2 (20%)
- memory: 0.1 (10%)

**Default Expected Values** (for normalization):
- perplexity: 15.0 (typical for 25M param model)
- speed: 1200 tokens/sec (on GTX 1660)
- memory: 500 MB (25M params × 4 bytes/param × 2)

**NASA POT10**: ≤60 LOC per function
- Main function: ~55 LOC

---

### Task 6: Fitness Caching System (1 hour)

**File**: `src/phase2_evomerge/fitness/cache.py`

**Requirements**:
- Hash model parameters for cache key
- Store fitness results
- LRU eviction policy (max 100 entries)
- Thread-safe (for parallel evaluation)

**Function Signature**:
```python
class FitnessCache:
    """LRU cache for fitness evaluation results."""

    def __init__(self, max_size: int = 100):
        """Initialize cache with max size."""

    def hash_model(self, model: nn.Module) -> str:
        """
        Compute hash of model parameters.

        Returns:
            SHA256 hash of flattened parameters
        """

    def get(self, model: nn.Module) -> Optional[Dict[str, Any]]:
        """Get cached fitness, or None if not found."""

    def put(self, model: nn.Module, fitness: Dict[str, Any]) -> None:
        """Store fitness in cache."""

    def clear(self) -> None:
        """Clear all cached entries."""

    def size(self) -> int:
        """Return number of cached entries."""
```

**NASA POT10**: ≤60 LOC per method
- hash_model: ~30 LOC
- get/put: ~15 LOC each
- clear/size: ~5 LOC each

---

### Task 7: FitnessEvaluator Main API (1 hour)

**File**: `src/phase2_evomerge/fitness/__init__.py`

**Requirements**:
- Unified API for fitness evaluation
- Batch processing support
- Caching integration
- Configuration management

**Class Design**:
```python
class FitnessEvaluator:
    """Main API for fitness evaluation."""

    def __init__(
        self,
        validation_dataset: DataLoader,
        test_dataset: Optional[DataLoader] = None,
        fitness_weights: Optional[Dict[str, float]] = None,
        expected_values: Optional[Dict[str, float]] = None,
        cache_enabled: bool = True,
        device: str = 'cuda',
        mixed_precision: bool = True,
        max_batches: Optional[int] = None
    ):
        """Initialize fitness evaluator with datasets and config."""

    def evaluate(self, model: nn.Module) -> Dict[str, Any]:
        """
        Evaluate single model fitness.

        Returns:
            Dictionary with composite score and components
        """

    def evaluate_batch(self, models: List[nn.Module]) -> List[float]:
        """
        Evaluate batch of models (parallel if possible).

        Returns:
            List of composite fitness scores
        """

    def clear_cache(self) -> None:
        """Clear fitness cache."""
```

**NASA POT10**: ≤60 LOC per method
- __init__: ~40 LOC
- evaluate: ~50 LOC
- evaluate_batch: ~45 LOC

---

### Task 8: Test Suite (1.5 hours)

**File**: `tests/unit/test_fitness_evaluation.py`

**Requirements**:
- Test each component function
- Test composite scoring
- Test caching behavior
- Test edge cases (NaN, Inf, zero values)
- Test batch evaluation
- ≥90% coverage

**Test Classes**:
```python
@pytest.mark.phase2
@pytest.mark.fitness
class TestPerplexityCalculation:
    def test_perplexity_single_batch(self, simple_model, dummy_data)
    def test_perplexity_multiple_batches(self, simple_model, dummy_data)
    def test_perplexity_nan_handling(self, broken_model, dummy_data)
    def test_perplexity_max_batches_limit(self, simple_model, dummy_data)

@pytest.mark.phase2
@pytest.mark.fitness
class TestAccuracyCalculation:
    def test_accuracy_next_token(self, simple_model, dummy_data)
    def test_accuracy_zero(self, wrong_model, dummy_data)
    def test_accuracy_perfect(self, perfect_model, dummy_data)

@pytest.mark.phase2
@pytest.mark.fitness
class TestSpeedBenchmark:
    def test_speed_warmup(self, simple_model)
    def test_speed_synchronization(self, simple_model)
    def test_speed_batch_normalization(self, simple_model)

@pytest.mark.phase2
@pytest.mark.fitness
class TestMemoryMeasurement:
    def test_memory_measurement(self, simple_model)
    def test_memory_cache_cleared(self, simple_model)

@pytest.mark.phase2
@pytest.mark.fitness
class TestCompositeFitness:
    def test_default_weights(self)
    def test_custom_weights(self)
    def test_normalization(self)
    def test_edge_cases(self)

@pytest.mark.phase2
@pytest.mark.fitness
class TestFitnessCache:
    def test_cache_hit(self, simple_model)
    def test_cache_miss(self, simple_model)
    def test_cache_eviction(self, simple_models)
    def test_cache_clear(self)

@pytest.mark.phase2
@pytest.mark.fitness
class TestFitnessEvaluator:
    def test_evaluate_single(self, evaluator, simple_model)
    def test_evaluate_batch(self, evaluator, simple_models)
    def test_cache_integration(self, evaluator, simple_model)
```

**Test Count**: 20+ tests
**Target Coverage**: ≥90%

---

## Success Criteria

### Functional
- ✅ All 4 component metrics implemented (perplexity, accuracy, speed, memory)
- ✅ Composite fitness scoring working
- ✅ Caching system functional
- ✅ Batch evaluation supported
- ✅ All tests passing

### Quality
- ✅ NASA POT10 compliant (all functions ≤60 LOC)
- ✅ Test coverage ≥90%
- ✅ Type hints 100%
- ✅ Docstrings 100%

### Performance
- ✅ Single model evaluation: <5 seconds (on validation subset)
- ✅ Batch evaluation (8 models): <40 seconds (with caching)
- ✅ Cache hit rate: >80% in typical evolution loop

---

## Estimated LOC

**Production Code**:
- perplexity.py: ~80 LOC (2 functions)
- accuracy.py: ~40 LOC (1 function)
- speed.py: ~50 LOC (1 function)
- memory.py: ~35 LOC (1 function)
- composite.py: ~55 LOC (1 function)
- cache.py: ~70 LOC (5 methods)
- __init__.py: ~135 LOC (3 methods)
- **Total**: ~465 LOC

**Test Code**:
- test_fitness_evaluation.py: ~600 LOC (20+ tests)

**Documentation**:
- Day 6 summary: ~300 lines
- Updated progress tracking: ~50 lines

**Total Day 6 Output**: ~1,415 lines

---

## Integration Points

### With Week 1 (Merge Techniques)
- Fitness evaluation uses models created by merge combos
- Tests use `simple_model` fixture from Week 1

### With Week 2 (Evolution Loop - Day 7+)
- Population management will call `evaluate_batch()` each generation
- Champion tracking uses composite fitness scores
- Early stopping uses fitness improvement tracking

---

## Risks & Mitigation

### Risk 1: Perplexity Calculation Slow
**Impact**: Fitness evaluation becomes bottleneck
**Mitigation**:
- Use subset of validation data (1K samples)
- Mixed precision (torch.amp)
- Early stopping (max_batches limit)
- Caching (80% hit rate in evolution)

### Risk 2: Memory Measurement Unreliable
**Impact**: Memory scores inconsistent
**Mitigation**:
- Clear cache before each measurement
- Reset peak stats
- Run multiple measurements and average

### Risk 3: Caching False Positives
**Impact**: Different models get same cached score
**Mitigation**:
- Use SHA256 hash of all parameters (collision resistant)
- Include model architecture in hash
- Add validation (warn if same hash, different model)

---

## Next Steps (After Day 6)

### Day 7: Population Management
- Initialize 8-model population
- Elite preservation logic
- Loser merging strategy
- Generation transition
- Uses `FitnessEvaluator.evaluate_batch()` from Day 6

### Day 8: Genetic Operations
- Mutation function (Gaussian noise)
- Selection algorithms
- Diversity metrics
- Re-seeding strategy

---

## References

### V1 Implementation
- [phases/phase2/PHASE2_COMPLETE_GUIDE.md](../../phases/phase2/PHASE2_COMPLETE_GUIDE.md) - Lines 602-780 (Fitness Evaluation section)
- [phases/phase2/LOGICAL_UNDERSTANDING.md](../../phases/phase2/LOGICAL_UNDERSTANDING.md) - Lines 170-190 (Fitness Formula)

### Week 1 Code
- [src/phase2_evomerge/merge/](../../src/phase2_evomerge/merge/) - Merge techniques for testing
- [tests/unit/test_merge_techniques.py](../../tests/unit/test_merge_techniques.py) - Test fixtures to reuse

---

**Document Version**: 1.0
**Created**: 2025-10-17
**Author**: Phase 2 Implementation Team
**Status**: Ready for Implementation 🚀

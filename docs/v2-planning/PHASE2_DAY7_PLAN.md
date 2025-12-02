# Phase 2: Day 7 Implementation Plan - Population Management & Evolution Loop

**Date**: 2025-10-17
**Focus**: Population Management, Genetic Operations, Evolution Loop
**Duration**: 1 day (estimated 6-8 hours)

---

## Objectives

Create a production-ready evolution loop that:
1. Initializes 8-model population from 3 base models (all binary combos)
2. Implements elite preservation (top 2 → 6 children via mutation)
3. Implements loser merging (bottom 6 → 2 children via combo merging)
4. Tracks diversity and implements re-seeding
5. Runs 50-generation evolution with early stopping
6. Integrates with Day 6 fitness evaluation
7. Achieves 23.5% fitness improvement target

---

## V1 Reference (Evolution Strategy)

```python
# Generation 0: 8 models (all binary combos from 3 Phase 1 models)
# Generations 1-50:
#   1. Evaluate fitness (sort by score)
#   2. Elite preservation: Top 2 → mutate 3× each → 6 children
#   3. Loser merging: Bottom 6 → 2 groups of 3 → merge → 2 children
#   4. New population: 6 + 2 = 8 models
#   5. Update champion if fitness improved
#   6. Check early stopping (if improvement < 0.1% for 5 gens)
```

**Key Parameters**:
- Population size: 8 models
- Elite count: 2 models
- Mutation rate: 0.01 (1% of weights)
- Mutation sigma: 0.01 (1% noise)
- Diversity threshold: 0.3 (healthy), 0.2 (re-seed)
- Early stopping: 0.001 improvement for 5 generations

---

## Architecture Design

### File Structure

```
src/phase2_evomerge/
├── evolution/
│   ├── __init__.py (API)
│   ├── population.py (Population initialization)
│   ├── mutation.py (Gaussian mutation)
│   ├── diversity.py (Diversity tracking)
│   ├── evolution_loop.py (Main evolution loop)
│   └── config.py (Evolution configuration)

tests/unit/
└── test_evolution.py (Comprehensive tests)
```

### API Design

```python
from src.phase2_evomerge.evolution import EvolutionLoop, EvolutionConfig

# Initialize
config = EvolutionConfig(
    generations=50,
    population_size=8,
    elite_count=2,
    mutation_sigma=0.01,
    mutation_rate=0.01,
    min_diversity=0.3,
    early_stopping=True
)

evolution = EvolutionLoop(
    config=config,
    fitness_evaluator=evaluator
)

# Run evolution
result = evolution.evolve(
    base_models=[model1, model2, model3]
)

# Returns: {
#     'champion': best_model,
#     'fitness': 0.185,
#     'improvement': 0.235,  # 23.5%
#     'generations': 38,
#     'convergence_reason': 'threshold_met'
# }
```

---

## Implementation Tasks (Day 7)

### Task 1: Population Initialization (1 hour)

**File**: `src/phase2_evomerge/evolution/population.py`

**Requirements**:
- Create 8 initial models using all binary combos (000-111)
- Validate 3 base models
- Use MergeTechniques from Week 1

**Function Signature**:
```python
def initialize_population(
    base_models: List[nn.Module]
) -> List[nn.Module]:
    """
    Create initial population of 8 models using all binary combinations.

    Args:
        base_models: List of 3 Phase 1 models

    Returns:
        List of 8 merged models (one per binary combo)

    Raises:
        ValueError: If not exactly 3 base models
    """
```

**Implementation**:
```python
from src.phase2_evomerge.merge import MergeTechniques

merger = MergeTechniques()
population = []

for combo_id in range(8):  # 000 to 111
    model = merger.apply_combo(base_models, combo_id)
    population.append(model)

return population
```

**NASA POT10**: ~25 LOC

---

### Task 2: Mutation Function (1 hour)

**File**: `src/phase2_evomerge/evolution/mutation.py`

**Requirements**:
- Gaussian noise injection
- Configurable sigma and rate
- Sparse mutation (1% of weights)
- Return new model (don't modify original)

**Function Signature**:
```python
def mutate_model(
    model: nn.Module,
    sigma: float = 0.01,
    rate: float = 0.01,
    device: str = 'cuda'
) -> nn.Module:
    """
    Apply Gaussian noise mutation to model parameters.

    Args:
        model: Model to mutate
        sigma: Standard deviation of Gaussian noise (default: 0.01)
        rate: Fraction of weights to mutate (default: 0.01 = 1%)
        device: Device for computation

    Returns:
        New mutated model (original unchanged)

    Example:
        >>> mutated = mutate_model(elite1, sigma=0.01, rate=0.01)
        >>> # elite1 unchanged, mutated is new model with noise
    """
```

**Implementation Steps**:
1. Deep copy model
2. For each parameter:
   - Create random mask (Bernoulli with p=rate)
   - Generate Gaussian noise (mean=0, std=sigma)
   - Add noise where mask==1
3. Return mutated model

**NASA POT10**: ~35 LOC

---

### Task 3: Diversity Tracking (1 hour)

**File**: `src/phase2_evomerge/evolution/diversity.py`

**Requirements**:
- Compute pairwise L2 distance between models
- Normalize to [0, 1] range
- Return average diversity score

**Function Signature**:
```python
def compute_diversity(
    population: List[nn.Module],
    expected_distance: Optional[float] = None
) -> float:
    """
    Compute population diversity via average pairwise L2 distance.

    Args:
        population: List of models
        expected_distance: Normalization factor (None = auto-compute)

    Returns:
        Diversity score (0.0 = identical, 1.0 = maximally diverse)

    Example:
        >>> diversity = compute_diversity(population)
        >>> if diversity < 0.2:
        ...     print("Warning: Low diversity, re-seeding needed")
    """
```

**Implementation**:
```python
distances = []
for i in range(len(population)):
    for j in range(i + 1, len(population)):
        # Flatten parameters
        params_i = flatten_model_params(population[i])
        params_j = flatten_model_params(population[j])

        # L2 distance
        dist = torch.norm(params_i - params_j).item()
        distances.append(dist)

avg_distance = np.mean(distances)

# Normalize
if expected_distance is None:
    expected_distance = avg_distance  # First generation baseline

normalized = avg_distance / expected_distance
return normalized
```

**NASA POT10**: ~45 LOC (including flatten helper)

---

### Task 4: Evolution Configuration (30 min)

**File**: `src/phase2_evomerge/evolution/config.py`

**Requirements**:
- Configuration dataclass
- Default values from V1
- Validation

**Class Design**:
```python
@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization."""

    # Evolution parameters
    generations: int = 50
    population_size: int = 8
    elite_count: int = 2

    # Mutation
    mutation_sigma: float = 0.01  # Noise std
    mutation_rate: float = 0.01   # Fraction of weights (1%)

    # Diversity
    min_diversity: float = 0.3
    diversity_reseed_threshold: float = 0.2

    # Convergence
    early_stopping: bool = True
    convergence_threshold: float = 0.001  # 0.1% improvement
    convergence_patience: int = 5  # Generations

    # Device
    device: str = 'cuda'

    def validate(self) -> None:
        """Validate configuration values."""
        assert self.population_size == 8, "Population must be 8"
        assert self.elite_count == 2, "Elite count must be 2"
        assert 0 < self.mutation_rate < 1
        assert self.mutation_sigma > 0
```

**NASA POT10**: ~30 LOC

---

### Task 5: Evolution Loop (2.5 hours)

**File**: `src/phase2_evomerge/evolution/evolution_loop.py`

**Requirements**:
- Main evolution loop (50 generations)
- Elite preservation + mutation
- Loser merging
- Diversity tracking + re-seeding
- Early stopping
- Champion tracking

**Class Design**:
```python
class EvolutionLoop:
    """Main evolutionary optimization loop."""

    def __init__(
        self,
        config: EvolutionConfig,
        fitness_evaluator: FitnessEvaluator
    ):
        """Initialize evolution loop."""

    def evolve(
        self,
        base_models: List[nn.Module]
    ) -> Dict[str, Any]:
        """
        Run complete evolution loop.

        Returns:
            Dictionary with champion model and metrics
        """

    def _elite_preservation(
        self,
        population: List[nn.Module],
        fitness_scores: List[float]
    ) -> List[nn.Module]:
        """Create 6 children from top 2 elites via mutation."""

    def _loser_merging(
        self,
        population: List[nn.Module],
        fitness_scores: List[float]
    ) -> List[nn.Module]:
        """Create 2 children from bottom 6 via combo merging."""

    def _check_convergence(
        self,
        fitness_history: List[float]
    ) -> bool:
        """Check if evolution has converged."""

    def _reseed_if_needed(
        self,
        population: List[nn.Module],
        diversity: float,
        base_models: List[nn.Module]
    ) -> List[nn.Module]:
        """Re-seed bottom 2 if diversity too low."""
```

**Main Evolution Loop**:
```python
def evolve(self, base_models):
    # Initialize
    population = initialize_population(base_models)
    champion = None
    champion_fitness = -float('inf')
    fitness_history = []

    for generation in range(1, self.config.generations + 1):
        # 1. Evaluate fitness
        fitness_scores = self.evaluator.evaluate_batch(population)

        # 2. Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # 3. Update champion
        if fitness_scores[0] > champion_fitness:
            champion = population[0]
            champion_fitness = fitness_scores[0]

        fitness_history.append(champion_fitness)

        # 4. Check convergence
        if self.config.early_stopping:
            if self._check_convergence(fitness_history):
                break

        # 5. Elite preservation (top 2 → 6 children)
        elite_children = self._elite_preservation(population, fitness_scores)

        # 6. Loser merging (bottom 6 → 2 children)
        loser_children = self._loser_merging(population, fitness_scores)

        # 7. New population
        population = elite_children + loser_children

        # 8. Diversity management
        diversity = compute_diversity(population)
        population = self._reseed_if_needed(population, diversity, base_models)

    return {
        'champion': champion,
        'fitness': champion_fitness,
        'improvement': ...,
        'generations': generation
    }
```

**NASA POT10**: Split into methods ≤60 LOC each
- evolve(): ~55 LOC
- _elite_preservation(): ~30 LOC
- _loser_merging(): ~40 LOC
- _check_convergence(): ~25 LOC
- _reseed_if_needed(): ~30 LOC

---

### Task 6: Evolution API (__init__.py) (30 min)

**File**: `src/phase2_evomerge/evolution/__init__.py`

**Requirements**:
- Export main classes
- Simple API

**Content**:
```python
"""
Evolution module for Phase 2 (EvoMerge).

Implements evolutionary optimization with:
- Population initialization from binary combos
- Elite preservation via Gaussian mutation
- Loser merging via combo application
- Diversity tracking and re-seeding
- Early stopping on convergence
"""

from .config import EvolutionConfig
from .population import initialize_population
from .mutation import mutate_model
from .diversity import compute_diversity
from .evolution_loop import EvolutionLoop

__all__ = [
    'EvolutionConfig',
    'EvolutionLoop',
    'initialize_population',
    'mutate_model',
    'compute_diversity'
]
```

**NASA POT10**: ~20 LOC

---

### Task 7: Comprehensive Tests (2 hours)

**File**: `tests/unit/test_evolution.py`

**Test Classes**:
```python
@pytest.mark.phase2
@pytest.mark.evolution
class TestPopulationInit:
    def test_init_creates_8_models(self, base_models)
    def test_init_validates_3_base_models(self)
    def test_init_all_combos_unique(self, base_models)

@pytest.mark.phase2
@pytest.mark.evolution
class TestMutation:
    def test_mutation_returns_new_model(self, simple_model)
    def test_mutation_applies_noise(self, simple_model)
    def test_mutation_rate_controls_sparsity(self, simple_model)
    def test_mutation_sigma_controls_magnitude(self, simple_model)

@pytest.mark.phase2
@pytest.mark.evolution
class TestDiversity:
    def test_diversity_identical_models_zero(self, simple_model)
    def test_diversity_different_models_positive(self)
    def test_diversity_normalization(self)

@pytest.mark.phase2
@pytest.mark.evolution
class TestEvolutionLoop:
    def test_evolution_runs_to_completion(self, base_models, evaluator)
    def test_elite_preservation_creates_6_children(self)
    def test_loser_merging_creates_2_children(self)
    def test_early_stopping_works(self)
    def test_diversity_reseeding(self)
    def test_champion_tracking(self)
```

**Test Count**: 15+ tests
**Target Coverage**: ≥90%

---

## Success Criteria

### Functional
- ✅ Population initialization creates 8 models
- ✅ Elite preservation works (2 → 6)
- ✅ Loser merging works (6 → 2)
- ✅ Diversity tracking functional
- ✅ Evolution loop completes 50 generations
- ✅ Early stopping triggers correctly
- ✅ Champion tracking works

### Quality
- ✅ NASA POT10 compliant
- ✅ Test coverage ≥90%
- ✅ Type hints 100%
- ✅ Docstrings 100%

### Performance
- ✅ Single generation: <30 seconds (with caching)
- ✅ 50 generations: <25 minutes (fast convergence ~38 gens)
- ✅ Diversity maintained >0.3

---

## Estimated LOC

**Production Code**:
- population.py: ~30 LOC
- mutation.py: ~40 LOC
- diversity.py: ~50 LOC
- config.py: ~35 LOC
- evolution_loop.py: ~180 LOC (5 methods)
- __init__.py: ~20 LOC
- **Total**: ~355 LOC

**Test Code**:
- test_evolution.py: ~450 LOC (15+ tests)

**Total Day 7 Output**: ~805 LOC + documentation

---

## Integration Points

### With Week 1 (Merge Techniques)
- Uses `MergeTechniques.apply_combo()` for population init and loser merging

### With Day 6 (Fitness Evaluation)
- Uses `FitnessEvaluator.evaluate_batch()` for fitness scoring

### With Future (Day 8-10)
- Evolution loop will be wrapped in higher-level API
- W&B logging will be added
- Checkpointing will be added

---

## Risks & Mitigation

### Risk 1: Evolution Doesn't Converge
**Mitigation**: Tune mutation parameters, increase patience

### Risk 2: Diversity Collapse
**Mitigation**: Re-seeding implemented, can adjust threshold

### Risk 3: Slow Evolution (>2 hours)
**Mitigation**: Fitness caching (Day 6), early stopping

---

## Next Steps (After Day 7)

### Day 8: W&B Integration & Logging
- Log per-generation metrics
- Track combo usage stats
- Visualize fitness curves

### Day 9: Checkpointing & Resume
- Save every 10 generations
- Resume from checkpoint
- Model persistence

### Day 10: Integration Testing & Documentation
- End-to-end Phase 2 test
- Performance validation
- Week 2 completion summary

---

**Document Version**: 1.0
**Created**: 2025-10-17
**Status**: Ready for Implementation 🚀

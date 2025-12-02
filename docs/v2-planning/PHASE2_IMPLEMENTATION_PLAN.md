# Phase 2 (EvoMerge) Implementation Plan

**Version**: 1.0
**Date**: 2025-10-16
**Status**: Approved - Ready for Implementation
**Estimated Duration**: 4 weeks (20 days)
**Target Completion**: Week 17 of overall project timeline

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Context](#project-context)
3. [Implementation Phases](#implementation-phases)
4. [Cross-Phase Integration](#cross-phase-integration)
5. [Testing Strategy](#testing-strategy)
6. [UI Implementation](#ui-implementation)
7. [W&B Integration](#wb-integration)
8. [File Structure](#file-structure)
9. [Success Criteria](#success-criteria)
10. [Risk Mitigation](#risk-mitigation)
11. [Timeline & Milestones](#timeline--milestones)
12. [Dependencies](#dependencies)

---

## Executive Summary

### What is Phase 2?

Phase 2 (EvoMerge) implements **evolutionary model optimization** that takes 3 specialized 25M-parameter models from Phase 1 (Cognate) and evolves them over **50 generations** using **6 merge techniques** across **8 binary combinations** to produce a single, highly-optimized merged model.

### Key Objectives

- **Fitness Improvement**: Achieve ≥20% improvement (target: **23.5%**)
- **Evolution Time**: Complete in ≤90 minutes on GTX 1660
- **Memory Efficiency**: Maintain <6GB VRAM usage (12 models max in memory)
- **Population Diversity**: Keep diversity >0.3 throughout evolution
- **Quality Assurance**: ≥90% test coverage, NASA POT10 compliance

### Current Status

**Infrastructure**: ✅ **75% Complete** (Weeks 1-12 done)
- Cross-phase systems (model registry, orchestrator, config manager)
- UI framework (Streamlit with 5 pages)
- CI/CD pipeline (6 jobs, quality gates)
- Testing framework (pytest, 47 tests passing)
- W&B integration (676 metrics configured)

**Phase 2 Implementation**: ❌ **0% Complete** (needs full build)
- 26 new files to create
- 91+ tests to write
- 370 W&B metrics to implement
- 7 API endpoints + 4 WebSocket events

---

## Project Context

### The 8-Phase Pipeline

```
Phase 1 (Cognate) → Phase 2 (EvoMerge) → Phase 3 (Quiet-STaR) → ... → Phase 8 (Compression)
     ↓                    ↓                      ↓
  3× 25M models    1 evolved model      Reasoning-enhanced
  (reasoning,      (+23.5% fitness)          model
   memory,
   speed)
```

### Phase 2's Role

**Input**: 3 specialized TRM × Titans-MAG models (~25M params each)
**Process**: Evolutionary optimization via model merging
**Output**: 1 champion model with significant fitness improvement
**Next Phase**: Champion model goes to Phase 3 for reasoning enhancement

### Why Model Merging?

**Advantages**:
- No gradient-based training (pure weight manipulation)
- Combines strengths of 3 specialized models
- Faster than training from scratch (90 min vs days)
- Preserves architectural compatibility (same size output)

**Research Basis**:
- Model Soups (Wortsman et al.)
- TIES-Merging (Yadav et al., NeurIPS 2023)
- DARE (Yu et al., arXiv 2024)
- FrankenMerge (used in Goliath-120B, SOLAR-10.7B)

---

## Implementation Phases

### Week 1: Core Merge Techniques

**Duration**: 5 days
**Files Created**: 7
**Tests Written**: 18+
**Lines of Code**: ~700 production + ~400 tests

#### Day 1-3: Implement 6 Merge Techniques

**File**: `src/phase2_evomerge/merge/__init__.py`
```python
"""
Unified merge technique interface for Phase 2 EvoMerge.

Implements 6 research-validated merging methods organized into 3 mutually
exclusive pairs for binary pairing strategy (2³ = 8 combinations).
"""

from .linear_merge import LinearMerge
from .slerp_merge import SLERPMerge
from .dare_merge import DAREMerge
from .ties_merge import TIESMerge
from .frankenmerge import FrankenMerge
from .dfs_merge import DFSMerge

class MergeTechniques:
    """
    Main API for applying merge techniques.

    Binary Pairing Strategy:
        Bit 0: Interpolation (0=Linear, 1=SLERP)
        Bit 1: Task Arithmetic (0=DARE, 1=TIES)
        Bit 2: Selection (0=FrankenMerge, 1=DFS)

    Example:
        >>> merger = MergeTechniques()
        >>> models = [model1, model2, model3]
        >>> # Binary 111 (SLERP + TIES + DFS)
        >>> result = merger.apply_combo(models, combo_id=7)
    """

    def apply_combo(self, models: List[Model], combo_id: int) -> Model:
        """Apply 3-stage sequential merge pipeline."""
        # Stage 1: Interpolation
        # Stage 2: Task Arithmetic
        # Stage 3: Selection
        pass
```

**Technique 1: Linear Merge** (`linear_merge.py`, ~80 LOC)
- Simple weighted average: `merged = 0.33*m1 + 0.33*m2 + 0.33*m3`
- Baseline technique, always works
- Fast, predictable results

**Technique 2: SLERP Merge** (`slerp_merge.py`, ~120 LOC)
- Spherical linear interpolation preserving magnitude
- θ = arccos(dot(w1, w2))
- **Edge case**: θ=0 (identical models) → fallback to linear
- Better preserves parameter geometry

**Technique 3: DARE Merge** (`dare_merge.py`, ~100 LOC)
- Drop And REscale: Drop 90% of delta randomly, rescale by 10×
- `delta = model - base`
- `sparse_delta = delta * bernoulli(p=0.1)`
- `result = base + sparse_delta * 10`
- Reduces task interference

**Technique 4: TIES Merge** (`ties_merge.py`, ~130 LOC)
- TrIm, Elect Sign, Merge
- Keep top 20% magnitude parameters
- Vote on sign (+/-) for each parameter
- Merge only params with matching elected sign
- Resolves sign conflicts

**Technique 5: FrankenMerge** (`frankenmerge.py`, ~110 LOC)
- Layer-wise selection from best-performing model
- Patterns: "ABC" (alternate), "ABBA" (symmetric), random
- Mix-and-match approach

**Technique 6: DFS Merge** (`dfs_merge.py`, ~100 LOC)
- Deep Feature Selection via inverse-variance weighting
- `importance = 1 / variance(param_across_models)`
- Stable features get higher weight

**Sequential Pipeline**:
```python
def apply_combo(models, combo_id):
    bit0 = (combo_id >> 0) & 1
    bit1 = (combo_id >> 1) & 1
    bit2 = (combo_id >> 2) & 1

    # Stage 1: Interpolation
    stage1 = linear_merge(models) if bit0==0 else slerp_merge(models)

    # Stage 2: Task Arithmetic
    stage2 = dare_merge(stage1, models[0]) if bit1==0 else ties_merge(stage1, models)

    # Stage 3: Selection
    stage3 = frankenmerge(stage2, models) if bit2==0 else dfs_merge(stage2, models)

    return stage3
```

#### Day 4-5: Unit Tests for Merging

**File**: `tests/unit/test_merge_techniques.py` (~400 LOC, 18+ tests)

**Test Structure**:
```python
class TestLinearMerge:
    def test_identical_models(self):
        """Linear merge of identical models returns identical model."""

    def test_random_models(self):
        """Linear merge produces weighted average."""

    def test_opposite_models(self):
        """Linear merge of opposite weights produces zeros."""

class TestSLERPMerge:
    def test_identical_models_fallback(self):
        """θ=0 edge case falls back to linear merge."""

    def test_orthogonal_models(self):
        """θ=90° produces expected interpolation."""

    def test_magnitude_preservation(self):
        """SLERP preserves parameter magnitude better than linear."""

class TestDAREMerge:
    def test_stochasticity(self):
        """Different random masks produce different results."""

    def test_sparsity(self):
        """90% of delta is indeed dropped."""

    def test_rescaling(self):
        """Remaining 10% is rescaled by 10×."""

class TestTIESMerge:
    def test_sign_voting(self):
        """3 models voting: majority sign wins."""

    def test_trimming(self):
        """Only top 20% magnitude params kept."""

    def test_conflict_resolution(self):
        """Conflicting signs are resolved correctly."""

class TestFrankenMerge:
    def test_layer_selection(self):
        """Best layer selected per position."""

    def test_dimension_compatibility(self):
        """Layer dimensions match after merge."""

    def test_abc_pattern(self):
        """ABC alternation pattern works."""

class TestDFSMerge:
    def test_variance_weighting(self):
        """Inverse variance computed correctly."""

    def test_stable_features_prioritized(self):
        """Low-variance features get higher weight."""

class TestBinaryCombinations:
    def test_all_8_combos_unique(self):
        """All 8 binary combos produce unique models."""

    def test_sequential_pipeline(self):
        """3-stage pipeline applied in correct order."""

    def test_combo_000_vs_111(self):
        """Conservative vs aggressive combos differ significantly."""
```

**Coverage Target**: ≥98%

**Deliverable**: All 6 merge techniques working with comprehensive tests

---

### Week 2: Fitness Evaluation & Population Management

**Duration**: 5 days
**Files Created**: 11
**Tests Written**: 37+
**Lines of Code**: ~900 production + ~600 tests

#### Day 6-8: Fitness Evaluation System

**File**: `src/phase2_evomerge/fitness/evaluator.py` (~150 LOC)

```python
class FitnessEvaluator:
    """
    Composite fitness evaluation for evolved models.

    Fitness = 0.4 * (1/perplexity) + 0.3 * accuracy + 0.2 * (1/time) + 0.1 * (1/memory)

    Components:
        - Perplexity (40%): Language modeling quality
        - Accuracy (30%): Task performance
        - Speed (20%): Inference time
        - Memory (10%): VRAM usage

    Example:
        >>> evaluator = FitnessEvaluator(config)
        >>> score = evaluator.evaluate(model)
        >>> print(score.composite)  # 0.185 (target: >0.150 + 20%)
    """

    def __init__(self, config: FitnessConfig):
        self.perplexity_calc = PerplexityCalculator(config.validation_set)
        self.accuracy_calc = AccuracyCalculator(config.test_set)
        self.speed_bench = SpeedBenchmark(config.batch_size)
        self.memory_prof = MemoryProfiler()
        self.weights = config.fitness_weights  # {perplexity: 0.4, ...}

    def evaluate(self, model: nn.Module) -> CompositeScore:
        """Evaluate model on all 4 components."""
        perplexity = self.perplexity_calc.compute(model)
        accuracy = self.accuracy_calc.compute(model)
        speed = self.speed_bench.compute(model)
        memory = self.memory_prof.compute(model)

        composite = (
            self.weights['perplexity'] * (1 / perplexity) +
            self.weights['accuracy'] * accuracy +
            self.weights['speed'] * (1 / speed) +
            self.weights['memory'] * (1 / memory)
        )

        return CompositeScore(
            perplexity=perplexity,
            accuracy=accuracy,
            speed=speed,
            memory=memory,
            composite=composite
        )
```

**Component Files**:

1. **Perplexity** (`perplexity.py`, ~100 LOC)
   - Cross-entropy loss on validation set
   - Convert to perplexity: `exp(loss)`
   - Normalize to [0, 1] range

2. **Accuracy** (`accuracy.py`, ~90 LOC)
   - Task-specific metric (classification, QA, etc.)
   - Direct percentage (0-1 range)

3. **Speed** (`speed.py`, ~110 LOC)
   - Tokens per second: `(batch_size × seq_len × 100) / total_time`
   - Normalized to expected throughput

4. **Memory** (`memory.py`, ~80 LOC)
   - Peak VRAM during inference
   - Inverse-normalized (lower is better)

**Caching**:
```python
class FitnessCache:
    """Cache fitness results to avoid re-evaluation."""
    def __init__(self):
        self.cache = {}  # model_hash → CompositeScore

    def get(self, model):
        model_hash = hash_model_weights(model)
        return self.cache.get(model_hash)

    def set(self, model, score):
        model_hash = hash_model_weights(model)
        self.cache[model_hash] = score
```

#### Day 9-10: Population Management

**File**: `src/phase2_evomerge/population/manager.py` (~180 LOC)

```python
class PopulationManager:
    """
    Manages population of 8 models across 50 generations.

    Binary Pairing Strategy:
        - Generation 0: All 8 binary combos (000-111)
        - Generation 1-50: Elite mutation + loser merging

    Memory Management:
        - Only 12 models in memory max:
          * 3 original Phase 1 models (archival)
          * 1 current champion (best seen so far)
          * 8 current generation population
    """

    def initialize_generation_0(self, phase1_models):
        """Create all 8 binary combinations."""
        population = []
        for combo_id in range(8):
            model = self.merger.apply_combo(phase1_models, combo_id)
            population.append(model)
        return population

    def evolve_generation(self, current_pop, fitness_scores):
        """
        Create next generation from current population.

        Elite Preservation:
            - Top 2 models → mutate 3× each → 6 children

        Loser Merging:
            - Bottom 6 models → 2 groups of 3 → merge with random combos → 2 children

        Returns: 8 new models
        """
        # Sort by fitness
        sorted_pop = sort_by_fitness(current_pop, fitness_scores)

        # Elite mutation
        elite1, elite2 = sorted_pop[:2]
        elite_children = []
        for elite in [elite1, elite2]:
            for _ in range(3):
                child = self.genetic_ops.mutate(elite, sigma=0.01, rate=0.01)
                elite_children.append(child)

        # Loser merging
        losers = sorted_pop[-6:]
        group1, group2 = losers[:3], losers[3:]
        combo1 = random.randint(0, 7)
        combo2 = random.randint(0, 7)
        loser_child1 = self.merger.apply_combo(group1, combo1)
        loser_child2 = self.merger.apply_combo(group2, combo2)

        return elite_children + [loser_child1, loser_child2]
```

**File**: `src/phase2_evomerge/population/genetic_ops.py` (~120 LOC)

```python
class GeneticOperations:
    """Mutation and selection operators."""

    def mutate(self, model, sigma=0.01, rate=0.01):
        """
        Apply Gaussian noise to weights.

        Args:
            sigma: Standard deviation (~1% of weight magnitude)
            rate: Fraction of weights to mutate (1%)
        """
        mutated = copy.deepcopy(model)
        for param in mutated.parameters():
            if random.random() < rate:
                noise = torch.randn_like(param) * sigma
                param.data += noise
        return mutated
```

**File**: `src/phase2_evomerge/population/diversity.py` (~100 LOC)

```python
class DiversityTracker:
    """
    Monitor population diversity via pairwise L2 distances.

    Thresholds:
        - > 0.3: Healthy ✅
        - 0.2-0.3: Warning (increase mutation)
        - < 0.2: Critical (re-seed bottom 2)
    """

    def compute_diversity(self, population):
        """Average pairwise L2 distance."""
        distances = []
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                flat_i = flatten_parameters(population[i])
                flat_j = flatten_parameters(population[j])
                dist = torch.norm(flat_i - flat_j, p=2)
                distances.append(dist)
        return torch.mean(torch.stack(distances)).item()

    def check_and_reseed(self, population, diversity):
        """Re-seed bottom 2 models if diversity < 0.2."""
        if diversity < 0.2:
            # Replace bottom 2 with random merges
            population[-2] = random_merge()
            population[-1] = random_merge()
        return population
```

#### Tests for Week 2

**File**: `tests/unit/test_fitness_evaluation.py` (~300 LOC, 12 tests)
- Perplexity calculation correctness
- Accuracy scoring correctness
- Speed benchmarking correctness
- Memory profiling correctness
- Composite fitness formula validation
- Edge cases: NaN, Inf handling

**File**: `tests/unit/test_population_management.py` (~250 LOC, 15 tests)
- Generation 0 initialization (8 unique models)
- Elite preservation (top 2 kept)
- Elite mutation (6 children from 2 elites)
- Loser merging (6 worst → 2 children)
- Population size consistency (always 8)
- Memory limit enforcement (≤12 models)

**File**: `tests/unit/test_diversity.py` (~200 LOC, 10 tests)
- Pairwise distance computation
- Diversity threshold detection
- Re-seeding triggers correctly
- Diversity increases after re-seeding

**Coverage Target**: ≥95%

**Deliverable**: Complete fitness + population systems with 37+ tests passing

---

### Week 3: Evolution Engine & Integration

**Duration**: 5 days
**Files Created**: 6
**Tests Written**: 14+
**Lines of Code**: ~700 production + ~500 tests

#### Day 11-13: Evolution Engine

**File**: `src/phase2_evomerge/evolution/engine.py` (~250 LOC)

```python
class EvolutionEngine:
    """
    Main 50-generation evolution loop.

    Process:
        1. Initialize Generation 0 (8 binary combos)
        2. For each generation 1-50:
            a. Evaluate fitness (all 8 models)
            b. Update champion (track best ever)
            c. Check convergence (early stop?)
            d. Evolve next generation (elite + loser)
            e. Check diversity (re-seed if needed)
            f. Log to W&B
        3. Return champion model

    Typical runtime: 90 minutes (35-40 generations convergence)
    """

    def __init__(self, config: EvoMergeConfig):
        self.config = config
        self.population_mgr = PopulationManager(config)
        self.fitness_eval = FitnessEvaluator(config)
        self.diversity_tracker = DiversityTracker()
        self.convergence_detector = ConvergenceDetector(config)
        self.checkpoint_mgr = CheckpointManager(config)
        self.wandb_logger = WandBLogger(config)

    def run(self, phase1_models):
        """Run full evolution for 50 generations."""
        # Generation 0
        population = self.population_mgr.initialize_generation_0(phase1_models)
        fitness_scores = [self.fitness_eval.evaluate(m) for m in population]
        champion = max(zip(population, fitness_scores), key=lambda x: x[1].composite)[0]

        for gen in range(1, self.config.generations + 1):
            # Evaluate fitness
            fitness_scores = [self.fitness_eval.evaluate(m) for m in population]

            # Update champion
            best_this_gen = max(zip(population, fitness_scores), key=lambda x: x[1].composite)
            if best_this_gen[1].composite > self.fitness_eval.evaluate(champion).composite:
                champion = best_this_gen[0]

            # Check convergence
            if self.convergence_detector.should_stop(fitness_scores):
                break

            # Evolve next generation
            population = self.population_mgr.evolve_generation(population, fitness_scores)

            # Check diversity
            diversity = self.diversity_tracker.compute_diversity(population)
            if diversity < self.config.min_diversity:
                population = self.diversity_tracker.check_and_reseed(population, diversity)

            # Log to W&B
            self.wandb_logger.log_generation(gen, population, fitness_scores, diversity, champion)

            # Checkpoint
            if gen % self.config.checkpoint_every == 0:
                self.checkpoint_mgr.save(gen, population, champion)

        return champion, gen
```

**File**: `src/phase2_evomerge/evolution/convergence.py` (~80 LOC)

```python
class ConvergenceDetector:
    """
    Detect early convergence to stop evolution.

    Criteria:
        - Improvement < 0.1% for 5 consecutive generations
    """

    def __init__(self, threshold=0.001, patience=5):
        self.threshold = threshold
        self.patience = patience
        self.history = []

    def should_stop(self, fitness_scores):
        best_fitness = max(score.composite for score in fitness_scores)
        self.history.append(best_fitness)

        if len(self.history) < self.patience + 1:
            return False

        recent = self.history[-self.patience-1:]
        improvements = [(recent[i+1] - recent[i]) / recent[i] for i in range(len(recent)-1)]

        return all(imp < self.threshold for imp in improvements)
```

**File**: `src/phase2_evomerge/evolution/checkpoint.py` (~120 LOC)

```python
class CheckpointManager:
    """
    Save and resume evolution state.

    Checkpoint includes:
        - Current generation number
        - Population (8 models)
        - Champion model
        - Fitness scores
        - Evolution history
    """

    def save(self, generation, population, champion):
        checkpoint = {
            'generation': generation,
            'population': [model.state_dict() for model in population],
            'champion': champion.state_dict(),
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        path = self.config.checkpoint_dir / f"gen_{generation}.pt"
        torch.save(checkpoint, path)

    def resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generation = checkpoint['generation']
        # ... restore population and champion
        return generation, population, champion
```

#### Day 14-15: Phase Handoff Integration

**File**: `src/phase2_evomerge/phase2_pipeline.py` (~200 LOC)

```python
class Phase2Pipeline:
    """
    Main orchestrator for Phase 2 (EvoMerge).

    Responsibilities:
        - Load 3 Phase 1 models
        - Validate inputs
        - Run evolution
        - Validate outputs
        - Save champion model
        - Log to W&B
    """

    def __init__(self, config: EvoMergeConfig):
        self.config = config
        self.evolution_engine = EvolutionEngine(config)
        self.model_storage = ModelStorage()
        self.wandb_logger = WandBLogger(config)

    def run(self, session_id: str):
        """
        Run Phase 2 pipeline.

        Args:
            session_id: Session ID from Phase 1

        Returns:
            champion_model, metrics_dict
        """
        # Load Phase 1 models
        phase1_models = self.load_phase1_models(session_id)

        # Validate inputs
        self.validate_phase1_models(phase1_models)

        # Run evolution
        champion, generations = self.evolution_engine.run(phase1_models)

        # Validate outputs
        improvement = self.validate_champion(champion, phase1_models)

        # Save champion
        self.save_champion(champion, session_id)

        # Log final results
        self.wandb_logger.log_final(champion, improvement, generations)

        return champion, {
            'improvement': improvement,
            'generations': generations,
            'success': improvement >= 0.20
        }

    def load_phase1_models(self, session_id):
        """Load 3 models from Phase 1."""
        models = []
        for model_name in ['reasoning', 'memory_integration', 'adaptive_computation']:
            model = self.model_storage.load_model(session_id, 'cognate', model_name)
            models.append(model)
        return models

    def validate_phase1_models(self, models):
        """Ensure exactly 3 TRM × Titans-MAG models, ~25M params each."""
        assert len(models) == 3, f"Expected 3 models, got {len(models)}"
        for model in models:
            param_count = sum(p.numel() for p in model.parameters())
            assert 23_000_000 <= param_count <= 27_000_000, \
                f"Model size {param_count} out of range [23M, 27M]"
            assert hasattr(model, 'trm_wrapper'), "Missing TRM wrapper"
            assert hasattr(model, 'titans_mag'), "Missing Titans-MAG backbone"

    def validate_champion(self, champion, phase1_models):
        """Ensure champion has ≥20% fitness improvement."""
        champion_fitness = self.evolution_engine.fitness_eval.evaluate(champion).composite
        phase1_fitness = [self.evolution_engine.fitness_eval.evaluate(m).composite for m in phase1_models]
        baseline = max(phase1_fitness)
        improvement = (champion_fitness - baseline) / baseline

        assert improvement >= 0.20, f"Improvement {improvement:.1%} < 20%"
        return improvement

    def save_champion(self, champion, session_id):
        """Save champion to storage."""
        self.model_storage.save_model(
            champion,
            session_id,
            phase='evomerge',
            model_name='champion',
            metadata={
                'improvement': improvement,
                'generations': generations,
                'phase': 'evomerge'
            }
        )
```

#### Integration Tests for Week 3

**File**: `tests/integration/test_phase2_evolution.py` (~200 LOC, 4 tests)

```python
@pytest.mark.integration
@pytest.mark.phase2
class TestPhase2Evolution:
    def test_full_evolution_loop(self, mock_phase1_models):
        """Test 50-generation evolution completes successfully."""
        engine = EvolutionEngine(config)
        champion, gens = engine.run(mock_phase1_models)

        assert gens <= 50
        assert champion is not None
        assert isinstance(champion, nn.Module)

    def test_early_stopping(self, mock_phase1_models):
        """Test convergence triggers early stopping."""
        config = EvoMergeConfig(convergence_patience=5, convergence_threshold=0.001)
        engine = EvolutionEngine(config)
        champion, gens = engine.run(mock_phase1_models)

        assert gens < 50  # Should converge early

    def test_diversity_management(self, mock_phase1_models):
        """Test diversity maintained throughout evolution."""
        engine = EvolutionEngine(config)
        diversities = []

        # Monitor diversity each generation
        for gen in range(10):
            pop = engine.population_mgr.current_population
            diversity = engine.diversity_tracker.compute_diversity(pop)
            diversities.append(diversity)

        assert all(d > 0.25 for d in diversities)  # Never collapse

    def test_checkpoint_resume(self, mock_phase1_models):
        """Test checkpoint save/resume works."""
        engine = EvolutionEngine(config)

        # Run 10 generations
        champion1, _ = engine.run(mock_phase1_models, max_gens=10)

        # Save checkpoint
        engine.checkpoint_mgr.save(10, engine.population, champion1)

        # Resume from checkpoint
        gen, pop, champion2 = engine.checkpoint_mgr.resume('gen_10.pt')

        assert gen == 10
        assert torch.allclose(champion1.state_dict(), champion2.state_dict())
```

**File**: `tests/integration/test_phase1_to_phase2_handoff.py` (~150 LOC, 3 tests)

```python
@pytest.mark.integration
@pytest.mark.handoff
class TestPhase1ToPhase2Handoff:
    def test_load_phase1_models(self, session_id):
        """Test loading 3 Phase 1 models."""
        pipeline = Phase2Pipeline(config)
        models = pipeline.load_phase1_models(session_id)

        assert len(models) == 3
        assert all(isinstance(m, nn.Module) for m in models)

    def test_validate_phase1_models_correct(self, valid_phase1_models):
        """Test validation passes for correct models."""
        pipeline = Phase2Pipeline(config)
        pipeline.validate_phase1_models(valid_phase1_models)  # Should not raise

    def test_validate_phase1_models_incorrect(self, invalid_phase1_models):
        """Test validation fails for incorrect models."""
        pipeline = Phase2Pipeline(config)
        with pytest.raises(AssertionError):
            pipeline.validate_phase1_models(invalid_phase1_models)
```

**File**: `tests/integration/test_phase2_output.py` (~120 LOC, 3 tests)

```python
@pytest.mark.integration
@pytest.mark.handoff
class TestPhase2Output:
    def test_champion_saved_correctly(self, session_id, champion):
        """Test champion model saved to storage."""
        pipeline = Phase2Pipeline(config)
        pipeline.save_champion(champion, session_id)

        # Verify file exists
        path = f"storage/{session_id}/evomerge_champion_*/model.pt"
        assert len(glob.glob(path)) == 1

    def test_improvement_validation(self, champion, phase1_models):
        """Test ≥20% improvement validation."""
        pipeline = Phase2Pipeline(config)
        improvement = pipeline.validate_champion(champion, phase1_models)

        assert improvement >= 0.20

    def test_output_format(self, session_id):
        """Test output metadata format correct."""
        pipeline = Phase2Pipeline(config)
        champion, metrics = pipeline.run(session_id)

        assert 'improvement' in metrics
        assert 'generations' in metrics
        assert 'success' in metrics
        assert metrics['success'] is True
```

**File**: `tests/integration/test_phase2_wandb.py` (~150 LOC, 4 tests)

```python
@pytest.mark.integration
@pytest.mark.wandb
class TestPhase2WandBIntegration:
    def test_370_metrics_logged(self, mock_wandb):
        """Test all 370 metrics logged correctly."""
        logger = WandBLogger(config)

        # Run evolution
        engine = EvolutionEngine(config)
        champion, gens = engine.run(mock_phase1_models)

        # Check W&B calls
        assert mock_wandb.log.call_count >= 50  # At least 50 generations

        # Check metric names
        logged_metrics = [call[0][0] for call in mock_wandb.log.call_args_list]
        assert 'evolution/best_fitness' in logged_metrics
        assert 'evolution/diversity' in logged_metrics

    def test_artifacts_saved(self, mock_wandb):
        """Test champion model artifact saved."""
        logger = WandBLogger(config)
        logger.log_final(champion, improvement=0.235, generations=38)

        assert mock_wandb.log_artifact.called

    def test_config_logged(self, mock_wandb):
        """Test config logged to W&B."""
        wandb.init(config=config.to_dict())

        assert mock_wandb.config == config.to_dict()

    def test_metrics_per_generation(self, mock_wandb):
        """Test metrics logged every generation."""
        logger = WandBLogger(config)

        for gen in range(10):
            logger.log_generation(gen, population, fitness_scores, diversity, champion)

        assert mock_wandb.log.call_count == 10
```

**Coverage Target**: ≥90% for integration tests

**Deliverable**: Full evolution pipeline with Phase 1→2→3 handoff working

---

### Week 4: W&B Integration, UI, Performance Testing

**Duration**: 5 days
**Files Created**: 2
**Tests Written**: 4+
**Lines of Code**: ~400 production + ~200 tests

#### Day 16-17: W&B Logging Integration

**File**: `src/phase2_evomerge/monitoring/wandb_logger.py` (~200 LOC)

```python
class WandBLogger:
    """
    Weights & Biases integration for Phase 2.

    Logs 370 metrics total:
        - Per-generation: 50 generations × ~15 metrics = 750 logs
        - Per-model: 8 models × 50 gens × 6 metrics = 2400 logs
        - Final summary: ~20 metrics

    Total W&B API calls: ~3,170
    """

    def __init__(self, config: EvoMergeConfig):
        self.config = config
        wandb.init(
            project="agent-forge-v2",
            name=f"phase2_evomerge_{config.session_id}_{timestamp}",
            config=config.to_dict(),
            tags=["phase2", "evomerge", "evolution"]
        )

    def log_generation(self, gen, population, fitness_scores, diversity, champion):
        """Log per-generation metrics."""
        best_fitness = max(score.composite for score in fitness_scores)
        avg_fitness = sum(score.composite for score in fitness_scores) / len(fitness_scores)
        worst_fitness = min(score.composite for score in fitness_scores)

        wandb.log({
            'evolution/generation': gen,
            'evolution/best_fitness': best_fitness,
            'evolution/avg_fitness': avg_fitness,
            'evolution/worst_fitness': worst_fitness,
            'evolution/fitness_std': np.std([s.composite for s in fitness_scores]),
            'evolution/diversity': diversity,
            'evolution/champion_fitness': self.fitness_eval.evaluate(champion).composite,
            'time/generation_duration_seconds': gen_time,
            'time/estimated_completion_minutes': estimated_time,
        })

        # Per-model metrics
        for idx, (model, score) in enumerate(zip(population, fitness_scores)):
            wandb.log({
                f'models/model_{idx}/fitness': score.composite,
                f'models/model_{idx}/perplexity': score.perplexity,
                f'models/model_{idx}/accuracy': score.accuracy,
                f'models/model_{idx}/inference_time': score.speed,
                f'models/model_{idx}/memory_usage': score.memory,
                f'models/model_{idx}/combo_id': model.combo_id,
            })

        # Combo usage
        combo_counts = self.count_combo_usage(population)
        for combo_id, count in combo_counts.items():
            wandb.log({f'evolution/combo_{combo_id:03d}_count': count})

    def log_final(self, champion, improvement, generations):
        """Log final summary metrics."""
        wandb.log({
            'final/best_fitness': self.fitness_eval.evaluate(champion).composite,
            'final/improvement_pct': improvement,
            'final/generations_completed': generations,
            'final/best_combo': champion.combo_id,
            'final/total_evolution_time_minutes': total_time / 60,
        })

        # Save champion as artifact
        artifact = wandb.Artifact('phase2-champion', type='model')
        artifact.add_file('champion.pt')
        wandb.log_artifact(artifact)
```

**Metrics Breakdown**:

**Per-Generation (50× logs)**:
- `evolution/generation` (int)
- `evolution/best_fitness` (float)
- `evolution/avg_fitness` (float)
- `evolution/worst_fitness` (float)
- `evolution/fitness_std` (float)
- `evolution/diversity` (float)
- `evolution/champion_fitness` (float)
- `evolution/combo_000_count` through `combo_111_count` (8× int)
- `time/generation_duration_seconds` (float)
- `time/estimated_completion_minutes` (float)

**Per-Model (8 models × 50 gens = 400× logs)**:
- `models/model_{idx}/fitness` (float)
- `models/model_{idx}/perplexity` (float)
- `models/model_{idx}/accuracy` (float)
- `models/model_{idx}/inference_time` (float)
- `models/model_{idx}/memory_usage` (float)
- `models/model_{idx}/combo_id` (int)

**Final Summary (1× log)**:
- `final/best_fitness` (float)
- `final/improvement_pct` (float)
- `final/generations_completed` (int)
- `final/best_combo` (int)
- `final/total_evolution_time_minutes` (float)

**Total Metrics**: 50×17 + 400×6 + 5 = 850 + 2400 + 5 = **3,255 individual metric logs**

*Note: 370 unique metric names, logged 3,255 times total*

#### Day 18-19: UI Backend Integration

**File**: `src/ui/pages/phase2_evolution.py` (~200 LOC)

```python
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_phase2_evolution():
    """
    Phase 2 Evolution Dashboard.

    Components:
        1. Evolution Monitor (fitness curve)
        2. 3D Merge Visualization (Three.js tree)
        3. Combo Statistics Panel (8 binary combos)
        4. Population Diversity Chart
        5. Intervention Modal (manual controls)
    """
    st.title("Phase 2: EvoMerge Evolution")

    # Evolution Monitor
    st.header("Evolution Progress")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Generation", f"{current_gen}/{total_gens}")
    with col2:
        st.metric("Best Fitness", f"{best_fitness:.4f}", delta=f"+{improvement_pct:.1%}")
    with col3:
        st.metric("Diversity", f"{diversity:.3f}", delta_color="normal")
    with col4:
        st.metric("ETA", f"{eta_minutes} min")

    # Fitness Curve
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Fitness Evolution", "Diversity Evolution"])

    fig.add_trace(
        go.Scatter(x=generations, y=best_fitness_history, name="Best", line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=generations, y=avg_fitness_history, name="Avg", line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=generations, y=worst_fitness_history, name="Worst", line=dict(color='red')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=generations, y=diversity_history, name="Diversity", line=dict(color='purple')),
        row=2, col=1
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3D Merge Visualization
    st.header("3D Evolution Tree")
    st.info("3D visualization shows parent→child relationships across generations")

    # Three.js rendering (placeholder - needs JS integration)
    st.components.v1.html(render_3d_tree_html(evolution_tree_data), height=500)

    # Combo Statistics Panel
    st.header("Merge Technique Usage")
    combo_df = pd.DataFrame({
        'Combo ID': [f"Binary {i:03b}" for i in range(8)],
        'Technique': [decode_combo(i) for i in range(8)],
        'Usage Count': combo_usage_counts,
        'Avg Fitness': combo_avg_fitness,
    })
    st.dataframe(combo_df, use_container_width=True)

    # Bar chart
    fig_combo = go.Figure(data=[
        go.Bar(x=combo_df['Combo ID'], y=combo_df['Usage Count'], marker_color='lightblue')
    ])
    fig_combo.update_layout(title="Combo Usage Distribution", xaxis_title="Combo", yaxis_title="Count")
    st.plotly_chart(fig_combo, use_container_width=True)

    # Intervention Modal
    if diversity < 0.3:
        st.warning(f"⚠️ Low diversity detected ({diversity:.3f} < 0.3)")
        if st.button("Trigger Diversity Boost"):
            trigger_diversity_intervention()

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Pause Evolution"):
            pause_evolution()
    with col2:
        if st.button("Resume Evolution"):
            resume_evolution()

def decode_combo(combo_id):
    """Decode binary combo to technique names."""
    bit0 = (combo_id >> 0) & 1
    bit1 = (combo_id >> 1) & 1
    bit2 = (combo_id >> 2) & 1

    interp = "SLERP" if bit0 else "Linear"
    task = "TIES" if bit1 else "DARE"
    select = "DFS" if bit2 else "Franken"

    return f"{interp} + {task} + {select}"
```

**API Endpoints** (added to `src/ui/api.py`):

1. **POST /api/phases/2/configure**
   - Set evolution parameters (generations, mutation rate, etc.)
   - Returns: `{success: bool, config: dict}`

2. **POST /api/phases/2/start**
   - Start evolution (async background task)
   - Returns: `{success: bool, task_id: str}`

3. **GET /api/phases/2/status**
   - Current generation progress
   - Returns: `{generation: int, best_fitness: float, diversity: float, eta_minutes: int}`

4. **GET /api/phases/2/population**
   - Current 8 models + fitness scores
   - Returns: `{models: [model_info×8], fitness_scores: [float×8]}`

5. **GET /api/phases/2/diversity**
   - Population diversity metrics
   - Returns: `{diversity: float, pairwise_distances: [[float]], threshold: 0.3}`

6. **POST /api/phases/2/pause**
   - Pause evolution
   - Returns: `{success: bool, paused_at_generation: int}`

7. **POST /api/phases/2/intervene**
   - Trigger diversity boost
   - Returns: `{success: bool, new_diversity: float}`

**WebSocket Events** (added to `src/ui/websocket.py`):

```python
@socketio.on('phase:progress')
def handle_progress(data):
    """Emit every generation."""
    emit('phase:progress', {
        'generation': data['generation'],
        'best_fitness': data['best_fitness'],
        'avg_fitness': data['avg_fitness'],
        'diversity': data['diversity'],
        'eta_seconds': data['eta_seconds']
    })

@socketio.on('phase:metric')
def handle_metric(data):
    """Emit per-generation detailed metrics."""
    emit('phase:metric', {
        'combo_usage': data['combo_usage'],  # {0: 12, 1: 8, ...}
        'best_combo': data['best_combo'],
        'generations_complete': data['generations_complete'],
        'diversity_warning': data['diversity'] < 0.3
    })

@socketio.on('phase:alert')
def handle_alert(data):
    """Emit for critical events."""
    emit('phase:alert', {
        'alert_type': data['alert_type'],  # 'low_diversity'|'convergence'|'error'
        'message': data['message'],
        'recommended_action': data['recommended_action']
    })

@socketio.on('phase:complete')
def handle_complete(data):
    """Emit once at end."""
    emit('phase:complete', {
        'success': data['success'],
        'best_fitness': data['best_fitness'],
        'improvement_pct': data['improvement_pct'],
        'best_combo': data['best_combo'],
        'generations_run': data['generations_run']
    })
```

#### Day 20: Performance Testing

**File**: `tests/performance/test_phase2_performance.py` (~150 LOC, 2 tests)

```python
@pytest.mark.slow
@pytest.mark.performance
class TestPhase2Performance:
    def test_90_minute_benchmark(self, phase1_models):
        """Test evolution completes in ≤90 minutes on GTX 1660."""
        config = EvoMergeConfig(generations=50)
        engine = EvolutionEngine(config)

        start_time = time.time()
        champion, gens = engine.run(phase1_models)
        end_time = time.time()

        duration_minutes = (end_time - start_time) / 60

        assert duration_minutes <= 90, f"Evolution took {duration_minutes:.1f} min > 90 min"
        assert gens <= 50

    def test_parallel_evaluation_speedup(self, phase1_models):
        """Test parallel evaluation speeds up fitness computation."""
        config_serial = EvoMergeConfig(enable_parallel=False, num_workers=1)
        config_parallel = EvoMergeConfig(enable_parallel=True, num_workers=4)

        # Serial
        engine_serial = EvolutionEngine(config_serial)
        start = time.time()
        _ = engine_serial.run(phase1_models, max_gens=5)
        serial_time = time.time() - start

        # Parallel
        engine_parallel = EvolutionEngine(config_parallel)
        start = time.time()
        _ = engine_parallel.run(phase1_models, max_gens=5)
        parallel_time = time.time() - start

        speedup = serial_time / parallel_time
        assert speedup >= 2.0, f"Speedup {speedup:.2f}× < 2.0× expected"
```

**File**: `tests/performance/test_phase2_memory.py` (~100 LOC, 2 tests)

```python
@pytest.mark.slow
@pytest.mark.gpu
class TestPhase2Memory:
    def test_vram_usage_under_6gb(self, phase1_models):
        """Test VRAM usage <6GB throughout evolution."""
        config = EvoMergeConfig(generations=50)
        engine = EvolutionEngine(config)

        max_vram = 0

        def monitor_vram():
            nonlocal max_vram
            while engine.running:
                current_vram = torch.cuda.memory_allocated() / 1e9  # GB
                max_vram = max(max_vram, current_vram)
                time.sleep(1)

        monitor_thread = threading.Thread(target=monitor_vram)
        monitor_thread.start()

        champion, gens = engine.run(phase1_models)

        monitor_thread.join()

        assert max_vram < 6.0, f"Peak VRAM {max_vram:.2f}GB > 6.0GB"

    def test_memory_cleanup(self, phase1_models):
        """Test only 12 models in memory max."""
        config = EvoMergeConfig(generations=10)
        engine = EvolutionEngine(config)

        # Count models in memory throughout
        model_counts = []

        for gen in range(10):
            engine.step()  # Run one generation

            # Count all model tensors
            num_models = len(engine.population_mgr.population)
            num_models += 1  # champion
            num_models += 3  # original Phase 1 models

            model_counts.append(num_models)

        assert all(count <= 12 for count in model_counts), \
            f"Model count exceeded 12: {max(model_counts)}"
```

**Coverage Target**: Performance tests ensure production readiness

**Deliverable**: ✅ **PHASE 2 COMPLETE** - All 91+ tests pass, UI functional, performance validated

---

## Cross-Phase Integration

### Phase 1 → Phase 2 Handoff

**Input Contract**:
```python
{
    'models': [model1, model2, model3],  # Exactly 3 models
    'architecture': 'TRM_Titans_MAG',
    'parameters_per_model': ~25_000_000,
    'total_parameters': ~75_000_000,
}
```

**Input Validation**:
```python
def validate_phase1_output(models):
    assert len(models) == 3, "Must be exactly 3 models"
    for model in models:
        params = sum(p.numel() for p in model.parameters())
        assert 23_000_000 <= params <= 27_000_000, f"Params {params} out of range"
        assert hasattr(model, 'trm_wrapper'), "Missing TRM wrapper"
        assert hasattr(model, 'titans_mag'), "Missing Titans-MAG backbone"
        vram = measure_vram(model)
        assert vram < 6_000_000_000, f"VRAM {vram/1e9:.2f}GB > 6GB"
```

**Loading Interface**:
```python
from src.phase1_cognate.cognate import load_phase1_models

model1, model2, model3 = load_phase1_models(session_id="my_run")
# Returns 3 TRM × Titans-MAG models
```

---

### Phase 2 → Phase 3 Handoff

**Output Contract**:
```python
{
    'success': True,
    'model': champion_model,  # Single evolved model
    'phase_name': 'evomerge',
    'metrics': {
        'best_fitness': 0.185,
        'initial_fitness': 0.150,
        'improvement': 0.035,
        'improvement_pct': 0.235,  # 23.5%
        'generations_run': 38,
        'convergence_reason': 'early_stop',
        'final_diversity': 0.35,
        'best_combo': 7,  # Binary 111
    },
    'artifacts': {
        'evolution_log': 'logs/evomerge_generation_log.json',
        'final_checkpoint': 'checkpoints/phase2/final.pt',
    },
    'duration_seconds': 5432.1
}
```

**Output Validation**:
```python
def validate_phase2_output(champion, phase1_models):
    # Fitness improvement
    champion_fitness = evaluate_fitness(champion)
    baseline_fitness = max(evaluate_fitness(m) for m in phase1_models)
    improvement = (champion_fitness - baseline_fitness) / baseline_fitness

    assert improvement >= 0.20, f"Improvement {improvement:.1%} < 20%"

    # Model size
    params = sum(p.numel() for p in champion.parameters())
    assert 23_000_000 <= params <= 27_000_000, f"Params {params} out of range"

    # VRAM for Phase 3
    vram = measure_vram(champion)
    assert vram < 6_000_000_000, f"VRAM {vram/1e9:.2f}GB > 6GB"
```

**Storage Strategy**:
```
storage/{session_id}/
└── evomerge_champion_{timestamp}/
    ├── model.pt                      # Evolved model
    ├── metadata.json                 # Metrics + specs
    ├── config.json                   # Architecture config
    └── evolution_log.json            # Gen-by-gen progress
```

**Cleanup Process**:
```python
# After Phase 2 completion
delete(original_phase1_models)        # 3 models (archival)
delete(all_generation_populations)    # Gens 1-50, 8 models each
keep(champion_model)                  # Only keep best
```

---

### Shared Infrastructure Usage

| Component | How Phase 2 Uses It |
|-----------|---------------------|
| **Model Registry** | Track 12 models (3 original + 1 champion + 8 current) |
| **W&B Integration** | Log 370 metrics via `wandb_integration.log_phase2_metrics()` |
| **Config Manager** | Load from `config/pipeline_config.yaml` lines 99-128 |
| **Device Manager** | GPU/CPU handling, VRAM monitoring |
| **Orchestrator** | Chain Phase 1→2→3 seamlessly |
| **Utils** | Model-size detection, batch sizing |
| **Logger** | Structured logging throughout evolution |

---

## Testing Strategy

### Unit Tests (73+ tests, ≥95% coverage)

**File**: `tests/unit/test_merge_techniques.py` (18 tests)
- Linear merge: identical, random, opposite models
- SLERP merge: θ=0 fallback, magnitude preservation
- DARE merge: stochasticity, sparsity, rescaling
- TIES merge: sign voting, trimming, conflict resolution
- FrankenMerge: layer selection, dimension compatibility
- DFS merge: variance weighting, stable features

**File**: `tests/unit/test_fitness_evaluation.py` (12 tests)
- Perplexity calculation correctness
- Accuracy scoring correctness
- Speed benchmarking correctness
- Memory profiling correctness
- Composite fitness formula validation
- Edge cases: NaN, Inf handling

**File**: `tests/unit/test_population_management.py` (15 tests)
- Generation 0 initialization (8 unique models)
- Elite preservation (top 2 kept)
- Elite mutation (6 children from 2 elites)
- Loser merging (6 worst → 2 children)
- Population size consistency (always 8)
- Memory limit enforcement (≤12 models)

**File**: `tests/unit/test_diversity.py` (10 tests)
- Pairwise distance computation
- Diversity threshold detection
- Re-seeding triggers correctly
- Diversity increases after re-seeding

**File**: `tests/unit/test_genetic_operations.py` (12 tests)
- Mutation applies Gaussian noise
- Mutation rate correctness (1% of weights)
- Mutation sigma correctness (~1% magnitude)
- Mutation preserves model architecture

**File**: `tests/unit/test_error_handling.py` (6 tests)
- NaN weights detection and fallback
- Dimension mismatch detection
- SLERP θ=0 singularity handling
- Failed merge detection

---

### Integration Tests (14+ tests)

**File**: `tests/integration/test_phase2_evolution.py` (4 tests)
- Full 50-gen evolution completes
- Early stopping triggers correctly
- Diversity maintained throughout
- Checkpoint save/resume works

**File**: `tests/integration/test_phase1_to_phase2_handoff.py` (3 tests)
- Load 3 Phase 1 models successfully
- Validation passes for correct models
- Validation fails for incorrect models

**File**: `tests/integration/test_phase2_output.py` (3 tests)
- Champion saved to storage correctly
- Improvement ≥20% validation
- Output metadata format correct

**File**: `tests/integration/test_phase2_wandb.py` (4 tests)
- All 370 metrics logged
- Artifacts saved correctly
- Config logged to W&B
- Metrics logged every generation

---

### Performance Tests (4+ tests)

**File**: `tests/performance/test_phase2_performance.py` (2 tests)
- 90-minute benchmark on GTX 1660
- Parallel evaluation speedup ≥2.0×

**File**: `tests/performance/test_phase2_memory.py` (2 tests)
- VRAM usage <6GB throughout
- Memory cleanup (≤12 models max)

---

### CI/CD Integration

**Existing Workflow** (`.github/workflows/ci.yml`):
```yaml
jobs:
  lint:
    # Black, isort, flake8, pylint, NASA POT10

  type-check:
    # mypy strict mode

  test:
    # pytest on Python 3.10, 3.11
    # Coverage ≥90%

  security:
    # bandit security scan

  build:
    # Package distribution validation

  quality-gate:
    # All jobs must pass
```

**Phase 2 Tests Run**:
```bash
# Unit tests (fast)
pytest tests/unit -m phase2 -v --cov=src/phase2_evomerge

# Integration tests
pytest tests/integration -m phase2 -v

# Performance tests (slow, on-demand)
pytest tests/performance -m phase2 -v --slow
```

**Quality Gates Enforced**:
- ✅ NASA POT10: All functions ≤60 LOC
- ✅ Type checking: mypy strict passes
- ✅ Test coverage: ≥90% overall
- ✅ Formatting: Black + isort
- ✅ Security: bandit passes

---

## UI Implementation

### Phase 2 Dashboard Components

**1. Evolution Monitor** (`src/ui/pages/phase2_evolution.py`, lines 1-50)
- **Display**: Fitness curve (best/avg/worst), generation #/50, ETA
- **Update Frequency**: Real-time (every generation)
- **Data Source**: WebSocket `phase:progress` event

**2. 3D Merge Visualization** (lines 51-100)
- **Technology**: Three.js for 3D rendering
- **Nodes**: One node per generation (50 nodes max)
- **Edges**: Parent→child relationships
- **Node Color**: Red (bad fitness) → Green (good fitness)
- **Node Size**: Proportional to diversity
- **Interactions**: Rotate, zoom, hover for details

**3. Combo Statistics Panel** (lines 101-150)
- **Display**: 8 binary combos usage %, avg fitness per combo
- **Charts**: Bar chart (usage count), table (detailed stats)
- **Highlight**: Best combo (typically Binary 111)

**4. Population Diversity Chart** (lines 151-180)
- **Display**: Real-time diversity tracking (line chart)
- **Threshold Lines**: 0.3 (healthy), 0.2 (critical)
- **Warning**: Alert if diversity <0.3

**5. Intervention Modal** (lines 181-200)
- **Trigger**: Automatic when diversity <0.3
- **Actions**: "Inject Random Models", "Increase Mutation Rate", "Continue"
- **Confirmation**: User confirms intervention

---

### API Endpoints (7 new)

**POST /api/phases/2/configure**
```python
@app.route('/api/phases/2/configure', methods=['POST'])
def configure_phase2():
    config = request.json  # {generations: 50, mutation_rate: 0.01, ...}
    phase2_config = EvoMergeConfig(**config)
    return {'success': True, 'config': phase2_config.to_dict()}
```

**POST /api/phases/2/start**
```python
@app.route('/api/phases/2/start', methods=['POST'])
def start_phase2():
    session_id = request.json['session_id']
    task = evolution_engine.run_async(session_id)
    return {'success': True, 'task_id': task.id}
```

**GET /api/phases/2/status**
```python
@app.route('/api/phases/2/status', methods=['GET'])
def get_phase2_status():
    status = evolution_engine.get_status()
    return {
        'generation': status.current_generation,
        'best_fitness': status.best_fitness,
        'diversity': status.diversity,
        'eta_minutes': status.eta_minutes
    }
```

**GET /api/phases/2/population**
```python
@app.route('/api/phases/2/population', methods=['GET'])
def get_population():
    pop = evolution_engine.get_current_population()
    return {
        'models': [model_to_dict(m) for m in pop.models],
        'fitness_scores': pop.fitness_scores
    }
```

**POST /api/phases/2/pause**
```python
@app.route('/api/phases/2/pause', methods=['POST'])
def pause_evolution():
    evolution_engine.pause()
    return {'success': True, 'paused_at_generation': evolution_engine.current_generation}
```

**POST /api/phases/2/intervene**
```python
@app.route('/api/phases/2/intervene', methods=['POST'])
def trigger_intervention():
    action = request.json['action']  # 'inject_random' | 'increase_mutation'
    new_diversity = evolution_engine.intervene(action)
    return {'success': True, 'new_diversity': new_diversity}
```

**GET /api/phases/2/results**
```python
@app.route('/api/phases/2/results', methods=['GET'])
def get_results():
    results = evolution_engine.get_final_results()
    return {
        'champion': model_to_dict(results.champion),
        'improvement_pct': results.improvement,
        'generations_run': results.generations,
        'best_combo': results.best_combo
    }
```

---

### WebSocket Events (4 types)

**Event: `phase:progress`** (every generation)
```javascript
socket.on('phase:progress', (data) => {
    // data = {
    //   generation: 38,
    //   best_fitness: 0.185,
    //   avg_fitness: 0.172,
    //   diversity: 0.35,
    //   eta_seconds: 720
    // }
    updateFitnessCurve(data);
    updateGenerationCounter(data.generation);
});
```

**Event: `phase:metric`** (every generation, detailed)
```javascript
socket.on('phase:metric', (data) => {
    // data = {
    //   combo_usage: {0: 12, 1: 8, 2: 7, ...},
    //   best_combo: 7,
    //   generations_complete: 38,
    //   diversity_warning: false
    // }
    updateComboStatistics(data.combo_usage);
    highlightBestCombo(data.best_combo);
});
```

**Event: `phase:alert`** (immediate)
```javascript
socket.on('phase:alert', (data) => {
    // data = {
    //   alert_type: 'low_diversity',
    //   message: 'Population diversity dropped to 0.25',
    //   recommended_action: 'Consider triggering diversity boost'
    // }
    showAlertModal(data);
});
```

**Event: `phase:complete`** (once)
```javascript
socket.on('phase:complete', (data) => {
    // data = {
    //   success: true,
    //   best_fitness: 0.185,
    //   improvement_pct: 0.235,
    //   best_combo: 7,
    //   generations_run: 38
    // }
    showCompletionSummary(data);
    stopLoadingIndicator();
});
```

---

## W&B Integration

### Metric Categories (370 unique metrics)

**1. Per-Generation Metrics** (17 metrics × 50 gens = 850 logs)

```python
wandb.log({
    # Evolution metrics
    'evolution/generation': int,                    # 0-50
    'evolution/best_fitness': float,                # 0.150 → 0.185
    'evolution/avg_fitness': float,                 # Population average
    'evolution/worst_fitness': float,               # Worst in population
    'evolution/fitness_std': float,                 # Standard deviation
    'evolution/diversity': float,                   # 0.35-0.45 typical
    'evolution/champion_fitness': float,            # Best seen so far

    # Combo usage (8 combos)
    'evolution/combo_000_count': int,               # Linear + DARE + Franken
    'evolution/combo_001_count': int,               # Linear + DARE + DFS
    'evolution/combo_010_count': int,               # Linear + TIES + Franken
    'evolution/combo_011_count': int,               # Linear + TIES + DFS
    'evolution/combo_100_count': int,               # SLERP + DARE + Franken
    'evolution/combo_101_count': int,               # SLERP + DARE + DFS
    'evolution/combo_110_count': int,               # SLERP + TIES + Franken
    'evolution/combo_111_count': int,               # SLERP + TIES + DFS ← WINNER

    # Time tracking
    'time/generation_duration_seconds': float,      # ~108 sec avg
    'time/estimated_completion_minutes': float,     # ETA
})
```

**2. Per-Model Metrics** (6 metrics × 8 models × 50 gens = 2,400 logs)

```python
for idx, (model, score) in enumerate(zip(population, fitness_scores)):
    wandb.log({
        f'models/model_{idx}/fitness': score.composite,        # 0.150-0.185
        f'models/model_{idx}/perplexity': score.perplexity,    # 12-18
        f'models/model_{idx}/accuracy': score.accuracy,        # 0.40-0.50
        f'models/model_{idx}/inference_time': score.speed,     # 0.05-0.08 sec
        f'models/model_{idx}/memory_usage': score.memory,      # 5.2GB
        f'models/model_{idx}/combo_id': model.combo_id,        # 0-7
    })
```

**3. Final Summary Metrics** (5 metrics × 1 log = 5 logs)

```python
wandb.log({
    'final/best_fitness': float,                    # 0.185 (target)
    'final/improvement_pct': float,                 # 0.235 (23.5%)
    'final/generations_completed': int,             # 38 (typical)
    'final/best_combo': int,                        # 7 (Binary 111 typical)
    'final/total_evolution_time_minutes': float,    # 90 (target)
})
```

**Total Logs**: 850 + 2,400 + 5 = **3,255 individual metric logs**
**Unique Metric Names**: 17 + 48 (6×8) + 5 = **370 unique metrics**

---

### W&B Project Structure

```
agent-forge-v2/
└── phase2-evomerge/
    ├── run-{session_id}-{timestamp}/
    │   ├── config.json                 # EvoMergeConfig
    │   ├── metrics/
    │   │   ├── evolution/*             # Per-generation logs
    │   │   ├── models/*                # Per-model logs
    │   │   └── final/*                 # Final summary
    │   └── artifacts/
    │       ├── phase2-champion.pt      # Best model
    │       ├── evolution-log.json      # Complete history
    │       └── combo-usage.json        # Statistics
```

---

### W&B Dashboard Configuration

**Dashboard Name**: "Phase 2 EvoMerge - Evolution Dashboard"

**Panels**:
1. **Fitness Evolution** (Line chart)
   - X: Generation (0-50)
   - Y: Fitness (0.150-0.185)
   - Lines: Best (green), Avg (blue), Worst (red)

2. **Diversity Evolution** (Line chart)
   - X: Generation
   - Y: Diversity (0.25-0.45)
   - Threshold lines: 0.3 (warning), 0.2 (critical)

3. **Combo Usage Heatmap** (Heatmap)
   - X: Generation (0-50)
   - Y: Combo ID (0-7)
   - Color: Usage count (0-8)

4. **Final Improvement** (Metric)
   - `final/improvement_pct` (target: ≥20%)
   - Color: Green if ≥20%, Red otherwise

5. **Evolution Time** (Metric)
   - `final/total_evolution_time_minutes` (target: ≤90)

---

## File Structure

### Production Code (26 files, ~2,500 LOC)

```
src/phase2_evomerge/
├── __init__.py                         (50 LOC)
├── phase2_pipeline.py                  (200 LOC)
├── config.py                           (80 LOC)
│
├── merge/
│   ├── __init__.py                     (50 LOC)
│   ├── linear_merge.py                 (80 LOC)
│   ├── slerp_merge.py                  (120 LOC)
│   ├── dare_merge.py                   (100 LOC)
│   ├── ties_merge.py                   (130 LOC)
│   ├── frankenmerge.py                 (110 LOC)
│   └── dfs_merge.py                    (100 LOC)
│
├── fitness/
│   ├── __init__.py                     (50 LOC)
│   ├── evaluator.py                    (150 LOC)
│   ├── perplexity.py                   (100 LOC)
│   ├── accuracy.py                     (90 LOC)
│   ├── speed.py                        (110 LOC)
│   └── memory.py                       (80 LOC)
│
├── population/
│   ├── __init__.py                     (50 LOC)
│   ├── manager.py                      (180 LOC)
│   ├── genetic_ops.py                  (120 LOC)
│   └── diversity.py                    (100 LOC)
│
├── evolution/
│   ├── __init__.py                     (50 LOC)
│   ├── engine.py                       (250 LOC)
│   ├── convergence.py                  (80 LOC)
│   └── checkpoint.py                   (120 LOC)
│
└── monitoring/
    ├── __init__.py                     (50 LOC)
    └── wandb_logger.py                 (200 LOC)
```

### Test Code (10 files, ~1,800 LOC)

```
tests/
├── unit/
│   ├── test_merge_techniques.py        (400 LOC, 18 tests)
│   ├── test_fitness_evaluation.py      (300 LOC, 12 tests)
│   ├── test_population_management.py   (250 LOC, 15 tests)
│   ├── test_diversity.py               (200 LOC, 10 tests)
│   ├── test_genetic_operations.py      (200 LOC, 12 tests)
│   └── test_error_handling.py          (100 LOC, 6 tests)
│
├── integration/
│   ├── test_phase2_evolution.py        (200 LOC, 4 tests)
│   ├── test_phase1_to_phase2_handoff.py (150 LOC, 3 tests)
│   ├── test_phase2_output.py           (120 LOC, 3 tests)
│   └── test_phase2_wandb.py            (150 LOC, 4 tests)
│
└── performance/
    ├── test_phase2_performance.py      (150 LOC, 2 tests)
    └── test_phase2_memory.py           (100 LOC, 2 tests)
```

### UI Code (1 file, ~200 LOC)

```
src/ui/pages/
└── phase2_evolution.py                 (200 LOC)
```

**Total**: 26 production files + 10 test files + 1 UI file = **37 new files**
**Total Lines**: 2,500 production + 1,800 tests + 200 UI = **4,500 LOC**

---

## Success Criteria

### Technical Validation ✅

- [ ] All 6 merge techniques implemented (100% spec-compliant)
- [ ] 8 binary combinations working (2³ = Linear/SLERP × DARE/TIES × Franken/DFS)
- [ ] Fitness improvement ≥20% (target: 23.5%)
- [ ] Evolution completes in ≤90 minutes on GTX 1660
- [ ] Memory usage <6GB VRAM (12 models max)
- [ ] Diversity maintained >0.3 throughout
- [ ] Early stopping triggers (0.1% improvement threshold, 5-gen patience)
- [ ] All 8 combos used at least once

### Testing Validation ✅

- [ ] 91+ tests pass (73 unit + 14 integration + 4 performance)
- [ ] ≥90% overall coverage (≥95% critical paths)
- [ ] NASA POT10 compliance (all functions ≤60 LOC)
- [ ] mypy strict type checking passes
- [ ] All CI/CD quality gates pass (lint, type-check, test, security)
- [ ] Black + isort formatting passes

### Integration Validation ✅

- [ ] Phase 1→2 handoff: Load 3 TRM × Titans-MAG models
- [ ] Input validation: Exactly 3 models, ~25M params each, <6GB VRAM
- [ ] Phase 2→3 handoff: Output champion with ≥20% improvement
- [ ] 370 W&B metrics logged correctly (3,255 total logs)
- [ ] UI displays real-time progress (7 API endpoints, 4 WebSocket events)
- [ ] Model registry tracks all 12 models
- [ ] Champion saved to storage with metadata

### Performance Validation ✅

- [ ] 50 generations complete in 90 minutes (typical: 35-40 gens converge early)
- [ ] VRAM peak <6GB throughout evolution
- [ ] Parallel evaluation works (4 workers, ≥2.0× speedup)
- [ ] Fitness caching speeds up evaluation (≥30% speedup)
- [ ] Diversity re-seeding works (triggered when <0.2)
- [ ] Checkpoint save/resume works correctly

### Documentation Validation ✅

- [ ] All functions have docstrings (≥95% coverage)
- [ ] README.md updated with Phase 2 usage
- [ ] API documentation generated (Sphinx)
- [ ] GraphViz diagrams updated (phase-flow-v2.dot)

---

## Risk Mitigation

### High-Risk Areas

**1. GPU Memory (12 models × 25M params = ~2.4GB base + gradients)**
- **Probability**: Medium
- **Impact**: High (blocks evolution)
- **Mitigation**:
  - Adaptive batch sizing implemented in `src/cross_phase/utils.py`
  - Model offloading to CPU if VRAM exceeds 5.5GB
  - Garbage collection after each generation

**2. Merge Failures (NaN weights, dimension mismatches)**
- **Probability**: Medium
- **Impact**: High (breaks evolution)
- **Mitigation**:
  - Validation after every merge operation
  - Fallback to linear merge if any merge fails
  - NaN/Inf detection with automatic replacement
  - Dimension compatibility checks before layer-wise operations

**3. Diversity Collapse (population converges to single solution)**
- **Probability**: Medium
- **Impact**: High (poor final model)
- **Mitigation**:
  - Active diversity tracking every generation
  - Re-seeding at <0.2 threshold
  - Diversity monitoring in W&B dashboard
  - Manual intervention controls in UI

**4. Slow Convergence (50 generations not enough)**
- **Probability**: Low
- **Impact**: Medium (extends runtime)
- **Mitigation**:
  - Early stopping at 0.1% improvement for 5 consecutive gens
  - Typical convergence: 35-40 gens (validated in V1)
  - Configurable max generations (can increase if needed)

---

### Medium-Risk Areas

**5. Fitness Evaluation Speed (8 models × 50 gens = 400 evaluations)**
- **Probability**: Medium
- **Impact**: Medium (extends runtime)
- **Mitigation**:
  - Parallel evaluation (4 workers)
  - Fitness caching (avoid re-evaluation of identical models)
  - Subset validation (500 samples instead of full dataset)
  - GPU batching for perplexity/accuracy

**6. SLERP Singularity (θ=0 when models identical)**
- **Probability**: Low
- **Impact**: Low (edge case)
- **Mitigation**:
  - Detect θ<1e-6, fallback to linear merge
  - Warning logged to W&B
  - Automatic technique switching

**7. W&B Logging Overhead (370 metrics per run)**
- **Probability**: Low
- **Impact**: Low (minor slowdown)
- **Mitigation**:
  - Batch logging (log every 5 generations if needed)
  - Offline mode (local W&B server)
  - Async logging (non-blocking)

---

### Low-Risk Areas

**8. UI Latency (WebSocket updates)**
- **Probability**: Low
- **Impact**: Low (cosmetic)
- **Mitigation**:
  - Debounce WebSocket events (max 1 update/sec)
  - Client-side caching
  - Progressive rendering

**9. Checkpoint File Size**
- **Probability**: Low
- **Impact**: Low (disk space)
- **Mitigation**:
  - Compress checkpoints (gzip)
  - Keep only last 5 checkpoints
  - Automatic cleanup after completion

---

## Timeline & Milestones

### Week 1 (Days 1-5): Core Merge Techniques

**Monday (Day 1)**
- [ ] Implement Linear + SLERP merge (2 files, 200 LOC)
- [ ] Unit tests for interpolation methods (6 tests)

**Tuesday (Day 2)**
- [ ] Implement DARE + TIES merge (2 files, 230 LOC)
- [ ] Unit tests for task arithmetic (6 tests)

**Wednesday (Day 3)**
- [ ] Implement FrankenMerge + DFS (2 files, 210 LOC)
- [ ] Unit tests for selection methods (6 tests)

**Thursday (Day 4)**
- [ ] Implement binary combination pipeline (1 file, 80 LOC)
- [ ] Integration tests for all 8 combos (3 tests)

**Friday (Day 5)**
- [ ] Code review & refactoring
- [ ] Run all merge tests (18+), ensure ≥98% coverage
- [ ] **Milestone**: All merge techniques working

---

### Week 2 (Days 6-10): Fitness & Population

**Monday (Day 6)**
- [ ] Implement fitness evaluator + perplexity (2 files, 250 LOC)
- [ ] Unit tests for fitness (4 tests)

**Tuesday (Day 7)**
- [ ] Implement accuracy + speed + memory (3 files, 280 LOC)
- [ ] Unit tests for components (8 tests)

**Wednesday (Day 8)**
- [ ] Implement population manager (1 file, 180 LOC)
- [ ] Unit tests for population (8 tests)

**Thursday (Day 9)**
- [ ] Implement genetic operations + diversity (2 files, 220 LOC)
- [ ] Unit tests for genetic ops + diversity (17 tests)

**Friday (Day 10)**
- [ ] Integration tests for fitness + population (6 tests)
- [ ] Run all Week 2 tests (37+), ensure ≥95% coverage
- [ ] **Milestone**: Fitness + population systems complete

---

### Week 3 (Days 11-15): Evolution Engine & Integration

**Monday (Day 11)**
- [ ] Implement evolution engine (1 file, 250 LOC)
- [ ] Unit tests for engine (5 tests)

**Tuesday (Day 12)**
- [ ] Implement convergence + checkpoint (2 files, 200 LOC)
- [ ] Unit tests for convergence + checkpoint (4 tests)

**Wednesday (Day 13)**
- [ ] Implement phase2_pipeline (1 file, 200 LOC)
- [ ] Integration tests for Phase 1→2 handoff (3 tests)

**Thursday (Day 14)**
- [ ] Integration tests for full evolution (4 tests)
- [ ] Integration tests for Phase 2→3 handoff (3 tests)

**Friday (Day 15)**
- [ ] End-to-end test: Run full 50-gen evolution on dummy data
- [ ] Fix any integration issues
- [ ] **Milestone**: Full evolution pipeline working

---

### Week 4 (Days 16-20): W&B, UI, Performance

**Monday (Day 16)**
- [ ] Implement W&B logger (1 file, 200 LOC)
- [ ] Integration tests for W&B (4 tests)

**Tuesday (Day 17)**
- [ ] Test 370 metrics logging
- [ ] Validate W&B dashboard configuration

**Wednesday (Day 18)**
- [ ] Implement UI backend (7 API endpoints)
- [ ] Implement WebSocket events (4 types)

**Thursday (Day 19)**
- [ ] Implement UI frontend (phase2_evolution.py, 200 LOC)
- [ ] Test UI real-time updates

**Friday (Day 20)**
- [ ] Performance testing (90-min benchmark, 6GB VRAM)
- [ ] Final integration test: Full pipeline (Phase 1→2→3)
- [ ] Documentation update
- [ ] **Milestone**: ✅ **PHASE 2 COMPLETE**

---

### Milestone Checkpoints

**Milestone 1 (End of Week 1)**: Core Merge Techniques
- ✅ All 6 merge techniques implemented
- ✅ 8 binary combinations working
- ✅ 18+ unit tests passing
- ✅ ≥98% test coverage

**Milestone 2 (End of Week 2)**: Fitness & Population
- ✅ Fitness evaluation system complete
- ✅ Population management working
- ✅ 37+ unit tests passing
- ✅ ≥95% test coverage

**Milestone 3 (End of Week 3)**: Evolution Engine
- ✅ 50-generation evolution loop working
- ✅ Phase handoff integration complete
- ✅ 14+ integration tests passing
- ✅ End-to-end dummy test passes

**Milestone 4 (End of Week 4)**: Production Ready
- ✅ W&B logging complete (370 metrics)
- ✅ UI functional (7 endpoints, 4 events)
- ✅ Performance validated (90 min, 6GB)
- ✅ All 91+ tests passing
- ✅ Documentation complete
- ✅ **PHASE 2 PRODUCTION READY**

---

## Dependencies & Prerequisites

### Already Complete ✅

**Infrastructure** (Weeks 1-12 done):
- Cross-phase systems (model registry, orchestrator, config manager)
- MuGrokfast optimizer (not used in Phase 2, available for other phases)
- Prompt baking system (not used in Phase 2, available for Phase 3+)
- W&B integration framework (370 metrics pre-defined)
- UI framework (Streamlit with 5 pages)
- CI/CD pipeline (6 jobs, quality gates)
- Testing framework (pytest, 47 tests passing)

**Configuration**:
- Phase 2 config in `config/pipeline_config.yaml` (lines 99-128)
- All hyperparameters defined (50 gens, 8 pop, fitness weights, etc.)

**Documentation**:
- Phase 2 complete guide (1,213 lines)
- Logical understanding (599 lines)
- Binary pairing strategy (170 lines)
- Merge techniques (6 techniques fully documented)
- GraphViz diagrams (phase-flow-v2.dot)

---

### Required Before Start ⚠️

**Phase 1 Completion**:
- [ ] 3 trained TRM × Titans-MAG models (~25M params each)
- [ ] Models saved to `storage/{session_id}/cognate_*_{timestamp}/`
- [ ] Metadata includes: param count, architecture, seed, ACT threshold

**Hardware**:
- [ ] GTX 1660 or better GPU (6GB+ VRAM)
- [ ] CUDA 11.8+ installed
- [ ] 16GB+ system RAM
- [ ] 50GB+ disk space

**Software**:
- [ ] Python 3.10+
- [ ] PyTorch 2.0+ with CUDA support
- [ ] HuggingFace Transformers
- [ ] All deps from `requirements.txt` installed

**Environment**:
- [ ] W&B offline mode configured
- [ ] Pre-commit hooks installed
- [ ] pytest working
- [ ] All quality gates passing

---

### External Dependencies (Already in requirements.txt)

```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
wandb>=0.15.0
streamlit>=1.28.0
psutil>=5.9.0
pytest>=7.4.0
pytest-cov>=4.1.0
mypy>=1.8.0
black>=23.12.0
```

**No additional dependencies needed for Phase 2**

---

## Estimated Effort

**Total Duration**: 20 days (4 weeks × 5 days)

**Breakdown by Activity**:
- **Development**: 13 days
  - Merge techniques: 3 days
  - Fitness evaluation: 3 days
  - Population management: 2 days
  - Evolution engine: 3 days
  - Integration: 2 days
- **Testing**: 5 days
  - Unit tests: 3 days
  - Integration tests: 1 day
  - Performance tests: 1 day
- **UI/W&B**: 2 days
  - W&B logging: 1 day
  - UI backend/frontend: 1 day

**Lines of Code**:
- **Production**: ~2,500 LOC
- **Tests**: ~1,800 LOC
- **UI**: ~200 LOC
- **Total**: ~4,500 LOC

**Team Size**: 1 developer (full-time)
- **Parallelization**: Can reduce to 2 weeks with 2-3 developers
- **Dependencies**: Merge → Fitness → Evolution (sequential bottleneck)

---

## Conclusion

Phase 2 (EvoMerge) is a well-specified, production-ready implementation plan with:

✅ **Clear deliverables** (26 files, 4,500 LOC, 91+ tests)
✅ **Proven methodology** (research-validated merge techniques)
✅ **Strong infrastructure** (75% complete, ready to build on)
✅ **Comprehensive testing** (unit, integration, performance)
✅ **Production quality** (NASA POT10, ≥90% coverage, CI/CD)
✅ **Full observability** (370 W&B metrics, real-time UI)
✅ **Risk mitigation** (fallback strategies, edge case handling)

**Ready to implement!** 🚀

---

**Document Version**: 1.0
**Last Updated**: 2025-10-16
**Next Review**: After Week 1 completion
**Owner**: Phase 2 Implementation Team

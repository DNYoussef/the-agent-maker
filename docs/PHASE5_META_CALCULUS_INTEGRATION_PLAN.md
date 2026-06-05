# Phase 5: Meta-Calculus Integration Plan

## Executive Summary

This document outlines the comprehensive plan for integrating MOO (Multi-Objective Optimization) and meta-calculus capabilities into Phase 5 (Curriculum Learning) of Agent Forge V2.

**Status**: ALL TIERS COMPLETE (Quick Wins + Tier 1 + Tier 2 + Tier 3).

---

## Quick Wins Completed (5/5)

| # | Integration | File Modified | Change |
|---|-------------|---------------|--------|
| 1 | MetaGrokfast optimizer | `training_loop.py` | Replaced AdamW with MetaGrokfast (bigeometric gradient filtering) |
| 2 | Spectral gap advancement | `curriculum_engine.py` | Added gap monitoring + should_advance_stage() check before level transitions |
| 3 | k(level) dream samples | `dream_consolidation.py` | Dream sample count scales with k(L) - later levels get more consolidation |
| 4 | k(L) difficulty curve | `curriculum_generator.py` | Non-linear difficulty mapping using get_stage_difficulty() |
| 5 | Spectral gap self-modeling | `self_modeling.py` | Monitor representation health during self-prediction training |

---

## Phase 5 Architecture Overview

```
Phase 5: Curriculum Learning Pipeline
=====================================

                          +------------------+
                          |  CurriculumEngine |
                          |  (orchestrator)   |
                          +--------+---------+
                                   |
    +------------+---------+-------+-------+---------+-----------+
    |            |         |               |         |           |
    v            v         v               v         v           v
+-------+  +----------+ +--------+ +-----------+ +-------+ +-------+
|Assess-|  |Curriculum| |Training| |  Prompt   | | Self- | |Dream  |
| ment  |  |Generator | |  Loop  | |  Baking   | |Modeling| |Consol.|
+-------+  +----------+ +--------+ +-----------+ +-------+ +-------+
    |            |         |               |         |           |
    v            v         v               v         v           v
+-------+  +----------+ +--------+ +-----------+ +-------+ +-------+
|Binary |  |  k(L)    | |MetaGrok| | k(level)  | |Spectral| | k(L)  |
|Search |  |Difficulty | |fast   | | baking    | |Gap Mon.| |Samples|
+-------+  +----------+ +--------+ +-----------+ +-------+ +-------+
                                       |
                          +------------+------------+
                          |                         |
                          v                         v
                   +------------+           +-------------+
                   | Eudaimonia |           | Archetypes  |
                   |   Rules    |           | + OODA Loop |
                   +------------+           +-------------+
```

---

## Integration Tiers

### Tier 0: COMPLETE (Quick Wins)
Already implemented above.

### Tier 1: COMPLETE (Component Enhancements)

| # | Enhancement | File | Status |
|---|-------------|------|--------|
| 1.1 | MOO edge-of-chaos finder | `assessment.py` | DONE - `find_baseline_moo()` with 3 objectives |
| 1.2 | Adaptive mastery threshold | `training_loop.py` | DONE - `_get_mastery_threshold()` + `_get_max_hints()` |
| 1.3 | k(level) baking strength | `curriculum_engine.py` | DONE - `_get_baking_config()` with adaptive LoRA rank |
| 1.4 | Virtue weight MOO | `eudaimonia/rules.py` | DONE - `VirtueWeightOptimizer` class |

### Tier 2: COMPLETE (Advanced Integrations)

| # | Enhancement | File | Status |
|---|-------------|------|--------|
| 2.1 | MOO curriculum generator | `curriculum_generator.py` | DONE - `MOOCurriculumGenerator` with 4-objective optimization |
| 2.2 | MOO action selection | `eudaimonia/ooda_loop.py` | DONE - `select_action_moo()` with Pareto front selection |
| 2.3 | MOO temperature ranges | `self_modeling.py` | DONE - `MOOSelfModelingTrainer.optimize_temperature_ranges()` |

### Tier 3: COMPLETE (Deep Integrations)

| # | Enhancement | File | Status |
|---|-------------|------|--------|
| 3.1 | Full curriculum pipeline | (integrated across files) | DONE - MOO + k(L) throughout |
| 3.2 | Archetype weight learning | `eudaimonia/archetypes.py` | DONE - `ArchetypeWeightLearner` class |
| 3.3 | Dream quality gates | `dream_consolidation.py` | DONE - `DreamQualityGate` with spectral gap |

### Tier 1 Details (for reference):

#### 1.1 Assessment MOO Enhancement
**File**: `assessment.py`
**Current**: Binary search for 75% accuracy threshold
**Enhancement**: Multi-objective edge-of-chaos finder

```python
# assessment.py additions
from src.cross_phase.meta_calculus.moo_utils import HybridMOORunner

class EdgeOfChaosAssessment:
    """Multi-objective edge-of-chaos finder."""

    def find_edge_moo(self, model, test_cases) -> Dict:
        """
        Find edge-of-chaos zone using multi-objective optimization.

        Objectives:
        1. Maximize accuracy (want ~75%)
        2. Minimize confidence variance (want stable predictions)
        3. Minimize response time (want efficient)

        Returns:
            Dict with Pareto-optimal difficulty levels
        """
        def evaluate(difficulty_level):
            accuracy = self._test_at_difficulty(model, difficulty_level, test_cases)
            variance = self._measure_confidence_variance(model, difficulty_level)
            latency = self._measure_response_time(model, difficulty_level)

            return [
                -accuracy,         # maximize (negate for minimization)
                variance,          # minimize
                latency,           # minimize
            ]

        runner = HybridMOORunner.from_evaluator(
            evaluator=evaluate,
            n_vars=1,              # difficulty level
            n_objs=3,
            xl=[0],
            xu=[100],
        )
        result = runner.run(n_generations=20)

        return {
            "pareto_front": result.F,
            "optimal_difficulties": result.X,
            "recommended_baseline": self._select_balanced(result),
        }
```

**Benefit**: Instead of single binary search point, get Pareto-optimal "zone" of edge-of-chaos.

---

#### 1.2 Training Loop Adaptive Mastery
**File**: `training_loop.py`
**Current**: Fixed mastery threshold (3 consecutive successes)
**Enhancement**: k(difficulty) adaptive mastery

```python
# training_loop.py additions
def _get_mastery_threshold(self, question: Question) -> int:
    """
    Get adaptive mastery threshold based on question difficulty.

    Harder questions need MORE consecutive successes to prove mastery.
    Uses k(difficulty/100) formula.
    """
    if META_CALCULUS_AVAILABLE:
        difficulty_normalized = question.original_difficulty / 100.0
        k = meta_phase5.get_k_value(difficulty_normalized)

        # k is high for easy (small L), low for hard (large L)
        # We want MORE successes for hard questions
        # Base = 3, range = 2-5
        threshold = 3 + int((1 - k) * 2)  # 3 for easy, 5 for hard
        return max(2, min(5, threshold))

    return self.consecutive_for_mastery  # Default: 3

def _get_max_hints(self, question: Question) -> int:
    """
    Get adaptive max hints based on difficulty.

    Harder questions get more hints allowed.
    """
    if META_CALCULUS_AVAILABLE:
        difficulty_normalized = question.original_difficulty / 100.0
        k = meta_phase5.get_k_value(difficulty_normalized)

        # More hints for harder questions
        max_hints = 3 + int((1 - k) * 4)  # 3-7 hints
        return max(3, min(7, max_hints))

    return self.max_hints  # Default: 5
```

**Benefit**: Difficulty-appropriate mastery criteria; harder questions proven with more evidence.

---

#### 1.3 Prompt Baking k(level) Strength
**File**: `curriculum_engine.py` (method `_run_prompt_baking`)
**Current**: Fixed LoRA config for all levels
**Enhancement**: k(level) adaptive baking strength

```python
# curriculum_engine.py additions to _run_prompt_baking
def _get_baking_config(self, level: int) -> Dict:
    """
    Get level-adaptive baking configuration.

    Early levels: Light baking (learning still happening)
    Later levels: Stronger baking (embed more deeply)
    """
    if META_CALCULUS_AVAILABLE:
        L = level / self.config.num_levels
        k = meta_phase5.get_k_value(max(0.01, L))

        # k is high for early levels, low for late levels
        # We want STRONGER baking for later levels
        baking_strength = 0.3 + (1 - k) * 0.5  # 0.3 to 0.8

        # LoRA rank scales with baking strength
        lora_rank = 8 + int((1 - k) * 24)  # 8-32

        return {
            "lora_rank": lora_rank,
            "lora_alpha": lora_rank * 2,
            "baking_strength": baking_strength,
            "epochs": 3 + int((1 - k) * 2),  # 3-5 epochs
        }

    return {
        "lora_rank": 16,
        "lora_alpha": 32,
        "baking_strength": 0.5,
        "epochs": 3,
    }
```

**Benefit**: Progressive baking depth; early skills lightly encoded, later skills deeply embedded.

---

#### 1.4 Virtue Weight MOO Optimization
**File**: `eudaimonia/rules.py`
**Current**: Fixed virtue weights (40/20/20/20)
**Enhancement**: MOO-optimized virtue weights per specialization

```python
# rules.py additions
class AdaptiveEudaimoniaRules(EudaimoniaRuleSystem):
    """Eudaimonia rules with MOO-optimized virtue weights."""

    def optimize_weights_for_specialization(
        self,
        specialization: str,
        test_scenarios: List[Dict]
    ) -> Dict[str, float]:
        """
        Find Pareto-optimal virtue weights for a specialization.

        Objectives:
        1. Maximize correct ethical judgments
        2. Minimize false positives (unnecessary compass triggers)
        3. Minimize false negatives (missed ethical issues)
        """
        def evaluate(weights):
            # weights = [prime_directive, curiosity, esprit_de_corps, life_value]
            self._set_weights(weights)

            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for scenario in test_scenarios:
                score = self.assess(scenario["situation"], scenario["context"])
                expected_trigger = scenario["should_trigger_compass"]
                actual_trigger = not score.is_confident

                if expected_trigger and actual_trigger:
                    true_positives += 1
                elif not expected_trigger and actual_trigger:
                    false_positives += 1
                elif expected_trigger and not actual_trigger:
                    false_negatives += 1

            return [
                -true_positives,  # maximize
                false_positives,  # minimize
                false_negatives,  # minimize
            ]

        runner = HybridMOORunner.from_evaluator(
            evaluator=evaluate,
            n_vars=4,
            n_objs=3,
            xl=[0.1, 0.1, 0.1, 0.1],
            xu=[0.6, 0.4, 0.4, 0.4],
        )
        result = runner.run()

        # Select balanced solution from Pareto front
        best_weights = self._select_balanced_weights(result)

        return {
            "prime_directive": best_weights[0],
            "curiosity": best_weights[1],
            "esprit_de_corps": best_weights[2],
            "life_value": best_weights[3],
        }
```

**Benefit**: Data-driven virtue calibration per agent specialization.

---

### Tier 2: Advanced Integrations (8-16 hours each)

#### 2.1 Curriculum Generator MOO
**File**: `curriculum_generator.py`
**Enhancement**: Multi-objective curriculum design

```python
class MOOCurriculumGenerator(AdaptiveCurriculumGenerator):
    """Curriculum generator with multi-objective optimization."""

    def generate_optimal_curriculum(
        self,
        baseline_level: int,
        evaluator_model: nn.Module,
    ) -> Dict[int, List[Question]]:
        """
        Generate Pareto-optimal curriculum.

        Objectives:
        1. Maximize expected learning rate
        2. Maximize concept retention
        3. Minimize difficulty variance (smoothness)
        4. Minimize total questions needed
        """
        def evaluate(curriculum_params):
            # curriculum_params encodes:
            # - difficulty curve shape (alpha, beta for beta distribution)
            # - questions per level distribution
            # - frontier model allocation

            curriculum = self._generate_from_params(curriculum_params)

            learning_rate = self._estimate_learning_rate(curriculum, evaluator_model)
            retention = self._estimate_retention(curriculum)
            smoothness = self._measure_difficulty_variance(curriculum)
            total_questions = sum(len(q) for q in curriculum.values())

            return [
                -learning_rate,    # maximize
                -retention,        # maximize
                smoothness,        # minimize
                total_questions,   # minimize
            ]

        runner = HybridMOORunner.from_evaluator(
            evaluator=evaluate,
            n_vars=6,  # alpha, beta, level_distribution[4]
            n_objs=4,
            xl=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            xu=[5.0, 5.0, 2.0, 2.0, 2.0, 2.0],
        )
        result = runner.run(n_generations=50)

        # Generate curriculum from best parameters
        best_params = self._select_efficient_curriculum(result)
        return self._generate_from_params(best_params)
```

**Benefit**: Optimal curriculum shape, not just linear difficulty progression.

---

#### 2.2 OODA Loop Action MOO
**File**: `eudaimonia/ooda_loop.py`
**Enhancement**: Multi-objective action selection

```python
class MOOOODALoop(OODALoop):
    """OODA loop with MOO-optimized action selection."""

    def _select_best_action_moo(
        self,
        candidate_actions: List[str],
        moral_direction: Dict,
        context: Dict,
    ) -> SmallestMeasurableAction:
        """
        Select action using multi-objective optimization.

        Objectives:
        1. Maximize alignment with moral direction
        2. Maximize measurability
        3. Maximize reversibility
        4. Minimize risk
        """
        actions = [
            self._evaluate_action(desc, moral_direction, context)
            for desc in candidate_actions
        ]

        if not actions:
            return None

        # Build objective matrix
        objectives = np.array([
            [
                -a.alignment_score,   # maximize
                -a.measurability,     # maximize
                -a.reversibility,     # maximize
                a.risk_level,         # minimize
            ]
            for a in actions
        ])

        # Find Pareto-optimal actions
        pareto_mask = self._find_pareto_front(objectives)
        pareto_actions = [a for a, is_pareto in zip(actions, pareto_mask) if is_pareto]

        # Select from Pareto front using knee-point method
        if len(pareto_actions) == 1:
            return pareto_actions[0]

        # Prefer action with best alignment among Pareto-optimal
        return max(pareto_actions, key=lambda a: a.alignment_score)
```

**Benefit**: No single weighting scheme; find truly non-dominated actions.

---

#### 2.3 Self-Modeling Temperature MOO
**File**: `self_modeling.py`
**Enhancement**: MOO-optimized temperature range selection

```python
class MOOSelfModelingTrainer(SelfModelingTrainer):
    """Self-modeling with MOO-optimized temperature ranges."""

    def optimize_temperature_ranges(
        self,
        model: nn.Module,
        tokenizer: Any,
    ) -> List[TemperatureRange]:
        """
        Find Pareto-optimal temperature ranges for self-modeling.

        Objectives:
        1. Maximize self-prediction accuracy
        2. Maximize temperature coverage (breadth)
        3. Minimize training time
        4. Maximize representation diversity (spectral gap)
        """
        def evaluate(range_params):
            # range_params: [start, width, num_ranges, samples_per_range]
            ranges = self._create_ranges_from_params(range_params)

            # Quick training run
            accuracy = self._quick_train_and_eval(model, ranges, tokenizer)
            coverage = self._measure_temperature_coverage(ranges)
            time_cost = range_params[2] * range_params[3]  # num_ranges * samples
            gap = self._compute_spectral_gap(model)

            return [
                -accuracy,   # maximize
                -coverage,   # maximize
                time_cost,   # minimize
                -gap,        # maximize
            ]

        runner = HybridMOORunner.from_evaluator(
            evaluator=evaluate,
            n_vars=4,
            n_objs=4,
            xl=[0.0, 0.1, 5, 50],
            xu=[1.0, 0.5, 20, 200],
        )
        result = runner.run()

        best_params = self._select_efficient_ranges(result)
        return self._create_ranges_from_params(best_params)
```

**Benefit**: Optimal temperature coverage for meta-cognitive development.

---

### Tier 3: Deep Integrations (16-32 hours)

#### 3.1 Full Curriculum MOO + k(L) Pipeline
Complete integration of MOO search with k(L) shaped results.

#### 3.2 Eudaimonia + Archetype Weight Learning
MOO-optimized archetype weights that learn from interaction outcomes.

#### 3.3 Dream Consolidation Quality Gates
Spectral gap-based quality gates for memory consolidation.

---

## Implementation Priority Matrix

| Integration | MOO Impact | Meta-Calc Impact | Synergy | Effort | Priority |
|-------------|------------|------------------|---------|--------|----------|
| Training Loop k(mastery) | LOW | HIGH | MEDIUM | 4h | **P1** |
| Prompt Baking k(strength) | LOW | HIGH | HIGH | 4h | **P1** |
| Assessment MOO | HIGH | MEDIUM | MEDIUM | 8h | **P2** |
| Virtue Weight MOO | HIGH | LOW | MEDIUM | 8h | **P2** |
| Curriculum MOO | HIGH | HIGH | VERY HIGH | 16h | **P2** |
| OODA Action MOO | HIGH | LOW | LOW | 8h | **P3** |
| Self-Modeling Temp MOO | MEDIUM | HIGH | MEDIUM | 12h | **P3** |

---

## File Modification Summary

### Already Modified (Quick Wins)

| File | Changes |
|------|---------|
| `training_loop.py` | +META_CALCULUS_AVAILABLE, +MetaGrokfast optimizer |
| `curriculum_engine.py` | +gap_monitor, +gap_history, +spectral gap advancement check, +_compute_spectral_gap() |
| `dream_consolidation.py` | +level/total_levels params, +k(L) sample scaling |
| `curriculum_generator.py` | +META_CALCULUS_AVAILABLE, +k(L) difficulty curve in _map_to_original_difficulty() |
| `self_modeling.py` | +gap_monitor, +gap_history, +spectral gap epoch monitoring, +_compute_spectral_gap() |

### To Be Modified (Tier 1-2)

| File | Planned Changes |
|------|-----------------|
| `training_loop.py` | +_get_mastery_threshold(), +_get_max_hints() |
| `curriculum_engine.py` | +_get_baking_config() with k(level) |
| `assessment.py` | +EdgeOfChaosAssessment with MOO |
| `eudaimonia/rules.py` | +AdaptiveEudaimoniaRules with weight MOO |
| `curriculum_generator.py` | +MOOCurriculumGenerator |
| `eudaimonia/ooda_loop.py` | +MOOOODALoop with action selection MOO |
| `self_modeling.py` | +MOOSelfModelingTrainer with temperature MOO |

---

## Testing Strategy

### Unit Tests
```python
# tests/test_phase5_meta_integration.py

def test_metagrokfast_optimizer_used():
    """Verify MetaGrokfast is used when available."""
    loop = CurriculumTrainingLoop()
    # Mock META_CALCULUS_AVAILABLE = True
    optimizer = loop._create_optimizer(model)
    assert "MetaGrokfast" in type(optimizer).__name__

def test_k_difficulty_curve():
    """Verify k(L) creates non-linear difficulty."""
    gen = AdaptiveCurriculumGenerator(baseline_level=30, num_levels=10)
    difficulties = [gen._map_to_original_difficulty(i) for i in range(1, 11)]

    # Should NOT be linear
    linear_diff = [30 + (i-1) * 7.78 for i in range(1, 11)]
    assert difficulties != linear_diff

def test_spectral_gap_advancement():
    """Verify spectral gap affects level advancement."""
    engine = CurriculumEngine()
    engine.gap_history = [0.3, 0.3, 0.3]  # Stable
    assert meta_phase5.should_advance_stage(engine.gap_history, 1, 10)

    engine.gap_history = [0.3, 0.1, 0.02]  # Collapsing
    assert not meta_phase5.should_advance_stage(engine.gap_history, 1, 10)

def test_dream_sample_scaling():
    """Verify k(level) scales dream samples."""
    early = DreamConsolidator(num_samples=1000, level=1, total_levels=10)
    late = DreamConsolidator(num_samples=1000, level=10, total_levels=10)

    # Later levels should have MORE samples
    assert late.num_samples > early.num_samples
```

### Integration Tests
```python
def test_full_level_with_meta_calculus():
    """Test full level progression with all integrations."""
    engine = CurriculumEngine()
    model = create_test_model()
    tokenizer = create_test_tokenizer()

    result = engine.run(model, tokenizer)

    # Verify integrations were active
    assert len(engine.gap_history) > 0
    assert result.metrics.get("meta_calculus_enabled", False)
```

---

## Rollout Plan

### Phase 1: Validation (Week 1)
- Run existing tests with Quick Win integrations
- Verify no regressions
- Benchmark training speed with MetaGrokfast

### Phase 2: Tier 1 Implementation (Week 2)
- Implement k(mastery) in training_loop.py
- Implement k(baking_strength) in curriculum_engine.py
- Unit tests for each

### Phase 3: Tier 2 Implementation (Weeks 3-4)
- Assessment MOO
- Virtue Weight MOO
- Curriculum MOO

### Phase 4: Integration Testing (Week 5)
- End-to-end curriculum runs
- Performance benchmarking
- A/B comparison with/without meta-calculus

---

## Success Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Training convergence | 50 epochs/level | 35 epochs/level | Mean epochs to 90% accuracy |
| Representation health | Not tracked | >0.1 spectral gap | Gap never falls below threshold |
| Curriculum efficiency | Linear difficulty | Smoother learning curve | Variance in per-level accuracy |
| Level advancement | Fixed 75% threshold | Adaptive + stable | Fewer stuck levels |
| Memory retention | Not measured | 67% improvement | Post-consolidation test accuracy |

---

## Appendix: Import Patterns

All Phase 5 files should use this pattern:

```python
# At top of file, after standard library imports

# Import meta-calculus integration
try:
    from src.cross_phase.meta_calculus.phase_facades import phase5 as meta_phase5

    META_CALCULUS_AVAILABLE = True
except ImportError:
    META_CALCULUS_AVAILABLE = False
```

Then guard all meta-calculus usage:

```python
if META_CALCULUS_AVAILABLE:
    # Use meta-calculus features
    optimizer = meta_phase5.create_optimizer(model)
else:
    # Fallback to standard approach
    optimizer = torch.optim.AdamW(model.parameters())
```

This ensures Phase 5 works standalone but gains enhancements when meta-calculus is available.

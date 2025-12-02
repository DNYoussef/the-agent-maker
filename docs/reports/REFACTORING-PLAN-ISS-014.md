# ISS-014: Large File Refactoring Plan

## Overview

**Objective**: Split large files (>300 LOC) into smaller, focused modules
**Total Files to Refactor**: 10 priority files (from 36 identified)
**Estimated Effort**: 8-10 hours
**Strategy**: Parallel agent execution for independent files

---

## Priority 1: CRITICAL (phase_controller.py - 1,198 LOC)

### Current State
- Single file with 8 phase controllers + base class
- Each controller is 100-180 lines
- Hard to navigate, test, and maintain

### Target State
```
src/cross_phase/orchestrator/
    __init__.py              # Exports all controllers
    base_controller.py       # PhaseResult + PhaseController ABC (~80 LOC)
    phase1_controller.py     # Phase1Controller (~150 LOC)
    phase2_controller.py     # Phase2Controller (~120 LOC)
    phase3_controller.py     # Phase3Controller (~180 LOC)
    phase4_controller.py     # Phase4Controller (~150 LOC)
    phase5_controller.py     # Phase5Controller (~180 LOC)
    phase6_controller.py     # Phase6Controller (~170 LOC)
    phase7_controller.py     # Phase7Controller (~150 LOC)
    phase8_controller.py     # Phase8Controller (~160 LOC)
```

### Execution Steps
1. Create `base_controller.py` with PhaseResult + PhaseController ABC
2. Extract each PhaseNController to separate file
3. Update imports in each controller file
4. Create `__init__.py` that re-exports all controllers
5. Update any imports in other files

---

## Priority 2: HIGH (Phase 3 Architecture - 626 LOC)

### Current State
- `phase3_quietstar/architecture.py` has 5 neural network classes
- Mixed responsibilities: generation, scoring, mixing, injection

### Target State
```
src/phase3_quietstar/
    architecture/
        __init__.py           # Exports QuietSTaRModel
        dataclasses.py        # ThoughtOutput, CoherenceScores (~40 LOC)
        thought_generator.py  # ThoughtGenerator class (~150 LOC)
        coherence_scorer.py   # CoherenceScorer class (~140 LOC)
        mixing_head.py        # MixingHead class (~80 LOC)
        thought_injector.py   # ThoughtInjector class (~100 LOC)
        quiet_star_model.py   # QuietSTaRModel wrapper (~100 LOC)
```

---

## Priority 3: HIGH (Phase 5 Curriculum Engine - 536 LOC)

### Current State
- `phase5_curriculum/curriculum_engine.py` mixes config, engine, and results

### Target State
```
src/phase5_curriculum/
    engine/
        __init__.py
        config.py             # CurriculumConfig, SpecializationType (~80 LOC)
        progress.py           # LevelProgress tracking (~60 LOC)
        result.py             # Phase5Result (~40 LOC)
        curriculum_engine.py  # Main CurriculumEngine (~200 LOC)
```

---

## Priority 4: MODERATE (Phase 7 ADAS - 485 LOC)

### Current State
- `phase7_experts/adas_optimizer.py` has NSGA-II + routing + evaluation

### Target State
```
src/phase7_experts/
    adas/
        __init__.py
        config.py             # ADASConfig, Individual (~60 LOC)
        nsga2.py              # Ranking, crowding, selection (~150 LOC)
        operators.py          # Crossover, mutation (~100 LOC)
        evaluation.py         # Fitness evaluation (~80 LOC)
        adas_optimizer.py     # Main orchestrator (~150 LOC)
```

---

## Priority 5: MODERATE (Titans-MAG - 464 LOC)

### Current State
- `phase1_cognate/model/titans_mag.py` has 7 classes

### Target State
```
src/phase1_cognate/model/
    components/
        __init__.py
        normalization.py      # RMSNorm (~30 LOC)
        mlp.py                # SwiGLUMLP (~40 LOC)
        attention.py          # SlidingWindowAttention (~150 LOC)
        memory.py             # LongTermMemory (~80 LOC)
        gating.py             # MAGGate (~80 LOC)
    titans_mag.py             # TitansMAGLayer + TitansMAGBackbone (~150 LOC)
```

---

## Execution Strategy

### Parallel Streams (can run concurrently)

**Stream 1**: phase_controller.py split (CRITICAL)
- Agent: code-analyzer + coder
- Files: 9 new files in orchestrator/

**Stream 2**: architecture.py split (HIGH)
- Agent: coder
- Files: 7 new files in phase3_quietstar/architecture/

**Stream 3**: curriculum_engine.py + adas_optimizer.py (HIGH/MODERATE)
- Agent: coder
- Files: 5+5 new files

**Stream 4**: titans_mag.py (MODERATE)
- Agent: coder
- Files: 6 new files in components/

### Dependencies
- Stream 1 must complete before updating imports elsewhere
- Streams 2-4 are independent and can run in parallel

---

## Agent Assignments

| Task | Agent Type | Skill | Estimated Time |
|------|------------|-------|----------------|
| Split phase_controller.py | coder | sparc-methodology | 2h |
| Split architecture.py | coder | - | 1.5h |
| Split curriculum_engine.py | coder | - | 1h |
| Split adas_optimizer.py | coder | - | 1h |
| Split titans_mag.py | coder | - | 1h |
| Update imports | code-analyzer | - | 1h |
| Verify tests pass | tester | - | 1h |

**Total**: ~8.5 hours (parallelized to ~4 hours with 3 streams)

---

## Backward Compatibility

All refactored modules will maintain backward compatibility through `__init__.py` re-exports:

```python
# src/cross_phase/orchestrator/__init__.py
from .base_controller import PhaseResult, PhaseController
from .phase1_controller import Phase1Controller
from .phase2_controller import Phase2Controller
# ... etc

# Existing imports will continue to work:
# from cross_phase.orchestrator.phase_controller import Phase1Controller
```

---

## Validation Checklist

- [ ] All tests pass after refactoring
- [ ] No circular imports
- [ ] Each new file < 200 LOC
- [ ] `__init__.py` exports maintain compatibility
- [ ] Import statements updated throughout codebase

---

## Rollback Strategy

If issues arise:
1. Git revert to pre-refactoring commit
2. Original files preserved as `*.py.backup`
3. Incremental commits per file split

---

*Plan created: 2025-11-26*
*Estimated completion: 4-8 hours with parallel execution*

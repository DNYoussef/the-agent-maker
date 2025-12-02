# Phase 2 (EvoMerge) Implementation Checklist

**Status**: Ready to Begin
**Start Date**: TBD
**Target Completion**: 4 weeks (20 days)
**Last Updated**: 2025-10-16

---

## Quick Reference

**Plan Document**: [`PHASE2_IMPLEMENTATION_PLAN.md`](./PHASE2_IMPLEMENTATION_PLAN.md)
**Files to Create**: 37 (26 production + 10 tests + 1 UI)
**Lines of Code**: ~4,500 LOC
**Tests Required**: 91+ tests
**Success Criteria**: ≥20% fitness improvement, ≤90 min evolution, <6GB VRAM

---

## Week 1: Core Merge Techniques (Days 1-5)

### Day 1: Interpolation Methods
- [ ] Create `src/phase2_evomerge/merge/__init__.py`
- [ ] Implement `src/phase2_evomerge/merge/linear_merge.py` (~80 LOC)
  - [ ] LinearMerge class with `merge()` method
  - [ ] Weighted average: `0.33*m1 + 0.33*m2 + 0.33*m3`
- [ ] Implement `src/phase2_evomerge/merge/slerp_merge.py` (~120 LOC)
  - [ ] SLERPMerge class with spherical interpolation
  - [ ] Edge case: θ=0 fallback to linear
- [ ] Create `tests/unit/test_merge_techniques.py` (6 tests for interpolation)
  - [ ] `test_linear_identical_models()`
  - [ ] `test_linear_random_models()`
  - [ ] `test_linear_opposite_models()`
  - [ ] `test_slerp_identical_fallback()`
  - [ ] `test_slerp_orthogonal_models()`
  - [ ] `test_slerp_magnitude_preservation()`
- [ ] Run tests: `pytest tests/unit/test_merge_techniques.py -k "linear or slerp" -v`
- [ ] **Deliverable**: 2 interpolation techniques working

### Day 2: Task Arithmetic Methods
- [ ] Implement `src/phase2_evomerge/merge/dare_merge.py` (~100 LOC)
  - [ ] DAREMerge class
  - [ ] Drop 90% randomly, rescale by 10×
- [ ] Implement `src/phase2_evomerge/merge/ties_merge.py` (~130 LOC)
  - [ ] TIESMerge class
  - [ ] Trim top 20%, vote on signs, merge matching
- [ ] Add 6 tests to `test_merge_techniques.py` (task arithmetic)
  - [ ] `test_dare_stochasticity()`
  - [ ] `test_dare_sparsity()`
  - [ ] `test_dare_rescaling()`
  - [ ] `test_ties_sign_voting()`
  - [ ] `test_ties_trimming()`
  - [ ] `test_ties_conflict_resolution()`
- [ ] Run tests: `pytest tests/unit/test_merge_techniques.py -k "dare or ties" -v`
- [ ] **Deliverable**: 4 techniques total (Linear, SLERP, DARE, TIES)

### Day 3: Selection Methods
- [ ] Implement `src/phase2_evomerge/merge/frankenmerge.py` (~110 LOC)
  - [ ] FrankenMerge class
  - [ ] Layer-wise selection from best model
- [ ] Implement `src/phase2_evomerge/merge/dfs_merge.py` (~100 LOC)
  - [ ] DFSMerge class
  - [ ] Inverse-variance weighting
- [ ] Add 6 tests to `test_merge_techniques.py` (selection methods)
  - [ ] `test_frankenmerge_layer_selection()`
  - [ ] `test_frankenmerge_dimension_compatibility()`
  - [ ] `test_frankenmerge_abc_pattern()`
  - [ ] `test_dfs_variance_weighting()`
  - [ ] `test_dfs_stable_features_prioritized()`
  - [ ] `test_dfs_computation()`
- [ ] Run tests: `pytest tests/unit/test_merge_techniques.py -k "franken or dfs" -v`
- [ ] **Deliverable**: All 6 techniques implemented

### Day 4: Binary Combination Pipeline
- [ ] Update `src/phase2_evomerge/merge/__init__.py` with `MergeTechniques` class
  - [ ] `apply_combo(models, combo_id)` - Sequential 3-stage pipeline
  - [ ] Decode binary combo (bit 0=interp, bit 1=task, bit 2=select)
- [ ] Add 3 tests to `test_merge_techniques.py` (binary combinations)
  - [ ] `test_all_8_combos_unique()`
  - [ ] `test_sequential_pipeline()`
  - [ ] `test_combo_000_vs_111()`
- [ ] Run all merge tests: `pytest tests/unit/test_merge_techniques.py -v`
- [ ] **Deliverable**: 8 binary combos working

### Day 5: Week 1 Review
- [ ] Run all merge tests: `pytest tests/unit/test_merge_techniques.py -v`
- [ ] Check test coverage: `pytest tests/unit/test_merge_techniques.py --cov=src/phase2_evomerge/merge --cov-report=term-missing`
- [ ] Target: ≥98% coverage
- [ ] NASA POT10 check: `python scripts/check_function_length.py src/phase2_evomerge/merge/*.py`
- [ ] mypy check: `mypy src/phase2_evomerge/merge --strict`
- [ ] Code review & refactoring
- [ ] **Milestone 1**: ✅ All 6 merge techniques working, 18+ tests passing

---

## Week 2: Fitness Evaluation & Population Management (Days 6-10)

### Day 6: Fitness Evaluator Core
- [ ] Create `src/phase2_evomerge/fitness/__init__.py`
- [ ] Implement `src/phase2_evomerge/fitness/evaluator.py` (~150 LOC)
  - [ ] `FitnessEvaluator` class
  - [ ] Composite fitness: `0.4*ppl + 0.3*acc + 0.2*speed + 0.1*mem`
  - [ ] `CompositeScore` dataclass
- [ ] Implement `src/phase2_evomerge/fitness/perplexity.py` (~100 LOC)
  - [ ] `PerplexityCalculator` class
  - [ ] Cross-entropy → exp(loss)
- [ ] Create `tests/unit/test_fitness_evaluation.py`
- [ ] Add 4 tests: `test_perplexity_calculation()`, `test_composite_formula()`, `test_fitness_caching()`, `test_nan_handling()`
- [ ] Run tests: `pytest tests/unit/test_fitness_evaluation.py -v`
- [ ] **Deliverable**: Fitness evaluator + perplexity working

### Day 7: Fitness Components
- [ ] Implement `src/phase2_evomerge/fitness/accuracy.py` (~90 LOC)
  - [ ] `AccuracyCalculator` class
  - [ ] Task-specific accuracy (0-1 range)
- [ ] Implement `src/phase2_evomerge/fitness/speed.py` (~110 LOC)
  - [ ] `SpeedBenchmark` class
  - [ ] Tokens per second calculation
- [ ] Implement `src/phase2_evomerge/fitness/memory.py` (~80 LOC)
  - [ ] `MemoryProfiler` class
  - [ ] Peak VRAM tracking
- [ ] Add 8 tests to `test_fitness_evaluation.py`
  - [ ] `test_accuracy_scoring()`
  - [ ] `test_speed_benchmarking()`
  - [ ] `test_memory_profiling()`
  - [ ] `test_composite_weighting()`
  - [ ] `test_normalization()`
  - [ ] `test_edge_cases()`
  - [ ] `test_fitness_comparison()`
  - [ ] `test_parallel_evaluation()`
- [ ] Run tests: `pytest tests/unit/test_fitness_evaluation.py -v`
- [ ] **Deliverable**: All 4 fitness components working

### Day 8: Population Manager
- [ ] Create `src/phase2_evomerge/population/__init__.py`
- [ ] Implement `src/phase2_evomerge/population/manager.py` (~180 LOC)
  - [ ] `PopulationManager` class
  - [ ] `initialize_generation_0()` - Create 8 binary combos
  - [ ] `evolve_generation()` - Elite mutation + loser merging
- [ ] Create `tests/unit/test_population_management.py`
- [ ] Add 8 tests:
  - [ ] `test_generation_0_initialization()`
  - [ ] `test_elite_preservation()`
  - [ ] `test_elite_mutation()`
  - [ ] `test_loser_merging()`
  - [ ] `test_population_size_consistency()`
  - [ ] `test_memory_limit_enforcement()`
  - [ ] `test_model_tracking()`
  - [ ] `test_generation_evolution()`
- [ ] Run tests: `pytest tests/unit/test_population_management.py -v`
- [ ] **Deliverable**: Population manager working

### Day 9: Genetic Operations & Diversity
- [ ] Implement `src/phase2_evomerge/population/genetic_ops.py` (~120 LOC)
  - [ ] `GeneticOperations` class
  - [ ] `mutate()` - Gaussian noise (sigma=0.01, rate=0.01)
- [ ] Implement `src/phase2_evomerge/population/diversity.py` (~100 LOC)
  - [ ] `DiversityTracker` class
  - [ ] `compute_diversity()` - Pairwise L2 distances
  - [ ] `check_and_reseed()` - Re-seed if diversity <0.2
- [ ] Create `tests/unit/test_diversity.py`
- [ ] Add 10 tests to `test_diversity.py`:
  - [ ] `test_pairwise_distance_computation()`
  - [ ] `test_diversity_threshold_detection()`
  - [ ] `test_reseed_triggers()`
  - [ ] `test_diversity_increases_after_reseed()`
  - [ ] `test_diversity_range()`
  - [ ] `test_diversity_warning_threshold()`
  - [ ] `test_diversity_critical_threshold()`
  - [ ] `test_diversity_history_tracking()`
  - [ ] `test_identical_models_zero_diversity()`
  - [ ] `test_random_models_high_diversity()`
- [ ] Add 7 tests for genetic operations:
  - [ ] `test_mutation_applies_noise()`
  - [ ] `test_mutation_rate_correctness()`
  - [ ] `test_mutation_sigma_correctness()`
  - [ ] `test_mutation_preserves_architecture()`
  - [ ] `test_mutation_stochasticity()`
  - [ ] `test_mutation_magnitude()`
  - [ ] `test_mutation_distribution()`
- [ ] Run tests: `pytest tests/unit/test_diversity.py tests/unit/test_genetic_operations.py -v`
- [ ] **Deliverable**: Genetic ops + diversity management working

### Day 10: Week 2 Review
- [ ] Run all Week 2 tests: `pytest tests/unit/test_fitness_evaluation.py tests/unit/test_population_management.py tests/unit/test_diversity.py -v`
- [ ] Check coverage: `pytest tests/unit --cov=src/phase2_evomerge/fitness --cov=src/phase2_evomerge/population --cov-report=term-missing`
- [ ] Target: ≥95% coverage
- [ ] NASA POT10 check: `python scripts/check_function_length.py src/phase2_evomerge/fitness/*.py src/phase2_evomerge/population/*.py`
- [ ] mypy check: `mypy src/phase2_evomerge/fitness src/phase2_evomerge/population --strict`
- [ ] **Milestone 2**: ✅ Fitness + population systems complete, 37+ tests passing

---

## Week 3: Evolution Engine & Integration (Days 11-15)

### Day 11: Evolution Engine Core
- [ ] Create `src/phase2_evomerge/evolution/__init__.py`
- [ ] Implement `src/phase2_evomerge/evolution/engine.py` (~250 LOC)
  - [ ] `EvolutionEngine` class
  - [ ] `run()` - Main 50-generation loop
  - [ ] Generation 0 initialization
  - [ ] Per-generation: evaluate, update champion, check convergence, evolve, check diversity
- [ ] Add 5 tests to `tests/integration/test_phase2_evolution.py`:
  - [ ] `test_generation_0_creates_8_models()`
  - [ ] `test_evolution_updates_champion()`
  - [ ] `test_fitness_improves_over_generations()`
  - [ ] `test_diversity_maintained()`
  - [ ] `test_evolution_completes()`
- [ ] Run tests: `pytest tests/integration/test_phase2_evolution.py -v`
- [ ] **Deliverable**: Evolution engine core working

### Day 12: Convergence & Checkpointing
- [ ] Implement `src/phase2_evomerge/evolution/convergence.py` (~80 LOC)
  - [ ] `ConvergenceDetector` class
  - [ ] `should_stop()` - Check if improvement <0.1% for 5 gens
- [ ] Implement `src/phase2_evomerge/evolution/checkpoint.py` (~120 LOC)
  - [ ] `CheckpointManager` class
  - [ ] `save()` - Save generation, population, champion
  - [ ] `resume()` - Load checkpoint and continue
- [ ] Add 4 tests:
  - [ ] `test_convergence_detection()`
  - [ ] `test_early_stopping()`
  - [ ] `test_checkpoint_save()`
  - [ ] `test_checkpoint_resume()`
- [ ] Run tests: `pytest tests/integration/test_phase2_evolution.py -k "convergence or checkpoint" -v`
- [ ] **Deliverable**: Convergence + checkpointing working

### Day 13: Phase 2 Pipeline
- [ ] Create `src/phase2_evomerge/phase2_pipeline.py` (~200 LOC)
  - [ ] `Phase2Pipeline` class
  - [ ] `run(session_id)` - Main orchestrator
  - [ ] `load_phase1_models()` - Load 3 models from Phase 1
  - [ ] `validate_phase1_models()` - Check inputs
  - [ ] `validate_champion()` - Check ≥20% improvement
  - [ ] `save_champion()` - Save to storage
- [ ] Create `tests/integration/test_phase1_to_phase2_handoff.py`
- [ ] Add 3 tests:
  - [ ] `test_load_phase1_models()`
  - [ ] `test_validate_correct_models()`
  - [ ] `test_validate_incorrect_models_raises()`
- [ ] Run tests: `pytest tests/integration/test_phase1_to_phase2_handoff.py -v`
- [ ] **Deliverable**: Phase 2 pipeline orchestrator working

### Day 14: Phase Handoff Tests
- [ ] Create `tests/integration/test_phase2_output.py`
- [ ] Add 3 tests:
  - [ ] `test_champion_saved_correctly()`
  - [ ] `test_improvement_validation()`
  - [ ] `test_output_format()`
- [ ] Add 1 test to `test_phase2_evolution.py`:
  - [ ] `test_full_50_generation_evolution()`
- [ ] Run all integration tests: `pytest tests/integration -m phase2 -v`
- [ ] **Deliverable**: Phase 1→2→3 handoff validated

### Day 15: Week 3 Review
- [ ] Run end-to-end test with dummy Phase 1 models
- [ ] Verify evolution completes in reasonable time (~5 min for dummy data)
- [ ] Check champion fitness improves
- [ ] Verify diversity maintained throughout
- [ ] Run all integration tests: `pytest tests/integration -m phase2 -v`
- [ ] Target: 14+ integration tests passing
- [ ] NASA POT10 check: `python scripts/check_function_length.py src/phase2_evomerge/evolution/*.py src/phase2_evomerge/phase2_pipeline.py`
- [ ] **Milestone 3**: ✅ Full evolution pipeline working

---

## Week 4: W&B, UI, Performance (Days 16-20)

### Day 16: W&B Logger
- [ ] Create `src/phase2_evomerge/monitoring/__init__.py`
- [ ] Implement `src/phase2_evomerge/monitoring/wandb_logger.py` (~200 LOC)
  - [ ] `WandBLogger` class
  - [ ] `log_generation()` - Log 17 metrics per generation
  - [ ] `log_final()` - Log 5 final metrics
  - [ ] Per-model logging (6 metrics × 8 models)
  - [ ] Artifact saving (champion model)
- [ ] Create `tests/integration/test_phase2_wandb.py`
- [ ] Add 4 tests:
  - [ ] `test_370_metrics_logged()`
  - [ ] `test_artifacts_saved()`
  - [ ] `test_config_logged()`
  - [ ] `test_metrics_per_generation()`
- [ ] Run tests: `pytest tests/integration/test_phase2_wandb.py -v`
- [ ] **Deliverable**: W&B logging working (370 metrics)

### Day 17: W&B Integration Testing
- [ ] Run full evolution with W&B logging enabled
- [ ] Verify all 370 metrics logged to W&B
- [ ] Check W&B dashboard displays correctly
- [ ] Verify artifacts saved (champion.pt, evolution-log.json)
- [ ] Test offline mode (local W&B)
- [ ] **Deliverable**: W&B integration validated

### Day 18: UI Backend (API Endpoints)
- [ ] Add 7 API endpoints to `src/ui/api.py`:
  - [ ] `POST /api/phases/2/configure`
  - [ ] `POST /api/phases/2/start`
  - [ ] `GET /api/phases/2/status`
  - [ ] `GET /api/phases/2/population`
  - [ ] `POST /api/phases/2/pause`
  - [ ] `POST /api/phases/2/intervene`
  - [ ] `GET /api/phases/2/results`
- [ ] Add 4 WebSocket events to `src/ui/websocket.py`:
  - [ ] `phase:progress` (every generation)
  - [ ] `phase:metric` (detailed metrics)
  - [ ] `phase:alert` (low diversity, convergence)
  - [ ] `phase:complete` (final results)
- [ ] Test API endpoints with curl/Postman
- [ ] **Deliverable**: UI backend working

### Day 19: UI Frontend
- [ ] Implement `src/ui/pages/phase2_evolution.py` (~200 LOC)
  - [ ] Evolution monitor (fitness curve, generation counter, ETA)
  - [ ] 3D merge visualization (Three.js integration)
  - [ ] Combo statistics panel (8 binary combos usage)
  - [ ] Population diversity chart
  - [ ] Intervention modal (manual diversity boost)
- [ ] Test UI with dummy evolution
- [ ] Verify real-time updates work
- [ ] Test WebSocket events fire correctly
- [ ] **Deliverable**: UI functional

### Day 20: Performance Testing & Final Validation
- [ ] Create `tests/performance/test_phase2_performance.py`
- [ ] Add 2 tests:
  - [ ] `test_90_minute_benchmark()` - Full 50-gen evolution on GTX 1660
  - [ ] `test_parallel_evaluation_speedup()` - 4 workers vs 1 worker
- [ ] Create `tests/performance/test_phase2_memory.py`
- [ ] Add 2 tests:
  - [ ] `test_vram_usage_under_6gb()` - Peak VRAM <6GB
  - [ ] `test_memory_cleanup()` - Only 12 models in memory
- [ ] Run performance tests: `pytest tests/performance -m phase2 -v --slow`
- [ ] Run all tests: `pytest tests/ -m phase2 -v`
- [ ] Check coverage: `pytest tests/ --cov=src/phase2_evomerge --cov-report=html`
- [ ] Target: ≥90% overall coverage
- [ ] Final NASA POT10 check: `python scripts/check_function_length.py src/phase2_evomerge/**/*.py`
- [ ] Final mypy check: `mypy src/phase2_evomerge --strict`
- [ ] Documentation update:
  - [ ] Update `README.md` with Phase 2 usage
  - [ ] Generate API docs: `sphinx-build -b html docs/ docs/_build/`
- [ ] **Milestone 4**: ✅ **PHASE 2 PRODUCTION READY**

---

## Final Validation Checklist

### Technical Validation
- [ ] All 6 merge techniques implemented and tested
- [ ] 8 binary combinations working
- [ ] Fitness improvement ≥20% (target: 23.5%) on real Phase 1 models
- [ ] Evolution completes in ≤90 minutes on GTX 1660
- [ ] Memory usage <6GB VRAM throughout
- [ ] Diversity maintained >0.3 across generations
- [ ] Early stopping triggers correctly
- [ ] All 8 combos used at least once

### Testing Validation
- [ ] 91+ tests passing (73 unit + 14 integration + 4 performance)
- [ ] ≥90% overall coverage
- [ ] ≥95% coverage for critical paths (merge, fitness, evolution)
- [ ] NASA POT10 compliance (all functions ≤60 LOC)
- [ ] mypy strict passes (no type errors)
- [ ] All CI/CD quality gates pass

### Integration Validation
- [ ] Phase 1→2 handoff: Load 3 TRM × Titans-MAG models successfully
- [ ] Input validation: Exactly 3 models, ~25M params each, <6GB VRAM
- [ ] Phase 2→3 handoff: Output champion with ≥20% improvement
- [ ] 370 W&B metrics logged correctly (3,255 total logs)
- [ ] UI displays real-time progress
- [ ] 7 API endpoints working
- [ ] 4 WebSocket events firing
- [ ] Model registry tracks all 12 models
- [ ] Champion saved to storage with correct metadata

### Performance Validation
- [ ] 50 generations complete in 90 minutes (typical: 35-40 converge)
- [ ] VRAM peak <6GB (validated on GTX 1660)
- [ ] Parallel evaluation works (4 workers, ≥2.0× speedup)
- [ ] Fitness caching speeds up evaluation
- [ ] Diversity re-seeding works (triggered at <0.2)
- [ ] Checkpoint save/resume works

### Documentation Validation
- [ ] All functions have docstrings (≥95%)
- [ ] README.md updated with Phase 2 usage examples
- [ ] API documentation generated (Sphinx)
- [ ] GraphViz diagrams updated
- [ ] Implementation plan complete

---

## File Tracking

### Production Files (26 files, ~2,500 LOC)

**Merge** (7 files, ~690 LOC):
- [ ] `src/phase2_evomerge/merge/__init__.py` (50 LOC)
- [ ] `src/phase2_evomerge/merge/linear_merge.py` (80 LOC)
- [ ] `src/phase2_evomerge/merge/slerp_merge.py` (120 LOC)
- [ ] `src/phase2_evomerge/merge/dare_merge.py` (100 LOC)
- [ ] `src/phase2_evomerge/merge/ties_merge.py` (130 LOC)
- [ ] `src/phase2_evomerge/merge/frankenmerge.py` (110 LOC)
- [ ] `src/phase2_evomerge/merge/dfs_merge.py` (100 LOC)

**Fitness** (6 files, ~580 LOC):
- [ ] `src/phase2_evomerge/fitness/__init__.py` (50 LOC)
- [ ] `src/phase2_evomerge/fitness/evaluator.py` (150 LOC)
- [ ] `src/phase2_evomerge/fitness/perplexity.py` (100 LOC)
- [ ] `src/phase2_evomerge/fitness/accuracy.py` (90 LOC)
- [ ] `src/phase2_evomerge/fitness/speed.py` (110 LOC)
- [ ] `src/phase2_evomerge/fitness/memory.py` (80 LOC)

**Population** (4 files, ~450 LOC):
- [ ] `src/phase2_evomerge/population/__init__.py` (50 LOC)
- [ ] `src/phase2_evomerge/population/manager.py` (180 LOC)
- [ ] `src/phase2_evomerge/population/genetic_ops.py` (120 LOC)
- [ ] `src/phase2_evomerge/population/diversity.py` (100 LOC)

**Evolution** (4 files, ~500 LOC):
- [ ] `src/phase2_evomerge/evolution/__init__.py` (50 LOC)
- [ ] `src/phase2_evomerge/evolution/engine.py` (250 LOC)
- [ ] `src/phase2_evomerge/evolution/convergence.py` (80 LOC)
- [ ] `src/phase2_evomerge/evolution/checkpoint.py` (120 LOC)

**Monitoring** (2 files, ~250 LOC):
- [ ] `src/phase2_evomerge/monitoring/__init__.py` (50 LOC)
- [ ] `src/phase2_evomerge/monitoring/wandb_logger.py` (200 LOC)

**Core** (3 files, ~330 LOC):
- [ ] `src/phase2_evomerge/__init__.py` (50 LOC)
- [ ] `src/phase2_evomerge/phase2_pipeline.py` (200 LOC)
- [ ] `src/phase2_evomerge/config.py` (80 LOC)

### Test Files (10 files, ~1,800 LOC)

**Unit Tests** (6 files, ~1,250 LOC):
- [ ] `tests/unit/test_merge_techniques.py` (400 LOC, 18 tests)
- [ ] `tests/unit/test_fitness_evaluation.py` (300 LOC, 12 tests)
- [ ] `tests/unit/test_population_management.py` (250 LOC, 15 tests)
- [ ] `tests/unit/test_diversity.py` (200 LOC, 10 tests)
- [ ] `tests/unit/test_genetic_operations.py` (200 LOC, 12 tests)
- [ ] `tests/unit/test_error_handling.py` (100 LOC, 6 tests)

**Integration Tests** (4 files, ~620 LOC):
- [ ] `tests/integration/test_phase2_evolution.py` (200 LOC, 4 tests)
- [ ] `tests/integration/test_phase1_to_phase2_handoff.py` (150 LOC, 3 tests)
- [ ] `tests/integration/test_phase2_output.py` (120 LOC, 3 tests)
- [ ] `tests/integration/test_phase2_wandb.py` (150 LOC, 4 tests)

**Performance Tests** (2 files, ~250 LOC):
- [ ] `tests/performance/test_phase2_performance.py` (150 LOC, 2 tests)
- [ ] `tests/performance/test_phase2_memory.py` (100 LOC, 2 tests)

### UI Files (1 file, ~200 LOC)
- [ ] `src/ui/pages/phase2_evolution.py` (200 LOC)

**Total Progress**: 0/37 files created, 0/4,500 LOC written, 0/91 tests passing

---

## Quick Commands

### Running Tests
```bash
# All Phase 2 tests
pytest tests/ -m phase2 -v

# Unit tests only (fast)
pytest tests/unit -m phase2 -v

# Integration tests
pytest tests/integration -m phase2 -v

# Performance tests (slow)
pytest tests/performance -m phase2 -v --slow

# Coverage report
pytest tests/ --cov=src/phase2_evomerge --cov-report=html
open htmlcov/index.html
```

### Code Quality Checks
```bash
# NASA POT10 check (≤60 LOC/function)
python scripts/check_function_length.py src/phase2_evomerge/**/*.py

# Type checking
mypy src/phase2_evomerge --strict

# Formatting
black src/phase2_evomerge tests/
isort src/phase2_evomerge tests/

# Linting
flake8 src/phase2_evomerge
pylint src/phase2_evomerge --exit-zero
```

### Running Phase 2
```bash
# Via Python
python -m src.phase2_evomerge.phase2_pipeline --session-id my_run

# Via orchestrator
python -m src.cross_phase.orchestrator.pipeline --phase 2 --session-id my_run

# With UI
streamlit run src/ui/app.py
```

---

## Notes

- **Daily Progress**: Update checkboxes daily
- **Blockers**: Document any blockers immediately
- **Changes**: Note any deviations from plan
- **Performance**: Track actual vs. expected times

**Remember**: This is a 4-week plan. Stay focused, test frequently, commit often! 🚀

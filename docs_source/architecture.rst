Architecture
============

Overview
--------

Agent Forge V2 implements an 8-phase pipeline for creating small, efficient AI agents.

.. image:: _static/pipeline_overview.png
   :alt: Pipeline Overview
   :align: center

8-Phase Pipeline
---------------

Phase 1: Cognate (TRM × Titans-MAG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creates 3 specialized 25M parameter models:

* **Model 1**: Reasoning specialist
* **Model 2**: Memory specialist
* **Model 3**: General purpose

**Architecture**: TRM (Transformer with Routing Mechanism) × Titans-MAG

**Duration**: ~6 hours (3 models × 2 hours each)

Phase 2: EvoMerge
^^^^^^^^^^^^^^^^

50-generation evolutionary optimization using 6 merge techniques:

* Linear interpolation
* SLERP (Spherical Linear Interpolation)
* TIES (Trim, Elect, Sign, Merge)
* DARE (Drop And REscale)
* FrankenMerge (Layer stacking)
* DFS (Depth-First Search)

**Population**: 8 models per generation
**Fitness**: Task performance + diversity metrics
**Duration**: ~90 minutes

Phase 3: Quiet-STaR
^^^^^^^^^^^^^^^^^^

Reasoning enhancement via thought generation:

1. **Prompt Baking** (5 min): Bake CoT reasoning prompt
2. **RL Training**: Token-wise parallel thought sampling
3. **Coherence Scoring**: Semantic, syntactic, predictive
4. **Anti-Theater Detection**: Validate genuine reasoning

**Duration**: 5-10 hours

Phase 4: BitNet
^^^^^^^^^^^^^^

1.58-bit quantization for 8.2× compression:

* **STE (Straight-Through Estimator)**: Quantized forward, full-precision gradients
* **Target**: 8.2× compression, 3.8× speedup
* **Model size**: 95.4 MB → 11.8 MB

**Duration**: 3-5 hours

Phase 5: Curriculum Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^

7-stage adaptive curriculum:

* Edge-of-chaos assessment (75% accuracy threshold)
* 20,000 questions across 10 difficulty levels
* Tool use training with validation
* Eudaimonia baking (4-rule moral system)
* Dream consolidation (prevents catastrophic forgetting)

**Duration**: 120-240 hours

Phase 6: Tool & Persona Baking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A/B optimization loops:

* **A-Cycle**: Tool use via SWE-Bench
* **B-Cycle**: Self-guided persona generation (9 agents)
* **Half-Baking**: 50% strength per iteration
* **Plateau Detection**: Automatic cycle switching

**Duration**: 15-30 hours

Phase 7: Self-Guided Experts
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model-driven expert discovery:

1. Model analyzes own capabilities
2. Transformer² SVF training
3. Model-guided NSGA-II ADAS (5000 evaluations)

**Duration**: 78 hours

Phase 8: Final Compression
^^^^^^^^^^^^^^^^^^^^^^^^^

Triple compression pipeline:

* **SeedLM**: 2× compression
* **VPTQ**: 20× compression
* **Hypercompression**: 6.25× compression
* **Total**: 280× compression (100MB → 0.4MB)

**Quality Gates**: ≥95% retention per stage, ≥84% cumulative

Core Systems
-----------

SQLite Model Registry
^^^^^^^^^^^^^^^^^^^^

**WAL Mode**: Concurrent read/write support

.. code-block:: python

   registry = ModelRegistry()
   registry.create_session("session_001", {...})
   registry.register_model(...)
   registry.update_session_progress("session_001", "phase1", 25.0)

MuGrokfast Optimizer
^^^^^^^^^^^^^^^^^^^

Combines 3 techniques:

* **Grokfast**: EMA gradient filtering (α=0.98, λ varies by phase)
* **Muon**: Newton-Schulz orthogonalization (k=5 iterations)
* **QK-Clip**: Attention safety rails (τ=30.0)

**Phase-Specific Presets**:

* Phase 1: muon_lr=1e-3, grokfast_lambda=0.3
* Phase 3: muon_lr=5e-4, kl_coefficient=0.1 (RL stability)
* Phase 5: muon_ste_mode=True, grokfast_lambda=2.0 (BitNet STE)

Prompt Baking System
^^^^^^^^^^^^^^^^^^^^

KL divergence minimization:

.. math::

   θ_u = \arg\min D_{KL}(P_θ(·|u) || P_{θu}(·))

**Features**:

* Half-baking (50% strength)
* Sequential baking (compose prompts)
* Prompt pursuit (iterative amplification)

W&B Integration
^^^^^^^^^^^^^^

**676 Total Metrics** across 8 phases:

* Phase 1: 37 metrics
* Phase 2: 370 metrics
* Phase 3: 17 metrics
* Phase 4: 19 metrics
* Phase 5: 78 metrics
* Phase 6: 32 metrics
* Phase 7: 28 metrics
* Phase 8: 95 metrics

Pipeline Orchestrator
^^^^^^^^^^^^^^^^^^^^

**Phase Sequencing**: 1 → 2 → 3 → ... → 8

.. code-block:: python

   with PipelineOrchestrator(config) as pipeline:
       # Run all phases
       results = pipeline.run_full_pipeline()

       # Or single phase
       result = pipeline.run_single_phase(1)

Model-Size-Agnostic Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Runtime adaptation:

* **Size Detection**: Categorize as tiny/small/medium/large
* **Adaptive Batching**: VRAM-based batch size calculation
* **Diversity Validation**: Ensure model population diversity
* **Divergence Detection**: Training stability monitoring

Data Flow
---------

.. code-block:: text

   Phase 1 → [3 Models] → Phase 2 → [Champion Model] → Phase 3
      ↓
   [Reasoning-Enhanced] → Phase 4 → [Compressed Model] → Phase 5
      ↓
   [Curriculum-Trained] → Phase 6 → [Tool + Persona] → Phase 7
      ↓
   [Expert System] → Phase 8 → [Final Compressed Agent]

Quality Assurance
----------------

NASA POT10 Compliance
^^^^^^^^^^^^^^^^^^^^

All functions ≤60 lines of code (excluding docstrings)

.. code-block:: bash

   python .github/hooks/nasa_pot10_check.py src/**/*.py

Testing
^^^^^^^

* **47 tests**: 33 unit + 14 integration
* **Coverage target**: ≥90%
* **Fixtures**: temp_dir, sample_config, mock_model, mock_tokenizer

Pre-Commit Hooks
^^^^^^^^^^^^^^^

10 automated quality checks:

1. NASA POT10 enforcement
2. Black auto-formatter
3. isort import sorter
4. flake8 linter
5. mypy type checker
6. pylint advanced linter
7. YAML/TOML/JSON validation
8. File fixers
9. Security checks
10. pytest quick tests (on push)

Performance
----------

Training Times (Local GTX 1660, 6GB VRAM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Phase 1: ~6 hours
* Phase 2: ~1.5 hours
* Phase 3: ~8 hours
* Phase 4: ~4 hours
* Phase 5: ~200 hours
* Phase 6: ~20 hours
* Phase 7: ~78 hours
* Phase 8: ~45 hours

**Total**: ~362 hours (~15 days continuous)

Model Sizes
^^^^^^^^^^

* Phase 1 output: 3 × 95.4 MB = 286.2 MB
* Phase 2 output: 95.4 MB
* Phase 4 output: 11.8 MB (8.2× compression)
* Phase 8 output: 0.4 MB (280× compression)

Memory Requirements
^^^^^^^^^^^^^^^^^^

* **VRAM**: 6GB minimum (GPU)
* **RAM**: 16GB minimum (system)
* **Disk**: 50GB (models + checkpoints + datasets)

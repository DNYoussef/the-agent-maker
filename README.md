# Agent Forge

**8-Phase Pipeline for Training Small Language Models from Scratch**

[![CI](https://github.com/DNYoussef/the-agent-maker/actions/workflows/ci.yml/badge.svg)](https://github.com/DNYoussef/the-agent-maker/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

Agent Forge implements a complete 8-phase training pipeline that creates efficient 25M parameter language models optimized for edge deployment. Each phase builds on the previous, progressively adding reasoning capabilities, compression, and specialization.

**Target**: 25M parameters -> 0.4MB final model size (280x compression)

---

## Architecture

### Phase 1: Cognate (TRM x Titans-MAG)

Creates three specialized 25M parameter models using a novel architecture combining:

**Titans-MAG Backbone** (8 layers, ~20M params):
- Sliding Window Attention with O(n*w) complexity
- Long-range Memory Module (LMM) with factorized projections
- MAG Gate for memory-augmented output blending
- RMSNorm + SwiGLU MLP

**TRM Wrapper** (Test-time Reasoning):
- Multi-pass reasoning with configurable depth
- Deep supervision across reasoning steps
- Weighted step losses for curriculum-style training

**ACT Head** (Adaptive Computation Time):
- Learned halting probability per token
- Dynamic compute allocation based on input difficulty
- Configurable halt threshold

```
Input -> Token Emb -> [8x Titans-MAG Layers] -> LMM -> MAG Gate -> TRM -> ACT -> Output
                           |                      |       |
                      Sliding Window          Long-term  Gated
                       Attention              Memory     Blend
```

Three model variants trained on different objectives:
- **Reasoning**: Optimized for logical inference
- **Memory**: Enhanced long-term context retention
- **Speed**: Optimized for inference latency

**Research**: Titans (Google DeepMind), Test-time Compute Scaling

---

### Phase 2: EvoMerge (Evolutionary Model Merging)

50-generation evolutionary optimization using 6 merge techniques:

| Technique | Method | Use Case |
|-----------|--------|----------|
| **Linear** | Weighted averaging | Baseline blending |
| **SLERP** | Spherical interpolation | Smooth geometry preservation |
| **TIES** | Task-specific weight selection | Multi-task merging |
| **DARE** | Drop and rescale | Sparse merging with dropout |
| **FrankenMerge** | Layer-wise mixing | Architecture exploration |
| **DFS** | Depth-first search merging | Targeted optimization |

**Evolution Strategy**:
1. Initialize population of 8 models from binary combinations of 3 Phase 1 models
2. Elite preservation: Top 2 models -> mutate 3x each -> 6 children
3. Loser merging: Bottom 6 models -> merge in pairs -> 2 children
4. Diversity tracking with automatic re-seeding on convergence
5. Early stopping when fitness threshold met

**Target**: 23.5% fitness improvement over 50 generations

**Research**: Model Merging (MergeKit), TIES-Merging, DARE

---

### Phase 3: Quiet-STaR (Self-Taught Reasoning)

Enhances reasoning through internal thought generation:

**ThoughtGenerator**:
- Generates 4-8 parallel thought continuations per position
- Nucleus sampling (top-p) with temperature control
- Adaptive thought length (10-20 tokens)

**CoherenceScorer** (3-dimensional scoring):
- Semantic: 40% weight (embedding cosine similarity)
- Syntactic: 30% weight (learned grammar validity)
- Predictive: 30% weight (next-token prediction utility)

**MixingHead**:
- 8-head attention-based thought integration
- Gating mechanism for base/thought blending
- Residual connection + layer normalization

**ThoughtInjector**:
- Difficulty-based injection using entropy, attention dispersion, loss
- Minimum interval enforcement between injections
- Configurable threshold (default 0.6)

**Anti-Theater Detection**: Validates genuine reasoning vs memorization

**Research**: Quiet-STaR (Stanford), Chain-of-Thought Prompting

---

### Phase 4: BitNet (1.58-bit Quantization)

Compresses model to ternary weights {-1, 0, +1}:

**Quantization Process**:
1. Per-channel scale factor: alpha = mean(|W|)
2. Normalize by scale
3. Apply sparsity threshold
4. Quantize: Q(w) = sign(w) if |w| > threshold else 0

**Features**:
- Straight-Through Estimator (STE) for gradient flow
- Configurable sparsity threshold
- Layer-wise precision preservation for embeddings
- Calibration-based fine-tuning

**Targets**: 8.2x compression, 3.8x inference speedup

**Research**: BitNet b1.58 (Microsoft), Ternary Weight Networks

---

### Phase 5: Curriculum Learning

7-stage adaptive curriculum with frontier model integration:

**Stage 1 - Assessment**:
- Edge-of-chaos detection (75% accuracy threshold)
- Identifies optimal difficulty level

**Stage 2 - Curriculum Generation**:
- 20,000 questions across 10 difficulty levels
- Generated via OpenRouter (GPT-4o-mini, Claude-3.5 Haiku, Gemini 2.0 Flash)

**Stage 3 - Training Loop**:
- Recursive thinking with tool use
- Variants and hints for scaffolding
- Docker sandbox for safe code execution

**Stage 4 - Eudaimonia Baking**:
Four virtue rules baked into weights:
1. Honesty - Always tell the truth
2. Empathy - Consider impact on others
3. Growth - Learn and improve
4. Respect - Treat all with dignity

**Stage 5 - OODA Loop Integration**:
- Observe -> Orient -> Decide -> Act
- Archetype council for decision validation

**Stage 6 - Self-Modeling**:
- Temperature range prediction training
- Model learns its own behavior patterns

**Stage 7 - Dream Consolidation**:
- Memory preservation during sleep-like phase
- Prevents catastrophic forgetting across levels

**Research**: Intelligence at the Edge of Chaos, Self-Modeling in Neural Systems, Dreaming Is All You Need

---

### Phase 6: Tool & Persona Baking

Iterative A/B optimization loops:

**A-Cycle (Tool Optimization)**:
- SWE-Bench style evaluation
- Tool use proficiency training
- Code execution validation

**B-Cycle (Persona Optimization)**:
- Self-guided persona discovery
- Model determines own behavioral patterns
- Not pre-defined templates

**Half-Baking**:
- Gradual strength integration (50% per iteration)
- Strength scheduler for progressive learning

**Plateau Detection**:
- Adaptive monitoring for convergence
- Automatic cycle switching when plateaued

**Loss Functions**:
- KL divergence for distribution matching
- Reverse KL for mode-seeking
- Jensen-Shannon for symmetric divergence
- Distillation loss for knowledge transfer

**Research**: Prompt Baking (arXiv:2409.13697v1)

---

### Phase 7: Self-Guided Experts

Model-driven expert discovery and routing:

**Stage 1 - Expert Discovery**:
- Model self-analyzes capabilities
- Determines optimal expert count (N=3-10)
- Creates expert profiles

**Stage 2 - SVF Training**:
- Transformer^2 Singular Value Fine-tuning
- REINFORCE trainer for policy optimization
- Z-vector sparse routing

**Stage 3 - ADAS Optimization**:
- NSGA-II multi-objective search
- 100 generations x 50 population
- 5000 architecture evaluations

**Transformer^2 Architecture**:
- Two-pass inference: first pass generates routing, second uses experts
- Expert adapters with learned routing
- Sparse activation for efficiency

**Research**: Transformer^2, NSGA-II, Automated Design of Agentic Systems

---

### Phase 8: Final Compression

Triple compression pipeline with quality gates:

| Stage | Method | Compression | Cumulative |
|-------|--------|-------------|------------|
| SeedLM | Seed-based weight projection | 2x | 2x |
| VPTQ | Vector post-training quantization | 20x | 40x |
| Hypercompression | Parametric curve fitting | 6.25x | 280x |

**Quality Gates**:
- Per-stage: >95% accuracy retention
- Final cumulative: >84% accuracy retention
- Automatic rollback if quality fails

**Benchmarks**:
- MMLU (multi-task language understanding)
- GSM8K (grade school math)
- Phase-specific evaluations

**Grokfast Optimizer**:
- Accelerated grokking for fine-tuning
- Gradient filtering for stable compression

**Research**: SeedLM, VPTQ, Hyper-Compression of LLM Weights

---

## Installation

```bash
# Clone repository
git clone https://github.com/DNYoussef/the-agent-maker.git
cd the-agent-maker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

---

## Project Structure

```
src/
|-- phase1_cognate/
|   |-- model/            # TRM x Titans-MAG architecture
|   |   |-- titans_mag.py # 8-layer backbone with LMM + MAG
|   |   |-- trm_wrapper.py # Multi-pass reasoning
|   |   |-- act_head.py   # Adaptive computation
|   |   |-- components/   # Attention, memory, gating, MLP
|   |-- data/             # Dataset loading and processing
|   |-- training/         # Trainer and W&B integration
|
|-- phase2_evomerge/
|   |-- merge/            # 6 merge techniques
|   |-- evolution/        # Evolution loop and mutation
|   |-- fitness/          # Fitness evaluation
|
|-- phase3_quietstar/
|   |-- architecture/     # Thought generator, coherence, mixing
|   |-- anti_theater.py   # Reasoning validation
|   |-- step1_baking.py   # CoT prompt baking
|   |-- step2_rl.py       # REINFORCE training
|
|-- phase4_bitnet/
|   |-- quantizer.py      # Ternary quantization
|   |-- compressed_model.py # STE-enabled model
|   |-- calibration.py    # Calibration dataset
|   |-- fine_tuner.py     # Post-quantization tuning
|
|-- phase5_curriculum/
|   |-- assessment.py     # Edge-of-chaos detection
|   |-- curriculum_generator.py # Question generation
|   |-- eudaimonia.py     # Virtue rule system
|   |-- self_modeling.py  # Temperature prediction
|   |-- dream_consolidation.py # Memory preservation
|   |-- docker_sandbox.py # Safe code execution
|
|-- phase6_baking/
|   |-- a_cycle_tool.py   # Tool optimization
|   |-- b_cycle_persona.py # Persona discovery
|   |-- half_baking.py    # Gradual integration
|   |-- swe_bench_eval.py # Evaluation framework
|
|-- phase7_experts/
|   |-- expert_discovery.py # Self-guided discovery
|   |-- svf_trainer.py    # SVF + REINFORCE
|   |-- transformer2.py   # Two-pass inference
|   |-- adas_optimizer.py # NSGA-II search
|
|-- phase8_compression/
|   |-- seedlm.py         # Seed-based projection
|   |-- vptq.py           # Vector quantization
|   |-- hypercompression.py # Curve fitting
|   |-- benchmarks.py     # MMLU, GSM8K
|   |-- validation.py     # Quality gates
|
|-- cross_phase/
|   |-- mugrokfast/       # Grokfast x Muon optimizer
|   |-- prompt_baking/    # Prompt -> weight conversion
|   |-- storage/          # Model registry with SQLite WAL
|   |-- orchestrator/     # Phase controllers
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1660 (6GB) | RTX 3080 (10GB) |
| RAM | 16GB | 32GB |
| Storage | 50GB | 100GB SSD |
| CPU | 4 cores | 8+ cores |

---

## Configuration

Each phase has dedicated configuration:

```python
from phase1_cognate.model.model_config import Phase1Config, TitansMAGConfig
from phase4_bitnet.config import Phase4Config

# Phase 1: TRM x Titans-MAG
titans_config = TitansMAGConfig(
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_ff=2048,
    vocab_size=32768,
    max_seq_len=2048,
    sw_window=256,  # Sliding window size
)

# Phase 4: BitNet quantization
phase4_config = Phase4Config(
    sparsity_threshold=0.1,
    calibration_samples=1000,
    fine_tune_epochs=5,
    preserve_layers=["embedding", "lm_head"],
)
```

---

## Research Papers

This implementation is based on the following academic research:

### Phase 1: Architecture
- **Behrouz, A., Zhong, P., & Mirrokni, V.** (2025). "Titans: Learning to Memorize at Test Time." *arXiv:2501.00663v1*. Google Research.
- **Jolicoeur-Martineau, A.** (2025). "Less is More: Recursive Reasoning with Tiny Networks." *arXiv:2510.04871v1*. Samsung SAIL Montreal.

### Phase 2: Model Merging
- **Akiba, T., Shing, M., Tang, Y., Sun, Q., & Ha, D.** (2024). "Evolutionary Optimization of Model Merging Recipes." *arXiv:2403.13187v1*. Sakana AI.

### Phase 3: Reasoning Enhancement
- **Zelikman, E., Harik, G., Shao, Y., Jayasiri, V., Haber, N., & Goodman, N. D.** (2024). "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking." *arXiv:2403.09629v2*. Stanford University.

### Phase 4: Quantization
- **Wang, H. et al.** (2023). "BitNet: Scaling 1-bit Transformers for Large Language Models." *arXiv:2310.11453*. Microsoft Research.
- **Ma, S. et al.** (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." *arXiv:2402.17764*. Microsoft Research.
- **Mekkouri, M., Sun, M., & von Werra, L.** (2024). "Fine-tuning LLMs to 1.58bit: Extreme Quantization Made Easy." Hugging Face.

### Phase 5: Curriculum Learning
- **Ni, M. & Liu, W.** (2024). "Dreaming Is All You Need." *arXiv:2409.01633v1*. University of Technology Sydney.
- **Zhang, S., Patel, A., Rizvi, S. A. et al.** (2024). "Intelligence at the Edge of Chaos." *arXiv:2410.02536v1*. Yale/Columbia/Northwestern.
- **Premakumar, V. N. et al.** (2024). "Unexpected Benefits of Self-Modeling in Neural Systems." *arXiv:2407.10188v2*. Princeton Neuroscience Institute.

### Phase 6: Prompt Baking
- **Bhargava, A., Witkowski, C., Detkov, A., & Thomson, M.** (2024). "Prompt Baking." *arXiv:2409.13697v1*. Caltech.

### Phase 7: Expert Systems
- **Sun, Q., Cetin, E., & Tang, Y.** (2025). "Transformer-squared: Self-Adaptive LLMs." *ICLR 2025, arXiv:2501.06252v3*. Sakana AI.
- **Hu, S., Lu, C., & Clune, J.** (2024). "Automated Design of Agentic Systems." *arXiv:2408.08435v1*. University of British Columbia.

### Phase 8: Compression
- **Shafipour, R. et al.** (2024). "SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators." *arXiv:2410.10714v2*. Apple/Meta.
- **Liu, Y. et al.** (2024). "VPTQ: Extreme Low-Bit Vector Post-Training Quantization for Large Language Models." *arXiv:2409.17066v1*. Microsoft/USTC.
- **Fan, F.-L. et al.** (2024). "Hyper-Compression: Model Compression via Hyperfunction." *arXiv:2409.00592v1*. Chinese University of Hong Kong.

---

## Contributing

```bash
# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Run linting
black src/ tests/
isort src/ tests/
mypy src/

# Run tests
pytest tests/ -v --cov=src
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

<p align="center">
<b>Agent Forge</b> - From 25M parameters to 0.4MB, one phase at a time.
</p>

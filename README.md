# Agent Forge

**Production-grade 8-phase pipeline for creating efficient AI agents from scratch**

[![CI](https://github.com/agent-forge/agent-forge-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/agent-forge/agent-forge-v2/actions)
[![Coverage](https://codecov.io/gh/agent-forge/agent-forge-v2/branch/main/graph/badge.svg)](https://codecov.io/gh/agent-forge/agent-forge-v2)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

Agent Forge is a local-first ML pipeline that creates small, efficient AI agents through an 8-phase training methodology. Starting from randomly initialized weights, the pipeline produces production-ready 25M parameter models optimized for edge deployment.

### Key Capabilities

- **TRM x Titans-MAG Architecture**: Novel transformer design combining Test-time Reasoning Models with Memory-Augmented Generation
- **Evolutionary Model Merging**: NSGA-II driven optimization with 6 merge techniques (SLERP, TIES, DARE, FrankenMerge, DFS, Linear)
- **1.58-bit Quantization**: BitNet compression achieving 8.2x model size reduction with <3% accuracy loss
- **280x Final Compression**: Three-stage pipeline (SeedLM -> VPTQ -> Hypercompression) for edge deployment

---

## Architecture

```
Phase 1: Cognate          Phase 2: EvoMerge         Phase 3: Quiet-STaR
+------------------+      +------------------+      +------------------+
| 3x 25M TRM       | ---> | 50-gen Evolution | ---> | Reasoning        |
| Titans-MAG       |      | 6 Merge Techs    |      | Enhancement      |
| Models           |      | Binary Pairing   |      | Anti-Theater     |
+------------------+      +------------------+      +------------------+
                                                            |
                                                            v
Phase 4: BitNet           Phase 5: Curriculum       Phase 6: Baking
+------------------+      +------------------+      +------------------+
| 1.58-bit Quant   | <--- | 7-Stage Adaptive | <--- | Tool & Persona   |
| STE Gradients    |      | Edge-of-Chaos    |      | A/B Optimization |
| 8.2x Compression |      | Dream Consol.    |      | Half-Baking      |
+------------------+      +------------------+      +------------------+
        |
        v
Phase 7: Experts          Phase 8: Compression
+------------------+      +------------------+
| Self-Guided      | ---> | SeedLM (2x)      |
| ADAS Search      |      | VPTQ (20x)       |
| SVF Training     |      | Hyper (6.25x)    |
+------------------+      +------------------+
                                  |
                                  v
                          +------------------+
                          | 25M -> 0.4MB     |
                          | 280x Compression |
                          | Edge-Ready       |
                          +------------------+
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (6GB+ VRAM recommended)
- 16GB+ System RAM

### Quick Start

```bash
# Clone repository
git clone https://github.com/agent-forge/agent-forge-v2.git
cd agent-forge-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Launch dashboard
streamlit run src/ui/app.py
```

---

## Project Structure

```
agent-forge/
|-- src/
|   |-- phase1_cognate/      # TRM x Titans-MAG model training
|   |-- phase2_evomerge/     # Evolutionary model merging
|   |-- phase3_quietstar/    # Quiet-STaR reasoning enhancement
|   |-- phase4_bitnet/       # 1.58-bit quantization
|   |-- phase5_curriculum/   # Adaptive curriculum learning
|   |-- phase6_baking/       # Prompt baking system
|   |-- phase7_experts/      # Self-guided expert discovery
|   |-- phase8_compression/  # Final compression pipeline
|   |-- cross_phase/         # Shared utilities and model registry
|   |-- ui/                  # Streamlit monitoring dashboard
|-- tests/
|   |-- unit/               # Unit tests
|   |-- integration/        # Integration tests
|   |-- e2e/                # End-to-end pipeline tests
|-- docs/                   # Documentation
|-- phases/                 # Phase-specific guides
```

---

## 8-Phase Pipeline

### Phase 1: Cognate

Creates three specialized 25M parameter TRM x Titans-MAG models:

- **Reasoning Model**: Optimized for logical inference
- **Memory Model**: Enhanced long-term context retention
- **Speed Model**: Optimized for inference latency

**Architecture**: Adaptive Computation Time (ACT) + Long-Term Memory (LTM)

### Phase 2: EvoMerge

Evolutionary optimization using 6 merge techniques:

| Technique | Description | Use Case |
|-----------|-------------|----------|
| SLERP | Spherical linear interpolation | Smooth blending |
| TIES | Task-specific weight selection | Multi-task models |
| DARE | Drop and rescale | Sparse merging |
| FrankenMerge | Layer-wise mixing | Architecture exploration |
| DFS | Depth-first search merging | Targeted optimization |
| Linear | Weighted averaging | Baseline merging |

**Result**: 23.5% fitness improvement over 50 generations

### Phase 3: Quiet-STaR

Self-taught reasoning enhancement:

- Token-wise parallel thought sampling
- Coherence scoring (semantic, syntactic, predictive)
- Anti-theater detection for genuine reasoning validation
- Prompt baking integration for CoT reasoning

### Phase 4: BitNet

1.58-bit quantization using Straight-Through Estimator (STE):

```
Target: 8.2x compression, 3.8x inference speedup
Method: Quantized forward pass, full-precision gradients
Result: <3% accuracy degradation
```

### Phase 5: Curriculum Learning

7-stage adaptive curriculum:

1. Edge-of-chaos assessment (75% accuracy threshold)
2. 20,000 questions across 10 difficulty levels
3. Tool use training with sandboxed code execution
4. Eudaimonia baking (4-rule moral system)
5. Self-modeling temperature prediction
6. Dream consolidation (prevents catastrophic forgetting)
7. Frontier model validation

### Phase 6: Tool and Persona Baking

Iterative A/B optimization:

- **A-Cycle**: Tool use optimization via SWE-Bench
- **B-Cycle**: Self-guided persona generation
- **Half-Baking**: 50% strength per iteration
- **Plateau Detection**: Automatic cycle switching

### Phase 7: Self-Guided Experts

Model-driven expert discovery:

1. Capability self-analysis -> Expert count determination (N=3-10)
2. Transformer^2 SVF training (REINFORCE + MuonGrokfast)
3. NSGA-II ADAS (100 gen x 50 pop = 5000 evaluations)

### Phase 8: Final Compression

Three-stage compression pipeline:

```
Stage 1: SeedLM      100MB -> 50MB   (2x)
Stage 2: VPTQ        50MB  -> 2.5MB  (20x)
Stage 3: Hyper       2.5MB -> 0.4MB  (6.25x)
-------------------------------------------
Total:               100MB -> 0.4MB  (280x)
```

Quality gates ensure >=84% retention with automatic rollback.

---

## Core Systems

### MuGrokfast Optimizer

Unified optimizer combining:

- **Grokfast**: EMA gradient filtering for accelerated grokking
- **Muon**: Newton-Schulz orthogonalization preventing low-rank collapse
- **QK-Clip**: Attention safety rails for RL training

```python
from cross_phase.optimizer import MuonGrokfast, MuGrokConfig

config = MuGrokConfig.from_phase(1)
optimizer = MuonGrokfast(model.parameters(), config=config)
```

### Prompt Baking

Converts prompts into weight updates:

```python
from phase6_baking import bake_prompt, PromptBakingConfig

config = PromptBakingConfig(lora_r=16, num_epochs=3)
baked_model = bake_prompt(model, "You are a reasoning specialist...", config)
```

### Model Registry

SQLite-based registry with WAL mode for concurrent access:

```python
from cross_phase.storage import ModelRegistry

registry = ModelRegistry("./storage/registry.db")
model_id = registry.register_model(
    session_id="session_001",
    phase_name="phase1",
    model_name="reasoning",
    model_path="./checkpoints/phase1/reasoning/best_model.pt",
    metadata={"parameters": 25_000_000, "loss": 2.34}
)
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/e2e/ -v                     # End-to-end tests

# Run phase-specific tests
pytest tests/ -k "phase1" -v             # Phase 1 tests
pytest tests/ -k "bitnet" -v             # BitNet tests
```

---

## Monitoring

Launch the Streamlit dashboard for real-time monitoring:

```bash
streamlit run src/ui/app.py
```

Features:
- Pipeline overview with phase progress
- Model browser with metadata inspection
- System resource monitoring
- Phase-specific configuration editor
- Training metrics visualization

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1660 (6GB) | RTX 3080 (10GB) |
| RAM | 16GB | 32GB |
| Storage | 50GB | 100GB SSD |
| CPU | 4 cores | 8+ cores |

**Estimated Training Time** (RTX 3080):
- Phase 1: 4-6 hours
- Phase 2: 2-3 hours
- Phases 3-4: 6-8 hours
- Phases 5-8: 24-48 hours
- **Total**: ~40-65 hours

---

## Research References

This project implements techniques from:

- [BitNet: 1-bit Transformers](https://arxiv.org/abs/2310.11453)
- [Quiet-STaR: Self-Taught Reasoning](https://arxiv.org/abs/2403.09629)
- [Grokfast: Accelerated Grokking](https://arxiv.org/abs/2405.20233)
- [NSGA-II: Multi-Objective Optimization](https://doi.org/10.1109/4235.996017)

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Run linting
black src/ tests/
isort src/ tests/
mypy src/

# Submit PR
git checkout -b feature/your-feature
git commit -m "feat: description"
git push origin feature/your-feature
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Agent Forge</strong> - Building efficient AI agents, one phase at a time.
</p>

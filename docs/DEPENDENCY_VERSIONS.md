# Agent Forge V2 - Dependency Versions

**Purpose**: Pin all dependency versions for reproducible V2 builds

**Last Updated**: 2025-10-16

**Tested On**: Python 3.11, Ubuntu 22.04, CUDA 12.1

---

## Python Environment

```bash
python: ">=3.10,<3.12"  # Python 3.11 recommended for best compatibility
```

**Rationale**: Python 3.12 has limited support for some dependencies. Python 3.10-3.11 are stable and fully supported.

---

## Core ML/AI Frameworks

### PyTorch

```bash
torch>=2.1.0,<2.3.0  # CUDA 11.8+ or 12.1 support
torchvision>=0.16.0  # Matches torch version
torchaudio>=2.1.0    # Matches torch version
```

**CUDA Requirements**:
- CUDA 11.8 or 12.1 (PyTorch 2.1.0-2.2.x compatible)
- cuDNN 8.9+ (bundled with PyTorch)
- VRAM: 6GB minimum (GTX 1660), 8GB recommended (RTX 3070)

**Installation**:
```bash
# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### HuggingFace

```bash
transformers==4.38.0      # Pinned for BitNet API compatibility
datasets>=2.16.0          # Dataset loading and processing
accelerate>=0.26.0        # Multi-GPU training
peft>=0.8.0              # LoRA/QLoRA for prompt baking
tokenizers>=0.15.0        # Fast tokenization
```

**Rationale**: Transformers 4.39+ has breaking changes for BitNet implementation. Pin to 4.38.0.

### Quantization

```bash
bitsandbytes>=0.42.0  # 8-bit/4-bit quantization, BitNet support
```

---

## Experiment Tracking

### Weights & Biases

```bash
wandb>=0.16.0  # Local or cloud experiment tracking
```

**Configuration**:
```python
# Local W&B server (offline mode)
import wandb
wandb.init(mode="offline", project="agent-forge-v2")

# Cloud W&B (requires API key)
wandb.login(key="YOUR_API_KEY")
wandb.init(project="agent-forge-v2")
```

---

## Frontier Model Access (Phases 3, 5, 7)

### API Clients

```bash
openai>=1.10.0           # OpenAI API (GPT-4o-mini)
anthropic>=0.18.0        # Anthropic API (Claude-3.5 Haiku)
google-generativeai>=0.3.0  # Google Gemini API (Gemini 2.0 Flash)
# qwen via openrouter or direct API
```

### Unified Access (Recommended)

```bash
openrouter-py>=0.2.0  # Unified API for all frontier models
# OR
litellm>=1.30.0       # Alternative unified API
```

**Configuration**:
```python
# OpenRouter configuration (recommended)
import openai

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "YOUR_OPENROUTER_KEY"

frontier_models = {
    "gpt-4o-mini": "openai/gpt-4o-mini",              # $0.15/1M input tokens
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku", # $0.80/1M input tokens
    "gemini-2.0-flash": "google/gemini-2.0-flash-exp",# $0.075/1M input tokens
    "qwen-2.5": "qwen/qwen-2.5-72b-instruct",         # $0.40/1M input tokens
}
```

### Cost Management

```python
# Phase-specific budget limits
api_config = {
    "max_cost_phase3": 200,   # USD for Quiet-STaR data generation
    "max_cost_phase5": 800,   # USD for curriculum generation
    "max_cost_phase7": 250,   # USD for expert data generation
    "rate_limit_rpm": 500,    # Requests per minute
    "batch_size": 10,         # Parallel requests
    "retry_attempts": 3,      # On rate limit errors
    "timeout_seconds": 30,    # Per request
}
```

### Model Selection by Phase

#### Phase 3: Quiet-STaR Data Generation
**Usage**: Generate reasoning trajectories
**Cost**: $100-200
**Models**:
- Primary: GPT-4o-mini (fast, cheap)
- Fallback: Claude-3.5 Haiku (if quality issues)

#### Phase 5: Curriculum Question Generation
**Usage**: 20,000 questions across 10 levels + variants + hints
**Cost**: $600-800
**Models**:
- Questions: GPT-4o-mini (bulk generation)
- Variants: Claude-3.5 Haiku (reasoning quality)
- Hints: Gemini 2.0 Flash (multimodal if needed)
- Validation: Qwen 2.5 (open-source cross-check)

#### Phase 7: Self-Guided Expert Data
**Usage**: Model-generated training data for N=3-10 experts
**Cost**: $150-250
**Models**: Same as Phase 5

---

## Optimization & Training

### Optimizers

```bash
scipy>=1.11.0          # NSGA-II dependencies (Phase 7 ADAS)
scikit-learn>=1.3.0    # K-means for VPTQ (Phase 8)
pymoo>=0.6.0           # NSGA-II ADAS search (Phase 7)
```

### MuonGrokfast Optimizer

**Note**: Custom implementation in `v1-reference/implementation/`
- No external package required
- Integrated into codebase
- Uses PyTorch base optimizers (AdamW, Lion)

---

## Numerical Computing

```bash
numpy>=1.24.0,<2.0.0   # NumPy 1.x (avoid 2.0 breaking changes)
pandas>=2.0.0          # Data analysis
```

**Important**: NumPy 2.0 has breaking changes with PyTorch <2.2. Pin to 1.x.

---

## Utilities

```bash
tqdm>=4.66.0           # Progress bars
python-dotenv>=1.0.0   # Environment variables (.env)
pyyaml>=6.0            # YAML config parsing
requests>=2.31.0       # HTTP requests
aiohttp>=3.9.0         # Async HTTP (for batch API calls)
```

---

## Development & Testing

```bash
pytest>=7.4.0          # Unit testing
pytest-asyncio>=0.21.0 # Async test support
pytest-cov>=4.1.0      # Code coverage
black>=23.0.0          # Code formatting (NASA POT10 compliance)
mypy>=1.5.0            # Type checking
ruff>=0.1.0            # Fast linter
pre-commit>=3.5.0      # Git hooks for quality checks
```

---

## Phase-Specific Dependencies

### Phase 1 (Cognate)
```bash
# Uses: torch, transformers
# No additional dependencies
```

### Phase 2 (EvoMerge)
```bash
# Uses: torch (merging techniques in-house)
# No additional dependencies
```

### Phase 3 (Quiet-STaR)
```bash
openrouter-py>=0.2.0   # Frontier model data generation
peft>=0.8.0            # Prompt baking (LoRA)
```

### Phase 4 (BitNet)
```bash
bitsandbytes>=0.42.0   # Quantization
```

### Phase 5 (Curriculum Learning)
```bash
openrouter-py>=0.2.0   # $600-800 frontier model usage
peft>=0.8.0            # Eudaimonia prompt baking
# Dream consolidation uses pre-trained autoencoder (transformers)
```

### Phase 6 (Tool & Persona Baking)
```bash
peft>=0.8.0            # Iterative baking (A/B cycles)
scikit-learn>=1.3.0    # SWE-Bench analysis
```

### Phase 7 (Self-Guided Experts)
```bash
pymoo>=0.6.0           # NSGA-II ADAS search
openrouter-py>=0.2.0   # $150-250 expert data generation
```

### Phase 8 (Final Compression)
```bash
scipy>=1.11.0          # Polynomial fitting (hypercompression)
scikit-learn>=1.3.0    # K-means (VPTQ codebook learning)
```

---

## Installation

### Minimal (Phases 1-4 only, no frontier models)

```bash
pip install \
  torch==2.1.0 \
  transformers==4.38.0 \
  datasets>=2.16.0 \
  wandb>=0.16.0 \
  numpy>=1.24.0,<2.0.0 \
  tqdm>=4.66.0 \
  peft>=0.8.0 \
  bitsandbytes>=0.42.0
```

**Estimated size**: ~5GB
**Use case**: Local development, Phases 1-4 only

### Full (All phases, with frontier models)

```bash
pip install -r requirements.txt
```

**Estimated size**: ~7GB
**Use case**: Complete pipeline, Phases 1-8

### Requirements.txt

```txt
# Core ML/AI
torch>=2.1.0,<2.3.0
transformers==4.38.0
datasets>=2.16.0
accelerate>=0.26.0
peft>=0.8.0
bitsandbytes>=0.42.0

# Experiment Tracking
wandb>=0.16.0

# Frontier Models (Phases 3, 5, 7)
openai>=1.10.0
anthropic>=0.18.0
google-generativeai>=0.3.0
openrouter-py>=0.2.0

# Optimization
scipy>=1.11.0
scikit-learn>=1.3.0
pymoo>=0.6.0

# Numerical
numpy>=1.24.0,<2.0.0
pandas>=2.0.0

# Utilities
tqdm>=4.66.0
python-dotenv>=1.0.0
pyyaml>=6.0
requests>=2.31.0
aiohttp>=3.9.0

# Dev/Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.0.0
mypy>=1.5.0
ruff>=0.1.0
pre-commit>=3.5.0
```

---

## System Requirements

### Hardware

**Minimum**:
- GPU: GTX 1660 or equivalent (6GB VRAM, CUDA 11.8+)
- CPU: 4 cores, 3.0 GHz
- RAM: 16GB
- Storage: 50GB free space

**Recommended**:
- GPU: RTX 3070 or equivalent (8GB VRAM, CUDA 12.1)
- CPU: 8 cores, 3.5 GHz
- RAM: 32GB (Phase 5 curriculum training benefits from more RAM)
- Storage: 100GB free space (datasets, checkpoints, W&B logs)

### Operating Systems

**Supported**:
- Linux: Ubuntu 22.04+, Debian 11+, CentOS 8+
- macOS: 12+ (Apple Silicon M1/M2 supported via MPS backend)
- Windows: 10/11 (WSL2 recommended for better compatibility)

**Recommended**: Ubuntu 22.04 LTS for best stability

---

## Version Update Policy

### Security Patches
- ✅ **Update immediately**: Critical security vulnerabilities
- Examples: torch CVE patches, transformers security fixes

### Minor Versions
- ✅ **Update within 1 month**: Bug fixes, performance improvements
- Examples: torch 2.1.0 → 2.1.1, transformers 4.38.0 → 4.38.1

### Major Versions
- ⚠️ **Test on isolated branch first**: Breaking changes possible
- Examples: torch 2.1.x → 2.2.x, numpy 1.x → 2.x
- Create migration guide before updating production

### Breaking Changes
- ⚠️ **Document in migration guide**: API changes, deprecations
- Examples: transformers 4.38 → 4.39 (BitNet API changes)
- Test all 8 phases before merging to main

---

## Known Compatibility Issues

### Issue 1: NumPy 2.0
**Problem**: Breaking changes with PyTorch <2.2
**Solution**: Pin to `numpy>=1.24.0,<2.0.0`
**Status**: Documented, workaround applied

### Issue 2: Transformers 4.39+
**Problem**: API changes for BitNet implementation
**Solution**: Pin to `transformers==4.38.0`
**Status**: Documented, workaround applied

### Issue 3: Python 3.12
**Problem**: Some dependencies not yet compatible (bitsandbytes, pymoo)
**Solution**: Use Python 3.10 or 3.11
**Status**: Documented, workaround applied

### Issue 4: CUDA 12.2+
**Problem**: PyTorch 2.1.x officially supports CUDA 12.1 (12.2+ may work but untested)
**Solution**: Use CUDA 12.1 for guaranteed compatibility
**Status**: Documented, recommendation provided

---

## Environment Setup

### Conda (Recommended)

```bash
# Create environment
conda create -n agent-forge-v2 python=3.11 -y
conda activate agent-forge-v2

# Install PyTorch with CUDA
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

### venv (Alternative)

```bash
# Create environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Docker (Future)

```dockerfile
# Dockerfile (example for future use)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

WORKDIR /app
CMD ["python", "main.py"]
```

---

## Verification

### Test Installation

```bash
# Test PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Test Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test Frontier Models (requires API keys)
python -c "import openai; print('OpenAI client loaded')"

# Test W&B
python -c "import wandb; print(f'W&B: {wandb.__version__}')"
```

### Expected Output

```
PyTorch: 2.1.0
CUDA available: True
CUDA version: 12.1
Transformers: 4.38.0
OpenAI client loaded
W&B: 0.16.2
```

---

## Updating This Document

**When adding new phases or changing dependencies:**

1. Update version constraints in this file
2. Update `requirements.txt`
3. Test on clean environment (conda/venv)
4. Document any breaking changes in "Known Compatibility Issues"
5. Update "Last Updated" date at top
6. Commit with clear version message: `docs: update dependencies for Phase X`

---

## Related Documents

- [CLAUDE.md](../CLAUDE.md) - Project overview with Phase 5-8 V2 descriptions
- [docs/v2-planning/PHASE5-8_V1_VS_V2_RECONCILIATION.md](v2-planning/PHASE5-8_V1_VS_V2_RECONCILIATION.md) - V1 vs V2 comparison
- [README.md](../README.md) - Project README with getting started guide

---

**Maintainer**: Agent Forge V2 Team
**Support**: File issues at project repository

# Agent Forge V2: Implementation Guide

**Version**: 1.0
**Last Updated**: 2025-10-16
**Status**: Ready for Implementation

---

## Quick Links
- **Repository Overview**: [REPOSITORY_STATUS_REPORT.md](REPOSITORY_STATUS_REPORT.md)
- **Reading Order**: [DEVELOPER_READING_ORDER.md](DEVELOPER_READING_ORDER.md)
- **V2 Specification**: [v2-specification/AGENT_FORGE_V2_SPECIFICATION.md](v2-specification/AGENT_FORGE_V2_SPECIFICATION.md)
- **Master Index**: [INDEX.md](INDEX.md)

---

## Prerequisites

### Hardware Requirements
**Minimum**:
- CUDA-capable GPU: GTX 1660 or equivalent (6GB+ VRAM)
- System RAM: 16GB
- Storage: 50GB available disk space
- OS: Windows 10+, Linux (Ubuntu 20.04+), macOS 11+
- CPU: Modern multi-core (4+ cores)

**Recommended**:
- GPU: RTX 3060 or better (12GB+ VRAM)
- System RAM: 32GB
- Storage: 100GB SSD
- CPU: 8+ cores

### Software Requirements
- Python 3.10 or higher
- Git
- CUDA Toolkit (for GPU support)
- Visual Studio Code (recommended IDE)

---

## Step-by-Step Implementation

### Phase 0: Environment Setup (Week 1)

#### 1. Clone and Verify Repository
```bash
# Navigate to project directory
cd "c:\Users\17175\Desktop\the agent maker"

# Verify documentation completeness
ls docs/
ls phases/
ls v1-reference/

# Read critical files
cat README.md
cat CLAUDE.md
cat docs/INDEX.md
```

#### 2. Create Project Structure
```bash
# Create implementation directories
mkdir -p src/{phase1,phase2,phase3,phase4,phase5,phase6,phase7,phase8}
mkdir -p src/cross_phase
mkdir -p tests/{phase1,phase2,phase3,phase4,phase5,phase6,phase7,phase8}
mkdir -p tests/integration
mkdir -p examples
mkdir -p config
mkdir -p storage/{sessions,registry,checkpoints}

# Verify structure
tree -L 2 src/
```

#### 3. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### 4. Install Dependencies
```bash
# Create requirements.txt
cat > requirements.txt << EOF
# Core ML/AI
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
scipy>=1.10.0

# Infrastructure
pyyaml>=6.0
structlog>=23.1.0
psutil>=5.9.0

# Monitoring
wandb>=0.15.0

# Development
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.7.0
mypy>=1.4.0
pre-commit>=3.3.0

# Optional (for UI/dashboard)
streamlit>=1.25.0
plotly>=5.15.0
rich>=13.5.0
EOF

# Install dependencies
pip install -r requirements.txt
```

#### 5. Configure Git Hooks (NASA POT10 Enforcement)
```bash
# Copy pre-commit config
cp scripts/.pre-commit-config.yaml .pre-commit-config.yaml

# Install pre-commit hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

#### 6. Configure Weights & Biases (Local)
```bash
# Initialize W&B (offline mode for local)
wandb login --offline

# Configure W&B
cat > config/wandb_config.yaml << EOF
project: agent-forge-v2
mode: offline  # Local development
base_dir: ./wandb_logs
EOF
```

#### 7. Create Pipeline Configuration
```bash
cat > config/pipeline_config.yaml << EOF
# Agent Forge V2 Pipeline Configuration

# Global settings
project_name: "agent-forge-v2"
device: "cuda"  # or "cpu"
seed: 42

# Storage settings
storage:
  base_path: "./storage"
  sessions_dir: "./storage/sessions"
  registry_db: "./storage/registry/model_registry.db"
  checkpoints_dir: "./storage/checkpoints"

# Cleanup policy
cleanup_policy:
  max_session_age_days: 30
  max_failed_session_age_days: 7
  max_sessions_total: 100
  checkpoint_every_n_sessions: 10
  auto_cleanup_enabled: true

# W&B settings
wandb:
  enabled: true
  mode: "offline"
  project: "agent-forge-v2"

# Phase configurations (references to phase-specific configs)
phases:
  cognate:
    config_path: "./config/phase_configs/phase1_cognate.yaml"
  evomerge:
    config_path: "./config/phase_configs/phase2_evomerge.yaml"
  quietstar:
    config_path: "./config/phase_configs/phase3_quietstar.yaml"
  bitnet:
    config_path: "./config/phase_configs/phase4_bitnet.yaml"
  forge:
    config_path: "./config/phase_configs/phase5_forge.yaml"
  baking:
    config_path: "./config/phase_configs/phase6_baking.yaml"
  edge:
    config_path: "./config/phase_configs/phase7_edge.yaml"
  final_compression:
    config_path: "./config/phase_configs/phase8_final.yaml"
EOF
```

---

### Phase 1: Cognate Implementation (Weeks 1-2)

#### Week 1: Architecture Setup

**Goal**: Implement TRM × Titans-MAG architecture (25M parameters)

**Reference Documentation**:
- [phases/phase1/TRM_TITANS_ARCHITECTURE.md](../phases/phase1/TRM_TITANS_ARCHITECTURE.md)
- [phases/phase1/PHASE1_COMPLETE_GUIDE.md](../phases/phase1/PHASE1_COMPLETE_GUIDE.md)
- [phases/phase1/LOGICAL_UNDERSTANDING.md](../phases/phase1/LOGICAL_UNDERSTANDING.md)

**Tasks**:
1. Create Phase 1 configuration file
2. Implement TitansMAG backbone (8 layers, 512 dim)
3. Implement TRM wrapper (recursive refinement)
4. Implement ACT halting mechanism
5. Integrate MuGrokfast optimizer

**Step-by-Step**:

##### 1.1 Create Phase 1 Config
```bash
mkdir -p config/phase_configs

cat > config/phase_configs/phase1_cognate.yaml << EOF
# Phase 1: Cognate Configuration

model:
  d_model: 512               # Hidden dimension
  n_layers: 8                # Transformer layers
  n_heads: 8                 # Attention heads
  vocab_size: 32768          # BPE tokenizer
  max_seq_len: 2048

  attention:
    type: "sliding_window"
    sw_window: 1024          # ±512 tokens

  memory:
    type: "factorized"       # LMM (Long-range Memory Module)
    d_mem: 256               # Half of d_model
    decay: 0.99              # Exponential decay

gate:
  hidden: 256                # MAG gate network
  entropy_reg: 0.001         # Prevent saturation

trm:
  T_max: 3                   # Recursion depth
  micro_steps: 2             # Refinement steps
  deep_supervision: true     # Loss per step
  detach_between_steps: true # Memory efficiency

act:
  halt_thresh: 0.5           # ACT halting threshold
  ema_teacher: 0.98          # EMA calibration

optimizer:
  kind: "muon_grokfast"
  alpha: 0.98                # Grokfast EMA decay
  lam: 0.3                   # Grokfast amplification
  k: 3                       # Newton-Schulz iterations
  beta: 0.9                  # Muon momentum
  weight_decay: 0.05
  lr: 0.001
  qkclip_tau: 30.0           # QK-Clip threshold

training:
  epochs: 10
  batch_size: 16
  eval_interval: 500
  save_interval: 1000

specializations:
  - name: "reasoning"
    act_threshold: 0.95
    memory_capacity: 4096
    seed: 42
  - name: "memory_integration"
    act_threshold: 0.90
    memory_capacity: 8192
    seed: 1337
  - name: "adaptive_computation"
    act_threshold: 0.99
    memory_capacity: 2048
    seed: 2023
EOF
```

##### 1.2 Implement Model Architecture
```bash
# Create Phase 1 source directory
mkdir -p src/phase1/{model,training,data}

# Implement TitansMAG backbone
cat > src/phase1/model/titans_mag.py << 'EOF'
# Implementation here (refer to TRM_TITANS_ARCHITECTURE.md)
# Key components:
# - SlidingWindowAttention
# - FactorizedLongRangeMemory
# - MAGGate (Memory-Augmented Gating)
EOF

# Implement TRM wrapper
cat > src/phase1/model/trm_wrapper.py << 'EOF'
# Implementation here (refer to TRM_TITANS_ARCHITECTURE.md)
# Key components:
# - RecursiveRefinement (g_φ)
# - AnswerUpdate (h_ψ)
# - DeepSupervision
EOF

# Implement ACT halting
cat > src/phase1/model/act_halting.py << 'EOF'
# Implementation here
# Key components:
# - ACTHead (halt probability)
# - EMACalibration
EOF
```

##### 1.3 Integrate MuGrokfast Optimizer
```bash
# Copy from V1 reference or implement from scratch
mkdir -p src/cross_phase

# Implement MuGrokfast (based on v1-reference/implementation/MUGROKFAST_DEVELOPER_GUIDE.md)
cat > src/cross_phase/mugrokfast_optimizer.py << 'EOF'
# MuGrokfast optimizer implementation
# - Grokfast: EMA gradient filtering
# - Muon: Newton-Schulz orthogonalization
# - QK-Clip: Attention stability
# - Parameter routing: 2-D → Muon, 1-D → AdamW
EOF

cat > src/cross_phase/mugrokfast_config.py << 'EOF'
# MuGrokfast configuration with phase presets
EOF
```

##### 1.4 Create Training Loop
```bash
cat > src/phase1/training/train.py << 'EOF'
# Phase 1 training loop
# - Load config
# - Initialize models (3 specializations)
# - Train with MuGrokfast
# - W&B logging (37 metrics)
# - Save models
EOF
```

##### 1.5 Add Tests
```bash
mkdir -p tests/phase1

cat > tests/phase1/test_titans_mag.py << 'EOF'
import pytest
import torch
from src.phase1.model.titans_mag import TitansMAG

def test_titans_mag_forward():
    """Test forward pass"""
    model = TitansMAG(d_model=512, n_layers=8)
    input_ids = torch.randint(0, 32768, (2, 128))
    output = model(input_ids)
    assert output.shape == (2, 128, 512)

def test_parameter_count():
    """Test model has ~25M parameters"""
    model = TitansMAG(d_model=512, n_layers=8)
    params = sum(p.numel() for p in model.parameters())
    assert 22_500_000 <= params <= 27_500_000

# ... more tests
EOF

# Run tests
pytest tests/phase1/ -v --cov=src/phase1
```

#### Week 2: Training & Validation

**Tasks**:
1. Prepare datasets (ARC-Easy, GSM8K, Mini-MBPP)
2. Train 3 specialized models
3. Validate models fit in 6GB VRAM
4. Test inference <100ms
5. Implement Phase 1 → Phase 2 handoff

**Step-by-Step**:

##### 2.1 Prepare Datasets
```bash
# Create data directory
mkdir -p data/phase1

# Download datasets (placeholder - use actual dataset download)
cat > src/phase1/data/download_datasets.py << 'EOF'
# Download ARC-Easy, GSM8K, Mini-MBPP, PIQA, SVAMP
# Preprocess to common format
EOF

# Run download
python src/phase1/data/download_datasets.py
```

##### 2.2 Train Models
```bash
# Train 3 specialized models
python src/phase1/training/train.py \
  --config config/phase_configs/phase1_cognate.yaml \
  --session-id "phase1-run-001"

# Monitor with W&B
wandb sync ./wandb_logs
```

##### 2.3 Validate Performance
```bash
# Test VRAM usage
python src/phase1/training/validate_vram.py

# Test inference latency
python src/phase1/training/validate_inference.py

# Expected output:
# ✅ Model 1: 25.1M params, 1.9GB VRAM, 87ms inference
# ✅ Model 2: 25.0M params, 1.9GB VRAM, 89ms inference
# ✅ Model 3: 25.2M params, 1.9GB VRAM, 85ms inference
```

##### 2.4 Implement Phase Handoff
```bash
cat > src/cross_phase/phase_handoff.py << 'EOF'
# Phase handoff validation
# - Validate number of models
# - Validate parameter counts
# - Validate metadata
# - Register in model registry
EOF

# Test handoff
python src/cross_phase/phase_handoff.py \
  --source phase1 \
  --target phase2 \
  --session-id "phase1-run-001"
```

---

### Phase 2: EvoMerge Implementation (Weeks 3-4)

**Reference Documentation**:
- [phases/phase2/PHASE2_COMPLETE_GUIDE.md](../phases/phase2/PHASE2_COMPLETE_GUIDE.md)
- [phases/phase2/LOGICAL_UNDERSTANDING.md](../phases/phase2/LOGICAL_UNDERSTANDING.md)
- [phases/phase2/MERGE_TECHNIQUES_UPDATED.md](../phases/phase2/MERGE_TECHNIQUES_UPDATED.md)
- [phases/phase2/BINARY_PAIRING_STRATEGY.md](../phases/phase2/BINARY_PAIRING_STRATEGY.md)

**Tasks**:
1. Implement 6 merge techniques (Linear, SLERP, TIES, DARE, FrankenMerge, DFS)
2. Implement binary combination strategy (8 combinations)
3. Implement elite mutation + loser merging evolution
4. Run 50 generations
5. Validate fitness improvement >20%

**Step-by-Step**: (Similar structure to Phase 1)

---

### Phase 3-8: Implementation (Weeks 5-16)

Follow the same pattern for remaining phases:
- **Phase 3**: Weeks 5-6 (Quiet-STaR reasoning)
- **Phase 4**: Weeks 7-8 (BitNet compression)
- **Phase 5**: Weeks 9-10 (Curriculum learning)
- **Phase 6**: Weeks 11-12 (Tool & persona baking)
- **Phase 7**: Weeks 13-14 (Self-guided experts)
- **Phase 8**: Weeks 15-16 (Final compression + benchmark testing)

Each phase has:
- Configuration file in `config/phase_configs/`
- Implementation in `src/phaseN/`
- Tests in `tests/phaseN/`
- Integration validation

---

## Testing Strategy

### Unit Tests
- **Coverage Target**: ≥90% overall, ≥95% critical paths
- **Framework**: pytest
- **Location**: `tests/phaseN/test_*.py`

```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific phase
pytest tests/phase1/ -v --cov=src/phase1

# Generate coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Integration Tests
```bash
# Test phase handoffs
pytest tests/integration/test_phase_handoffs.py

# Test full pipeline
pytest tests/integration/test_full_pipeline.py
```

### Performance Tests
```bash
# Test VRAM usage
python tests/performance/test_vram_usage.py

# Test inference latency
python tests/performance/test_inference_latency.py
```

---

## Debugging & Troubleshooting

### Common Issues

#### CUDA Out of Memory
**Problem**: Model doesn't fit in GPU memory
**Solution**:
```python
# Reduce batch size
config.batch_size = 8  # Was 16

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision training
config.use_amp = True
```

#### Slow Training
**Problem**: Training taking too long
**Solution**:
```python
# Increase batch size (if memory allows)
config.batch_size = 32

# Reduce validation frequency
config.eval_interval = 1000  # Was 500

# Enable data prefetching
config.num_workers = 4
config.pin_memory = True
```

#### Test Failures
**Problem**: NASA POT10 violations (functions >60 LOC)
**Solution**:
```bash
# Check violations
python scripts/check_function_length.py

# Fix by splitting functions
# Example: split_long_function(func) → func_part1(), func_part2()
```

---

## Deployment & Production

### Model Export
```bash
# Export trained model
python src/export_model.py \
  --session-id "final-run" \
  --phase phase8 \
  --output "./models/agent_forge_v2_final.pt"
```

### Inference API (Optional)
```bash
# Create simple inference API
cat > serve.py << 'EOF'
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load("models/agent_forge_v2_final.pt")

@app.post("/generate")
def generate(prompt: str):
    return model.generate(prompt)
EOF

# Run API
uvicorn serve:app --host 0.0.0.0 --port 8000
```

---

## Performance Expectations

### Training Time Estimates (GTX 1660, 6GB VRAM)
| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1 | 45 min | 3 models, 10 epochs each |
| Phase 2 | 2.5 hours | 50 generations |
| Phase 3 | 1.5 hours | Reasoning enhancement |
| Phase 4 | 30 min | Quantization |
| Phase 5 | 4-6 hours | Curriculum training |
| Phase 6 | 3-4 hours | Tool/persona baking |
| Phase 7 | 6-8 hours | ADAS architecture search |
| Phase 8 | 27-50 hours | Compression + benchmarking |
| **Total** | **45-70 hours** | End-to-end pipeline |

### Model Size Progression
| After Phase | Size | Compression |
|------------|------|-------------|
| Phase 1 | 286 MB | 3 models × 95 MB |
| Phase 2 | 95 MB | 3:1 (merged) |
| Phase 3 | 95 MB | (no compression) |
| Phase 4 | 12 MB | 8:1 (BitNet) |
| Phase 8 (SeedLM) | 50 MB | 2:1 |
| Phase 8 (VPTQ) | 2.5 MB | 40:1 |
| Phase 8 (Final) | **0.4 MB** | **280:1** |

---

## Quality Metrics

### NASA POT10 Compliance
- **Requirement**: All functions ≤60 LOC
- **Enforcement**: Pre-commit hook
- **Check**: `python scripts/check_function_length.py`

### Test Coverage
- **Overall**: ≥90%
- **Critical Paths**: ≥95%
- **Check**: `pytest --cov=src --cov-report=term`

### Performance Targets
- **Inference Latency**: <100ms (GTX 1660)
- **GPU Utilization**: >90%
- **VRAM Usage**: <6GB per model

---

## Next Steps

1. **Complete Environment Setup**: Follow Phase 0 checklist
2. **Study Phase 1 Documentation**: Read TRM_TITANS_ARCHITECTURE.md
3. **Implement Phase 1**: Follow Week 1-2 guide above
4. **Validate Phase 1**: Test VRAM, inference, handoff
5. **Continue to Phase 2**: Repeat process for EvoMerge

**Questions?** Refer to:
- [REPOSITORY_STATUS_REPORT.md](REPOSITORY_STATUS_REPORT.md) for overview
- [DEVELOPER_READING_ORDER.md](DEVELOPER_READING_ORDER.md) for learning path
- [docs/INDEX.md](INDEX.md) for complete navigation

---

**Guide Version**: 1.0
**Last Updated**: 2025-10-16
**Status**: ✅ Ready for Implementation

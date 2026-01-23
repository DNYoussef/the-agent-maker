# Agent Forge V2 - Full 8-Phase Pipeline
# AGM-007: Updated to support full pipeline deployment, not just Phase 1
# GPU-enabled PyTorch container for training 25M parameter models

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy entire project first
COPY . .

# Install Python dependencies with compatible versions
# AGM-007: Added dependencies for all 8 phases
RUN pip install --no-cache-dir \
    "transformers>=4.35.0,<4.40.0" \
    "peft>=0.7.0" \
    "wandb>=0.16.0" \
    "pyyaml>=6.0" \
    "numpy>=1.24.0" \
    "tqdm>=4.65.0" \
    "safetensors>=0.4.0" \
    "datasets>=2.14.0" \
    "accelerate>=0.25.0" \
    "scipy>=1.10.0" \
    "networkx>=3.0" \
    && pip install --no-cache-dir pytest pytest-cov

# Create directories for data, checkpoints, and storage
RUN mkdir -p /app/checkpoints /app/data /app/wandb /app/storage/registry

# Set environment variables
ENV PYTHONPATH=/app/src
ENV WANDB_DIR=/app/wandb
ENV HF_HOME=/app/data/huggingface
ENV HF_DATASETS_OFFLINE=0

# AGM-007: Pipeline mode control via environment variable
# PIPELINE_MODE options:
#   full     - Run all 8 phases (default)
#   phase1   - Run Phase 1 only (legacy behavior)
#   single   - Run single phase specified by PIPELINE_PHASE
#   mock     - Run full pipeline with mock models (testing)
ENV PIPELINE_MODE=full
ENV PIPELINE_PHASE=1
ENV WANDB_MODE=offline

# AGM-007: Default command - full 8-phase pipeline
# Use PIPELINE_MODE=phase1 for legacy Phase 1 only behavior
CMD ["sh", "-c", "if [ \"$PIPELINE_MODE\" = 'phase1' ]; then python src/phase1_cognate/train_phase1.py --all --wandb-mode $WANDB_MODE; elif [ \"$PIPELINE_MODE\" = 'single' ]; then python scripts/run_pipeline.py --phase $PIPELINE_PHASE --wandb-mode $WANDB_MODE; elif [ \"$PIPELINE_MODE\" = 'mock' ]; then python scripts/run_pipeline.py --mock --wandb-mode $WANDB_MODE; else python scripts/run_pipeline.py --wandb-mode $WANDB_MODE; fi"]

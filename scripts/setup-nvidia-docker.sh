#!/bin/bash
# Setup NVIDIA Container Toolkit for Docker GPU support
# Run on the remote server with sudo

set -e

echo "=== NVIDIA Container Toolkit Setup ==="
echo ""

# Check if NVIDIA driver is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA driver not found. Install driver first."
    exit 1
fi

echo "[1/4] Adding NVIDIA Container Toolkit repository..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "[2/4] Installing NVIDIA Container Toolkit..."
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "[3/4] Configuring Docker..."
sudo nvidia-ctk runtime configure --runtime=docker

echo "[4/4] Restarting Docker..."
sudo systemctl restart docker

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Test with: docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
echo ""
echo "Then run Agent Forge Phase 1 with GPU:"
echo "  cd ~/agent-forge"
echo "  docker run -d --name agent-forge-phase1 --gpus all \\"
echo "    -v \$(pwd)/checkpoints:/app/checkpoints \\"
echo "    -v \$(pwd)/data:/app/data \\"
echo "    -v \$(pwd)/wandb:/app/wandb \\"
echo "    agent-forge:phase1 \\"
echo "    python src/phase1_cognate/train_phase1.py --all --epochs 10 --wandb-mode offline"

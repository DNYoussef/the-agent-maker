#!/bin/bash
# Agent Forge Remote Deployment Script
# Usage: ./scripts/deploy-remote.sh

set -e

REMOTE_HOST="david@w.m1el.eu"
REMOTE_PORT="2222"
REMOTE_DIR="/var/data/agent-forge"

echo "=== Agent Forge Remote Deployment ==="
echo "Target: $REMOTE_HOST:$REMOTE_PORT"
echo "Directory: $REMOTE_DIR"
echo ""

# Create remote directory
echo "[1/5] Creating remote directory..."
ssh -p $REMOTE_PORT $REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Clone or update repository
echo "[2/5] Setting up repository..."
ssh -p $REMOTE_PORT $REMOTE_HOST "
  cd $REMOTE_DIR
  if [ -d .git ]; then
    echo 'Updating existing repo...'
    git pull
  else
    echo 'Cloning repository...'
    git clone https://github.com/DNYoussef/the-agent-maker.git .
  fi
"

# Copy local configs and Dockerfile
echo "[3/5] Copying local configs..."
scp -P $REMOTE_PORT Dockerfile docker-compose.yml $REMOTE_HOST:$REMOTE_DIR/
scp -P $REMOTE_PORT config/server-access.json $REMOTE_HOST:$REMOTE_DIR/config/

# Build Docker image
echo "[4/5] Building Docker image..."
ssh -p $REMOTE_PORT $REMOTE_HOST "
  cd $REMOTE_DIR
  docker build -t agent-forge:phase1 .
"

# Create data directories
echo "[5/5] Creating data directories..."
ssh -p $REMOTE_PORT $REMOTE_HOST "
  cd $REMOTE_DIR
  mkdir -p checkpoints data wandb
"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "To start Phase 1 training:"
echo "  ssh -p $REMOTE_PORT $REMOTE_HOST"
echo "  cd $REMOTE_DIR"
echo "  docker compose up -d"
echo ""
echo "To monitor:"
echo "  docker logs -f agent-forge-phase1"

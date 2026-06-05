#!/bin/bash
# Monitor Agent Forge training progress

CONTAINER_NAME="agent-forge-phase1"

echo "=== Agent Forge Training Monitor ==="
echo "Container: $CONTAINER_NAME"
echo "Press Ctrl+C to exit"
echo ""

# Check if container exists
if ! docker ps -a | grep -q $CONTAINER_NAME; then
    echo "ERROR: Container $CONTAINER_NAME not found"
    exit 1
fi

# Show container status
echo "Container Status:"
docker ps -a --filter name=$CONTAINER_NAME --format "table {{.ID}}\t{{.Status}}\t{{.RunningFor}}"
echo ""

# Show GPU usage if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
fi

# Show resource usage
echo "Container Resources:"
docker stats $CONTAINER_NAME --no-stream --format "CPU: {{.CPUPerc}} | Memory: {{.MemUsage}} | Net I/O: {{.NetIO}}"
echo ""

# Show recent logs
echo "Recent Logs (last 30 lines):"
echo "---"
docker logs --tail 30 $CONTAINER_NAME 2>&1
echo "---"
echo ""

# Show checkpoint files if any
echo "Checkpoints:"
ls -lh ~/agent-forge/checkpoints/ 2>/dev/null || echo "No checkpoints yet"
echo ""

# Show W&B runs
echo "W&B Runs:"
ls ~/agent-forge/wandb/ 2>/dev/null | head -10 || echo "No W&B runs yet"

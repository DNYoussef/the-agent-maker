# Agent Forge Pipeline API Documentation

Comprehensive RESTful and WebSocket API for the Agent Forge 8-Phase Pipeline orchestration system.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [RESTful Endpoints](#restful-endpoints)
5. [WebSocket Channels](#websocket-channels)
6. [Data Models](#data-models)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

## Overview

The Agent Forge API provides complete control over the 8-phase pipeline execution system with real-time monitoring capabilities.

### Features

- **Pipeline Control**: Start, stop, pause, resume pipeline execution
- **Phase Configuration**: Configure individual phases with custom parameters
- **Real-time Monitoring**: WebSocket streams for agents, tasks, and metrics
- **Quality Gates**: Automated validation with theater detection
- **Checkpoint Management**: Save and restore pipeline state
- **Configuration Presets**: Reusable pipeline configurations
- **Execution History**: Track all pipeline runs

### 8 Pipeline Phases

1. **Cognate**: Model creation and initialization
2. **EvoMerge**: Evolutionary model optimization
3. **Quiet-STaR**: Reasoning enhancement
4. **BitNet**: Initial compression
5. **Training**: Main training loop with Grokfast
6. **Baking**: Tool and persona integration
7. **ADAS**: Architecture search
8. **Compression**: Final compression with SeedLM + VPTQ

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Application                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   REST API  │  │  WebSocket   │  │   Services    │  │
│  │   Routes    │  │   Routes     │  │   Layer       │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│         │                 │                  │          │
│         └─────────────────┴──────────────────┘          │
│                           │                             │
│                  ┌────────┴─────────┐                   │
│                  │ Pipeline Service │                   │
│                  └──────────────────┘                   │
│                           │                             │
├───────────────────────────┼─────────────────────────────┤
│                           │                             │
│  ┌────────────────────────▼──────────────────────────┐  │
│  │          Swarm Coordinator & Execution           │  │
│  └──────────────────────────────────────────────────┘  │
│                           │                             │
│  ┌────────────────────────▼──────────────────────────┐  │
│  │              Unified Pipeline                     │  │
│  │  (8-Phase Orchestration)                         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements_api.txt

# Or install individually
pip install fastapi uvicorn websockets pydantic psutil
```

### Starting the Server

```bash
# Start API server
python run_api_server.py

# Or using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Accessing Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **WebSocket Test Client**: http://localhost:8000/ws/client

## RESTful Endpoints

### Pipeline Control

#### Start Pipeline

```http
POST /api/v1/pipeline/start
Content-Type: application/json

{
  "phases": ["cognate", "evomerge", "quietstar"],
  "config": {
    "cognate": {
      "base_models": ["model1", "model2"],
      "init_strategy": "xavier_uniform"
    }
  },
  "enable_monitoring": true,
  "enable_checkpoints": true,
  "swarm_topology": "hierarchical",
  "max_agents": 50
}
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started"
}
```

#### Control Pipeline

```http
POST /api/v1/pipeline/control
Content-Type: application/json

{
  "action": "pause",  // pause, resume, stop, cancel
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "force": false
}
```

**Response:**
```json
{
  "status": "success",
  "action": "pause"
}
```

#### Get Pipeline Status

```http
GET /api/v1/pipeline/status/{session_id}
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "current_phase": "training",
  "phases_completed": ["cognate", "evomerge", "quietstar"],
  "phases_remaining": ["bitnet", "baking", "adas", "compression"],
  "total_progress_percent": 45.5,
  "elapsed_seconds": 1250.3,
  "estimated_remaining_seconds": 1500.0,
  "phase_metrics": [...],
  "agent_count": 12,
  "active_agents": [...]
}
```

### Swarm Management

#### Get Swarm Status

```http
GET /api/v1/pipeline/swarm/{session_id}
```

**Response:**
```json
{
  "topology": "hierarchical",
  "total_agents": 50,
  "active_agents": 12,
  "idle_agents": 38,
  "agents": [...],
  "memory_usage_mb": 4096.0,
  "cpu_usage_percent": 45.2,
  "gpu_usage_percent": 78.3
}
```

### Quality Gates

#### Validate Quality Gates

```http
POST /api/v1/pipeline/quality-gates/{session_id}?phase=training
```

**Response:**
```json
{
  "phase": "training",
  "all_gates_passed": true,
  "gates_passed": 8,
  "gates_failed": 0,
  "gate_results": [
    {
      "gate_name": "nasa_pot10_compliance",
      "passed": true,
      "score": 0.95,
      "threshold": 0.90,
      "details": {...},
      "recommendations": []
    }
  ],
  "theater_detection": {
    "theater_detected": false,
    "confidence": 0.92,
    "theater_score": 0.15,
    "indicators": {...}
  },
  "can_proceed": true
}
```

### Checkpoint Management

#### Save Checkpoint

```http
POST /api/v1/pipeline/checkpoint/save
Content-Type: application/json

{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "checkpoint_name": "phase3_complete",
  "include_model_state": true,
  "include_swarm_state": true
}
```

#### Load Checkpoint

```http
POST /api/v1/pipeline/checkpoint/load/{checkpoint_id}
```

### Configuration Presets

#### List Presets

```http
GET /api/v1/pipeline/presets
```

**Response:**
```json
["quick_test", "full_pipeline", "compression_only"]
```

#### Get Preset

```http
GET /api/v1/pipeline/preset/full_pipeline
```

#### Save Preset

```http
POST /api/v1/pipeline/preset/save
Content-Type: application/json

{
  "preset_name": "custom_training",
  "config": {
    "phases": ["training", "baking"],
    "max_agents": 30
  }
}
```

### Execution History

#### Get History

```http
GET /api/v1/pipeline/history?limit=10&status=completed
```

**Response:**
```json
[
  {
    "session_id": "...",
    "start_time": "2024-01-15T10:30:00Z",
    "end_time": "2024-01-15T12:45:00Z",
    "status": "completed",
    "phases": [...],
    "success": true,
    "duration_seconds": 8100.0
  }
]
```

### Health Check

```http
GET /api/v1/pipeline/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400.0,
  "active_sessions": 3,
  "total_memory_mb": 16384.0,
  "available_memory_mb": 8192.0
}
```

## WebSocket Channels

All WebSocket endpoints support optional `session_id` query parameter for session-specific updates.

### Agent Updates

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/agents?session_id=...');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // {
  //   "event_type": "agent_update",
  //   "agent_id": "agent_123",
  //   "state": "active",
  //   "task": "compression",
  //   "timestamp": "2024-01-15T10:30:00Z"
  // }
};
```

### Task Progress

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/tasks');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // {
  //   "event_type": "phase_complete",
  //   "phase": "training",
  //   "success": true,
  //   "duration_seconds": 3600.0,
  //   "metrics": {...}
  // }
};
```

### Performance Metrics

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/metrics?session_id=...');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // {
  //   "event_type": "metrics",
  //   "metrics": {
  //     "cpu_percent": 45.2,
  //     "memory_mb": 2048.5,
  //     "gpu_utilization": 78.3
  //   }
  // }
};
```

### Pipeline Progress

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/pipeline');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // {
  //   "event_type": "pipeline_progress",
  //   "phase": "training",
  //   "progress_percent": 65.5,
  //   "metrics": {...}
  // }
};
```

### Combined Dashboard

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/dashboard?session_id=...');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Combined updates for all metrics
};
```

## Data Models

### PipelinePhase (Enum)

- `cognate`: Model creation
- `evomerge`: Evolutionary optimization
- `quietstar`: Reasoning enhancement
- `bitnet`: Initial compression
- `training`: Main training
- `baking`: Tool/persona integration
- `adas`: Architecture search
- `compression`: Final compression

### PipelineStatus (Enum)

- `idle`: No active execution
- `initializing`: Setup in progress
- `running`: Active execution
- `paused`: Temporarily paused
- `completed`: Successfully finished
- `failed`: Execution failed
- `cancelled`: Manually stopped

### SwarmTopology (Enum)

- `hierarchical`: Tree structure with coordinator
- `mesh`: Peer-to-peer collaboration
- `star`: Centralized hub control
- `ring`: Sequential processing

## Error Handling

All endpoints return standard error responses:

```json
{
  "error": "Session not found",
  "error_code": "SESSION_NOT_FOUND",
  "details": {
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (session/resource not found)
- `500`: Internal Server Error

## Examples

### Complete Pipeline Execution

```python
import requests
import json

# Start pipeline
response = requests.post('http://localhost:8000/api/v1/pipeline/start', json={
    "phases": ["cognate", "evomerge", "training"],
    "enable_monitoring": True,
    "max_agents": 30
})

session_id = response.json()['session_id']

# Monitor progress
status = requests.get(f'http://localhost:8000/api/v1/pipeline/status/{session_id}')
print(f"Progress: {status.json()['total_progress_percent']}%")

# Validate quality gates
gates = requests.post(
    f'http://localhost:8000/api/v1/pipeline/quality-gates/{session_id}',
    params={"phase": "training"}
)

if gates.json()['all_gates_passed']:
    print("Quality gates passed!")
```

### WebSocket Monitoring

```python
import asyncio
import websockets
import json

async def monitor_pipeline(session_id):
    uri = f"ws://localhost:8000/ws/dashboard?session_id={session_id}"

    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            print(f"Status: {data['data']['pipeline_status']}")
            print(f"Phase: {data['data']['current_phase']}")
            print(f"Progress: {data['data']['progress_percent']}%")

asyncio.run(monitor_pipeline("your-session-id"))
```

### Using Configuration Presets

```python
# Get preset
preset = requests.get('http://localhost:8000/api/v1/pipeline/preset/full_pipeline')
config = preset.json()['config']

# Start with preset
response = requests.post('http://localhost:8000/api/v1/pipeline/start', json={
    **config,
    "enable_monitoring": True
})
```

## Production Considerations

### Security

- Implement authentication (JWT tokens recommended)
- Use HTTPS in production
- Validate all input parameters
- Rate limit endpoints
- Implement proper CORS policies

### Scalability

- Use Redis for session state (distributed deployments)
- Implement connection pooling
- Configure appropriate worker counts
- Monitor WebSocket connection limits

### Monitoring

- Track API response times
- Monitor WebSocket connection health
- Log all pipeline executions
- Set up alerting for failures

## Support

For issues and questions:
- GitHub Issues: [agent-forge/issues](https://github.com/agent-forge/issues)
- Documentation: [agent-forge/docs](https://github.com/agent-forge/docs)
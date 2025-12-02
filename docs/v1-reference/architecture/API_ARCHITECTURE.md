# Agent Forge API Architecture

## System Architecture Overview

```
Client Applications (React/Vue/CLI/SDK)
    |
    | HTTP/WebSocket
    v
FastAPI Application
    |
    |-- CORS Middleware
    |-- Request Validation (Pydantic)
    |-- Error Handling
    |
    |-- REST API Routes (/api/v1/pipeline/*)
    |-- WebSocket Routes (/ws/*)
    |
    v
Pipeline Service Layer
    |
    |-- Session Management
    |-- Pipeline Orchestration
    |-- Quality Gate Coordination
    |-- Checkpoint Management
    |
    v
Agent Forge Core
    |
    |-- Swarm Coordinator
    |-- Swarm Execution Manager
    |-- Swarm Monitor
    |-- Unified Pipeline (8 Phases)
```

## Component Responsibilities

### FastAPI Application
- CORS middleware configuration
- Route registration
- OpenAPI documentation
- Error handling

### Pipeline Service
- Session management (create/track/cleanup)
- Execution control (start/pause/resume/stop)
- State management (progress/metrics)
- Integration with swarm components

### Connection Manager
- Multi-channel WebSocket management
- Event broadcasting
- Session-specific streams
- Auto cleanup

### Swarm Coordinator
- Topology management
- Agent spawning
- Resource allocation
- Memory management

## API Endpoints Summary

**Pipeline Control (4 endpoints)**
- POST /api/v1/pipeline/start
- POST /api/v1/pipeline/control
- GET /api/v1/pipeline/status/{id}
- GET /api/v1/pipeline/swarm/{id}

**Quality Gates (1 endpoint)**
- POST /api/v1/pipeline/quality-gates/{id}

**State Management (6 endpoints)**
- POST /api/v1/pipeline/checkpoint/save
- POST /api/v1/pipeline/checkpoint/load/{id}
- GET /api/v1/pipeline/presets
- GET /api/v1/pipeline/preset/{name}
- POST /api/v1/pipeline/preset/save
- GET /api/v1/pipeline/history

**System (2 endpoints)**
- GET /api/v1/pipeline/health
- GET /api/v1/info

**WebSocket (6 channels)**
- WS /ws/agents
- WS /ws/tasks
- WS /ws/metrics
- WS /ws/pipeline
- WS /ws/dashboard
- GET /ws/stats (HTTP)
- GET /ws/client (test client)

## WebSocket Event Types

### PipelineProgressEvent
- Phase progress updates
- Overall pipeline status
- Metrics snapshots

### AgentUpdateEvent
- Agent state changes
- Task assignments
- Resource utilization

### PhaseCompletionEvent
- Phase completion notification
- Success/failure status
- Duration and metrics

### MetricsStreamEvent
- Real-time performance metrics
- CPU/GPU/Memory usage
- Custom phase metrics

### ErrorEvent
- Error notifications
- Recoverable/non-recoverable
- Phase-specific errors

## Security Features

### Current
- CORS middleware
- Pydantic input validation
- Session isolation (UUID)
- Error sanitization

### Production Enhancements
- JWT authentication
- Role-based access control
- Rate limiting
- API keys
- Audit logging
- HTTPS/TLS

## Scaling Architecture

### Horizontal Scaling
- Stateless API design
- Redis for session state
- Redis pub/sub for WebSocket
- Load balancing support

### Performance
- Async/await throughout
- Background task execution
- Connection pooling
- Efficient serialization

### Resource Management
- Per-agent memory limits
- GPU allocation control
- Automatic cleanup
- Checkpoint optimization
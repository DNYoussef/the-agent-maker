# Phases 1-4: UI Integration Quick Reference

**Version**: 2.0
**Date**: 2025-10-16
**Purpose**: Quick reference for UI integration with each phase

---

## Overview

This document provides the **essential UI integration information** for each phase. For complete UI specifications, see [PHASE1-4_UI_SPECIFICATION_V2.md](PHASE1-4_UI_SPECIFICATION_V2.md).

---

## Phase 1: Cognate (3 Foundation Models)

### UI Components

| Component | Purpose | Key Controls |
|-----------|---------|--------------|
| **Configuration Panel** | Pre-launch setup | ACT thresholds, LTM sizes, dataset selection, optimizer settings |
| **Real-Time Training View** | Monitor 3 models | Loss curves, diversity metrics, VRAM usage, progress bars |
| **Detailed Model View** | Per-model deep dive | ACT analysis, LTM usage, dataset progress |
| **Failure Intervention Modals** | Handle critical issues | Diversity failure, OOM, convergence problems |
| **Completion Summary** | Phase results | Model comparison, training summary, diversity score |

### API Endpoints

```
POST /api/phases/1/configure    # Set 3-model configs
POST /api/phases/1/start        # Start training
GET  /api/phases/1/status       # Real-time status
GET  /api/phases/1/diversity    # Diversity metrics
POST /api/phases/1/pause        # Pause training
POST /api/phases/1/resume       # Resume training
POST /api/phases/1/checkpoint   # Manual checkpoint
POST /api/phases/1/intervene    # Handle failure modes
```

### WebSocket Events

```javascript
socket.on('phase:progress', ...)   // Every step: loss, progress
socket.on('phase:metric', ...)     // Every 100 steps: detailed metrics
socket.on('phase:alert', ...)      // Immediate: failure alerts
socket.on('phase:complete', ...)   // Once: phase completion
socket.on('hardware:update', ...)  // Every 10s: VRAM, RAM, GPU, temp
```

### Key User Actions

1. **Configure Models**: Set ACT thresholds (0.95, 0.90, 0.99) for diversity
2. **Select Datasets**: Choose from 16 available datasets
3. **Enable Protections**: OOM prevention, diversity monitoring, convergence detection
4. **Monitor Training**: Watch loss curves, diversity metrics, resource usage
5. **Intervene When Needed**: Respond to diversity failures, OOM, convergence issues
6. **Validate Results**: Check diversity score >0.7, accuracy metrics
7. **Proceed to Phase 2**: Click "Start Phase 2" when satisfied

### Critical Metrics to Display

- **Per Model**: Loss, perplexity, ACT halting steps, LTM usage
- **Diversity**: Halting diversity (>2.0), memory diversity (>0.3), speed diversity (>10ms)
- **Hardware**: VRAM (should stay <5GB), RAM, GPU utilization, temperature
- **Progress**: Current epoch, total steps, ETA

---

## Phase 2: EvoMerge (Evolution 3 → 1)

### UI Components

| Component | Purpose | Key Controls |
|-----------|---------|--------------|
| **Evolution Dashboard** | Monitor 50 generations | Fitness curve, population table, diversity monitor |
| **3D Merge Visualization** | Visual evolution tree | Three.js 3D graph, rotate/zoom controls |
| **Combo Statistics Panel** | Merge technique analysis | Performance by combo, best techniques |
| **Diversity Intervention Modal** | Handle convergence | Random injection, mutation rate increase |
| **Completion Summary** | Champion results | Fitness improvement, evolution stats |

### API Endpoints

```
POST /api/phases/2/configure      # Set evolution params
POST /api/phases/2/start          # Start evolution
GET  /api/phases/2/status         # Generation status
GET  /api/phases/2/population     # Current 8 models
GET  /api/phases/2/diversity      # Population diversity
POST /api/phases/2/pause          # Pause evolution
POST /api/phases/2/resume         # Resume evolution
POST /api/phases/2/intervene      # Diversity boost
```

### WebSocket Events

```javascript
socket.on('phase:progress', ...)   // Every generation: fitness, population
socket.on('phase:metric', ...)     // Per generation: diversity, best model
socket.on('phase:alert', ...)      // Immediate: low diversity warning
socket.on('phase:complete', ...)   // Once: champion evolved
```

### Key User Actions

1. **Configure Evolution**: Set generation count (default: 50), mutation rate
2. **Monitor Fitness**: Watch fitness curve improve (target: +20%)
3. **Track Diversity**: Ensure population diversity >0.25
4. **View Merge Tree**: Visualize evolution in 3D
5. **Intervene if Needed**: Inject random models, increase mutation
6. **Analyze Results**: Check champion fitness, combo statistics
7. **Proceed to Phase 3**: Click "Start Phase 3" with evolved champion

### Critical Metrics to Display

- **Fitness**: Current best, average, improvement % from Phase 1
- **Population**: 8 models with ranks, combos, ages
- **Diversity**: Population diversity score (>0.25)
- **Evolution**: Current generation, best generation, plateau detection
- **Combos**: Usage statistics for 8 merge combinations

---

## Phase 3: Quiet-STaR (Add Reasoning)

### UI Components

| Component | Purpose | Key Controls |
|-----------|---------|--------------|
| **Two-Step Overview** | Show Baking → RL flow | Step 1 progress, Step 2 progress |
| **Data Generation Setup** | Configure OpenRouter | API settings, cost monitoring, strategy selection |
| **Data Generation Progress** | Monitor $100-200 spend | Progress by strategy, cost tracking, quality metrics |
| **Step 1 (Baking) View** | Monitor supervised learning | Strategy-specific accuracy, token usage |
| **Step 2 (RL) View** | Monitor reinforcement learning | Reward, coherence, anti-theater checks |
| **Theater Detection Modal** | Handle fake reasoning | Divergence, length, diversity tests, rollback options |
| **Completion Summary** | Reasoning results | Performance comparison, anti-theater validation |

### API Endpoints

```
POST /api/phases/3/generate-data    # Start data generation
GET  /api/phases/3/generation-status # Data gen progress
POST /api/phases/3/step1/start      # Start baking
GET  /api/phases/3/step1/status     # Baking status
POST /api/phases/3/step2/start      # Start RL
GET  /api/phases/3/step2/status     # RL status
POST /api/phases/3/anti-theater     # Run anti-theater check
POST /api/phases/3/pause            # Pause phase
POST /api/phases/3/resume           # Resume phase
```

### WebSocket Events

```javascript
socket.on('phase:progress', ...)   // Every step: loss, reward, progress
socket.on('phase:metric', ...)     // Per batch/episode: detailed metrics
socket.on('phase:alert', ...)      // Immediate: theater detection, cost limit
socket.on('phase:complete', ...)   // Once: reasoning enhanced
```

### Key User Actions

1. **Setup Data Generation**: Configure OpenRouter API, set cost limit ($200)
2. **Monitor Cost**: Track spending per model, stop if limit exceeded
3. **Validate Data Quality**: Check valid examples >99%
4. **Monitor Baking**: Watch strategy-specific accuracy (target: 85%)
5. **Monitor RL**: Track reward, coherence, thought generation
6. **Anti-Theater Checks**: Respond to theater alerts every 1000 steps
7. **Rollback if Needed**: Revert to checkpoint, adjust hyperparameters
8. **Validate Results**: Ensure anti-theater tests pass, accuracy +3%
9. **Proceed to Phase 4**: Click "Start Phase 4" with reasoning model

### Critical Metrics to Display

- **Data Generation**: Progress per strategy, total cost, valid examples
- **Baking (Step 1)**: Overall accuracy, strategy-specific accuracy, token usage
- **RL (Step 2)**: Average reward, success rate, coherence score, KL divergence
- **Thoughts**: Avg thoughts/token, avg thought length, diversity
- **Anti-Theater**: Divergence (>0.30), length (>5.0), diversity (>3.0), ablation (>2%)

---

## Phase 4: BitNet (1.58-bit Compression)

### UI Components

| Component | Purpose | Key Controls |
|-----------|---------|--------------|
| **Compression Configuration** | Adaptive setup | Sparsity threshold, target compression, preserved layers |
| **Layer-by-Layer Progress** | Monitor compression | Layer status, compression ratio, accuracy drop per layer |
| **Sparsity Heatmap** | Visualize weights | Weight distribution, sparsity by region |
| **Accuracy Drop Modal** | Handle >10% drop | Revert layer, conservative config, extended fine-tuning |
| **Completion Summary** | Compression results | Compression ratio, accuracy drop, speedup |

### API Endpoints

```
POST /api/phases/4/configure      # Set compression params
POST /api/phases/4/start          # Start compression
GET  /api/phases/4/status         # Layer-by-layer status
GET  /api/phases/4/sparsity       # Sparsity heatmap data
POST /api/phases/4/pause          # Pause compression
POST /api/phases/4/resume         # Resume compression
POST /api/phases/4/revert-layer   # Revert layer quantization
```

### WebSocket Events

```javascript
socket.on('phase:progress', ...)   // Every layer: compression, accuracy
socket.on('phase:metric', ...)     // Per layer: sparsity, accuracy drop
socket.on('phase:alert', ...)      // Immediate: high accuracy drop
socket.on('phase:complete', ...)   // Once: compression complete
```

### Key User Actions

1. **Configure Compression**: Set sparsity threshold (auto-adjusted by model size)
2. **Set Target**: Accept adaptive target (6x tiny → 12x large)
3. **Monitor Progress**: Watch layer-by-layer compression
4. **Check Accuracy**: Ensure cumulative drop <10%
5. **Intervene if Needed**: Revert problematic layers to FP16
6. **View Sparsity**: Visualize weight distribution heatmap
7. **Validate Results**: Check compression ratio, accuracy drop, speedup
8. **Proceed to Phase 5**: Click "Start Phase 5" with compressed model

### Critical Metrics to Display

- **Overall**: Compression ratio, target ratio, accuracy drop (must be <10%)
- **Per Layer**: Compression ratio, accuracy drop contribution, sparsity %
- **Weights**: -1 %, 0 %, +1 % distribution
- **Size**: Original size, current size, projected final size
- **Performance**: Inference time before/after, speedup ratio

---

## Cross-Phase UI Elements

### Master Dashboard

**Components**:
- Pipeline progress bar (0-100% for all 8 phases)
- Phase cards (Phases 1-8) with status indicators
- Active phase display with current step
- Resource summary (GPU, VRAM, RAM, disk, temp)
- Alerts & notifications panel

**API Endpoint**:
```
GET /api/pipeline/status
  Response: {
    "current_phase": "cognate",
    "overall_progress": 0.12,
    "phases": [
      {"id": 1, "status": "in_progress", "progress": 0.60},
      {"id": 2, "status": "pending"},
      ...
    ],
    "hardware": {"vram": 4.8, "ram": 8.2, "gpu": 95, "temp": 72},
    "alerts": [...]
  }
```

### W&B Integration Panel

**Components**:
- Connection status indicator
- Active runs list
- Quick links to phase dashboards
- Settings button

**API Endpoints**:
```
GET /api/wandb/runs
  Response: {
    "runs": [
      {"id": "run_123", "phase": "cognate", "status": "completed"},
      ...
    ]
  }

GET /api/wandb/settings
POST /api/wandb/settings
```

### Settings Panel

**Global Settings**:
- Hardware configuration (GPU selection, VRAM limit)
- Failure detection toggles (per phase)
- Thermal management (max temp, throttling)
- W&B configuration

**API Endpoint**:
```
GET /api/settings
POST /api/settings
  Body: {
    "hardware": {...},
    "failure_detection": {...},
    "thermal": {...},
    "wandb": {...}
  }
```

---

## Implementation Priority

### Phase 1 (Week 1-2)
1. Configuration panel with 3-model setup
2. Real-time training view with loss curves
3. Diversity metrics dashboard
4. OOM intervention modal
5. Completion summary

### Phase 2 (Week 3-4)
1. Evolution dashboard with fitness curve
2. Population table
3. Diversity monitoring
4. Combo statistics
5. Completion summary

### Phase 3 (Week 5-6)
1. Data generation setup & progress
2. Two-step overview
3. Step 1 (Baking) monitoring
4. Step 2 (RL) monitoring with anti-theater
5. Theater detection modal
6. Completion summary

### Phase 4 (Week 7-8)
1. Compression configuration (adaptive)
2. Layer-by-layer progress
3. Sparsity heatmap
4. Accuracy drop intervention
5. Completion summary

### Cross-Phase (Week 9-10)
1. Master dashboard
2. W&B integration panel
3. Global settings
4. Mobile responsiveness
5. Accessibility compliance

---

## Testing Checklist

### Functional Testing
- [ ] All API endpoints respond correctly
- [ ] WebSocket events fire at correct times
- [ ] Real-time updates display without lag
- [ ] Intervention modals trigger on failure modes
- [ ] User actions execute correctly (pause, resume, checkpoint)

### Visual Testing
- [ ] Loss curves render correctly (Recharts/Plotly)
- [ ] 3D merge visualization loads (Three.js)
- [ ] Sparsity heatmaps display properly (D3.js)
- [ ] All icons and status indicators visible
- [ ] Color scheme consistent across phases

### Responsive Testing
- [ ] Desktop layout (≥1440px) displays correctly
- [ ] Laptop layout (1024-1439px) adapts properly
- [ ] Tablet layout (768-1023px) stacks correctly
- [ ] Mobile layout (<768px) is single-column and swipeable

### Accessibility Testing
- [ ] Keyboard navigation works (Tab, Enter, Space)
- [ ] Screen reader announces all elements
- [ ] Status icons don't rely on color alone
- [ ] High contrast mode supported
- [ ] Font sizes adjustable

### Integration Testing
- [ ] Phase 1 → Phase 2 handoff works
- [ ] Phase 2 → Phase 3 handoff works
- [ ] Phase 3 → Phase 4 handoff works
- [ ] W&B dashboards link correctly
- [ ] All metrics logged to W&B

---

## Quick Reference Tables

### Phase Completion Criteria (User Validation)

| Phase | User Must Verify | Target Metrics | Action |
|-------|------------------|----------------|--------|
| **Phase 1** | Diversity score | Diversity >0.7, GSM8K >10% | Click "Start Phase 2" |
| **Phase 2** | Fitness improvement | Fitness +20%, diversity >0.3 | Click "Start Phase 3" |
| **Phase 3** | Anti-theater tests | All tests pass, accuracy +3% | Click "Start Phase 4" |
| **Phase 4** | Compression results | Compression >6x, drop <10% | Click "Start Phase 5" |

### Critical User Interventions

| Phase | Failure Mode | User Action | Frequency |
|-------|-------------|-------------|-----------|
| **Phase 1** | Diversity failure | Apply aggressive divergence or continue | Every epoch |
| **Phase 1** | OOM | Accept auto-recovery or manual config | On OOM |
| **Phase 2** | Low diversity | Inject random model or increase mutation | Every 5 generations |
| **Phase 2** | Degenerate merge | Automatic fallback to linear merge | Per merge |
| **Phase 3** | Theater detected | Rollback and adjust hyperparameters | Every 1000 steps |
| **Phase 4** | High accuracy drop | Revert layer or use conservative config | Per layer |

### Resource Monitoring Thresholds

| Resource | Warning | Critical | User Action |
|----------|---------|----------|-------------|
| **VRAM** | >80% (4.8GB) | >95% (5.7GB) | Reduce batch size or enable gradient checkpointing |
| **RAM** | >75% (12GB) | >90% (14.4GB) | Close other applications |
| **GPU Temp** | >75°C | >80°C | Pause training, check cooling |
| **Disk Space** | <10GB free | <5GB free | Clean up old checkpoints |

---

## Support and Troubleshooting

### Common Issues

**Issue**: UI not updating in real-time
- **Check**: WebSocket connection status
- **Fix**: Reconnect WebSocket, check API server running

**Issue**: Loss curves not displaying
- **Check**: Metrics being logged to W&B
- **Fix**: Verify W&B integration, check API response format

**Issue**: Intervention modals not appearing
- **Check**: Failure detection enabled in settings
- **Fix**: Enable automatic interventions, check alert logic

**Issue**: Phase handoff fails
- **Check**: Previous phase completed successfully
- **Fix**: Verify outputs from previous phase, check storage

---

**Version**: 2.0
**Date**: 2025-10-16
**Status**: ✅ Complete Quick Reference for UI Implementation
**See Also**: [PHASE1-4_UI_SPECIFICATION_V2.md](PHASE1-4_UI_SPECIFICATION_V2.md) for detailed mockups and specs

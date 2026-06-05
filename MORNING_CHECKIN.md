# Agent Forge - FULL OVERNIGHT RUN

## Current Status: Phase 2 FULL EvoMerge Running

**Started**: 2025-12-07 15:25 UTC
**Expected Duration**: 6-8 hours
**Container**: `agent-forge-evomerge-full`

### What's Different This Time

| Attempt | Issue | Fix |
|---------|-------|-----|
| V2 | Only Linear+SLERP | Added all 6 techniques |
| Proper | Fake 2-sec benchmarks | Real GSM8K + perplexity |
| **FULL** | **Correct!** | 50 GSM8K + 100 PPL per model, ~1 min each |

### Current Configuration

- **8 models** per generation (all binary combos)
- **50 generations** (NO early stopping)
- **50 GSM8K samples** per model (~40 sec)
- **100 perplexity samples** per model (~20 sec)
- **~1 minute** per model evaluation
- **~8 minutes** per generation
- **~6-8 hours** total

---

## Quick Commands

### 1. Check if still running
```bash
ssh -p 2222 david@w.m1el.eu "docker ps --format '{{.Names}} {{.Status}}' | grep evomerge"
```

### 2. Check progress
```bash
ssh -p 2222 david@w.m1el.eu "docker logs --tail 50 agent-forge-evomerge-full"
```

### 3. Check latest checkpoint
```bash
ssh -p 2222 david@w.m1el.eu "ls -la ~/agent-forge/checkpoints/phase2_full/"
```

### 4. Check metrics
```bash
ssh -p 2222 david@w.m1el.eu "cat ~/agent-forge/checkpoints/phase2_full/checkpoint_gen*_metrics.json | tail -20"
```

### 5. Watch live (Ctrl+C to stop)
```bash
ssh -p 2222 david@w.m1el.eu "docker logs -f agent-forge-evomerge-full"
```

---

## Server Details

| Field | Value |
|-------|-------|
| Host | w.m1el.eu |
| Port | 2222 |
| User | david |
| Connect | `ssh -p 2222 david@w.m1el.eu` |
| GPU | RTX 4080 16GB |

---

## Expected Output

### Files
```
~/agent-forge/checkpoints/phase2_full/
  checkpoint_gen000.safetensors    (initial)
  checkpoint_gen005.safetensors    (every 5 gens)
  checkpoint_gen010.safetensors
  ...
  checkpoint_gen050.safetensors
  final_champion.safetensors       (best model)
  final_champion_metrics.json      (final metrics)
```

### Expected Results

Based on the correct 3-stage pipeline:
- **GSM8K**: 4% -> 10%+ (2.5x improvement)
- **Perplexity**: Should improve if model produces valid outputs
- **Fitness**: +20%+ improvement over 50 generations

---

## The 3-Stage Pipeline (CORRECT)

```
Stage 1 (Bit 0): Interpolation
  0 = Linear (weighted average)
  1 = SLERP (spherical interpolation)

Stage 2 (Bit 1): Task Arithmetic
  0 = DARE (drop 90%, rescale 10%)
  1 = TIES (trim top 20%, elect sign)

Stage 3 (Bit 2): Selection
  0 = FrankenMerge (layer-wise)
  1 = DFS (importance-weighted)

8 Binary Combinations:
  000 = Linear + DARE + Franken
  001 = SLERP + DARE + Franken
  010 = Linear + TIES + Franken
  011 = SLERP + TIES + Franken
  100 = Linear + DARE + DFS
  101 = SLERP + DARE + DFS
  110 = Linear + TIES + DFS
  111 = SLERP + TIES + DFS  <-- Usually best
```

---

## Evolution Strategy

Each generation:
1. **Evaluate** all 8 models (GSM8K + perplexity)
2. **Sort** by fitness (40% PPL + 30% Acc + 20% Speed + 10% Memory)
3. **Elite**: Top 2 -> mutate 3x each = 6 children
4. **Loser**: Bottom 6 -> 2 groups of 3 -> merge = 2 children
5. **New population**: 6 elite + 2 loser = 8 models

---

## If Something Goes Wrong

### Container crashed
```bash
# Check error
ssh -p 2222 david@w.m1el.eu "docker logs agent-forge-evomerge-full 2>&1 | tail -100"

# Restart
ssh -p 2222 david@w.m1el.eu "docker rm -f agent-forge-evomerge-full; cd ~/agent-forge && docker run -d --name agent-forge-evomerge-full --gpus all -e PYTHONUNBUFFERED=1 -v \$(pwd)/checkpoints:/app/checkpoints -v \$(pwd)/scripts:/app/scripts -v \$(pwd)/src:/app/src agent-forge:phase1 python /app/scripts/run_evomerge_full.py"
```

### GPU out of memory
```bash
ssh -p 2222 david@w.m1el.eu "nvidia-smi"
```

---

## Next Steps After Completion

### 1. Check final results
```bash
ssh -p 2222 david@w.m1el.eu "cat ~/agent-forge/checkpoints/phase2_full/final_champion_metrics.json"
```

### 2. Benchmark champion
```bash
ssh -p 2222 david@w.m1el.eu "cd ~/agent-forge && docker run --rm --gpus all -v \$(pwd):/app agent-forge:phase1 python /app/scripts/benchmark_phase1.py --checkpoint /app/checkpoints/phase2_full/final_champion.safetensors --gsm8k-samples 200"
```

### 3. Download champion
```bash
scp -P 2222 david@w.m1el.eu:~/agent-forge/checkpoints/phase2_full/final_champion.safetensors ./
```

### 4. Continue to Phase 3 (Quiet-STaR)
```bash
# Phase 3 adds reasoning via thought generation
cat docs/phases/phase3/PHASE3_COMPLETE_GUIDE.md
```

---

## Scripts Reference

| Script | Purpose | Status |
|--------|---------|--------|
| `run_evomerge_full.py` | CORRECT - Real benchmarks, 50 gens | RUNNING |
| `run_evomerge_proper.py` | WRONG - Fake fast benchmarks | Superseded |
| `run_evomerge_overnight_v2.py` | WRONG - Only Linear+SLERP | Superseded |

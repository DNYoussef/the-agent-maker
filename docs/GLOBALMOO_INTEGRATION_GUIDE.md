# GlobalMOO + pymoo Integration Guide for Agent Forge V2

## Overview

This guide explains how to use the hybrid MOO (Multi-Objective Optimization) system
that combines **GlobalMOO** (cloud-based Bayesian optimization) with **pymoo**
(local NSGA-II) for Agent Forge's 8-phase pipeline.

**Source**: Adapted from meta-calculus-toolkit's proven MOO integration.

---

## Quick Start

### 1. Install Dependencies

```bash
# pymoo (always available, local)
pip install pymoo

# GlobalMOO SDK (optional, cloud-based)
pip install globalmoo-sdk
```

### 2. Set API Key (for GlobalMOO)

```bash
# Windows PowerShell
$env:GLOBALMOO_API_KEY = "your_api_key_here"

# Windows CMD
set GLOBALMOO_API_KEY=your_api_key_here

# Linux/Mac
export GLOBALMOO_API_KEY="your_api_key_here"
```

Get your API key from: https://app.globalmoo.com

### 3. Basic Usage

```python
from src.cross_phase.meta_calculus.moo_bridge import (
    MOORunner, MOOConfig, EvoMergeProblem, ExpertDiscoveryProblem
)

# Phase 2: EvoMerge optimization
def evaluate_merge(params):
    # Your evaluation logic
    return {"perplexity": 10.5, "spectral_gap": 0.85, ...}

problem = EvoMergeProblem(model_evaluator=evaluate_merge)
config = MOOConfig(n_generations=50, population_size=40)
result = MOORunner(config).optimize(problem)

# Access Pareto front
for sol in result["pareto_front"]:
    print(f"Params: {sol['params']}, Objectives: {sol['objectives']}")
```

---

## Architecture: Hybrid MOO System

```
                    +------------------+
                    |   Agent Forge    |
                    |   Phase N        |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   MOORunner      |
                    |   (Orchestrator) |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
     +----------------+            +------------------+
     |    pymoo       |            |   GlobalMOO      |
     | (Local NSGA-II)|            | (Cloud Bayesian) |
     +----------------+            +------------------+
           |                              |
           |  Always available            |  Requires API key
           |  Fast for <100 evals         |  Better for expensive evals
           |  No rate limits              |  Builds surrogate model
           |                              |
           +-------------+----------------+
                         |
                         v
                +------------------+
                |  Pareto Front    |
                | (Unified Output) |
                +------------------+
```

### When to Use Which

| Scenario | Use | Reason |
|----------|-----|--------|
| Local development | pymoo | No API key needed, fast |
| Expensive evaluations (>5 min each) | GlobalMOO | Surrogate model reduces evals |
| Need reproducibility | pymoo | Deterministic with seed |
| Want cloud scaling | GlobalMOO | Run from anywhere |
| CI/CD integration | pymoo | No external dependencies |
| Research publication | Both | Compare results |

---

## Phase-Specific MOO Configurations

### Phase 2: EvoMerge

**Objectives** (5):
1. `perplexity` - Minimize task loss
2. `spectral_gap` - Maximize representation diversity
3. `weight_norm` - Minimize for compression readiness
4. `invariance_score` - Maximize cross-seed agreement
5. `fragility` - Minimize solution sensitivity

**Decision Variables**:
- Per-layer merge ratios (0-1)
- Merge technique weights (Linear, SLERP, TIES, DARE, etc.)

```python
from src.cross_phase.meta_calculus.moo_bridge import EvoMergeProblem

problem = EvoMergeProblem(
    n_layers=12,
    n_techniques=6,
    model_evaluator=your_evaluator
)
```

### Phase 7: Expert Discovery

**Objectives** (5):
1. `task_loss` - Minimize primary performance
2. `expert_diversity` - Maximize spectral gap of expert weights
3. `routing_entropy` - Maximize balanced utilization
4. `compute_cost` - Minimize inference FLOPs
5. `robustness` - Maximize cross-seed stability

**Decision Variables**:
- Number of experts (4-64)
- Top-k selection (1-8)
- Capacity factor (1.0-2.0)
- Router temperature

```python
from src.cross_phase.meta_calculus.moo_bridge import ExpertDiscoveryProblem

problem = ExpertDiscoveryProblem(
    expert_evaluator=your_evaluator
)
```

---

## GlobalMOO API Integration

### SDK Documentation

- **Official Docs**: https://globalmoo.gitbook.io/globalmoo-documentation
- **GitHub SDK**: https://github.com/globalMOO/gmoo-sdk-suite
- **API Endpoint**: https://app.globalmoo.com/api/

### Setup Workflow

```python
import os
from src.cross_phase.meta_calculus.moo_bridge import GlobalMOOAdapter

# Set API key
os.environ['GLOBALMOO_API_KEY'] = 'your_key'

# Create adapter
adapter = GlobalMOOAdapter()

# Export configuration for GlobalMOO
config = adapter.generate_api_config()
print(config)

# Generate initial samples
samples = adapter.export_sample_data(n_samples=20)
```

### Full Optimization Loop

```python
from src.cross_phase.meta_calculus.moo_bridge import (
    GlobalMOOClient, GlobalMOOAdapter, PhysicsOracle
)

# Initialize
client = GlobalMOOClient(debug=True)
adapter = GlobalMOOAdapter()

# Check connection
connection = client.check_connection()
if not connection['connected']:
    print("Falling back to pymoo...")
    # Use pymoo instead

# Run optimization
result = client.run_optimization(
    adapter,
    n_iterations=50,
    verbose=True
)

# Process results
if result['success']:
    for sol in result['pareto_front']:
        print(f"Solution: {sol}")
```

---

## Constraint Handling

### Built-in Constraints

```python
from src.cross_phase.meta_calculus.moo_utils.constraints import (
    make_param_constraint,
    make_latency_constraint,
    make_accuracy_constraint,
    ConstraintSet
)

# Create constraint set
cs = ConstraintSet()
cs.add("max_params", make_param_constraint(25_000_000))
cs.add("max_latency", make_latency_constraint(100))  # ms

# Check feasibility
if cs.is_feasible(solution):
    print("Solution is valid!")
else:
    print(f"Violations: {cs.get_violations(cs.evaluate(solution))}")
```

### Agent Forge Phase Constraints

| Phase | Constraint | Value |
|-------|------------|-------|
| 1 (Cognate) | Max params | 25M |
| 1 (Cognate) | Max VRAM | 6GB |
| 4 (BitNet) | Min accuracy retention | 95% |
| 7 (Experts) | Min diversity gap | 0.1 |
| 8 (Compress) | Target size | 0.4MB |

---

## Pareto Selection Strategies

### Available Methods

```python
from src.cross_phase.meta_calculus.moo_utils.selection import (
    select_balanced,
    select_knee_point,
    select_by_constraint,
    select_by_preference
)

# 1. Balanced selection (weighted objectives)
best = select_balanced(pareto_front, weights={
    "perplexity": 2.0,
    "spectral_gap": 1.5,
    "compute_cost": 1.0
})

# 2. Knee point (maximum trade-off curvature)
knee = select_knee_point(pareto_front)

# 3. Constraint-based
valid = select_by_constraint(pareto_front, {
    "perplexity": (0, 15),
    "spectral_gap": (0.5, 1.0)
})

# 4. Preference-based (closest to target)
closest = select_by_preference(pareto_front, {
    "perplexity": 10.0,
    "spectral_gap": 0.9
})
```

---

## Best Practices

### 1. Start with pymoo

```python
# Always test locally first
from src.cross_phase.meta_calculus.moo_bridge import PymooAdapter

adapter = PymooAdapter()
result = adapter.run_optimization(n_gen=20, pop_size=20, verbose=True)
```

### 2. Use Appropriate Population Sizes

| Evaluation Cost | Population | Generations |
|-----------------|------------|-------------|
| Fast (<1 sec) | 100 | 100 |
| Medium (1-30 sec) | 40-60 | 50 |
| Slow (>30 sec) | 20-30 | 30 |

### 3. Handle API Failures Gracefully

```python
try:
    result = globalmoo_client.run_optimization(...)
except Exception as e:
    print(f"GlobalMOO failed: {e}")
    print("Falling back to pymoo...")
    result = pymoo_adapter.run_optimization(...)
```

### 4. Save Results

```python
import json
from datetime import datetime

filename = f"moo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(f"results/{filename}", 'w') as f:
    json.dump(result, f, indent=2, default=str)
```

---

## Troubleshooting

### "API connection failed"

1. Verify API key is set: `echo $GLOBALMOO_API_KEY`
2. Check endpoint URL: https://app.globalmoo.com/api/
3. Test connection: `client.check_connection()`

### "No feasible solutions"

1. Check constraint bounds are realistic
2. Widen search space bounds
3. Verify evaluation function returns valid values

### "Optimizer stuck"

1. Increase population size
2. Add more initial samples for GlobalMOO
3. Try different random seed
4. Check for numerical instabilities in objectives

### "pymoo not finding good solutions"

1. Increase generations (n_gen)
2. Adjust crossover/mutation rates
3. Ensure objectives are properly scaled

---

## References

- **GlobalMOO Documentation**: https://globalmoo.gitbook.io/
- **pymoo Documentation**: https://pymoo.org/
- **NSGA-II Paper**: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm"
- **Meta-calculus MOO Integration**: `meta-calculus-toolkit/meta_calculus/moo_integration.py`

---

## File Locations

```
src/cross_phase/meta_calculus/
    moo_bridge.py           # Core MOO infrastructure
    moo_utils/
        __init__.py         # Public API
        architecture.py     # Architecture search problems
        hyperparams.py      # Hyperparameter optimization
        selection.py        # Pareto selection strategies
        constraints.py      # Constraint handling

docs/
    GLOBALMOO_INTEGRATION_GUIDE.md  # This file
```

# Simulation vs Real Implementation Analysis

## 🔍 Current State (As of 2025-10-01)

### ⚠️ Critical Finding: UI API Routes are SIMULATIONS

The dashboard API routes (`/api/phases/cognate` and `/api/phases/evomerge`) are **simulating** training progress and not creating real models!

---

## 📊 Analysis by Component

### 1. Frontend API Routes (TypeScript) - ❌ SIMULATIONS

#### Cognate API (`src/web/dashboard/app/api/phases/cognate/route.ts`)

**Current Implementation:**
```typescript
function simulateTrainingProgress(sessionId: string): void {
  // Simulate progress every 2 seconds
  const interval = setInterval(() => {
    currentSession.metrics.iteration += 1;
    currentSession.metrics.epoch = Math.floor(currentSession.metrics.iteration / 100);

    // Update per-model metrics
    model.loss = Math.max(0.1, model.loss - Math.random() * 0.005);
    model.grokfast_acceleration = 1.0 + Math.random() * 2.0;

    // Complete after some iterations
    if (currentSession.metrics.iteration >= 500) {
      currentSession.status = 'completed';

      // Generate MOCK model IDs
      const modelIds = [
        `cognate_tinytitan_reasoning_${sessionId.slice(-8)}_${Date.now()}`,
        `cognate_tinytitan_memory_integration_${sessionId.slice(-8)}_${Date.now()}`,
        `cognate_tinytitan_adaptive_computation_${sessionId.slice(-8)}_${Date.now()}`
      ];
    }
  }, 2000);
}
```

**Issues:**
- ❌ No actual PyTorch models created
- ❌ No model files saved to disk
- ❌ Just incrementing counters and random numbers
- ❌ Generates fake model IDs (strings, not file paths)
- ❌ No real training happening

**What it SHOULD do:**
- ✅ Call Python backend (`CognatePhase.run()`)
- ✅ Create 3 real TinyTitan models (25M params each)
- ✅ Save `.pt` files to disk
- ✅ Return actual file paths/model IDs

---

#### EvoMerge API (`src/web/dashboard/app/api/phases/evomerge/route.ts`)

**Current Implementation:**
```typescript
function createInitialPopulation(config: EvoMergeConfig): EvoMergeModel[] {
  for (let i = 0; i < config.population_size; i++) {
    population.push({
      id: `gen0_model${i}`,
      generation: 0,
      parent_ids: config.input_models,  // Just strings!
      technique: config.techniques[i % config.techniques.length],
      fitness_score: Math.random() * 0.3 + 0.4,  // RANDOM!
      metrics: {
        perplexity: Math.random() * 30 + 20,  // FAKE
        accuracy: Math.random() * 0.2 + 0.5,   // FAKE
      }
    });
  }
}

function evolveGeneration(...): { population: EvoMergeModel[] } {
  // Evolve to next generation
  const newPopulation: EvoMergeModel[] = [];

  // Elite preservation (keep top 2)
  newPopulation.push(...winners);

  // Each winner creates 3 offspring via FAKE mutation
  const child: EvoMergeModel = {
    fitness_score: Math.min(1.0, winner.fitness_score + (Math.random() - 0.3) * 0.1),
    // Just adding random numbers, no real model merging!
  };
}
```

**Issues:**
- ❌ No actual model merging happening
- ❌ Creates JavaScript objects, not PyTorch models
- ❌ "Mutations" are just random number changes
- ❌ No real SLERP, TIES, DARE merging
- ❌ Fitness scores are random, not evaluated
- ❌ Population of 8 is just 8 JavaScript objects

**What it SHOULD do:**
- ✅ Load 3 actual `.pt` files from Cognate
- ✅ Create 8 real merged models using merge techniques
- ✅ Evaluate fitness on real data
- ✅ Mutate actual model weights
- ✅ Save merged models to disk

---

### 2. Backend Python Implementation - ✅ REAL

#### Cognate Phase (`phases/cognate/cognate_phase.py`)

**Implementation:**
```python
class TinyTitanModel(nn.Module):
    def __init__(self, config: CognateConfig, seed: int, specialization: str):
        super().__init__()
        torch.manual_seed(seed)

        # Core transformer layers
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        # ... ACT components, memory gates, output layer

async def run(self, model=None) -> PhaseResult:
    # Create 3 specialized models
    for i in range(self.cognate_config.num_models):
        model = TinyTitanModel(
            config=self.cognate_config,
            seed=seeds[i],
            specialization=specializations[i]
        )
        model = model.to(self.cognate_config.device)
        models.append(model)

    # Save all 3 models to storage
    if self.session_id:
        for i, (model, spec) in enumerate(zip(models, specializations)):
            model_id = self.save_output_model(
                model=model,
                metrics={...},
                phase_name='cognate',
                model_name=f'tinytitan_{spec.lower()}',
                tags=[spec, 'cognate', f'model_{i+1}']
            )
```

**Status:** ✅ REAL IMPLEMENTATION
- ✅ Creates actual PyTorch nn.Module instances
- ✅ Real transformer layers with embeddings, attention, FFN
- ✅ ACT memory components
- ✅ Saves models via `save_output_model()` which calls storage manager
- ✅ Returns PhaseResult with real model objects

**BUT:** Not connected to the frontend API!

---

#### EvoMerge Phase (`phases/phase2_evomerge/evomerge.py`)

**Merge Techniques (`merge_techniques.py`):**
```python
def linear_merge(self, models: List[nn.Module], weights: Optional[List[float]] = None) -> nn.Module:
    """Linear interpolation merge."""
    merged = models[0].__class__()
    merged.to(self.device)

    with torch.no_grad():
        # Merge parameters
        for name, param in merged.named_parameters():
            merged_param = torch.zeros_like(param, device=self.device)

            for model, weight in zip(models, weights):
                model_param = model.state_dict()[name].to(self.device)
                merged_param += weight * model_param  # REAL WEIGHT MERGING

            param.data = merged_param

    return merged

def slerp_merge(self, models: List[nn.Module], t: float = 0.5) -> nn.Module:
    """Spherical Linear Interpolation merge."""
    # Get parameters from both models
    p1 = model1.state_dict()[name].to(self.device).flatten()
    p2 = model2.state_dict()[name].to(self.device).flatten()

    # Compute angle between parameters
    dot_product = torch.dot(p1, p2)
    norm1 = torch.norm(p1)
    norm2 = torch.norm(p2)

    # SLERP formula
    cos_theta = dot_product / (norm1 * norm2)
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    sin_theta = torch.sin(theta)

    # Interpolate on hypersphere
    slerp_param = ((torch.sin((1 - t) * theta) / sin_theta) * p1 +
                   (torch.sin(t * theta) / sin_theta) * p2)
```

**Status:** ✅ REAL IMPLEMENTATION
- ✅ Actual tensor operations on model weights
- ✅ Real linear interpolation (weighted average)
- ✅ Real SLERP (spherical interpolation on hypersphere)
- ✅ TIES merge (trim, elect, sign merge)
- ✅ DARE merge (drop and rescale)
- ✅ Frankenmerge (layer-wise selection)
- ✅ DFS merge (depth-first search)

**BUT:** Not connected to the frontend API!

---

## 🎯 The Disconnect

### What Happens Now

**User clicks "Start Training" on Cognate page:**
```
Browser → POST /api/phases/cognate
        ↓
    TypeScript API Route
        ↓
    simulateTrainingProgress() // FAKE
        ↓
    Increments counters, random numbers
        ↓
    Returns fake model IDs (strings)
```

**User clicks "Start Evolution" on EvoMerge page:**
```
Browser → POST /api/phases/evomerge
        ↓
    TypeScript API Route
        ↓
    createInitialPopulation() // FAKE
        ↓
    Creates 8 JavaScript objects
        ↓
    evolveGeneration() // FAKE mutations
        ↓
    Returns fake fitness scores
```

### What SHOULD Happen

**Cognate:**
```
Browser → POST /api/phases/cognate
        ↓
    TypeScript API Route
        ↓
    Calls FastAPI backend → POST /python/cognate/run
        ↓
    Python: CognatePhase().run()
        ↓
    Creates 3 real TinyTitan models
        ↓
    Saves to ./models/cognate/{session_id}/
        ↓
    Returns real file paths
        ↓
    TypeScript returns model IDs to frontend
```

**EvoMerge:**
```
Browser → POST /api/phases/evomerge
        ↓
    TypeScript API Route
        ↓
    Calls FastAPI backend → POST /python/evomerge/evolve
        ↓
    Python: EvoMerge().evolve()
        ↓
    Loads 3 real .pt files
        ↓
    Creates initial population of 8 merged models
        ↓
    For each generation (50 total):
        - Evaluate fitness on real data
        - Select top 2 (tournament)
        - Breed 6 offspring using merge techniques
        - Mutate weights
        - Save checkpoints
        ↓
    Returns final merged model
        ↓
    TypeScript returns result to frontend
```

---

## 📝 Required Changes

### High Priority: Connect Frontend to Backend

#### 1. Update Cognate API Route
**File:** `src/web/dashboard/app/api/phases/cognate/route.ts`

**Change from:**
```typescript
function simulateTrainingProgress(sessionId: string): void {
  // Fake progress simulation
}
```

**Change to:**
```typescript
async function runRealCognateTraining(sessionId: string, config: any): Promise<void> {
  // Call Python backend
  const response = await fetch('http://localhost:8000/cognate/run', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, config })
  });

  const result = await response.json();

  // Update session with real results
  session.modelIds = result.model_ids;
  session.metrics = result.metrics;
  session.status = 'completed';
}
```

#### 2. Update EvoMerge API Route
**File:** `src/web/dashboard/app/api/phases/evomerge/route.ts`

**Change from:**
```typescript
function createInitialPopulation(config: EvoMergeConfig): EvoMergeModel[] {
  // Fake JavaScript objects
}
```

**Change to:**
```typescript
async function runRealEvoMerge(sessionId: string, config: any, inputModelIds: string[]): Promise<void> {
  // Call Python backend
  const response = await fetch('http://localhost:8000/evomerge/evolve', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      input_model_ids: inputModelIds,
      config
    })
  });

  const result = await response.json();

  // Stream real-time updates via WebSocket
  const ws = new WebSocket(`ws://localhost:8000/evomerge/progress/${sessionId}`);
  ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    session.metrics = update.metrics;
    session.status = update.status;
  };
}
```

#### 3. Add FastAPI Endpoints
**File:** `src/api/pipeline_server_fixed.py` or create new `src/api/phase_endpoints.py`

```python
from fastapi import FastAPI, HTTPException
from phases.cognate import CognatePhase, CognateConfig
from phases.phase2_evomerge.evomerge import EvoMerge, EvoMergeConfig

app = FastAPI()

@app.post("/cognate/run")
async def run_cognate(request: dict):
    config = CognateConfig(**request['config'])

    cognate = CognatePhase(config)
    cognate.set_session_id(request['session_id'])

    result = await cognate.run()

    return {
        'success': result.success,
        'model_ids': result.artifacts.get('model_ids', []),
        'metrics': result.metrics
    }

@app.post("/evomerge/evolve")
async def run_evomerge(request: dict):
    config = EvoMergeConfig(**request['config'])

    evomerge = EvoMerge(config)
    evomerge.set_session_id(request['session_id'])
    evomerge.set_input_models(request['input_model_ids'])

    result = await evomerge.evolve()

    return {
        'success': True,
        'model_id': result.model_id,
        'metrics': result.metrics,
        'fitness': result.fitness
    }
```

---

## 🧪 Test Script to Verify Real Implementation

Create: `tests/manual/test_real_cognate_evomerge.py`

```python
"""
Test script to verify real model creation and merging.
Run this to see actual .pt files being created and merged.
"""

import asyncio
from pathlib import Path
from phases.cognate import CognatePhase, CognateConfig
from phases.phase2_evomerge.evomerge import EvoMerge, EvoMergeConfig

async def test_real_flow():
    print("=" * 80)
    print("TESTING REAL COGNATE + EVOMERGE FLOW")
    print("=" * 80)

    # Phase 1: Create 3 models
    print("\n[Phase 1: Cognate] Creating 3 TinyTitan models...")
    cognate_config = CognateConfig(
        num_models=3,
        parameters_per_model=25_000_000,
        pretraining_epochs=2,  # Fast for testing
        device='cpu'
    )

    cognate = CognatePhase(cognate_config)
    cognate.set_session_id("test-real-001")

    cognate_result = await cognate.run()

    if not cognate_result.success:
        print(f"❌ Cognate failed: {cognate_result.error}")
        return

    model_ids = cognate_result.artifacts['model_ids']
    print(f"✅ Created {len(model_ids)} models:")
    for i, model_id in enumerate(model_ids):
        print(f"   {i+1}. {model_id}")

    # Check files exist
    print("\n[Storage Check] Verifying .pt files exist...")
    storage = get_storage_manager()
    for model_id in model_ids:
        model_path = storage.get_model_path(model_id)
        if model_path and model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {model_path.name}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {model_id}: FILE NOT FOUND")

    # Phase 2: Merge models
    print("\n[Phase 2: EvoMerge] Evolving 8 models over 5 generations...")
    evomerge_config = EvoMergeConfig(
        generations=5,  # Fast for testing
        population_size=8,
        device='cpu'
    )

    evomerge = EvoMerge(evomerge_config)
    evomerge.set_session_id("test-real-001")
    evomerge.set_input_models(model_ids)

    # Load models
    models = evomerge.load_input_models(device='cpu')
    print(f"✅ Loaded {len(models)} models from storage")

    # Run evolution
    evo_result = await evomerge.evolve()

    print(f"\n[Results]")
    print(f"   Final fitness: {evo_result.fitness:.4f}")
    print(f"   Technique used: {evo_result.technique}")
    print(f"   Generations: {evo_result.generation}")

    # Check final model
    final_model_path = storage.get_model_path(evo_result.model_id)
    if final_model_path and final_model_path.exists():
        size_mb = final_model_path.stat().st_size / (1024 * 1024)
        print(f"   Final model: {size_mb:.1f} MB")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_real_flow())
```

---

## 📊 Expected vs Actual Behavior

### Expected (After Fixes)

**Cognate Output:**
- 3 `.pt` files in `./models/cognate/{session_id}/`
- Each file ~95-100 MB (25M params in FP32)
- Real transformer layers with weights
- Total: ~285-300 MB

**EvoMerge Process:**
- Loads 3 real `.pt` files
- Creates initial population: 8 merged models
- Each generation:
  - Evaluates fitness on actual data
  - Creates 8 new checkpoint files (can optionally delete old ones)
  - Uses real merge techniques (SLERP, TIES, etc.)
- Final output: 1 merged model, ~95-100 MB

**Total Disk Usage:** ~1-2 GB if keeping all checkpoints, ~300-400 MB if cleanup enabled

### Actual (Current)

**Cognate Output:**
- 0 `.pt` files (nothing created)
- 3 fake model ID strings
- Random numbers for metrics
- Total: 0 MB

**EvoMerge Process:**
- 0 models loaded
- 8 JavaScript objects (not models)
- Random number "mutations"
- No real merging
- Total: 0 MB

---

## 🎯 Priority Actions

1. **Test Script** - Run the Python test script above to verify backend works ✅
2. **API Integration** - Connect TypeScript routes to Python backend
3. **WebSocket** - Add real-time progress streaming
4. **Storage Cleanup** - Add option to delete intermediate checkpoints
5. **Validation** - Verify model sizes and parameter counts match expected

---

*Analysis Date: 2025-10-01*
*Status: SIMULATION DETECTED - Real backend exists but not connected*

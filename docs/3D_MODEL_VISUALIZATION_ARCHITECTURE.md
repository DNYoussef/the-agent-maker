# 3D Model Comparison - Architecture Diagram

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL BROWSER PAGE                          │
│                  (src/ui/pages/model_browser.py)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ calls
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              render_model_browser_3d()                          │
│           (Streamlit Component Wrapper)                         │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  UI CONTROLS                                           │    │
│  │  - Phase filter (multiselect)                         │    │
│  │  - Champion highlighting (checkbox)                   │    │
│  │  - Pareto frontier (checkbox)                         │    │
│  └───────────────────────────────────────────────────────┘    │
│                             │                                   │
│                             │ configures                        │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  create_model_comparison_3d()                         │    │
│  │  (Core Visualization Function)                        │    │
│  └───────────────────────────────────────────────────────┘    │
│                             │                                   │
│                             │ returns                           │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  st.plotly_chart()                                    │    │
│  │  (Render Plotly Figure)                               │    │
│  └───────────────────────────────────────────────────────┘    │
│                             │                                   │
│                             │ displays                          │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  SUMMARY STATISTICS                                   │    │
│  │  - Total models, avg accuracy, avg latency           │    │
│  │  - Phase breakdown table                             │    │
│  └───────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌─────────────────┐
│  Model Registry │
│   (Database)    │
└────────┬────────┘
         │
         │ query
         ▼
┌─────────────────┐         ┌──────────────────┐
│  List[Dict]     │────────>│  pd.DataFrame    │
│  Raw Models     │ convert │  Structured Data │
└─────────────────┘         └────────┬─────────┘
                                     │
                                     │ validate
                                     ▼
                            ┌─────────────────┐
                            │  Data Validator │
                            │  - Required cols│
                            │  - Type checking│
                            │  - Defaults     │
                            └────────┬────────┘
                                     │
                                     │ process
                                     ▼
┌─────────────────────────────────────────────────────────┐
│              create_model_comparison_3d()               │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Filter by  │  │  Highlight   │  │   Compute    │ │
│  │   Phases    │  │  Champions   │  │   Pareto     │ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                │                  │         │
│         └────────────────┴──────────────────┘         │
│                          │                            │
│                          ▼                            │
│         ┌────────────────────────────────┐           │
│         │    Create Plotly Traces        │           │
│         │    (one per phase)             │           │
│         └────────────────┬───────────────┘           │
│                          │                            │
│                          ▼                            │
│         ┌────────────────────────────────┐           │
│         │    Add Visual Encoding         │           │
│         │    - Color by phase            │           │
│         │    - Size by compression       │           │
│         │    - Symbol by status          │           │
│         └────────────────┬───────────────┘           │
│                          │                            │
│                          ▼                            │
│         ┌────────────────────────────────┐           │
│         │    Configure Layout            │           │
│         │    - Dark theme                │           │
│         │    - Axis labels               │           │
│         │    - Camera angle              │           │
│         └────────────────┬───────────────┘           │
│                          │                            │
│                          ▼                            │
│         ┌────────────────────────────────┐           │
│         │    Add Animation Frames        │           │
│         │    (if enabled)                │           │
│         └────────────────┬───────────────┘           │
└──────────────────────────┼─────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Plotly Figure  │
                  │  (go.Figure)    │
                  └─────────────────┘
```

## Visual Encoding Pipeline

```
┌──────────────┐
│  Model Data  │
└──────┬───────┘
       │
       ├─> params (int)        ──────> X-axis (millions)
       │
       ├─> accuracy (float)    ──────> Y-axis (percentage)
       │
       ├─> latency (float)     ──────> Z-axis (ms, reversed)
       │
       ├─> phase (str)         ──────> Color (8 colors)
       │                                ├─> phase1: #00F5D4 (cyan)
       │                                ├─> phase2: #0099FF (blue)
       │                                ├─> phase3: #9D4EDD (purple)
       │                                ├─> phase4: #FF006E (magenta)
       │                                ├─> phase5: #FB5607 (orange)
       │                                ├─> phase6: #FFBE0B (yellow)
       │                                ├─> phase7: #06FFA5 (green)
       │                                └─> phase8: #F72585 (pink)
       │
       ├─> compression (float) ──────> Size (5-55px)
       │                                size = compression * 5 + 5
       │
       ├─> status (str)        ──────> Symbol
       │                                ├─> complete: ● (circle)
       │                                ├─> running:  ◆ (diamond)
       │                                ├─> failed:   ✕ (x)
       │                                └─> pending:  ■ (square)
       │
       └─> id in champions     ──────> Border
                                        ├─> Yes: 3px white border
                                        └─> No:  0.5px dark border
```

## Pareto Frontier Computation

```
┌────────────────────┐
│  All Models (N)    │
│  (params, acc, lat)│
└─────────┬──────────┘
          │
          ▼
┌──────────────────────────────┐
│  Normalize Metrics [0, 1]    │
│  - params_norm               │
│  - accuracy_norm             │
│  - latency_norm              │
└─────────┬────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Pareto Dominance Test       │
│  For each model i:           │
│    Check if any model j      │
│    dominates i:              │
│      acc[j] >= acc[i] AND    │
│      lat[j] <= lat[i] AND    │
│      params[j] <= params[i]  │
│      AND at least one strict │
└─────────┬────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Pareto-Optimal Set (M ≤ N)  │
│  Models that are not         │
│  dominated by any other      │
└─────────┬────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Create 2D Grid              │
│  (params × accuracy)         │
│  20 × 20 = 400 points        │
└─────────┬────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Interpolate Latency         │
│  Using scipy.griddata        │
│  (linear interpolation)      │
└─────────┬────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Create go.Surface           │
│  - Semi-transparent cyan     │
│  - Opacity: 0.3              │
│  - No colorbar               │
└──────────────────────────────┘
```

## Animation System

```
┌────────────────┐
│  Static Figure │
│  (30 traces)   │
└───────┬────────┘
        │
        ▼
┌────────────────────────────┐
│  Generate 31 Frames        │
│  (0% → 100% progress)      │
│                            │
│  For frame i (i=0..30):    │
│    progress = (i/30)²      │  # Quadratic ease-out
│                            │
│    For each trace:         │
│      x[t] = x_final * p    │
│      y[t] = y_final * p +  │
│             (1-p) * 50     │  # Start from middle
│      z[t] = z_final * p +  │
│             (1-p) * avg_z  │
│      opacity = base * p    │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Attach Frames to Figure   │
│  fig.frames = [frame_list] │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Add Play Button           │
│  - Auto-play on load       │
│  - 50ms per frame          │
│  - Total: 1.5 seconds      │
└────────────────────────────┘
```

## Component Class Diagram

```
┌─────────────────────────────────────────────┐
│         model_comparison_3d.py              │
├─────────────────────────────────────────────┤
│                                             │
│  Constants:                                 │
│  ├─ PHASE_COLORS: Dict[str, str]           │
│  ├─ STATUS_SYMBOLS: Dict[str, str]         │
│  ├─ BACKGROUND_COLOR: str                  │
│  ├─ GRID_COLOR: str                        │
│  └─ TEXT_COLOR: str                        │
│                                             │
│  Public Functions:                          │
│  ├─ create_model_comparison_3d()           │
│  │   Args: models_df, highlighted_ids,     │
│  │         show_phases, show_pareto,       │
│  │         animate                          │
│  │   Returns: go.Figure                    │
│  │                                          │
│  ├─ render_model_browser_3d()              │
│  │   Args: models, key                     │
│  │   Returns: None (renders in Streamlit)  │
│  │                                          │
│  └─ get_sample_data()                      │
│      Returns: List[Dict]                   │
│                                             │
│  Private Functions:                         │
│  └─ _compute_pareto_surface()              │
│      Args: models_df                       │
│      Returns: Optional[go.Surface]         │
│                                             │
└─────────────────────────────────────────────┘
```

## Integration Points

```
┌────────────────────────────────────────────────────────┐
│                   STREAMLIT APP                        │
│                  (src/ui/app.py)                       │
└─────────────────────┬──────────────────────────────────┘
                      │
                      ├─> Model Browser Page
                      │   (src/ui/pages/model_browser.py)
                      │
                      ├─> Phase Details Page
                      │   (src/ui/pages/phase_details.py)
                      │
                      └─> System Monitor
                          (src/ui/pages/system_monitor.py)

┌────────────────────────────────────────────────────────┐
│              MODEL BROWSER PAGE                        │
│           (src/ui/pages/model_browser.py)              │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────────────────────────────────┐        │
│  │  FILTERS SECTION                         │        │
│  │  - Phase multiselect                     │        │
│  │  - Size dropdown                         │        │
│  │  - Search text input                     │        │
│  └──────────────────────────────────────────┘        │
│                                                        │
│  ┌──────────────────────────────────────────┐        │
│  │  VIEW MODE SELECTOR                      │        │
│  │  ○ 3D Visualization                      │        │
│  │  ○ List View                             │        │
│  │  ● Both (default)                        │        │
│  └──────────────────────────────────────────┘        │
│                                                        │
│  if view in ["3D", "Both"]:                           │
│  ┌──────────────────────────────────────────┐        │
│  │  3D MODEL COMPARISON                     │        │
│  │  (model_comparison_3d.py)                │        │
│  │  ┌────────────────────────────────────┐  │        │
│  │  │  Interactive 3D Scatter Plot       │  │        │
│  │  │  - Orbit controls                  │  │        │
│  │  │  - Zoom/pan                        │  │        │
│  │  │  - Hover tooltips                  │  │        │
│  │  │  - Phase filtering                 │  │        │
│  │  │  - Champion highlighting           │  │        │
│  │  └────────────────────────────────────┘  │        │
│  │  ┌────────────────────────────────────┐  │        │
│  │  │  Summary Statistics                │  │        │
│  │  │  - Total models: 42                │  │        │
│  │  │  - Avg accuracy: 65.3%             │  │        │
│  │  │  - Avg latency: 87.2 ms            │  │        │
│  │  └────────────────────────────────────┘  │        │
│  └──────────────────────────────────────────┘        │
│                                                        │
│  if view in ["List", "Both"]:                         │
│  ┌──────────────────────────────────────────┐        │
│  │  LIST VIEW                               │        │
│  │  (existing model_browser.py code)        │        │
│  │  - Expandable model cards               │        │
│  │  - Action buttons                       │        │
│  │  - Detailed metrics                     │        │
│  └──────────────────────────────────────────┘        │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Dependency Graph

```
model_comparison_3d.py
│
├─> plotly.graph_objects (3D scatter, surface)
│   └─> go.Scatter3d
│   └─> go.Surface
│   └─> go.Figure
│   └─> go.Frame
│
├─> pandas (data handling)
│   └─> pd.DataFrame
│
├─> numpy (numerical operations)
│   └─> np.random (sample data)
│   └─> np.linspace (grids)
│   └─> np.meshgrid (surfaces)
│
├─> scipy.interpolate (optional, Pareto)
│   └─> griddata
│
└─> streamlit (UI framework)
    └─> st.plotly_chart
    └─> st.checkbox
    └─> st.multiselect
    └─> st.metric
    └─> st.dataframe
```

## Performance Optimization Strategy

```
┌────────────────────┐
│  Input: N models   │
└─────────┬──────────┘
          │
          ▼
     N < 20?  ────Yes───> ┌─────────────────────┐
          │               │  FULL FEATURES       │
          No              │  - Animation: ON     │
          │               │  - Pareto: ON        │
          ▼               │  - All phases        │
     N < 50?  ────Yes───> └─────────────────────┘
          │               ┌─────────────────────┐
          No              │  STANDARD FEATURES   │
          │               │  - Animation: ON     │
          ▼               │  - Pareto: OFF       │
     N < 100? ────Yes───> │  - All phases        │
          │               └─────────────────────┘
          No              ┌─────────────────────┐
          │               │  PERFORMANCE MODE    │
          ▼               │  - Animation: OFF    │
┌─────────────────────┐  │  - Pareto: OFF       │
│  LIMIT DATASET       │  │  - Phase filtering   │
│  Take top 100        │  └─────────────────────┘
│  by accuracy         │
└─────────┬────────────┘
          │
          └──────────────> Render with limited features
```

## Error Handling Flow

```
┌────────────────────┐
│  render_model_     │
│  browser_3d()      │
└─────────┬──────────┘
          │
          ▼
┌──────────────────────────┐
│  Convert List to DF      │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│  Validate Columns        │
└─────────┬────────────────┘
          │
     Missing columns?
          │
          ├─Yes─> ┌──────────────────┐
          │       │  st.error()      │
          │       │  Show error msg  │
          │       │  Return early    │
          │       └──────────────────┘
          │
          No
          │
          ▼
┌──────────────────────────┐
│  Add Defaults            │
│  - compression = 1.0     │
│  - status = 'complete'   │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│  create_model_           │
│  comparison_3d()         │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│  try: Pareto surface     │
└─────────┬────────────────┘
          │
     Exception?
          │
          ├─Yes─> ┌──────────────────┐
          │       │  Catch silently  │
          │       │  Return None     │
          │       │  Continue        │
          │       └──────────────────┘
          │
          No
          │
          ▼
┌──────────────────────────┐
│  Return go.Figure        │
└──────────────────────────┘
```

## File Size Breakdown

```
model_comparison_3d.py (601 lines)
├─ Imports & Constants      (55 lines,   9%)
├─ create_model_3d()        (280 lines, 47%)
│  ├─ Data filtering        (20 lines)
│  ├─ Trace creation        (180 lines)
│  ├─ Layout config         (50 lines)
│  └─ Animation frames      (30 lines)
├─ _compute_pareto()        (80 lines,  13%)
├─ render_browser_3d()      (120 lines, 20%)
├─ get_sample_data()        (50 lines,   8%)
└─ Demo/Documentation       (16 lines,   3%)
```

---

**Diagram Version**: 1.0.0
**Last Updated**: 2025-11-27
**Related Docs**: 3D_MODEL_COMPARISON_GUIDE.md

# 3D Model Comparison Visualization Guide

## Overview

The 3D Model Comparison component provides an interactive visualization for comparing models across all 8 phases of the Agent Forge pipeline. Models are displayed in a 3-dimensional space optimized for identifying performance trade-offs and champions.

## Features

### Core Visualization

- **3D Scatter Plot**: Models positioned by size (X), accuracy (Y), and latency (Z)
- **Phase Coloring**: 8 distinct futuristic colors for each phase
- **Status Symbols**: Visual indicators for complete/running/failed/pending states
- **Champion Highlighting**: Automatic detection and emphasis of best models per phase
- **Pareto Frontier**: Optional surface showing optimal trade-off regions

### Interactivity

- **Orbit Controls**: Click and drag to rotate, scroll to zoom
- **Phase Filtering**: Show/hide specific phases dynamically
- **Hover Details**: Comprehensive model information on mouseover
- **Entrance Animation**: Smooth reveal animation on initial load

### Design

- **Dark Theme**: Matches Agent Forge design system (#0D1B2A background)
- **Futuristic Colors**: Cyan/blue/purple/magenta/orange gradient across phases
- **Responsive**: Works on all screen sizes
- **Accessible**: High contrast, clear labels

## Installation

### Requirements

```bash
pip install plotly>=5.17.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scipy>=1.11.0  # For Pareto surface computation
```

### File Location

```
src/ui/components/
├── model_comparison_3d.py          # Main component
└── model_browser_integration_example.py  # Integration example
```

## Quick Start

### Standalone Usage

```python
from ui.components.model_comparison_3d import render_model_browser_3d

# Your model data
models = [
    {
        'id': 'phase1_model1',
        'name': 'Phase 1 Reasoning Model',
        'phase': 'phase1',
        'params': 25_000_000,
        'accuracy': 45.2,
        'latency': 120.5,
        'compression': 1.0,
        'status': 'complete'
    },
    # ... more models
]

# Render in Streamlit
render_model_browser_3d(models, key="my_3d_view")
```

### Integration with Model Browser

Replace the existing `model_browser.py` render function:

```python
# Original: src/ui/pages/model_browser.py
def render():
    # ... existing filters ...

    # ADD: Import the component
    from ui.components.model_comparison_3d import render_model_browser_3d

    # ADD: View mode selector
    view_mode = st.radio(
        "View Mode",
        ["3D Visualization", "List View", "Both"],
        horizontal=True
    )

    # ADD: 3D visualization
    if view_mode in ["3D Visualization", "Both"]:
        render_model_browser_3d(models, key="browser_3d")

    # Keep existing list view for "List View" or "Both"
    if view_mode in ["List View", "Both"]:
        # ... existing expander code ...
```

## API Reference

### `create_model_comparison_3d()`

Core function that creates the Plotly figure.

```python
def create_model_comparison_3d(
    models_df: pd.DataFrame,
    highlighted_ids: Optional[List[str]] = None,
    show_phases: Optional[List[str]] = None,
    show_pareto: bool = False,
    animate: bool = True
) -> go.Figure:
```

**Parameters:**

- `models_df` (pd.DataFrame): Model data with required columns:
  - `id` (str): Unique model identifier
  - `name` (str): Display name
  - `phase` (str): Phase identifier (phase1-phase8)
  - `params` (int): Model size in parameters
  - `accuracy` (float): Accuracy metric (0-100%)
  - `latency` (float): Inference speed in milliseconds
  - `status` (str): One of 'complete', 'running', 'failed', 'pending'
  - `compression` (float, optional): Compression ratio

- `highlighted_ids` (List[str], optional): Model IDs to highlight with emphasis

- `show_phases` (List[str], optional): Phases to display (e.g., ['phase1', 'phase3']). None = all phases

- `show_pareto` (bool): Whether to display Pareto frontier surface

- `animate` (bool): Enable entrance animation

**Returns:**

- `plotly.graph_objects.Figure`: Interactive 3D scatter plot

### `render_model_browser_3d()`

Streamlit component wrapper with controls.

```python
def render_model_browser_3d(
    models: List[Dict[str, Any]],
    key: str = "model_browser_3d"
) -> None:
```

**Parameters:**

- `models` (List[Dict]): List of model dictionaries (automatically converted to DataFrame)

- `key` (str): Unique Streamlit component key

**Features Added:**

- Phase filtering multiselect
- Champion highlighting checkbox
- Pareto frontier toggle
- Summary statistics table
- Phase breakdown table

### `get_sample_data()`

Generate sample data for testing.

```python
def get_sample_data() -> List[Dict[str, Any]]:
```

**Returns:**

- List of 30-60 sample models across all 8 phases

## Data Format

### Required Fields

```python
{
    'id': str,          # Unique identifier
    'name': str,        # Display name
    'phase': str,       # 'phase1' through 'phase8'
    'params': int,      # Total parameters (e.g., 25_000_000)
    'accuracy': float,  # 0-100 scale
    'latency': float,   # Milliseconds
    'status': str       # 'complete' | 'running' | 'failed' | 'pending'
}
```

### Optional Fields

```python
{
    'compression': float  # Compression ratio (default: 1.0)
}
```

### Missing Data Handling

If `latency` or `compression` are missing, the integration example provides estimation:

```python
# Estimate latency based on model size and phase
base_latency = 100  # ms
size_factor = params / 25_000_000
phase_num = int(phase.replace('phase', ''))
phase_factor = max(0.5, 1.5 - phase_num * 0.1)
latency = base_latency * size_factor * phase_factor

# Estimate compression based on phase
compression = 1.0 + (phase_num - 1) * 0.3
```

## Visual Encoding

### Axes

| Axis | Metric | Interpretation |
|------|--------|----------------|
| X | Model Size (M params) | Smaller = more efficient |
| Y | Accuracy (%) | Higher = better performance |
| Z | Latency (ms) | Lower = faster (top of axis) |

**Note**: Z-axis is reversed (lower latency appears higher) for intuitive "up = better" interpretation.

### Colors (Phase Mapping)

| Phase | Color | Hex Code | Semantic Meaning |
|-------|-------|----------|------------------|
| Phase 1 | Cyan | #00F5D4 | Initial/Foundation |
| Phase 2 | Blue | #0099FF | Evolution |
| Phase 3 | Purple | #9D4EDD | Enhancement |
| Phase 4 | Magenta | #FF006E | Compression |
| Phase 5 | Orange | #FB5607 | Training |
| Phase 6 | Yellow | #FFBE0B | Specialization |
| Phase 7 | Green | #06FFA5 | Optimization |
| Phase 8 | Pink | #F72585 | Finalization |

### Symbols (Status Mapping)

| Status | Symbol | Meaning |
|--------|--------|---------|
| complete | ● (circle) | Training finished successfully |
| running | ◆ (diamond) | Currently training |
| failed | ✕ (x) | Training failed |
| pending | ■ (square) | Queued for training |

### Size Encoding

Point size is proportional to compression ratio:

```python
size = compression * 5 + 5  # Pixels
```

- 1.0x compression → 10px (baseline)
- 5.0x compression → 30px (larger)
- 10.0x compression → 55px (much larger)

### Highlighting

Champion models (best accuracy per phase) get:
- 1.5x larger point size
- 100% opacity (vs 60-70% for others)
- White border (3px width)
- " [CHAMPION]" badge in hover text

## Pareto Frontier

### What It Shows

The Pareto frontier surface represents the optimal trade-off region where no model is strictly better across all metrics. Models on this surface are "Pareto-optimal."

### Interpretation

- **On the surface**: Model is optimal (can't improve one metric without sacrificing another)
- **Below the surface**: Model is dominated (another model is better in all ways)
- **Above the surface**: Best-in-class models (rare)

### Computation

Uses a 3-objective Pareto dominance test:

```python
Model A dominates Model B if:
  accuracy_A >= accuracy_B AND
  latency_A <= latency_B AND
  params_A <= params_B AND
  (at least one inequality is strict)
```

The surface is created by:
1. Finding all Pareto-optimal points
2. Creating a 2D grid (params × accuracy)
3. Interpolating latency values using scipy.interpolate.griddata
4. Rendering as a semi-transparent surface

### Enabling

```python
# In render_model_browser_3d
show_pareto = st.checkbox("Show Pareto Frontier", value=False)
```

**Note**: Requires scipy and at least 4 models to compute.

## Animation

### Entrance Animation

30-frame quadratic ease-out animation:

1. Points start at center (50% Y-axis)
2. Gradually move to final positions
3. Opacity fades in simultaneously
4. Total duration: 1.5 seconds (50ms per frame)

### Controls

Auto-plays on load. User can replay with "Play" button in top-right corner.

### Disabling

```python
fig = create_model_comparison_3d(..., animate=False)
```

## Customization

### Theme Colors

Edit `BACKGROUND_COLOR`, `GRID_COLOR`, `TEXT_COLOR` at top of file:

```python
BACKGROUND_COLOR = '#0D1B2A'  # Dark navy
GRID_COLOR = '#1B263B'        # Slate
TEXT_COLOR = '#E0E1DD'        # Off-white
```

### Phase Colors

Edit `PHASE_COLORS` dictionary:

```python
PHASE_COLORS = {
    'phase1': '#YOUR_HEX_COLOR',
    # ... etc
}
```

### Camera Angle

Adjust initial view in layout:

```python
scene=dict(
    camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.3),  # Camera position
        center=dict(x=0, y=0, z=0),     # Look-at point
        up=dict(x=0, y=0, z=1)          # Up direction
    )
)
```

### Status Symbols

Edit `STATUS_SYMBOLS` dictionary:

```python
STATUS_SYMBOLS = {
    'complete': 'circle',
    'running': 'diamond',
    'failed': 'x',
    'pending': 'square',
    # Add custom statuses
}
```

## Performance Optimization

### For Large Datasets (>100 models)

1. **Limit models per phase**:
```python
models_per_phase = 10
filtered = df.groupby('phase').head(models_per_phase)
```

2. **Disable animation**:
```python
fig = create_model_comparison_3d(..., animate=False)
```

3. **Simplify Pareto surface** (reduce grid resolution):
```python
# In _compute_pareto_surface()
xi = np.linspace(x.min(), x.max(), 10)  # Was 20
yi = np.linspace(y.min(), y.max(), 10)  # Was 20
```

### For Real-time Updates

Use Streamlit's `st.plotly_chart` with `key` parameter:

```python
st.plotly_chart(fig, use_container_width=True, key=f"plot_{timestamp}")
```

Changing `key` forces re-render with new animation.

## Troubleshooting

### "Missing required columns" error

Ensure all required fields are present:

```python
required = ['id', 'name', 'phase', 'params', 'accuracy', 'latency', 'status']
print(set(required) - set(df.columns))  # Shows missing columns
```

### Pareto surface not showing

Causes:
- Fewer than 4 models
- scipy not installed
- All models have identical metrics (degenerate case)

Fix:
```bash
pip install scipy
```

### Animation not playing

Causes:
- `animate=False` in function call
- Browser JavaScript disabled
- Too many models (>200)

Fix: Reduce model count or disable animation for performance.

### Colors not matching design system

Colors are hardcoded in `PHASE_COLORS`. Update to match your theme:

```python
from ui.design_system import COLORS

PHASE_COLORS = {
    'phase1': COLORS['accent'],
    'phase2': COLORS['info'],
    # ... etc
}
```

## Examples

### Example 1: Basic Integration

```python
import streamlit as st
from ui.components.model_comparison_3d import render_model_browser_3d

st.title("Model Comparison")

models = [
    {'id': 'model1', 'name': 'Model 1', 'phase': 'phase1',
     'params': 25_000_000, 'accuracy': 45.0, 'latency': 120.0,
     'status': 'complete', 'compression': 1.0},
    # ... more models
]

render_model_browser_3d(models)
```

### Example 2: Champion Highlighting

```python
from ui.components.model_comparison_3d import create_model_comparison_3d
import pandas as pd

df = pd.DataFrame(models)

# Find champions (best accuracy per phase)
champions = df.loc[df.groupby('phase')['accuracy'].idxmax()]
champion_ids = champions['id'].tolist()

# Create plot with highlights
fig = create_model_comparison_3d(
    models_df=df,
    highlighted_ids=champion_ids,
    animate=True
)

st.plotly_chart(fig, use_container_width=True)
```

### Example 3: Filtered View

```python
# Show only Phase 3-6 models with Pareto frontier
selected_phases = ['phase3', 'phase4', 'phase5', 'phase6']

fig = create_model_comparison_3d(
    models_df=df,
    show_phases=selected_phases,
    show_pareto=True,
    animate=False  # Faster for large datasets
)

st.plotly_chart(fig)
```

### Example 4: Dynamic Filtering

```python
# User controls
phases = st.multiselect(
    "Select Phases",
    options=[f'phase{i}' for i in range(1, 9)],
    default=['phase1', 'phase2', 'phase3']
)

show_champions = st.checkbox("Highlight Champions", value=True)
show_pareto = st.checkbox("Show Pareto Frontier", value=False)

# Dynamic champion detection
champions = None
if show_champions:
    filtered_df = df[df['phase'].isin(phases)]
    champions = filtered_df.loc[filtered_df.groupby('phase')['accuracy'].idxmax()]['id'].tolist()

# Render
fig = create_model_comparison_3d(
    models_df=df,
    highlighted_ids=champions,
    show_phases=phases,
    show_pareto=show_pareto
)

st.plotly_chart(fig, use_container_width=True)
```

## Best Practices

### Data Preparation

1. **Consistent units**: Ensure all models use same units (params as integers, accuracy as %, latency as ms)
2. **Handle missing data**: Provide defaults or estimates for optional fields
3. **Normalize IDs**: Use consistent naming (e.g., `phase1_model1_session001`)

### Performance

1. **Limit models**: Show top 10-20 per phase for responsiveness
2. **Disable animation**: For datasets >50 models
3. **Cache figures**: Use `@st.cache_data` for expensive computations

### UX

1. **Start with overview**: Show all phases by default
2. **Progressive disclosure**: Put advanced features (Pareto) behind checkboxes
3. **Provide context**: Include legend and axis labels
4. **Mobile-friendly**: Test on tablet/phone (may need simplified view)

### Accessibility

1. **High contrast**: Use design system colors (already implemented)
2. **Text alternatives**: Provide summary statistics table
3. **Keyboard navigation**: Plotly supports arrow keys for rotation

## Future Enhancements

### Planned Features

- [ ] Multi-model selection for detailed comparison
- [ ] Export to PNG/SVG
- [ ] Custom metric axes (user-selectable)
- [ ] Time-series animation (models appearing as they're created)
- [ ] Clustering visualization (k-means on model space)
- [ ] Side-by-side 2D projections (XY, XZ, YZ planes)

### Integration Opportunities

- Link to Phase Details page on model click
- Show training curves on hover (requires historical data)
- Export selected models for batch operations
- Integration with Model Registry for live updates

## Related Documentation

- [Design System](../src/ui/design_system.py) - Color palette and typography
- [Model Registry](../src/cross_phase/storage/model_registry.py) - Data source
- [Model Browser](../src/ui/pages/model_browser.py) - Parent page
- [Plotly Documentation](https://plotly.com/python/3d-scatter-plots/) - 3D scatter plots

## License

Part of Agent Forge V2 project. See main LICENSE file.

---

**Created**: 2025-11-27
**Version**: 1.0.0
**Author**: Data Visualization Specialist
**Maintainer**: Agent Forge Team

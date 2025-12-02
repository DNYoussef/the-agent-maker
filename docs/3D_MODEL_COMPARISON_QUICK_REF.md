# 3D Model Comparison - Quick Reference

## One-Liner Integration

```python
from ui.components.model_comparison_3d import render_model_browser_3d
render_model_browser_3d(models, key="my_3d_view")
```

## Required Data Format

```python
models = [
    {
        'id': str,          # Unique identifier
        'name': str,        # Display name
        'phase': str,       # 'phase1' to 'phase8'
        'params': int,      # Total parameters
        'accuracy': float,  # 0-100 scale
        'latency': float,   # Milliseconds
        'status': str       # 'complete' | 'running' | 'failed' | 'pending'
        'compression': float  # Optional, defaults to 1.0
    },
    # ... more models
]
```

## Visual Encoding Cheat Sheet

| Element | Encoding | Meaning |
|---------|----------|---------|
| **X-axis** | Model size (M params) | Smaller = more efficient |
| **Y-axis** | Accuracy (%) | Higher = better |
| **Z-axis** | Latency (ms) | Lower = faster (top) |
| **Color** | Phase (cyan→pink) | Pipeline progression |
| **Size** | Compression ratio | Larger = more compressed |
| **Shape** | Status | ● = complete, ◆ = running, ✕ = failed, ■ = pending |
| **Border** | Champion highlight | White 3px = best in phase |

## Phase Colors

```python
phase1: #00F5D4  # Cyan
phase2: #0099FF  # Blue
phase3: #9D4EDD  # Purple
phase4: #FF006E  # Magenta
phase5: #FB5607  # Orange
phase6: #FFBE0B  # Yellow
phase7: #06FFA5  # Green
phase8: #F72585  # Pink
```

## Common Patterns

### Basic Usage

```python
render_model_browser_3d(models)
```

### With Champion Highlighting

```python
# Find best model per phase
champions = df.loc[df.groupby('phase')['accuracy'].idxmax()]['id'].tolist()

fig = create_model_comparison_3d(
    models_df=df,
    highlighted_ids=champions
)
```

### Filtered View

```python
fig = create_model_comparison_3d(
    models_df=df,
    show_phases=['phase3', 'phase4', 'phase5']
)
```

### With Pareto Frontier

```python
fig = create_model_comparison_3d(
    models_df=df,
    show_pareto=True
)
```

### Performance Mode (Large Datasets)

```python
fig = create_model_comparison_3d(
    models_df=df.head(50),  # Limit models
    animate=False,          # Disable animation
    show_pareto=False       # Skip expensive computation
)
```

## API Quick Ref

### `create_model_comparison_3d()`

```python
create_model_comparison_3d(
    models_df: pd.DataFrame,          # Required
    highlighted_ids: List[str] = None,  # Optional
    show_phases: List[str] = None,      # Optional
    show_pareto: bool = False,          # Optional
    animate: bool = True                # Optional
) -> go.Figure
```

### `render_model_browser_3d()`

```python
render_model_browser_3d(
    models: List[Dict],     # Required
    key: str = "browser_3d"  # Optional
) -> None
```

### `get_sample_data()`

```python
get_sample_data() -> List[Dict]  # Returns 30-60 test models
```

## Installation

```bash
pip install plotly>=5.17.0 pandas>=2.0.0 numpy>=1.24.0
pip install scipy>=1.11.0  # For Pareto surface (optional)
```

## File Locations

```
src/ui/components/
├── model_comparison_3d.py                    # Main component
└── model_browser_integration_example.py      # Full example

docs/
├── 3D_MODEL_COMPARISON_GUIDE.md             # Complete guide
└── 3D_MODEL_COMPARISON_QUICK_REF.md         # This file

tests/
└── test_model_comparison_3d.py              # Unit tests
```

## Troubleshooting One-Liners

```python
# Missing columns error
print(set(['id','name','phase','params','accuracy','latency','status']) - set(df.columns))

# Empty plot
print(f"Models: {len(df)}, Phases: {df['phase'].unique()}")

# Pareto not showing
print(f"Models: {len(df)} (need >=4), scipy: {importlib.util.find_spec('scipy') is not None}")

# Animation not playing
fig = create_model_comparison_3d(df, animate=False)  # Disable animation
```

## Sample Test Data

```python
from ui.components.model_comparison_3d import get_sample_data
models = get_sample_data()  # Returns 30-60 realistic models
```

## Integration with Model Browser

```python
# In src/ui/pages/model_browser.py
from ui.components.model_comparison_3d import render_model_browser_3d

def render():
    # ... existing code ...

    # Add view selector
    view = st.radio("View", ["3D", "List", "Both"], horizontal=True)

    # Add 3D view
    if view in ["3D", "Both"]:
        render_model_browser_3d(models)

    # Keep existing list view
    if view in ["List", "Both"]:
        # ... existing expander code ...
```

## Performance Tips

| Models | Animation | Pareto | Load Time |
|--------|-----------|--------|-----------|
| <20 | ✅ On | ✅ On | <1s |
| 20-50 | ✅ On | ❌ Off | 1-2s |
| 50-100 | ❌ Off | ❌ Off | 2-3s |
| >100 | ❌ Off | ❌ Off | 3-5s |

## Keyboard Controls (Plotly)

- **Click + Drag**: Rotate view
- **Scroll**: Zoom in/out
- **Shift + Click + Drag**: Pan
- **Double-click**: Reset view
- **Hover**: Show model details

## Custom Theme Example

```python
# Override colors
PHASE_COLORS['phase1'] = '#YOUR_COLOR'
BACKGROUND_COLOR = '#YOUR_BG'
GRID_COLOR = '#YOUR_GRID'
TEXT_COLOR = '#YOUR_TEXT'

# Then use normally
fig = create_model_comparison_3d(df)
```

## Testing

```bash
# Run tests
cd "C:/Users/17175/Desktop/the agent maker"
pytest tests/test_model_comparison_3d.py -v

# Run with coverage
pytest tests/test_model_comparison_3d.py --cov=src.ui.components.model_comparison_3d

# Run standalone demo
streamlit run src/ui/components/model_comparison_3d.py
```

## Links

- **Full Guide**: docs/3D_MODEL_COMPARISON_GUIDE.md
- **Plotly Docs**: https://plotly.com/python/3d-scatter-plots/
- **Design System**: src/ui/design_system.py

---

**Version**: 1.0.0
**Last Updated**: 2025-11-27

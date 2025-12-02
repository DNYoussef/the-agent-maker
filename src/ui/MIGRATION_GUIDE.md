# Design System Migration Guide

## Overview

This guide shows how to migrate existing Streamlit pages to use the new design system.

## Quick Migration Steps

### 1. Update app.py

**Before:**
```python
# Old custom CSS (lines 21-47)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    # ... more old CSS ...
</style>
""", unsafe_allow_html=True)
```

**After:**
```python
from design_system import get_custom_css

# Apply complete design system
st.markdown(get_custom_css(theme="dark"), unsafe_allow_html=True)
```

### 2. Update Class Names

#### Headers

**Before:**
```python
st.markdown('<div class="main-header">Agent Forge V2</div>', unsafe_allow_html=True)
```

**After:**
```python
st.markdown('<h1 class="main-header">Agent Forge V2</h1>', unsafe_allow_html=True)
# Or use section headers
st.markdown('<h2 class="section-header">Pipeline Status</h2>', unsafe_allow_html=True)
```

#### Status Badges

**Before:**
```python
st.markdown('<span class="status-success">SUCCESS</span>', unsafe_allow_html=True)
st.markdown('<span class="status-running">RUNNING</span>', unsafe_allow_html=True)
st.markdown('<span class="status-failed">FAILED</span>', unsafe_allow_html=True)
```

**After:**
```python
# Same classes work! But with enhanced styling
st.markdown('<span class="status-success">SUCCESS</span>', unsafe_allow_html=True)
st.markdown('<span class="status-running">RUNNING</span>', unsafe_allow_html=True)
st.markdown('<span class="status-failed">FAILED</span>', unsafe_allow_html=True)
# New: Also add .status-pending and programmatic badges
```

#### Cards

**Before:**
```python
st.markdown('''
<div class="metric-card">
    <h3>Phase 1</h3>
    <p>Content here</p>
</div>
''', unsafe_allow_html=True)
```

**After:**
```python
# Enhanced with glassmorphism
st.markdown('''
<div class="glass-card">
    <h3>Phase 1</h3>
    <p>Content here</p>
</div>
''', unsafe_allow_html=True)

# Or use programmatic approach
from design_system import get_card_styles, css_dict_to_string

styles = css_dict_to_string(get_card_styles("glass"))
st.markdown(f'<div style="{styles}"><h3>Phase 1</h3></div>', unsafe_allow_html=True)
```

### 3. Enhance Existing Components

#### Metrics (Native Streamlit)

**Before:**
```python
st.metric("Accuracy", "94.7%", delta="2.3%")
```

**After:**
```python
# Same code, auto-styled by design system!
st.metric("Accuracy", "94.7%", delta="2.3%")
# Now has glassmorphism background and accent colors
```

#### Buttons

**Before:**
```python
st.button("Start Training")
```

**After:**
```python
# Same code, auto-styled!
st.button("Start Training")
# Now has gradient background and glow on hover
```

## Page-by-Page Migration

### pipeline_overview.py

```python
def render():
    # Add at top of function
    st.markdown('<h2 class="section-header">Pipeline Overview</h2>', unsafe_allow_html=True)

    # Wrap metrics in glass cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("Current Phase", "3/8")
        st.markdown('</div>', unsafe_allow_html=True)

    # Status badges (keep existing code)
    st.markdown('<span class="status-running">RUNNING</span>', unsafe_allow_html=True)

    # Progress bars (auto-styled)
    st.progress(0.75)
```

### phase_details.py

```python
def render():
    st.markdown('<h2 class="section-header">Phase Details</h2>', unsafe_allow_html=True)

    # Phase cards with glassmorphism
    phases = [
        {"name": "Phase 1: Cognate", "status": "completed"},
        {"name": "Phase 2: EvoMerge", "status": "completed"},
        {"name": "Phase 3: Quiet-STaR", "status": "running"},
    ]

    for phase in phases:
        status_class = f"status-{phase['status']}"
        st.markdown(f'''
        <div class="glass-card">
            <h3>{phase['name']}</h3>
            <span class="{status_class}">{phase['status'].upper()}</span>
        </div>
        ''', unsafe_allow_html=True)
```

### system_monitor.py

```python
def render():
    st.markdown('<h2 class="section-header">System Monitor</h2>', unsafe_allow_html=True)

    # Metrics in glass cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">GPU Memory</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">4.2 GB</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-delta-negative">-0.8 GB</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Charts (auto-styled)
    import pandas as pd
    import numpy as np

    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['CPU', 'GPU', 'Memory']
    )
    st.line_chart(chart_data)
```

## Common Patterns

### Pattern 1: Glass Card with Status

```python
st.markdown(f'''
<div class="glass-card">
    <h3>Training Status</h3>
    <p>Epoch 42/100</p>
    <span class="status-running">RUNNING</span>
    <div class="metric-value">94.7%</div>
    <div class="metric-label">Accuracy</div>
</div>
''', unsafe_allow_html=True)
```

### Pattern 2: Metrics Grid

```python
col1, col2, col3, col4 = st.columns(4)

metrics = [
    {"label": "Accuracy", "value": "94.7%", "delta": "+2.3%"},
    {"label": "Loss", "value": "0.23", "delta": "-0.05"},
    {"label": "F1 Score", "value": "0.89", "delta": "+0.12"},
    {"label": "Inference", "value": "23ms", "delta": "-5ms"}
]

for col, metric in zip([col1, col2, col3, col4], metrics):
    with col:
        st.metric(
            label=metric["label"],
            value=metric["value"],
            delta=metric["delta"]
        )
```

### Pattern 3: Status Badge Grid

```python
col1, col2, col3, col4 = st.columns(4)

statuses = [
    ("Phase 1", "completed"),
    ("Phase 2", "completed"),
    ("Phase 3", "running"),
    ("Phase 4", "pending")
]

for col, (phase, status) in zip([col1, col2, col3, col4], statuses):
    with col:
        st.markdown(f'''
        <div class="glass-card text-center">
            <h4>{phase}</h4>
            <span class="status-{status}">{status.upper()}</span>
        </div>
        ''', unsafe_allow_html=True)
```

### Pattern 4: Progress with Label

```python
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="metric-label">Training Progress</div>', unsafe_allow_html=True)
st.progress(0.75)
st.markdown('<div class="metric-value">75%</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
```

## Color Updates

### Old Colors → New Colors

| Old | New | Usage |
|-----|-----|-------|
| `#1f77b4` (blue) | `#00F5D4` (cyan) | Primary accent |
| `#28a745` (green) | `#39FF14` (neon green) | Success |
| `#ffc107` (yellow) | `#FFB703` (amber) | Warning |
| `#dc3545` (red) | `#FF006E` (hot pink) | Error |
| `#f0f2f6` (light gray) | `rgba(27, 38, 59, 0.6)` (glass) | Card backgrounds |

### Using New Colors

```python
from design_system import COLORS

# In Python
primary_color = COLORS["primary"]
accent_color = COLORS["accent"]

# In HTML/CSS
st.markdown(f'<div style="color: {COLORS["accent"]}">Accent text</div>', unsafe_allow_html=True)

# With alpha
from design_system import get_color_with_alpha
semi_transparent = get_color_with_alpha("accent", 0.5)
st.markdown(f'<div style="background: {semi_transparent}">Semi-transparent</div>', unsafe_allow_html=True)
```

## Testing Migration

### Checklist

- [ ] Replace old CSS with `get_custom_css()`
- [ ] Update header classes to `.main-header` and `.section-header`
- [ ] Replace `.metric-card` with `.glass-card` (or keep for backward compatibility)
- [ ] Test status badges (`.status-success`, `.status-running`, etc.)
- [ ] Verify Streamlit native components (metrics, buttons) auto-style correctly
- [ ] Check charts and graphs render properly
- [ ] Test both dark and light themes (if applicable)
- [ ] Verify sidebar styling
- [ ] Check form elements (selectbox, text_input, etc.)
- [ ] Test responsive layout on different screen sizes

### Visual Regression

**Before Migration:**
1. Take screenshots of all pages
2. Note current color scheme
3. Document any custom styling

**After Migration:**
1. Compare new screenshots
2. Verify improved visual hierarchy
3. Check for any broken layouts
4. Test interactivity (hover, focus states)

## Gradual Migration Strategy

### Phase 1: Foundation (5 minutes)
1. Update `app.py` to use `get_custom_css()`
2. Test that existing pages still render

### Phase 2: Low-Hanging Fruit (10 minutes)
1. Update headers to use `.section-header`
2. Native Streamlit components auto-style (no code changes)

### Phase 3: Component Updates (30 minutes)
1. Migrate cards to `.glass-card`
2. Update custom metrics to use design system classes
3. Add status badges where appropriate

### Phase 4: Polish (20 minutes)
1. Add animations (`.pulse`, `.glow`)
2. Improve spacing consistency
3. Add gradient text effects where appropriate
4. Fine-tune responsive layout

## Rollback Plan

If issues arise:

```python
# Quick rollback - keep old CSS
OLD_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    # ... rest of old CSS ...
</style>
"""

st.markdown(OLD_CSS, unsafe_allow_html=True)
```

## Support

- See `design_system_demo.py` for live examples
- Check `README_DESIGN_SYSTEM.md` for API reference
- Review `design_system.py` source for all available utilities

## Next Steps

After migration:

1. **Optimize**: Remove unused custom CSS
2. **Enhance**: Add animations and micro-interactions
3. **Extend**: Create custom components using design system tokens
4. **Document**: Update page-specific documentation
5. **Test**: Comprehensive testing across pages and themes

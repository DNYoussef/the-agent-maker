# Design System Cheat Sheet

Quick reference for the Agent Forge V2 design system.

## Setup (One-time)

```python
from design_system import get_custom_css

st.markdown(get_custom_css(theme="dark"), unsafe_allow_html=True)
```

## Colors

```python
from design_system import COLORS

COLORS["primary"]    # #0D1B2A - Deep navy
COLORS["accent"]     # #00F5D4 - Electric cyan
COLORS["accent_2"]   # #F72585 - Magenta
COLORS["success"]    # #39FF14 - Neon green
COLORS["warning"]    # #FFB703 - Amber
COLORS["error"]      # #FF006E - Hot pink
```

## Headers

```html
<h1 class="main-header">Page Title</h1>
<h2 class="section-header">Section Title</h2>
```

## Cards

```html
<!-- Glass card (glassmorphism) -->
<div class="glass-card">
    <h3>Title</h3>
    <p>Content</p>
</div>

<!-- Solid card -->
<div class="solid-card">Content</div>

<!-- Elevated card -->
<div class="elevated-card">Content</div>
```

## Status Badges

```html
<span class="status-success">SUCCESS</span>
<span class="status-running">RUNNING</span>
<span class="status-failed">FAILED</span>
<span class="status-pending">PENDING</span>
```

## Metrics

```python
# Native Streamlit (auto-styled)
st.metric("Label", "Value", delta="Change")
```

```html
<!-- Custom large metric -->
<div class="metric-card">
    <div class="metric-label">Label</div>
    <div class="metric-value">25M</div>
    <div class="metric-delta-positive">+5%</div>
</div>
```

## Buttons

```python
# Native (auto-styled)
st.button("Click Me")
```

```html
<!-- Custom buttons -->
<button class="custom-button">Primary</button>
<button class="custom-button-secondary">Secondary</button>
<button class="custom-button-accent">Accent</button>
```

## Text Utilities

```html
<p class="text-accent">Cyan text</p>
<p class="text-accent-2">Magenta text</p>
<p class="text-success">Green text</p>
<p class="text-warning">Amber text</p>
<p class="text-error">Pink text</p>
<p class="gradient-text">Gradient effect</p>
<p class="uppercase">UPPERCASE</p>
<span class="font-code">Monospace</span>
```

## Animations

```html
<span class="pulse">Pulsing text</span>
<div class="glow">Glowing border</div>
```

## Programmatic

```python
from design_system import (
    get_card_styles,
    get_button_styles,
    get_badge_styles,
    css_dict_to_string,
    get_color_with_alpha
)

# Get component styles
card = css_dict_to_string(get_card_styles("glass"))
st.markdown(f'<div style="{card}">Content</div>', unsafe_allow_html=True)

# Color with alpha
semi = get_color_with_alpha("accent", 0.5)
# Returns: "rgba(0, 245, 212, 0.5)"
```

## Common Patterns

### Glass Card with Status
```html
<div class="glass-card">
    <h3>Phase 3: Quiet-STaR</h3>
    <p>Reasoning enhancement</p>
    <span class="status-running">RUNNING</span>
</div>
```

### Metrics Grid
```python
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", "94.7%", delta="+2.3%")
with col2:
    st.metric("Loss", "0.23", delta="-0.05")
# ... etc
```

### Progress Card
```html
<div class="glass-card">
    <div class="metric-label">Training Progress</div>
    <div class="metric-value">75%</div>
</div>
```
```python
st.progress(0.75)
```

## Files

- **design_system.py** - Core module
- **design_system_demo.py** - Interactive demo
- **README_DESIGN_SYSTEM.md** - Full documentation
- **MIGRATION_GUIDE.md** - Migration help
- **DESIGN_SYSTEM_COMPLETE.md** - Summary

## Quick Start

```bash
# Run demo
cd "C:/Users/17175/Desktop/the agent maker/src/ui"
streamlit run design_system_demo.py
```

## Help

```python
# See all colors
from design_system import COLORS
print(COLORS)

# See all typography
from design_system import TYPOGRAPHY
print(TYPOGRAPHY)

# See all spacing
from design_system import SPACING
print(SPACING)
```

# Agent Forge V2 - Design System

## Overview

A comprehensive design system for the Agent Forge V2 Streamlit UI featuring a futuristic command center theme with glassmorphism effects, neon accents, and smooth animations.

## Quick Start

### Basic Usage

```python
import streamlit as st
from design_system import get_custom_css

# Apply the complete design system
st.markdown(get_custom_css(theme="dark"), unsafe_allow_html=True)

# Now use custom CSS classes in your Streamlit app
st.markdown('<h1 class="main-header">Agent Forge V2</h1>', unsafe_allow_html=True)
st.markdown('<div class="glass-card"><p>Your content here</p></div>', unsafe_allow_html=True)
```

### Run the Demo

```bash
cd "C:/Users/17175/Desktop/the agent maker/src/ui"
streamlit run design_system_demo.py
```

## Theme

### Color Palette

**Futuristic Command Center Theme**

| Color | Hex | Usage |
|-------|-----|-------|
| Primary | `#0D1B2A` | Deep navy - Main background |
| Secondary | `#1B263B` | Slate - Secondary surfaces |
| Accent | `#00F5D4` | Electric cyan - Primary highlights |
| Accent 2 | `#F72585` | Magenta - Secondary highlights |
| Surface | `#415A77` | Dark slate - Elevated components |
| Text Primary | `#E0E1DD` | Off-white - Main text |
| Success | `#39FF14` | Neon green - Success states |
| Warning | `#FFB703` | Amber - Warning states |
| Error | `#FF006E` | Hot pink - Error states |

### Typography

**Font Families:**
- **Display/Headers**: Space Grotesk
- **Body**: Inter
- **Code**: JetBrains Mono

**Type Scale:**
- Display: 3rem (48px)
- H1: 2.5rem (40px)
- H2: 2rem (32px)
- H3: 1.5rem (24px)
- Body: 1rem (16px)
- Small: 0.875rem (14px)
- Tiny: 0.75rem (12px)

### Spacing Scale

8px base unit:
- xs: 4px
- sm: 8px
- md: 16px
- lg: 24px
- xl: 32px
- xxl: 48px
- xxxl: 64px

## Components

### 1. Cards

**Glass Card** (Glassmorphism with blur):
```python
from design_system import get_card_styles, css_dict_to_string

styles = css_dict_to_string(get_card_styles("glass"))
st.markdown(f'<div style="{styles}">Content</div>', unsafe_allow_html=True)

# Or use CSS class
st.markdown('<div class="glass-card">Content</div>', unsafe_allow_html=True)
```

**Variants:** `"glass"`, `"solid"`, `"elevated"`

### 2. Badges

**Status Badges:**
```python
# Predefined classes
st.markdown('<span class="status-success">SUCCESS</span>', unsafe_allow_html=True)
st.markdown('<span class="status-running">RUNNING</span>', unsafe_allow_html=True)
st.markdown('<span class="status-failed">FAILED</span>', unsafe_allow_html=True)
st.markdown('<span class="status-pending">PENDING</span>', unsafe_allow_html=True)

# Programmatic
from design_system import get_badge_styles, css_dict_to_string

styles = css_dict_to_string(get_badge_styles("info"))
st.markdown(f'<span style="{styles}">INFO</span>', unsafe_allow_html=True)
```

**Variants:** `"success"`, `"warning"`, `"error"`, `"info"`, `"pending"`

### 3. Metrics

**Large Metric Display:**
```python
from design_system import get_metric_styles

st.markdown('<div class="metric-card">', unsafe_allow_html=True)
st.markdown('<div class="metric-label">Total Parameters</div>', unsafe_allow_html=True)
st.markdown('<div class="metric-value">25M</div>', unsafe_allow_html=True)
st.markdown('<div class="metric-delta-positive">+5%</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
```

**Sizes:** `"small"`, `"medium"`, `"large"`

### 4. Buttons

```python
# Streamlit native (auto-styled)
st.button("Primary Button")

# Custom variants
st.markdown('<button class="custom-button">Primary</button>', unsafe_allow_html=True)
st.markdown('<button class="custom-button-secondary">Secondary</button>', unsafe_allow_html=True)
st.markdown('<button class="custom-button-accent">Accent</button>', unsafe_allow_html=True)

# Programmatic
from design_system import get_button_styles, css_dict_to_string

styles = css_dict_to_string(get_button_styles("primary"))
st.markdown(f'<button style="{styles}">Click Me</button>', unsafe_allow_html=True)
```

**Variants:** `"primary"`, `"secondary"`, `"accent"`, `"ghost"`

## Utility Classes

### Text Utilities

```html
<p class="text-accent">Accent colored text</p>
<p class="text-accent-2">Secondary accent text</p>
<p class="text-success">Success text</p>
<p class="text-warning">Warning text</p>
<p class="text-error">Error text</p>
<p class="gradient-text">Gradient text effect</p>
<p class="uppercase">Uppercase with letter spacing</p>
```

### Font Utilities

```html
<span class="font-code">Code font</span>
<span class="font-heading">Heading font</span>
```

### Layout Utilities

```html
<div class="text-center">Centered text</div>
```

## Programmatic Access

### Access Design Tokens

```python
from design_system import COLORS, TYPOGRAPHY, SPACING, RADII, SHADOWS

# Use in Python
primary_color = COLORS["primary"]
body_size = TYPOGRAPHY["size_body"]
medium_spacing = SPACING["md"]

# Custom colors with alpha
from design_system import get_color_with_alpha
semi_transparent_accent = get_color_with_alpha("accent", 0.5)
# Returns: "rgba(0, 245, 212, 0.5)"
```

### Generate Component Styles

```python
from design_system import (
    get_card_styles,
    get_button_styles,
    get_badge_styles,
    get_metric_styles,
    css_dict_to_string
)

# Get style dictionary
card_styles = get_card_styles("glass")
# Returns: {"background": "...", "border": "...", ...}

# Convert to CSS string
css_string = css_dict_to_string(card_styles)
# Returns: "background: ...; border: ...; ..."
```

## Themes

### Dark Mode (Default)

```python
st.markdown(get_custom_css(theme="dark"), unsafe_allow_html=True)
```

### Light Mode

```python
st.markdown(get_custom_css(theme="light"), unsafe_allow_html=True)
```

## Complete Example

```python
import streamlit as st
from design_system import get_custom_css, get_card_styles, css_dict_to_string, COLORS

# Page config
st.set_page_config(
    page_title="Agent Forge V2",
    page_icon="🤖",
    layout="wide"
)

# Apply design system
st.markdown(get_custom_css(theme="dark"), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Agent Forge V2</h1>', unsafe_allow_html=True)

# Section
st.markdown('<h2 class="section-header">Pipeline Status</h2>', unsafe_allow_html=True)

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Phase", "3/8", delta="Running")

with col2:
    st.metric("Accuracy", "94.7%", delta="2.3%")

with col3:
    st.metric("GPU Memory", "4.2 GB", delta="-0.8 GB")

with col4:
    st.metric("Time", "23ms", delta="-5ms")

# Card with status badge
glass_styles = css_dict_to_string(get_card_styles("glass"))
st.markdown(f'''
<div style="{glass_styles}">
    <h3>Phase 3: Quiet-STaR</h3>
    <p>Reasoning enhancement via thought generation</p>
    <span class="status-running">RUNNING</span>
</div>
''', unsafe_allow_html=True)

# Progress
st.progress(0.75)

# Button
st.button("Start Training", key="start")
```

## Animations

### Built-in Animations

- **Pulse**: `class="pulse"` - Pulsing opacity (2s loop)
- **Glow**: `class="glow"` - Glowing border (2s loop)
- **Hover effects**: All cards and buttons have smooth hover transitions

### Usage

```html
<span class="status-running pulse">RUNNING</span>
<div class="glass-card glow">Important content</div>
```

## Customization

### Extend Colors

```python
from design_system import COLORS

# Add custom color
custom_color = "#FF5733"

# Use in inline styles
st.markdown(f'<div style="color: {custom_color};">Custom text</div>', unsafe_allow_html=True)
```

### Override CSS

```python
# Get base CSS
base_css = get_custom_css(theme="dark")

# Add custom overrides
custom_css = """
<style>
.custom-component {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 12px;
}
</style>
"""

st.markdown(base_css, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)
```

## Best Practices

1. **Always inject CSS first**: Call `get_custom_css()` before other content
2. **Use semantic classes**: Prefer `.status-success` over inline styles
3. **Maintain consistency**: Stick to the color palette and spacing scale
4. **Leverage programmatic access**: Use `COLORS`, `SPACING` constants for consistency
5. **Test both themes**: Verify components work in dark and light modes
6. **Performance**: Inject CSS once per page, not per component

## File Structure

```
src/ui/
├── design_system.py           # Main design system module
├── design_system_demo.py      # Interactive demo
└── README_DESIGN_SYSTEM.md    # This file
```

## API Reference

### Functions

#### `get_custom_css(theme: str = "dark") -> str`
Returns complete CSS string for Streamlit injection.

**Args:**
- `theme` (str): `"dark"` or `"light"`

**Returns:**
- Complete CSS string with all styles

#### `get_card_styles(variant: str = "glass") -> Dict[str, str]`
Returns card component styles as dictionary.

**Args:**
- `variant` (str): `"glass"`, `"solid"`, or `"elevated"`

**Returns:**
- Dict of CSS properties

#### `get_button_styles(variant: str = "primary") -> Dict[str, str]`
Returns button component styles.

**Args:**
- `variant` (str): `"primary"`, `"secondary"`, `"accent"`, or `"ghost"`

#### `get_badge_styles(status: str = "info") -> Dict[str, str]`
Returns badge component styles.

**Args:**
- `status` (str): `"success"`, `"warning"`, `"error"`, `"info"`, or `"pending"`

#### `get_metric_styles(size: str = "medium") -> Dict[str, str]`
Returns metric display styles (nested dict).

**Args:**
- `size` (str): `"small"`, `"medium"`, or `"large"`

**Returns:**
- Dict with keys: `"value"`, `"label"`, `"delta_positive"`, `"delta_negative"`

#### `get_color_with_alpha(color_name: str, alpha: float = 1.0) -> str`
Get color from palette with custom alpha.

**Args:**
- `color_name` (str): Color key from `COLORS` dict
- `alpha` (float): Alpha value (0.0 - 1.0)

**Returns:**
- RGBA color string

#### `css_dict_to_string(styles: Dict[str, str]) -> str`
Convert CSS property dict to string.

**Args:**
- `styles` (dict): Dict of CSS properties

**Returns:**
- CSS string for inline styles

### Constants

- `COLORS` (dict): Complete color palette
- `COLORS_LIGHT` (dict): Light mode overrides
- `TYPOGRAPHY` (dict): Font families, sizes, weights, line heights
- `SPACING` (dict): Spacing scale (xs to xxxl)
- `RADII` (dict): Border radius values
- `SHADOWS` (dict): Box shadow definitions

## Support

For issues or questions:
1. Check `design_system_demo.py` for examples
2. Review this README
3. Inspect `design_system.py` source code

## Version

**Version:** 1.0.0
**Last Updated:** 2025-11-27
**Author:** Agent Forge V2 Team

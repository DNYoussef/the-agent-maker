# Streamlit HTML Rendering Guide

## Quick Decision Tree

### When to use `st.markdown()` with `unsafe_allow_html=True`:
- ✅ Simple HTML (single tags, minimal nesting)
- ✅ Basic formatting (bold, italic, links)
- ✅ Simple divs with 1-2 inline styles
- ✅ No animations or complex CSS

**Example**:
```python
st.markdown('<div style="color: #00F5D4;">Hello World</div>', unsafe_allow_html=True)
```

---

### When to use `st.components.v1.html()`:
- ✅ Complex nested divs (3+ levels deep)
- ✅ CSS animations and transitions
- ✅ Gradient backgrounds (linear-gradient, radial-gradient)
- ✅ Flexbox/Grid layouts
- ✅ Custom fonts with multiple weights
- ✅ Box shadows and glow effects
- ✅ Glassmorphism (backdrop-filter)

**Example**:
```python
import streamlit.components.v1 as components

html = """
<div style="
    background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
    border: 2px solid #00F5D4;
    padding: 30px;
    box-shadow: 0 8px 32px rgba(0, 245, 212, 0.2);
">
    <h1 style="color: #00F5D4;">Complex Card</h1>
</div>
"""
components.html(html, height=200)
```

---

## Common Issues & Fixes

### Issue 1: Raw HTML Tags Showing as Text
**Symptom**: You see `<div>`, `</div>`, etc. as text on the page

**Diagnosis**: HTML is too complex for `st.markdown()`

**Fix**:
```python
# BEFORE (broken):
st.markdown(create_complex_card(), unsafe_allow_html=True)

# AFTER (working):
import streamlit.components.v1 as components
card_html = create_complex_card()
components.html(card_html, height=350)
```

---

### Issue 2: CSS Not Applying
**Symptom**: No colors, gradients, or borders visible

**Diagnosis**: Streamlit sanitizing CSS properties

**Fix**: Use `components.html()` instead of `st.markdown()`

---

### Issue 3: Animations Not Working
**Symptom**: CSS keyframes animations don't run

**Fix**:
```python
html = """
<style>
@keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.6; }
}
</style>
<div style="animation: pulse 3s infinite;">Animated</div>
"""
components.html(html, height=100)
```

---

### Issue 4: Content Cut Off in components.html()
**Symptom**: Bottom of HTML is hidden

**Fix**: Increase `height` parameter
```python
# Too small:
components.html(html, height=200)

# Better:
components.html(html, height=350)

# Calculate dynamically if needed:
lines = html.count('<div')
height = max(200, lines * 50)
components.html(html, height=height)
```

---

### Issue 5: Emoji Not Rendering
**Symptom**: Emojis show as � or boxes

**Fix**: Use Unicode escape or remove emojis
```python
# Option 1: Remove emojis
"Phase 6: Tool & Persona Baking"  # instead of "🔥 Phase 6..."

# Option 2: Use Unicode (if absolutely needed)
html = "\U0001F525 Fire"  # 🔥
```

---

## Agent Maker Specific Patterns

### Phase Header Cards (Used in Phase 2, 6)
```python
import streamlit.components.v1 as components

def render_hero_section():
    hero_html = """
    <div style="
        background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
        border: 2px solid #00F5D4;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 0 40px rgba(0, 245, 212, 0.2);
    ">
        <h1 style="color: #00F5D4; font-size: 42px;">PHASE TITLE</h1>
        <p style="color: #8B9DAF; font-size: 16px;">Subtitle text</p>
    </div>
    """
    components.html(hero_html, height=150)
```

**Height**: 150px for title + subtitle

---

### Model Cards (Used in Phase 1)
```python
def create_model_card(...):
    html = f"""
    <div style="
        background: linear-gradient(135deg, #1B283822 0%, #2E3F4F22 100%);
        border: 2px solid {status_color}44;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    ">
        <!-- Animated background -->
        <div style="animation: pulse 3s infinite;">...</div>

        <!-- Card content -->
        <div style="position: relative; z-index: 1;">
            {content}
        </div>
    </div>
    """
    return html

# Usage:
card_html = create_model_card(...)
components.html(card_html, height=400)
```

**Height**: 400px for full model card

---

### Glassmorphism Cards (Used in Phase 6)
```python
glass_html = """
<div style="
    background: rgba(27, 40, 56, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 245, 212, 0.2);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
">
    <h3 style="color: #00F5D4;">Title</h3>
    <p style="color: #E0E1DD;">Content</p>
</div>
"""
components.html(glass_html, height=300)
```

**Height**: 300-360px depending on content

---

## Color Palette Reference

```python
# Agent Maker Theme Colors
COLORS = {
    'background_dark': '#0D1B2A',
    'background_medium': '#1B2838',
    'background_light': '#2E3F4F',
    'text_primary': '#E0E1DD',
    'text_secondary': '#8B9DAF',
    'accent_cyan': '#00F5D4',
    'accent_purple': '#8338EC',
    'accent_magenta': '#FF006E',
    'accent_orange': '#FB5607',
    'accent_yellow': '#FFBE0B'
}
```

---

## Common CSS Patterns

### Gradient Background
```css
background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
```

### Glowing Border
```css
border: 2px solid #00F5D4;
box-shadow: 0 0 20px rgba(0, 245, 212, 0.3);
```

### Glassmorphism Effect
```css
background: rgba(27, 40, 56, 0.6);
backdrop-filter: blur(10px);
border: 1px solid rgba(0, 245, 212, 0.2);
```

### Text Shadow (Glow)
```css
text-shadow: 0 0 20px rgba(0, 245, 212, 0.5);
```

### Smooth Transitions
```css
transition: all 0.3s ease;
```

### Pulse Animation
```css
@keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.6; }
}
animation: pulse 3s ease-in-out infinite;
```

---

## Testing Checklist

After implementing complex HTML:

- [ ] Import `streamlit.components.v1 as components` added
- [ ] `st.markdown()` replaced with `components.html()`
- [ ] Height parameter specified and tested
- [ ] No raw HTML tags visible in browser
- [ ] Colors and gradients render correctly
- [ ] Animations play smoothly
- [ ] Text is readable and properly sized
- [ ] No content cut off at bottom
- [ ] Tested in Chrome/Firefox/Edge
- [ ] No console errors in browser DevTools

---

## Height Guidelines by Component Type

| Component | Typical Height | Notes |
|-----------|---------------|-------|
| Hero section (title + subtitle) | 150px | Fixed content |
| Small card (1-3 lines) | 200-250px | Metric cards |
| Medium card (4-8 lines) | 300-400px | Model cards |
| Large card (9+ lines) | 400-600px | Detailed info |
| Architecture diagram | 600-800px | Complex layouts |
| Full-page section | 800-1000px | Rarely needed |

**Formula**: `height = (line_count * 40) + (padding * 2) + border_space`

---

## Debugging Tips

### Problem: Content not visible
1. Check height is sufficient: `components.html(html, height=500)`
2. Inspect with browser DevTools (right-click iframe → Inspect)
3. Check CSS for `display: none` or `visibility: hidden`
4. Verify background colors aren't white-on-white

### Problem: Layout broken
1. Check for unclosed HTML tags: `</div>` count matches `<div>` count
2. Verify CSS syntax (semicolons, colons, quotes)
3. Test HTML in standalone HTML file first
4. Simplify HTML to find the breaking element

### Problem: Slow rendering
1. Reduce iframe height if possible
2. Minimize number of `components.html()` calls (combine HTML)
3. Remove complex animations if performance is poor
4. Consider using `st.markdown()` for simpler sections

---

## Best Practices

1. **Prefer `st.markdown()` when possible** (faster, lighter)
2. **Only use `components.html()` for complex HTML** (animations, nested structures)
3. **Always specify height** (no auto-sizing in iframes)
4. **Test in multiple browsers** (Chrome, Firefox, Edge)
5. **Keep HTML in separate functions** (maintainability)
6. **Use f-strings for dynamic content** (`f"{variable}"`)
7. **Avoid external CSS files** (use inline styles)
8. **Comment height choices** (`# height=400: fits model card + metrics`)

---

## Example: Converting Broken Code

### Before (Broken)
```python
def render_card():
    st.markdown("""
    <div class="complex-card">
        <div class="nested-content">
            <h1 style="background: linear-gradient(...);">Title</h1>
            <div class="animated-box" style="animation: pulse 3s;">...</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

**Problems**:
- CSS classes won't work
- Gradient background won't render
- Animation won't play
- Nested divs may break

### After (Fixed)
```python
import streamlit.components.v1 as components

def render_card():
    html = """
    <style>
    @keyframes pulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }
    </style>
    <div style="
        background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
        border-radius: 12px;
        padding: 20px;
    ">
        <div style="position: relative;">
            <h1 style="color: #00F5D4; text-shadow: 0 0 10px rgba(0,245,212,0.5);">
                Title
            </h1>
            <div style="animation: pulse 3s infinite;">
                Animated content
            </div>
        </div>
    </div>
    """
    components.html(html, height=250)
```

**Fixes**:
- ✅ Inline styles instead of classes
- ✅ Animation keyframes in `<style>` block
- ✅ Explicit height (250px)
- ✅ Proper nesting maintained

---

## Reference: Fixed Files in Agent Maker

| File | Components | Heights | Notes |
|------|-----------|---------|-------|
| `phase1_cognate.py` | Model cards, Architecture | 400px, 650px | Animations work |
| `phase2_evomerge.py` | Hero, Champion card | 150px, 320px | Gradients render |
| `phase6_baking.py` | Hero section | 150px | Simplified HTML |

See `HTML_RENDERING_FIX_SUMMARY.md` for detailed change log.

---

**Last Updated**: Current session
**Maintainer**: Claude Code

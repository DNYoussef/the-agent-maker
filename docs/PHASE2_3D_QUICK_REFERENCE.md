# Phase 2: 3D Visualization Quick Reference

## 60-Second Integration

### 1. Install Dependencies
```bash
pip install plotly pandas numpy
```

### 2. Apply Patch

**Manual** (lines 154-156 in `src/ui/pages/phase_details.py`):

Replace:
```python
    # Binary pairing tree
    with st.expander("View Binary Pairing Tree"):
        st.info("3D merge visualization coming soon (Three.js integration)")
```

With:
```python
    # 3D Merge Tree Visualization
    try:
        from ui.components.merge_tree_3d import render_phase2_3d_visualization
        render_phase2_3d_visualization(
            generations=50, models_per_gen=8, height=800, show_controls=True
        )
    except ImportError as e:
        st.warning(f"3D visualization unavailable: {e}")
```

**OR Automated**:
```bash
python docs/PHASE2_INTEGRATION_PATCH.py --apply
```

### 3. Test
```bash
streamlit run src/ui/app.py
```
Navigate to: **Phase Details > Phase 2: EvoMerge**

---

## Merge Technique Colors

| Technique | Color | Symbol |
|-----------|-------|--------|
| Linear | Blue | Circle |
| SLERP | Green | Diamond |
| TIES (best) | Orange | Square |
| DARE | Purple | Cross |
| FrankenMerge | Red | Triangle |
| DFS | Cyan | Star |
| Phase 1 | Magenta | Large Diamond |

---

## Interactive Controls

- **Highlight Lineage**: Select model from dropdown
- **Filter Generations**: Use slider (e.g., 10-30)
- **Regenerate**: Click button for new tree
- **Camera**: Drag (rotate), Scroll (zoom), Shift+Drag (pan)

---

## Files Created

```
src/ui/components/merge_tree_3d.py              # Main component (650 lines)
docs/PHASE2_3D_VISUALIZATION_INTEGRATION.md     # Full guide (350 lines)
docs/PHASE2_INTEGRATION_PATCH.py                # Patch script (90 lines)
docs/PHASE2_3D_VISUALIZATION_SUMMARY.md         # Complete summary (440 lines)
```

---

## Standalone Testing

```bash
streamlit run src/ui/components/merge_tree_3d.py
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| ImportError: plotly | `pip install plotly` |
| Visualization blank | `pip install --upgrade streamlit` |
| Path errors | Run from project root |
| Performance slow | Reduce height parameter or filter generations |

---

## Full Documentation

See: `docs/PHASE2_3D_VISUALIZATION_SUMMARY.md`

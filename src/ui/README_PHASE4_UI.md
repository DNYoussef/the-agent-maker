# Phase 4 BitNet Compression - Streamlit UI

## Overview

The Phase 4 dashboard provides real-time visualization of the BitNet 1.58-bit compression process, including:

- **Real-time compression progress** - Layer-by-layer quantization tracking
- **Sparsity heatmaps** - Visual representation of zero-injection across layers
- **Pre/post compression metrics** - Side-by-side comparison tables
- **Fine-tuning loss curves** - MuGrokfast training progress
- **Quality validation** - Automated quality gates and gradient flow testing
- **Dual model outputs** - Comparison of quantized (12MB) vs dequantized (50MB) models

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit plotly pandas numpy
```

### 2. Run the Dashboard

From the project root:

```bash
streamlit run src/ui/app.py
```

Then navigate to **"Phase 4: BitNet Compression"** in the sidebar.

### 3. Configuration

The sidebar provides controls for:
- **Model size detection** - Auto-adapt compression targets
- **Compression ratio** - Target 6-12× compression
- **Sparsity injection** - Threshold and target sparsity
- **Fine-tuning settings** - Epochs, Grokfast λ
- **Output options** - Save quantized and/or dequantized models

## Features

### Tab 1: Overview
- Phase status and key metrics
- 7-step compression pipeline visualization
- Feature highlights

### Tab 2: Real-Time Progress
- Layer-by-layer compression heatmap
- Sparsity distribution bar chart
- Progress tracking with ETA

### Tab 3: Metrics & Analysis
- Pre/post compression comparison table
- Fine-tuning loss curve
- Model size and performance metrics

### Tab 4: Quality Validation
- 5 automated quality gates:
  - Compression ratio ≥6.0×
  - Accuracy preservation ≥95%
  - Perplexity degradation ≤10%
  - Sparsity range 25-45%
  - Gradient flow test PASS
- Layer-wise gradient norm distribution

### Tab 5: Dual Model Outputs
- Side-by-side comparison of quantized vs dequantized models
- Output directory structure
- Phase 4→5 handoff validation checklist

## Integration with Phase Controller

The UI can be connected to the actual Phase 4 controller via WebSocket for real-time updates:

```python
from src.phase4_bitnet import Phase4Controller, Phase4Config

config = Phase4Config(
    model_path="phase3_output/",
    output_path="phase4_output/",
    target_compression_ratio=8.0,
    enable_fine_tuning=True
)

controller = Phase4Controller(config)

# Execute with progress callback
results = controller.execute(
    phase3_output_path="phase3_output/",
    wandb_logger=None,
    progress_callback=update_streamlit_ui  # Custom callback
)
```

## Customization

### Adding Custom Metrics

To add new metrics to the dashboard, edit `phase4_bitnet.py`:

```python
def render_custom_metric():
    st.metric("Custom Metric", value, delta)
```

### Modifying Visualizations

All charts use Plotly. To customize:

```python
fig = go.Figure(...)
fig.update_layout(
    title="New Title",
    xaxis_title="X",
    yaxis_title="Y"
)
st.plotly_chart(fig, use_container_width=True)
```

## Performance Considerations

- **Mock Data**: Current implementation uses mock data for demonstration
- **Real-Time Updates**: For production, connect to Phase4Controller via WebSocket
- **Large Models**: Heatmaps may be slow for >100 layers; consider pagination

## Troubleshooting

### Issue: Dashboard not loading
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r config/requirements-dev.txt
```

### Issue: Import errors
**Solution**: Run from project root with proper PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
streamlit run src/ui/app.py
```

### Issue: Slow rendering
**Solution**: Reduce plot complexity or enable caching:
```python
@st.cache_data
def load_data():
    return data
```

## Future Enhancements

- [ ] WebSocket integration for real-time Phase4Controller updates
- [ ] Export charts as PNG/SVG
- [ ] Historical compression run comparison
- [ ] Model diff visualization (pre vs post)
- [ ] Interactive sparsity threshold tuning
- [ ] Automatic quality gate alerting

## Related Files

- **Implementation**: [src/phase4_bitnet/phase_controller.py](../../phase4_bitnet/phase_controller.py)
- **API Reference**: [docs/phases/phase4/API_REFERENCE.md](../../../docs/phases/phase4/API_REFERENCE.md)
- **Implementation Guide**: [docs/phases/phase4/IMPLEMENTATION_GUIDE.md](../../../docs/phases/phase4/IMPLEMENTATION_GUIDE.md)

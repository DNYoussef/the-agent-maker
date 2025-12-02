"""
3D Model Comparison Visualization
Interactive 3D scatter plot for comparing models across 8 phases
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import List, Optional, Dict, Any


# Phase colors - futuristic theme matching design system
PHASE_COLORS = {
    'phase1': '#00F5D4',  # Cyan
    'phase2': '#0099FF',  # Blue
    'phase3': '#9D4EDD',  # Purple
    'phase4': '#FF006E',  # Magenta
    'phase5': '#FB5607',  # Orange
    'phase6': '#FFBE0B',  # Yellow
    'phase7': '#06FFA5',  # Green
    'phase8': '#F72585',  # Pink
}

# Status symbols
STATUS_SYMBOLS = {
    'complete': 'circle',
    'running': 'diamond',
    'failed': 'x',
    'pending': 'square',
}

# Dark theme background
BACKGROUND_COLOR = '#0D1B2A'
GRID_COLOR = '#1B263B'
TEXT_COLOR = '#E0E1DD'


def create_model_comparison_3d(
    models_df: pd.DataFrame,
    highlighted_ids: Optional[List[str]] = None,
    show_phases: Optional[List[str]] = None,
    show_pareto: bool = False,
    animate: bool = True
) -> go.Figure:
    """
    Create 3D model comparison scatter plot.

    Args:
        models_df: DataFrame with columns:
            - id: Model identifier
            - name: Model name
            - phase: Phase identifier (phase1-phase8)
            - params: Model size in parameters (millions)
            - accuracy: Accuracy/performance metric (0-100%)
            - latency: Inference speed (milliseconds)
            - compression: Compression ratio (optional)
            - status: Model status (complete/running/failed/pending)
        highlighted_ids: List of model IDs to highlight (champion models)
        show_phases: List of phases to display (1-8), None for all
        show_pareto: Draw Pareto frontier surface
        animate: Enable entrance animation

    Returns:
        plotly.graph_objects.Figure
    """
    # Filter by phases if specified
    if show_phases:
        models_df = models_df[models_df['phase'].isin(show_phases)]

    # Create figure
    fig = go.Figure()

    # Add traces for each phase
    for phase in sorted(models_df['phase'].unique()):
        phase_data = models_df[models_df['phase'] == phase]

        # Get phase-specific data
        x_data = phase_data['params'] / 1_000_000  # Convert to millions
        y_data = phase_data['accuracy']
        z_data = phase_data['latency']

        # Size based on compression ratio (if available)
        if 'compression' in phase_data.columns:
            sizes = phase_data['compression'] * 5 + 5  # Scale for visibility
        else:
            sizes = 10

        # Color for this phase
        color = PHASE_COLORS.get(phase, '#FFFFFF')

        # Symbol based on status
        symbols = [STATUS_SYMBOLS.get(s, 'circle') for s in phase_data['status']]

        # Hover text with full details
        hover_text = []
        for _, row in phase_data.iterrows():
            is_highlighted = highlighted_ids and row['id'] in highlighted_ids
            champion_badge = ' [CHAMPION]' if is_highlighted else ''

            text = (
                f"<b>{row['name']}{champion_badge}</b><br>"
                f"ID: {row['id']}<br>"
                f"Phase: {phase.upper()}<br>"
                f"<br>"
                f"Parameters: {row['params']:,}<br>"
                f"Accuracy: {row['accuracy']:.2f}%<br>"
                f"Latency: {row['latency']:.1f} ms<br>"
            )

            if 'compression' in row:
                text += f"Compression: {row['compression']:.2f}x<br>"

            text += f"<br>Status: {row['status']}"
            hover_text.append(text)

        # Determine if any models in this phase are highlighted
        if highlighted_ids:
            highlighted_mask = phase_data['id'].isin(highlighted_ids)

            # Add non-highlighted models first
            if (~highlighted_mask).any():
                non_highlighted = phase_data[~highlighted_mask]
                fig.add_trace(go.Scatter3d(
                    x=x_data[~highlighted_mask],
                    y=y_data[~highlighted_mask],
                    z=z_data[~highlighted_mask],
                    mode='markers',
                    name=f'{phase.upper()}',
                    marker=dict(
                        size=sizes if isinstance(sizes, int) else sizes[~highlighted_mask],
                        color=color,
                        opacity=0.6,
                        symbol=[STATUS_SYMBOLS.get(s, 'circle')
                               for s in non_highlighted['status']],
                        line=dict(color=BACKGROUND_COLOR, width=0.5)
                    ),
                    text=[hover_text[i] for i in range(len(hover_text))
                          if not highlighted_mask.iloc[i]],
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=True
                ))

            # Add highlighted models with emphasis
            if highlighted_mask.any():
                highlighted = phase_data[highlighted_mask]
                fig.add_trace(go.Scatter3d(
                    x=x_data[highlighted_mask],
                    y=y_data[highlighted_mask],
                    z=z_data[highlighted_mask],
                    mode='markers',
                    name=f'{phase.upper()} (Champion)',
                    marker=dict(
                        size=sizes if isinstance(sizes, int) else sizes[highlighted_mask] * 1.5,
                        color=color,
                        opacity=1.0,
                        symbol=[STATUS_SYMBOLS.get(s, 'circle')
                               for s in highlighted['status']],
                        line=dict(color='#FFFFFF', width=3)
                    ),
                    text=[hover_text[i] for i in range(len(hover_text))
                          if highlighted_mask.iloc[i]],
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=True
                ))
        else:
            # No highlighting - add all models for this phase
            fig.add_trace(go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode='markers',
                name=f'{phase.upper()}',
                marker=dict(
                    size=sizes,
                    color=color,
                    opacity=0.7,
                    symbol=symbols,
                    line=dict(color=BACKGROUND_COLOR, width=0.5)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ))

    # Add Pareto frontier surface (optional)
    if show_pareto and len(models_df) > 3:
        pareto_surface = _compute_pareto_surface(models_df)
        if pareto_surface is not None:
            fig.add_trace(pareto_surface)

    # Layout configuration
    layout_kwargs = dict(
        scene=dict(
            xaxis=dict(
                title='Model Size (M params)',
                backgroundcolor=BACKGROUND_COLOR,
                gridcolor=GRID_COLOR,
                showbackground=True,
                zerolinecolor=GRID_COLOR,
                color=TEXT_COLOR
            ),
            yaxis=dict(
                title='Accuracy (%)',
                backgroundcolor=BACKGROUND_COLOR,
                gridcolor=GRID_COLOR,
                showbackground=True,
                zerolinecolor=GRID_COLOR,
                color=TEXT_COLOR
            ),
            zaxis=dict(
                title='Inference Speed (ms)',
                backgroundcolor=BACKGROUND_COLOR,
                gridcolor=GRID_COLOR,
                showbackground=True,
                zerolinecolor=GRID_COLOR,
                color=TEXT_COLOR,
                autorange='reversed'  # Lower latency is better (top)
            ),
            bgcolor=BACKGROUND_COLOR,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(
            family="'Inter', sans-serif",
            size=12,
            color=TEXT_COLOR
        ),
        legend=dict(
            bgcolor='rgba(27, 38, 59, 0.8)',
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_COLOR)
        ),
        hovermode='closest',
        margin=dict(l=0, r=0, b=0, t=30),
    )

    # Add entrance animation
    if animate and len(models_df) > 0:
        # Create frames for animation
        n_frames = 30
        frames = []

        for i in range(n_frames + 1):
            frame_data = []
            progress = (i / n_frames) ** 2  # Ease-out quadratic

            for trace in fig.data:
                # Gradually reveal points by scaling from center
                frame_trace = go.Scatter3d(
                    x=trace.x * progress,
                    y=trace.y * progress + (1 - progress) * 50,  # Start from middle
                    z=trace.z * progress + (1 - progress) *
                      (models_df['latency'].mean() if len(models_df) > 0 else 100),
                    mode=trace.mode,
                    marker=dict(
                        **trace.marker,
                        opacity=trace.marker.opacity * progress
                    ),
                    text=trace.text,
                    hovertemplate=trace.hovertemplate,
                    showlegend=trace.showlegend,
                    name=trace.name
                )
                frame_data.append(frame_trace)

            frames.append(go.Frame(data=frame_data, name=str(i)))

        fig.frames = frames

        # Add animation controls
        layout_kwargs['updatemenus'] = [dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(label='Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=50, redraw=True),
                                     fromcurrent=True,
                                     mode='immediate',
                                     transition=dict(duration=0))])
            ],
            x=0.1,
            xanchor='right',
            y=1.0,
            yanchor='top'
        )]

    fig.update_layout(**layout_kwargs)

    return fig


def _compute_pareto_surface(models_df: pd.DataFrame) -> Optional[go.Surface]:
    """
    Compute Pareto frontier surface for model comparison.

    The Pareto frontier represents models that are not dominated by any other
    model in the optimization space (max accuracy, min latency, min size).

    Args:
        models_df: DataFrame with model data

    Returns:
        Plotly Surface trace or None if insufficient data
    """
    if len(models_df) < 4:
        return None

    try:
        # Normalize metrics to [0, 1] for Pareto computation
        params_norm = (models_df['params'] - models_df['params'].min()) / \
                     (models_df['params'].max() - models_df['params'].min() + 1e-8)
        accuracy_norm = (models_df['accuracy'] - models_df['accuracy'].min()) / \
                       (models_df['accuracy'].max() - models_df['accuracy'].min() + 1e-8)
        latency_norm = (models_df['latency'] - models_df['latency'].min()) / \
                      (models_df['latency'].max() - models_df['latency'].min() + 1e-8)

        # Find Pareto-optimal points
        # A point is Pareto-optimal if no other point dominates it
        # (better in at least one objective, not worse in others)
        is_pareto = np.ones(len(models_df), dtype=bool)

        for i in range(len(models_df)):
            for j in range(len(models_df)):
                if i != j:
                    # Check if j dominates i
                    # Better: higher accuracy, lower latency, lower params
                    if (accuracy_norm.iloc[j] >= accuracy_norm.iloc[i] and
                        latency_norm.iloc[j] <= latency_norm.iloc[i] and
                        params_norm.iloc[j] <= params_norm.iloc[i] and
                        (accuracy_norm.iloc[j] > accuracy_norm.iloc[i] or
                         latency_norm.iloc[j] < latency_norm.iloc[i] or
                         params_norm.iloc[j] < params_norm.iloc[i])):
                        is_pareto[i] = False
                        break

        # Get Pareto points
        pareto_points = models_df[is_pareto]

        if len(pareto_points) < 3:
            return None

        # Create a mesh surface through Pareto points
        # Use convex hull or interpolation
        x = pareto_points['params'].values / 1_000_000
        y = pareto_points['accuracy'].values
        z = pareto_points['latency'].values

        # Sort by x-axis for surface creation
        sorted_idx = np.argsort(x)
        x = x[sorted_idx]
        y = y[sorted_idx]
        z = z[sorted_idx]

        # Create grid for surface
        xi = np.linspace(x.min(), x.max(), 20)
        yi = np.linspace(y.min(), y.max(), 20)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate z values
        from scipy.interpolate import griddata
        zi_grid = griddata((x, y), z, (xi_grid, yi_grid), method='linear')

        # Create surface trace
        surface = go.Surface(
            x=xi_grid,
            y=yi_grid,
            z=zi_grid,
            colorscale=[[0, 'rgba(0, 245, 212, 0.1)'],
                       [1, 'rgba(0, 245, 212, 0.3)']],
            showscale=False,
            name='Pareto Frontier',
            hoverinfo='skip',
            opacity=0.3
        )

        return surface

    except Exception as e:
        # Silently fail if Pareto computation fails
        print(f"Could not compute Pareto surface: {e}")
        return None


def render_model_browser_3d(
    models: List[Dict[str, Any]],
    key: str = "model_browser_3d"
) -> None:
    """
    Streamlit component for 3D model browser.

    Args:
        models: List of model dictionaries with required fields:
            - model_id or id
            - name
            - phase
            - params
            - accuracy
            - latency (ms)
            - status
            - compression (optional)
        key: Unique key for Streamlit component
    """
    st.markdown("### 3D Model Space")

    # Convert to DataFrame
    df = pd.DataFrame(models)

    # Normalize column names
    if 'model_id' in df.columns:
        df['id'] = df['model_id']

    # Ensure required columns exist
    required_cols = ['id', 'name', 'phase', 'params', 'accuracy', 'latency', 'status']
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return

    # Add default compression if not present
    if 'compression' not in df.columns:
        df['compression'] = 1.0

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        show_phases = st.multiselect(
            "Filter Phases",
            options=[f'phase{i}' for i in range(1, 9)],
            default=[f'phase{i}' for i in range(1, 9)],
            key=f"{key}_phases"
        )

    with col2:
        highlight_champions = st.checkbox(
            "Highlight Champions",
            value=True,
            key=f"{key}_champions"
        )

    with col3:
        show_pareto = st.checkbox(
            "Show Pareto Frontier",
            value=False,
            key=f"{key}_pareto"
        )

    # Find champion models (best accuracy per phase)
    highlighted_ids = None
    if highlight_champions:
        champions = df.loc[df.groupby('phase')['accuracy'].idxmax()]
        highlighted_ids = champions['id'].tolist()

    # Create 3D plot
    fig = create_model_comparison_3d(
        models_df=df,
        highlighted_ids=highlighted_ids,
        show_phases=show_phases if show_phases else None,
        show_pareto=show_pareto,
        animate=True
    )

    # Render
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_chart")

    # Summary stats
    st.markdown("---")
    st.markdown("#### Model Space Statistics")

    col1, col2, col3, col4 = st.columns(4)

    filtered_df = df[df['phase'].isin(show_phases)] if show_phases else df

    with col1:
        st.metric("Total Models", len(filtered_df))

    with col2:
        st.metric("Avg Accuracy", f"{filtered_df['accuracy'].mean():.1f}%")

    with col3:
        st.metric("Avg Latency", f"{filtered_df['latency'].mean():.1f} ms")

    with col4:
        st.metric("Avg Size", f"{filtered_df['params'].mean() / 1e6:.1f}M")

    # Phase breakdown
    st.markdown("#### Phase Breakdown")

    phase_stats = filtered_df.groupby('phase').agg({
        'accuracy': 'mean',
        'latency': 'mean',
        'params': 'count'
    }).round(2)
    phase_stats.columns = ['Avg Accuracy (%)', 'Avg Latency (ms)', 'Model Count']

    st.dataframe(
        phase_stats,
        use_container_width=True
    )


def get_sample_data() -> List[Dict[str, Any]]:
    """Generate sample data for testing 3D visualization."""
    np.random.seed(42)

    models = []

    for phase_num in range(1, 9):
        phase = f'phase{phase_num}'
        n_models = np.random.randint(3, 8)

        # Base metrics that improve with phases
        base_accuracy = 40 + phase_num * 5
        base_latency = 150 - phase_num * 10
        base_compression = 1.0 + phase_num * 0.5

        for i in range(n_models):
            # Add variation
            accuracy = base_accuracy + np.random.randn() * 3
            latency = max(10, base_latency + np.random.randn() * 15)
            compression = max(1.0, base_compression + np.random.randn() * 0.3)

            # Random status
            status = np.random.choice(
                ['complete', 'running', 'failed', 'pending'],
                p=[0.7, 0.15, 0.1, 0.05]
            )

            models.append({
                'id': f'{phase}_model_{i}',
                'name': f'{phase.upper()} Model {i+1}',
                'phase': phase,
                'params': 25_000_000,
                'accuracy': accuracy,
                'latency': latency,
                'compression': compression,
                'status': status
            })

    return models


# Demo/Testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="3D Model Browser",
        page_icon=":rocket:",
        layout="wide"
    )

    st.title("3D Model Comparison Visualization")

    # Generate sample data
    sample_models = get_sample_data()

    # Render component
    render_model_browser_3d(sample_models)

    # Add documentation
    with st.expander("About this visualization"):
        st.markdown("""
        ### 3D Model Space Visualization

        This interactive 3D scatter plot shows all models across the 8-phase pipeline
        in a unified comparison space.

        **Axes:**
        - **X-axis**: Model size (parameters in millions)
        - **Y-axis**: Accuracy/performance metric (0-100%)
        - **Z-axis**: Inference speed in milliseconds (lower is better, shown at top)

        **Visual Encoding:**
        - **Point size**: Proportional to compression ratio
        - **Point color**: Phase identifier (8 distinct colors)
        - **Point shape**: Model status
            - Circle: Complete
            - Diamond: Running
            - X: Failed
            - Square: Pending
        - **Border highlight**: Champion models (best accuracy per phase)

        **Features:**
        - **Orbit controls**: Click and drag to rotate
        - **Phase filtering**: Show/hide specific phases
        - **Champion highlighting**: Emphasize best models
        - **Pareto frontier**: Optional surface showing optimal trade-offs
        - **Hover details**: Full model information on mouseover

        **Interpretation:**
        Models in the upper-left-front region represent the best trade-offs:
        high accuracy, low latency, and small model size.
        """)

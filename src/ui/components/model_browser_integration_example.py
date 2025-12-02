"""
Example: Integrating 3D Model Comparison into Model Browser Page

This shows how to add the 3D visualization to the existing model_browser.py page.
"""

import sys
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cross_phase.storage.model_registry import ModelRegistry
from ui.components.model_comparison_3d import get_sample_data, render_model_browser_3d


def render_model_browser_with_3d():
    """
    Enhanced model browser with 3D visualization.

    This function shows how to integrate the 3D view into the existing
    model browser page.
    """
    st.markdown('<h1 class="main-header">Model Browser</h1>', unsafe_allow_html=True)

    # Initialize registry
    registry = ModelRegistry()

    # === FILTERS SECTION ===
    st.markdown("### Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        phase_filter = st.multiselect(
            "Filter by Phase",
            ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "phase8"],
            default=[],
        )

    with col2:
        size_filter = st.selectbox(
            "Model Size",
            ["All", "Tiny (<50M)", "Small (50-500M)", "Medium (500M-2B)", "Large (>2B)"],
        )

    with col3:
        search = st.text_input("Search by name or ID", "")

    st.markdown("---")

    # === VIEW SELECTION ===
    view_mode = st.radio(
        "View Mode",
        ["3D Visualization", "List View", "Both"],
        horizontal=True,
        index=2,  # Default to both
    )

    st.markdown("---")

    # === GET MODEL DATA ===
    try:
        models = registry.get_all_models(
            phase_filter=phase_filter if phase_filter else None, limit=100
        )

        # Apply size filter
        if size_filter != "All":
            if size_filter == "Tiny (<50M)":
                models = [m for m in models if m["params"] < 50_000_000]
            elif size_filter == "Small (50-500M)":
                models = [m for m in models if 50_000_000 <= m["params"] < 500_000_000]
            elif size_filter == "Medium (500M-2B)":
                models = [m for m in models if 500_000_000 <= m["params"] < 2_000_000_000]
            elif size_filter == "Large (>2B)":
                models = [m for m in models if m["params"] >= 2_000_000_000]

        # Apply search filter
        if search:
            models = [
                m
                for m in models
                if search.lower() in m["name"].lower()
                or search.lower() in m.get("model_id", "").lower()
            ]

    except Exception as e:
        st.warning(f"Could not load models from registry: {e}")
        st.info("Using sample data for demonstration")
        models = get_sample_data()

        # Apply filters to sample data
        if phase_filter:
            models = [m for m in models if m["phase"] in phase_filter]
        if search:
            models = [
                m
                for m in models
                if search.lower() in m["name"].lower() or search.lower() in m.get("id", "").lower()
            ]

    # === 3D VISUALIZATION ===
    if view_mode in ["3D Visualization", "Both"]:
        # Ensure models have required fields for 3D view
        models_for_3d = []
        for m in models:
            # Add latency if missing (estimate based on size and phase)
            if "latency" not in m:
                # Simple estimation: larger models = slower
                base_latency = 100
                size_factor = m["params"] / 25_000_000  # Relative to 25M baseline
                phase_num = int(m.get("phase", "phase1").replace("phase", ""))
                phase_factor = max(0.5, 1.5 - phase_num * 0.1)  # Later phases optimize
                m["latency"] = base_latency * size_factor * phase_factor

            # Add compression if missing
            if "compression" not in m:
                phase_num = int(m.get("phase", "phase1").replace("phase", ""))
                m["compression"] = 1.0 + (phase_num - 1) * 0.3

            # Add status if missing
            if "status" not in m:
                m["status"] = "complete"

            models_for_3d.append(m)

        # Render 3D component
        render_model_browser_3d(models_for_3d, key="main_browser_3d")

    # === LIST VIEW ===
    if view_mode in ["List View", "Both"]:
        st.markdown("---")
        st.markdown(f"### Model List ({len(models)} models)")

        for model in models:
            with st.expander(f"📦 {model['name']} ({model.get('phase', 'unknown')})"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Model Details**")
                    model_id = model.get("model_id", model.get("id", "unknown"))
                    st.text(f"ID: {model_id}")
                    st.text(f"Phase: {model.get('phase', 'unknown')}")
                    st.text(f"Parameters: {model.get('params', 0):,}")
                    st.text(f"Size: {model.get('size_mb', 0):.1f} MB")

                with col2:
                    st.markdown("**Performance**")
                    st.text(f"Loss: {model.get('loss', 0):.3f}")
                    st.text(f"Accuracy: {model.get('accuracy', 0):.1f}%")
                    st.text(f"Perplexity: {model.get('perplexity', 0):.2f}")
                    if "latency" in model:
                        st.text(f"Latency: {model['latency']:.1f} ms")

                with col3:
                    st.markdown("**Metadata**")
                    st.text(f"Created: {model.get('created', 'unknown')}")
                    st.text(f"Session: {model.get('session_id', 'unknown')}")
                    st.text(f"Status: {model.get('status', 'unknown')}")
                    if "compression" in model:
                        st.text(f"Compression: {model['compression']:.2f}x")

                # Action buttons
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.button(f"Load", key=f"load_{model_id}"):
                        st.success(f"Loaded {model['name']}")

                with col2:
                    if st.button(f"Export", key=f"export_{model_id}"):
                        st.info("Export functionality coming soon")

                with col3:
                    if st.button(f"Compare", key=f"compare_{model_id}"):
                        st.info("Comparison view coming soon")

                with col4:
                    if st.button(f"Delete", key=f"delete_{model_id}"):
                        if registry.delete_model(model_id):
                            st.success(f"Deleted {model['name']}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {model['name']}")

    registry.close()


# === STANDALONE DEMO ===
if __name__ == "__main__":
    st.set_page_config(page_title="Model Browser with 3D View", page_icon=":rocket:", layout="wide")

    # Apply custom styling
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00F5D4;
        margin-bottom: 1rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    render_model_browser_with_3d()

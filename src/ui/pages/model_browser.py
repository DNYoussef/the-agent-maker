"""
Model Browser Page
Browse all models in the registry with filtering and search
"""
import sys
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cross_phase.storage.model_registry import ModelRegistry


def render() -> None:
    """Render model browser page"""
    st.markdown('<h1 class="main-header">Model Browser</h1>', unsafe_allow_html=True)

    # Initialize registry
    registry = ModelRegistry()

    # Filters
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

    # Get all models from registry with filters
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
                if search.lower() in m["name"].lower() or search.lower() in m["model_id"].lower()
            ]

    except Exception as e:
        st.warning(f"Could not load models from registry: {e}")
        st.info("Showing example data instead")
        models = _get_example_models()

        # Apply filters to example data
        if phase_filter:
            models = [m for m in models if m["phase"] in phase_filter]
        if search:
            models = [
                m
                for m in models
                if search.lower() in m["name"].lower() or search.lower() in m["model_id"].lower()
            ]

    # Display models
    st.subheader(f"Models Found: {len(models)}")

    for model in models:
        with st.expander(f"📦 {model['name']} ({model['phase']})"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Model Details**")
                st.text(f"ID: {model['model_id']}")
                st.text(f"Phase: {model['phase']}")
                st.text(f"Parameters: {model['params']:,}")
                st.text(f"Size: {model['size_mb']:.1f} MB")

            with col2:
                st.markdown("**Performance**")
                st.text(f"Loss: {model['loss']:.3f}")
                st.text(f"Accuracy: {model['accuracy']:.1f}%")
                st.text(f"Perplexity: {model['perplexity']:.2f}")

            with col3:
                st.markdown("**Metadata**")
                st.text(f"Created: {model['created']}")
                st.text(f"Session: {model['session_id']}")
                st.text(f"Status: {model['status']}")

            # Action buttons
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button(f"Load {model['model_id'][:8]}", key=f"load_{model['model_id']}"):
                    st.success(f"Loaded {model['name']}")

            with col2:
                if st.button(f"Export {model['model_id'][:8]}", key=f"export_{model['model_id']}"):
                    st.info("Export functionality coming soon")

            with col3:
                if st.button(
                    f"Compare {model['model_id'][:8]}", key=f"compare_{model['model_id']}"
                ):
                    st.info("Comparison view coming soon")

            with col4:
                if st.button(f"Delete {model['model_id'][:8]}", key=f"delete_{model['model_id']}"):
                    if registry.delete_model(model["model_id"]):
                        st.success(f"Deleted {model['name']}")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete {model['name']}")

    registry.close()


def _get_example_models() -> list:
    """Get example model data (placeholder)"""
    return [
        {
            "model_id": "phase1_model1_reasoning_session001",
            "name": "Model 1: Reasoning",
            "phase": "phase1",
            "params": 25_000_000,
            "size_mb": 95.4,
            "loss": 2.34,
            "accuracy": 45.2,
            "perplexity": 12.3,
            "created": "2025-10-16 10:30",
            "session_id": "session_001",
            "status": "complete",
        },
        {
            "model_id": "phase1_model2_memory_session001",
            "name": "Model 2: Memory",
            "phase": "phase1",
            "params": 25_000_000,
            "size_mb": 95.4,
            "loss": 2.12,
            "accuracy": 48.5,
            "perplexity": 11.2,
            "created": "2025-10-16 11:45",
            "session_id": "session_001",
            "status": "complete",
        },
        {
            "model_id": "phase2_champion_generation25_session001",
            "name": "Generation 25 Champion",
            "phase": "phase2",
            "params": 25_000_000,
            "size_mb": 95.4,
            "loss": 1.87,
            "accuracy": 56.8,
            "perplexity": 9.4,
            "created": "2025-10-16 15:20",
            "session_id": "session_001",
            "status": "complete",
        },
    ]

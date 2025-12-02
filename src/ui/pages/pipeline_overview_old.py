"""
Pipeline Overview Page
Shows current pipeline status, progress, and session information
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cross_phase.storage.model_registry import ModelRegistry
from cross_phase.orchestrator.pipeline import PipelineOrchestrator


def render():
    """Render pipeline overview page"""
    st.markdown('<h1 class="main-header">Pipeline Overview</h1>',
                unsafe_allow_html=True)

    # Initialize registry
    registry = ModelRegistry()

    # Session selection
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Session")
        sessions = registry.list_sessions() if hasattr(registry, 'list_sessions') else []

        if sessions:
            session_id = st.selectbox(
                "Select Session",
                sessions,
                format_func=lambda x: f"Session: {x}"
            )
        else:
            st.info("No active sessions. Create a new session to begin.")
            session_id = None

    with col2:
        st.subheader("Quick Actions")
        if st.button("Create New Session", type="primary"):
            new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            registry.create_session(new_session_id, {
                "created": datetime.now().isoformat(),
                "pipeline": "agent-forge-v2"
            })
            st.success(f"Created session: {new_session_id}")
            st.rerun()

    st.markdown("---")

    if session_id:
        # Get session info
        session_info = registry.get_session(session_id) if hasattr(registry, 'get_session') else None

        if session_info:
            # Session status
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Status",
                    session_info.get('status', 'unknown').upper()
                )

            with col2:
                st.metric(
                    "Current Phase",
                    session_info.get('current_phase', 'N/A')
                )

            with col3:
                progress = session_info.get('progress_percent', 0.0)
                st.metric(
                    "Progress",
                    f"{progress:.1f}%"
                )

            with col4:
                models_count = len(registry.get_session_models(session_id)) if hasattr(registry, 'get_session_models') else 0
                st.metric(
                    "Models Created",
                    models_count
                )

            # Progress bar
            st.progress(progress / 100.0)

            st.markdown("---")

            # Phase pipeline visualization
            st.subheader("8-Phase Pipeline Status")

            phases = [
                ("Phase 1", "Cognate (TRM × Titans-MAG)", "25M params"),
                ("Phase 2", "EvoMerge (50 generations)", "Evolutionary optimization"),
                ("Phase 3", "Quiet-STaR", "Reasoning enhancement"),
                ("Phase 4", "BitNet", "1.58-bit compression"),
                ("Phase 5", "Curriculum Learning", "7-stage adaptive"),
                ("Phase 6", "Tool & Persona Baking", "A/B optimization"),
                ("Phase 7", "Self-Guided Experts", "Model-driven discovery"),
                ("Phase 8", "Final Compression", "280× compression")
            ]

            current_phase = session_info.get('current_phase', 'phase1')

            for phase_name, description, detail in phases:
                phase_key = phase_name.lower().replace(" ", "")

                # Determine status
                if phase_key < current_phase:
                    status = "✅ Complete"
                    status_class = "status-success"
                elif phase_key == current_phase:
                    status = "⏳ Running"
                    status_class = "status-running"
                else:
                    status = "⏸️ Pending"
                    status_class = "status-pending"

                col1, col2, col3 = st.columns([2, 3, 2])

                with col1:
                    st.markdown(f"**{phase_name}**")

                with col2:
                    st.markdown(f"{description} • {detail}")

                with col3:
                    st.markdown(f'<span class="{status_class}">{status}</span>',
                               unsafe_allow_html=True)

            st.markdown("---")

            # Recent activity log
            st.subheader("Recent Activity")

            # Placeholder for activity log
            activity_log = [
                {"time": "10:45 AM", "event": "Phase 1 Model 3 training started"},
                {"time": "10:30 AM", "event": "Phase 1 Model 2 completed (loss: 2.34)"},
                {"time": "10:15 AM", "event": "Phase 1 Model 1 completed (loss: 2.56)"},
            ]

            for entry in activity_log:
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.text(entry["time"])
                with col2:
                    st.text(entry["event"])

        else:
            st.warning(f"Session {session_id} not found.")

    # Auto-refresh
    if st.sidebar.checkbox("Auto-refresh (5s)", value=False):
        import time
        time.sleep(5)
        st.rerun()

    registry.close()


# Auto-run when accessed directly via Streamlit multipage
render()

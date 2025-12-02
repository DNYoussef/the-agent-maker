"""
Pipeline Overview Page - Enhanced Visual Edition
Shows current pipeline status, progress, and session information with modern UI
"""
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cross_phase.orchestrator.pipeline import PipelineOrchestrator
from cross_phase.storage.model_registry import ModelRegistry


def inject_custom_css() -> None:
    """Inject custom CSS for enhanced visual components"""
    css = """
    <style>
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #0a0e27 0%%, #1a1f3a 50%%, #0a0e27 100%%);
        border: 2px solid rgba(0, 255, 255, 0.3);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%%;
        left: -50%%;
        width: 200%%;
        height: 200%%;
        background: radial-gradient(circle, rgba(0, 255, 255, 0.1) 0%%, transparent 70%%);
        animation: pulse-glow 4s ease-in-out infinite;
    }

    @keyframes pulse-glow {
        0%%, 100%% { transform: scale(1); opacity: 0.5; }
        50%% { transform: scale(1.1); opacity: 0.8; }
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ffff 0%%, #0099ff 100%%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 1;
    }

    .hero-status {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid rgba(0, 255, 255, 0.4);
        border-radius: 24px;
        font-weight: 600;
        color: #00ffff;
        font-size: 1rem;
        position: relative;
        z-index: 1;
    }

    /* Enhanced Metric Cards */
    .metric-card-enhanced {
        background: linear-gradient(135deg, #1a1f3a 0%%, #0f1429 100%%);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card-enhanced:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 255, 255, 0.2);
        border-color: rgba(0, 255, 255, 0.4);
    }

    .metric-card-enhanced::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00ffff 0%%, #0099ff 100%%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card-enhanced:hover::before {
        opacity: 1;
    }

    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }

    .metric-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00ffff;
        animation: value-fade-in 0.5s ease;
    }

    @keyframes value-fade-in {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Phase Timeline */
    .phase-timeline {
        background: rgba(10, 14, 39, 0.6);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
    }

    .phase-item {
        display: flex;
        align-items: center;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        position: relative;
        border-left: 4px solid transparent;
    }

    .phase-item.complete {
        background: linear-gradient(90deg, rgba(0, 255, 255, 0.15) 0%%, rgba(0, 255, 255, 0.05) 100%%);
        border-left-color: #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    }

    .phase-item.running {
        background: linear-gradient(90deg, rgba(0, 153, 255, 0.15) 0%%, rgba(0, 153, 255, 0.05) 100%%);
        border-left-color: #0099ff;
        animation: pulse-border 2s ease-in-out infinite;
    }

    @keyframes pulse-border {
        0%%, 100%% { box-shadow: 0 0 10px rgba(0, 153, 255, 0.3); }
        50%% { box-shadow: 0 0 20px rgba(0, 153, 255, 0.6); }
    }

    .phase-item.pending {
        background: rgba(255, 255, 255, 0.02);
        border-left-color: rgba(255, 255, 255, 0.1);
        opacity: 0.6;
    }

    .phase-number {
        width: 48px;
        height: 48px;
        border-radius: 50%%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.1rem;
        margin-right: 1.5rem;
        flex-shrink: 0;
    }

    .phase-item.complete .phase-number {
        background: linear-gradient(135deg, #00ffff 0%%, #00cccc 100%%);
        color: #0a0e27;
        box-shadow: 0 4px 12px rgba(0, 255, 255, 0.4);
    }

    .phase-item.running .phase-number {
        background: linear-gradient(135deg, #0099ff 0%%, #0066cc 100%%);
        color: white;
        animation: rotate-gradient 3s linear infinite;
    }

    @keyframes rotate-gradient {
        0%% { transform: rotate(0deg); }
        100%% { transform: rotate(360deg); }
    }

    .phase-item.pending .phase-number {
        background: rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.4);
    }

    .phase-content {
        flex: 1;
    }

    .phase-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .phase-item.complete .phase-title {
        color: #00ffff;
    }

    .phase-item.running .phase-title {
        color: #0099ff;
    }

    .phase-item.pending .phase-title {
        color: rgba(255, 255, 255, 0.5);
    }

    .phase-description {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
        margin-bottom: 0.5rem;
    }

    .phase-detail {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.4);
    }

    .phase-status-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .status-badge-complete {
        background: rgba(0, 255, 255, 0.2);
        color: #00ffff;
        border: 1px solid rgba(0, 255, 255, 0.4);
    }

    .status-badge-running {
        background: rgba(0, 153, 255, 0.2);
        color: #0099ff;
        border: 1px solid rgba(0, 153, 255, 0.4);
        animation: pulse-badge 1.5s ease-in-out infinite;
    }

    @keyframes pulse-badge {
        0%%, 100%% { opacity: 1; }
        50%% { opacity: 0.6; }
    }

    .status-badge-pending {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .phase-progress {
        width: 100%%;
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        margin-top: 0.75rem;
        overflow: hidden;
    }

    .phase-progress-bar {
        height: 100%%;
        background: linear-gradient(90deg, #00ffff 0%%, #0099ff 100%%);
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* Activity Log */
    .activity-log {
        background: rgba(10, 14, 39, 0.6);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
    }

    .activity-log-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 0.5rem;
    }

    .activity-log-container::-webkit-scrollbar {
        width: 8px;
    }

    .activity-log-container::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }

    .activity-log-container::-webkit-scrollbar-thumb {
        background: rgba(0, 255, 255, 0.3);
        border-radius: 4px;
    }

    .activity-log-container::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 255, 255, 0.5);
    }

    .log-entry {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid rgba(0, 255, 255, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }

    .log-entry:hover {
        background: rgba(255, 255, 255, 0.06);
        border-left-color: rgba(0, 255, 255, 0.6);
        transform: translateX(4px);
    }

    .log-entry.success {
        border-left-color: rgba(0, 255, 128, 0.5);
    }

    .log-entry.info {
        border-left-color: rgba(0, 153, 255, 0.5);
    }

    .log-timestamp {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.4);
        margin-bottom: 0.25rem;
    }

    .log-event {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.9);
    }

    .log-type-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 0.5rem;
        text-transform: uppercase;
    }

    .log-type-success {
        background: rgba(0, 255, 128, 0.2);
        color: #00ff80;
    }

    .log-type-info {
        background: rgba(0, 153, 255, 0.2);
        color: #0099ff;
    }

    /* Progress Bar Enhancement */
    .custom-progress-container {
        width: 100%%;
        height: 24px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        overflow: hidden;
        margin: 1.5rem 0;
        position: relative;
    }

    .custom-progress-bar {
        height: 100%%;
        background: linear-gradient(90deg, #00ffff 0%%, #0099ff 50%%, #00ffff 100%%);
        background-size: 200%% 100%%;
        animation: shimmer 2s linear infinite;
        border-radius: 12px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 1rem;
    }

    @keyframes shimmer {
        0%% { background-position: 200%% 0; }
        100%% { background-position: -200%% 0; }
    }

    .progress-text {
        font-size: 0.8rem;
        font-weight: 700;
        color: #0a0e27;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00ffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 255, 255, 0.2);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_hero_section(session_id, session_info) -> None:
    """Render enhanced hero section with gradient and status"""
    status = session_info.get("status", "unknown").upper() if session_info else "NO SESSION"
    current_phase = session_info.get("current_phase", "N/A") if session_info else "N/A"

    st.markdown(
        f"""
    <div class="hero-section">
        <div class="hero-title">Agent Forge V2 Pipeline</div>
        <div class="hero-subtitle">8-Phase AI Agent Creation System</div>
        <div class="hero-status">Status: {status} | Current: {current_phase}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_enhanced_metrics(session_info, models_count) -> None:
    """Render enhanced metric cards with icons and animations"""
    col1, col2, col3, col4 = st.columns(4)

    status = session_info.get("status", "unknown").upper()
    current_phase = session_info.get("current_phase", "N/A")
    progress = session_info.get("progress_percent", 0.0)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card-enhanced">
            <div class="metric-icon">⚡</div>
            <div class="metric-label">Pipeline Status</div>
            <div class="metric-value">{status}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card-enhanced">
            <div class="metric-icon">🔄</div>
            <div class="metric-label">Current Phase</div>
            <div class="metric-value">{current_phase}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card-enhanced">
            <div class="metric-icon">📊</div>
            <div class="metric-label">Progress</div>
            <div class="metric-value">{progress:.1f}%%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card-enhanced">
            <div class="metric-icon">🤖</div>
            <div class="metric-label">Models Created</div>
            <div class="metric-value">{models_count}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Enhanced progress bar
    st.markdown(
        f"""
    <div class="custom-progress-container">
        <div class="custom-progress-bar" style="width: {progress}%%">
            <span class="progress-text">{progress:.1f}%%</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_phase_timeline(current_phase) -> None:
    """Render visual 8-phase timeline with status indicators"""
    phases = [
        {
            "number": 1,
            "name": "Cognate",
            "description": "TRM x Titans-MAG",
            "detail": "25M params, 3 specialized models",
            "key": "phase1",
            "progress": 100,
        },
        {
            "number": 2,
            "name": "EvoMerge",
            "description": "50 generations evolutionary optimization",
            "detail": "6 merge techniques, binary pairing",
            "key": "phase2",
            "progress": 65,
        },
        {
            "number": 3,
            "name": "Quiet-STaR",
            "description": "Reasoning enhancement",
            "detail": "Token-wise thought generation",
            "key": "phase3",
            "progress": 0,
        },
        {
            "number": 4,
            "name": "BitNet",
            "description": "1.58-bit compression",
            "detail": "8.2x compression, 3.8x speedup",
            "key": "phase4",
            "progress": 0,
        },
        {
            "number": 5,
            "name": "Curriculum Learning",
            "description": "7-stage adaptive curriculum",
            "detail": "Edge-of-chaos, frontier models",
            "key": "phase5",
            "progress": 0,
        },
        {
            "number": 6,
            "name": "Tool & Persona Baking",
            "description": "A/B optimization loops",
            "detail": "SWE-Bench, self-guided personas",
            "key": "phase6",
            "progress": 0,
        },
        {
            "number": 7,
            "name": "Self-Guided Experts",
            "description": "Model-driven discovery",
            "detail": "Transformer(2) SVF, NSGA-II ADAS",
            "key": "phase7",
            "progress": 0,
        },
        {
            "number": 8,
            "name": "Final Compression",
            "description": "280x compression",
            "detail": "SeedLM -> VPTQ -> Hypercompression",
            "key": "phase8",
            "progress": 0,
        },
    ]

    st.markdown('<div class="section-header">8-Phase Pipeline Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="phase-timeline">', unsafe_allow_html=True)

    for phase in phases:
        # Determine status
        if cast(int, phase["key"]) < current_phase:
            status_class = "complete"
            badge_class = "status-badge-complete"
            badge_text = "Complete"
            badge_icon = "✓"
        elif cast(str, phase["key"]) == current_phase:
            status_class = "running"
            badge_class = "status-badge-running"
            badge_text = "Running"
            badge_icon = "⏳"
        else:
            status_class = "pending"
            badge_class = "status-badge-pending"
            badge_text = "Pending"
            badge_icon = "○"

        progress_html = (
            f'<div class="phase-progress"><div class="phase-progress-bar" style="width: {phase["progress"]}%%"></div></div>'
            if status_class == "running"
            else ""
        )

        st.markdown(
            f"""
        <div class="phase-item {status_class}">
            <div class="phase-number">{phase['number']}</div>
            <div class="phase-content">
                <div class="phase-title">Phase {phase['number']}: {phase['name']}</div>
                <div class="phase-description">{phase['description']}</div>
                <div class="phase-detail">{phase['detail']}</div>
                {progress_html}
            </div>
            <div class="phase-status-badge {badge_class}">
                {badge_icon} {badge_text}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_activity_log() -> None:
    """Render enhanced activity log with card-based entries"""
    st.markdown('<div class="section-header">Recent Activity</div>', unsafe_allow_html=True)

    activity_log = [
        {
            "time": "2 minutes ago",
            "event": "Phase 2 Model 5 evolution completed successfully",
            "type": "success",
            "absolute_time": "10:45 AM",
        },
        {
            "time": "5 minutes ago",
            "event": "Phase 2 Generation 15/50 started",
            "type": "info",
            "absolute_time": "10:42 AM",
        },
        {
            "time": "8 minutes ago",
            "event": "Champion model fitness: 0.876 (+23.5%% from baseline)",
            "type": "success",
            "absolute_time": "10:39 AM",
        },
        {
            "time": "12 minutes ago",
            "event": "Phase 1 Model 3 handoff validation complete",
            "type": "success",
            "absolute_time": "10:35 AM",
        },
        {
            "time": "15 minutes ago",
            "event": "SLERP merge technique applied (interpolation: 0.6)",
            "type": "info",
            "absolute_time": "10:32 AM",
        },
        {
            "time": "18 minutes ago",
            "event": "Binary pairing strategy: Group A vs Group B",
            "type": "info",
            "absolute_time": "10:29 AM",
        },
        {
            "time": "22 minutes ago",
            "event": "MuGrokfast optimizer checkpoint saved",
            "type": "success",
            "absolute_time": "10:25 AM",
        },
        {
            "time": "25 minutes ago",
            "event": "W&B logging: 370 metrics tracked for Phase 2",
            "type": "info",
            "absolute_time": "10:22 AM",
        },
    ]

    st.markdown(
        '<div class="activity-log"><div class="activity-log-container">', unsafe_allow_html=True
    )

    for entry in activity_log:
        type_class = entry["type"]
        type_badge_class = f"log-type-{type_class}"
        type_label = entry["type"].upper()

        st.markdown(
            f"""
        <div class="log-entry {type_class}">
            <div class="log-timestamp">{entry['time']} ({entry['absolute_time']})</div>
            <div class="log-event">
                {entry['event']}
                <span class="log-type-badge {type_badge_class}">{type_label}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div></div>", unsafe_allow_html=True)


def render() -> None:
    """Render enhanced pipeline overview page"""
    inject_custom_css()
    registry = ModelRegistry()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Session Management</div>', unsafe_allow_html=True)
        sessions = registry.list_sessions() if hasattr(registry, "list_sessions") else []

        if sessions:
            session_id = st.selectbox(
                "Select Active Session",
                sessions,
                format_func=lambda x: f"Session: {x}",
                key="session_selector",
            )
        else:
            st.info("No active sessions. Create a new session to begin.")
            session_id = None

    with col2:
        st.markdown('<div class="section-header">Quick Actions</div>', unsafe_allow_html=True)
        if st.button("🚀 Create New Session", type="primary", use_container_width=True):
            new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            registry.create_session(
                new_session_id,
                {
                    "created": datetime.now().isoformat(),
                    "pipeline": "agent-forge-v2",
                    "status": "active",
                    "current_phase": "phase1",
                    "progress_percent": 0.0,
                },
            )
            st.success(f"Created session: {new_session_id}")
            st.rerun()

        if session_id:
            if st.button("📊 View Metrics", use_container_width=True):
                st.info("Opening W&B dashboard...")

            if st.button("⏸️ Pause Session", use_container_width=True):
                st.warning("Session paused")

    st.markdown("---")

    if session_id:
        session_info = (
            registry.get_session(session_id) if hasattr(registry, "get_session") else None
        )

        if session_info:
            render_hero_section(session_id, session_info)

            models_count = (
                len(registry.get_session_models(session_id))
                if hasattr(registry, "get_session_models")
                else 0
            )
            render_enhanced_metrics(session_info, models_count)

            st.markdown("---")

            current_phase = session_info.get("current_phase", "phase1")
            render_phase_timeline(current_phase)

            st.markdown("---")

            render_activity_log()
        else:
            st.warning(f"Session {session_id} not found.")
    else:
        st.markdown(
            """
        <div class="hero-section">
            <div class="hero-title">Welcome to Agent Forge V2</div>
            <div class="hero-subtitle">Create your first session to begin the 8-phase pipeline</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        ### Getting Started

        1. Click **Create New Session** to begin
        2. Monitor progress through the 8-phase pipeline
        3. Track metrics and model performance
        4. Export final compressed agent

        ### Pipeline Overview

        **Phase 1-2**: Foundation models (Cognate + EvoMerge)
        **Phase 3-4**: Enhancement (Quiet-STaR + BitNet)
        **Phase 5-6**: Training (Curriculum + Baking)
        **Phase 7-8**: Optimization (Experts + Compression)
        """
        )

    if st.sidebar.checkbox("Auto-refresh (5s)", value=False):
        import time

        time.sleep(5)
        st.rerun()

    registry.close()


# Auto-run when accessed directly via Streamlit multipage
render()

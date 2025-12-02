"""
Agent Forge V2 - Streamlit Dashboard (WITH DESIGN SYSTEM)
Updated version of app.py using the new design system

This is an example of how to integrate the design system into the existing app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import design system
from design_system import get_custom_css

# Page configuration
st.set_page_config(
    page_title="Agent Forge V2", page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)

# ===== APPLY DESIGN SYSTEM =====
# Replace the old custom CSS with the new design system
st.markdown(get_custom_css(theme="dark"), unsafe_allow_html=True)

# ===== SIDEBAR NAVIGATION =====
st.sidebar.markdown('<h1 class="gradient-text">Agent Forge V2</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Pipeline Overview",
        "Phase Details",
        "Phase 4: BitNet Compression",
        "Model Browser",
        "System Monitor",
        "Configuration Editor",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    """
    <div class="glass-card" style="padding: 12px; font-size: 0.875rem;">
    <strong>Agent Forge V2</strong><br/>
    8-phase AI agent creation pipeline.<br/>
    Local-first architecture with 25M parameter models.
    </div>
    """,
    unsafe_allow_html=True,
)

# ===== MAIN HEADER =====
st.markdown('<h1 class="main-header">Agent Forge V2</h1>', unsafe_allow_html=True)

# ===== LOAD SELECTED PAGE =====
if page == "Pipeline Overview":
    from pages import pipeline_overview

    pipeline_overview.render()
elif page == "Phase Details":
    from pages import phase_details

    phase_details.render()
elif page == "Phase 4: BitNet Compression":
    from pages import phase4_bitnet

    phase4_bitnet.render_phase4_dashboard()
elif page == "Model Browser":
    from pages import model_browser

    model_browser.render()
elif page == "System Monitor":
    from pages import system_monitor

    system_monitor.render()
elif page == "Configuration Editor":
    from pages import config_editor

    config_editor.render()

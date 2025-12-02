"""
Agent Forge V2 - Streamlit Dashboard
Main entry point for the web UI
Enhanced with futuristic command center aesthetics
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enhanced page configuration
st.set_page_config(
    page_title="Agent Forge V2 - Command Center",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Agent Forge V2: 8-phase AI agent creation pipeline"
    }
)

# Comprehensive Custom CSS - Futuristic Command Center Theme
st.markdown("""
<style>
    /* CSS Custom Properties - Dark Theme */
    :root {
        --primary-bg: #0D1B2A;
        --secondary-bg: #1B263B;
        --accent-cyan: #00F5D4;
        --accent-blue: #00D9FF;
        --accent-purple: #9D4EDD;
        --text-primary: #E0E1DD;
        --text-secondary: #778DA9;
        --card-bg: rgba(27, 38, 59, 0.6);
        --card-border: rgba(0, 245, 212, 0.2);
        --glow-cyan: rgba(0, 245, 212, 0.4);
        --glow-blue: rgba(0, 217, 255, 0.3);
        --success: #06FFA5;
        --warning: #FFD60A;
        --error: #FF006E;
    }

    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', 'Space Grotesk', sans-serif;
    }

    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, var(--primary-bg) 0%, #1a1a2e 100%);
        color: var(--text-primary);
    }

    /* Page Entry Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .main .block-container {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Header Styling */
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-blue) 50%, var(--accent-purple) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 30px var(--glow-cyan);
        letter-spacing: -0.02em;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-primary);
    }

    /* Glassmorphism Card Styling */
    .metric-card, .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-card:hover, .glass-card:hover {
        transform: translateY(-5px);
        border-color: var(--accent-cyan);
        box-shadow: 0 12px 48px 0 var(--glow-cyan);
    }

    /* Enhanced Metric Cards */
    div[data-testid="metric-container"] {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    div[data-testid="metric-container"]:hover {
        border-color: var(--accent-cyan);
        box-shadow: 0 0 20px var(--glow-cyan);
        transform: scale(1.02);
    }

    div[data-testid="metric-container"] label {
        color: var(--text-secondary) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: var(--accent-cyan) !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 0 0 10px var(--glow-cyan);
    }

    /* Animated Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue), var(--accent-purple));
        background-size: 200% 100%;
        animation: progressGlow 2s ease infinite;
        border-radius: 10px;
        box-shadow: 0 0 20px var(--glow-cyan);
    }

    @keyframes progressGlow {
        0%, 100% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
    }

    /* Glowing Status Badges */
    .status-success {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        background: rgba(6, 255, 165, 0.1);
        border: 1px solid var(--success);
        color: var(--success);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(6, 255, 165, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }

    .status-running {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        background: rgba(255, 214, 10, 0.1);
        border: 1px solid var(--warning);
        color: var(--warning);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(255, 214, 10, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }

    .status-failed {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        background: rgba(255, 0, 110, 0.1);
        border: 1px solid var(--error);
        color: var(--error);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(255, 0, 110, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }

    .status-pending {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        background: rgba(119, 141, 169, 0.1);
        border: 1px solid var(--text-secondary);
        color: var(--text-secondary);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--secondary-bg) 0%, var(--primary-bg) 100%);
        border-right: 1px solid var(--card-border);
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    /* Sidebar Title */
    section[data-testid="stSidebar"] h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent-cyan);
        text-shadow: 0 0 20px var(--glow-cyan);
        margin-bottom: 1rem;
    }

    /* Sidebar Radio Buttons */
    section[data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        color: var(--accent-cyan) !important;
        text-shadow: 0 0 10px var(--glow-cyan);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--primary-bg);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-cyan), var(--accent-blue));
        border-radius: 10px;
        border: 2px solid var(--primary-bg);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-blue), var(--accent-purple));
        box-shadow: 0 0 10px var(--glow-cyan);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
        color: var(--primary-bg);
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px var(--glow-cyan);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--glow-blue);
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div > select {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 8px;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent-cyan);
        box-shadow: 0 0 15px var(--glow-cyan);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--accent-cyan);
        border-color: var(--accent-cyan);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
        color: var(--primary-bg) !important;
        border: none;
        box-shadow: 0 4px 15px var(--glow-cyan);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 8px;
        color: var(--text-primary);
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .streamlit-expanderHeader:hover {
        border-color: var(--accent-cyan);
        box-shadow: 0 0 15px var(--glow-cyan);
    }

    /* Info/Warning/Error Boxes */
    .stAlert {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        color: var(--text-primary);
    }

    /* Data Tables */
    .stDataFrame {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 10px;
    }

    /* Code Blocks */
    .stCodeBlock {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
        margin: 2rem 0;
    }

    /* Phase Cards */
    .phase-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .phase-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue), var(--accent-purple));
        background-size: 200% 100%;
        animation: shimmer 3s ease infinite;
    }

    @keyframes shimmer {
        0%, 100% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
    }

    .phase-card:hover {
        transform: translateY(-8px);
        border-color: var(--accent-cyan);
        box-shadow: 0 15px 60px var(--glow-cyan);
    }

    /* Typography Enhancements */
    p, li, span {
        color: var(--text-primary);
    }

    a {
        color: var(--accent-cyan);
        text-decoration: none;
        transition: all 0.3s ease;
    }

    a:hover {
        color: var(--accent-blue);
        text-shadow: 0 0 10px var(--glow-cyan);
    }

    /* Loading Spinner */
    .stSpinner > div {
        border-color: var(--accent-cyan) transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Sidebar navigation with enhanced styling
st.sidebar.markdown('<h1 style="text-align: center;">🧬 Agent Forge V2</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Pipeline Overview",
        "Phase 1: Cognate",
        "Phase 2: EvoMerge",
        "Phase 3: Quiet-STaR",
        "Phase 4: BitNet",
        "Phase 5: Curriculum",
        "Phase 6: Baking",
        "Phase 7: Experts",
        "Phase 8: Compression",
        "W&B Monitor",
        "Model Browser",
        "System Monitor",
        "Configuration"
    ],
    key="navigation"
)

st.sidebar.markdown("---")

# Enhanced About Section
st.sidebar.markdown("### ABOUT")
st.sidebar.markdown(
    """
    <div style='
        background: rgba(27, 38, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 245, 212, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    '>
        <p style='margin: 0; font-size: 0.9rem; line-height: 1.6;'>
            <strong style='color: #00F5D4;'>Agent Forge V2</strong><br/>
            8-phase AI agent creation pipeline<br/>
            <span style='color: #778DA9;'>Local-first architecture</span><br/>
            <span style='color: #778DA9;'>25M parameter models</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# System Stats in Sidebar
st.sidebar.markdown("### SYSTEM STATUS")
st.sidebar.markdown(
    """
    <div style='
        background: rgba(27, 38, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 245, 212, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    '>
        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
            <span style='color: #778DA9;'>GPU</span>
            <span class='status-success'>ONLINE</span>
        </div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
            <span style='color: #778DA9;'>W&B</span>
            <span class='status-success'>CONNECTED</span>
        </div>
        <div style='display: flex; justify-content: space-between;'>
            <span style='color: #778DA9;'>Pipeline</span>
            <span class='status-running'>ACTIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align: center; color: #778DA9; font-size: 0.75rem; font-family: \"JetBrains Mono\", monospace;'>v2.0.0 | Command Center</p>",
    unsafe_allow_html=True
)

# Load selected page
if page == "Pipeline Overview":
    from pages import pipeline_overview
    pipeline_overview.render()
elif page == "Phase 1: Cognate":
    from pages import phase1_cognate
    phase1_cognate.render_phase1_cognate()
elif page == "Phase 2: EvoMerge":
    from pages import phase2_evomerge
    phase2_evomerge.render_phase2_dashboard()
elif page == "Phase 3: Quiet-STaR":
    from pages import phase3_quietstar
    phase3_quietstar.render_phase3_dashboard()
elif page == "Phase 4: BitNet":
    from pages import phase4_bitnet
    phase4_bitnet.render_phase4_dashboard()
elif page == "Phase 5: Curriculum":
    from pages import phase5_curriculum
    phase5_curriculum.render_phase5_dashboard()
elif page == "Phase 6: Baking":
    from pages import phase6_baking
    phase6_baking.render_phase6_dashboard()
elif page == "Phase 7: Experts":
    from pages import phase7_experts
    phase7_experts.render_phase7_dashboard()
elif page == "Phase 8: Compression":
    from pages import phase8_compression
    phase8_compression.render()
elif page == "W&B Monitor":
    from pages import wandb_monitor
    wandb_monitor.render()
elif page == "Model Browser":
    from pages import model_browser
    model_browser.render()
elif page == "System Monitor":
    from pages import system_monitor
    system_monitor.render()
elif page == "Configuration":
    from pages import config_editor
    config_editor.render()

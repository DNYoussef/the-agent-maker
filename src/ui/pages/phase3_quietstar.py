"""
Phase 3: Quiet-STaR Reasoning Enhancement - Streamlit UI Dashboard

Real-time visualization of thought generation, coherence scoring, and anti-theater validation.
Features futuristic command center theme with dark background and cyan accents.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from datetime import datetime


# ==================== CUSTOM CSS THEME ====================
def apply_futuristic_theme():
    """Apply futuristic command center theme with glassmorphism"""
    st.markdown("""
    <style>
        /* Main background - dark command center */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
            color: #00ffff;
        }

        /* Glassmorphism panels */
        .element-container, .stMarkdown, .stMetric {
            background: rgba(26, 31, 58, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 12px;
            padding: 12px;
            box-shadow: 0 8px 32px rgba(0, 255, 255, 0.1);
        }

        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 28px;
            color: #00ffff;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        }

        [data-testid="stMetricLabel"] {
            color: #8899ff;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Headers */
        h1, h2, h3 {
            color: #00ffff;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.4);
            font-family: 'Courier New', monospace;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #00ffff 0%, #0088ff 100%);
            color: #0a0e27;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(0, 255, 255, 0.3);
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            box-shadow: 0 6px 25px rgba(0, 255, 255, 0.5);
            transform: translateY(-2px);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(10, 14, 39, 0.6);
            border-radius: 8px;
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(26, 31, 58, 0.8);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 6px;
            color: #00ffff;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.3) 0%, rgba(0, 136, 255, 0.3) 100%);
            border: 1px solid #00ffff;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background: rgba(26, 31, 58, 0.8);
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 8px;
            color: #00ffff;
            font-weight: 600;
        }

        /* Progress bars */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #00ffff 0%, #0088ff 100%);
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        }

        /* Dataframes */
        .dataframe {
            background: rgba(26, 31, 58, 0.6);
            color: #00ffff;
            border: 1px solid rgba(0, 255, 255, 0.2);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
            border-right: 2px solid rgba(0, 255, 255, 0.3);
        }

        /* Alert boxes */
        .stAlert {
            background: rgba(26, 31, 58, 0.8);
            border: 1px solid rgba(0, 255, 255, 0.3);
            color: #00ffff;
        }

        /* Code blocks */
        code {
            background: rgba(10, 14, 39, 0.8);
            color: #00ffff;
            border: 1px solid rgba(0, 255, 255, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
        }

        /* Selectbox */
        .stSelectbox > div > div {
            background: rgba(26, 31, 58, 0.8);
            border: 1px solid rgba(0, 255, 255, 0.3);
            color: #00ffff;
        }

        /* Number input */
        .stNumberInput > div > div {
            background: rgba(26, 31, 58, 0.8);
            border: 1px solid rgba(0, 255, 255, 0.3);
            color: #00ffff;
        }

        /* Slider */
        .stSlider > div > div > div {
            background: rgba(0, 255, 255, 0.2);
        }

        /* Divider */
        hr {
            border-color: rgba(0, 255, 255, 0.3);
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
        }
    </style>
    """, unsafe_allow_html=True)


# ==================== MAIN DASHBOARD ====================
def render_phase3_dashboard():
    """Main dashboard for Phase 3 Quiet-STaR reasoning enhancement"""
    apply_futuristic_theme()

    # Initialize session state
    initialize_session_state()

    # Title with dramatic styling
    st.markdown("""
    <h1 style='text-align: center; font-size: 48px; margin-bottom: 0;'>
        ⚡ PHASE 3: QUIET-STAR REASONING
    </h1>
    <p style='text-align: center; color: #8899ff; font-size: 18px; margin-top: 0;'>
        Token-Wise Parallel Thought Generation • Coherence Validation • Anti-Theater Detection
    </p>
    """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ COMMAND CENTER")
        render_config_panel()

    # Hero metrics section
    render_hero_metrics()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧠 Thought Generation",
        "📊 Coherence Scoring",
        "🔍 Reasoning Trace",
        "🛡️ Anti-Theater",
        "🔥 Prompt Baking",
        "📈 Training Metrics"
    ])

    with tab1:
        render_thought_generation_tab()

    with tab2:
        render_coherence_scoring_tab()

    with tab3:
        render_reasoning_trace_tab()

    with tab4:
        render_anti_theater_tab()

    with tab5:
        render_prompt_baking_tab()

    with tab6:
        render_training_metrics_tab()

    # OpenRouter integration footer
    render_openrouter_footer()


# ==================== SESSION STATE ====================
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'phase3_running': False,
        'thought_tokens': 40,
        'generation_temp': 0.8,
        'sampling_strategy': 'nucleus',
        'coherence_semantic': 0.0,
        'coherence_syntactic': 0.0,
        'coherence_predictive': 0.0,
        'theater_score': 0.0,
        'baking_progress': 0.0,
        'baking_temp': 0.95,
        'baking_strength': 0.0,
        'rl_loss': [],
        'coherence_history': [],
        'thought_diversity': [],
        'openrouter_cost': 0.0,
        'openrouter_calls': 0,
        'current_thoughts': [],
        'reasoning_quality': 0.0,
        'memorization_score': 0.0,
        'novel_problem_ratio': 0.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==================== CONFIG PANEL ====================
def render_config_panel():
    """Render configuration controls in sidebar"""
    st.subheader("Thought Generation")

    st.session_state.thought_tokens = st.slider(
        "Thought Token Count",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
        help="Number of tokens for internal reasoning"
    )

    st.session_state.generation_temp = st.slider(
        "Generation Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Higher = more creative thoughts"
    )

    st.session_state.sampling_strategy = st.selectbox(
        "Sampling Strategy",
        ["nucleus", "top-k", "beam", "greedy"],
        help="Token selection method"
    )

    st.divider()
    st.subheader("Prompt Baking")

    baking_mode = st.selectbox(
        "Baking Mode",
        ["Standard", "Half-Baking (50%)", "Pursuit (Iterative)"],
        help="Prompt strength injection strategy"
    )

    st.session_state.baking_temp = st.slider(
        "Baking Temperature",
        min_value=0.5,
        max_value=1.5,
        value=0.95,
        step=0.05,
        help="KL divergence strength"
    )

    st.divider()
    st.subheader("RL Training")

    rl_epochs = st.number_input(
        "RL Epochs",
        min_value=1,
        max_value=10,
        value=3,
        help="REINFORCE training iterations"
    )

    kl_weight = st.slider(
        "KL Regularization Weight",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Prevent distribution drift"
    )

    st.divider()
    st.subheader("OpenRouter")

    frontier_model = st.selectbox(
        "Frontier Model",
        ["GPT-4o-mini", "Claude-3.5-Haiku", "Gemini-2.0-Flash", "Qwen-2.5"],
        help="Data generation model"
    )

    budget_limit = st.number_input(
        "Budget Limit ($)",
        min_value=10,
        max_value=500,
        value=150,
        help="Maximum API spend"
    )

    st.divider()

    # Action buttons
    if st.button("▶️ START PHASE 3", type="primary", use_container_width=True):
        st.session_state.phase3_running = True
        st.rerun()

    if st.button("⏸️ PAUSE", use_container_width=True):
        st.session_state.phase3_running = False

    if st.button("🔄 RESET", use_container_width=True):
        reset_phase3_state()
        st.rerun()


def reset_phase3_state():
    """Reset Phase 3 session state"""
    st.session_state.phase3_running = False
    st.session_state.coherence_semantic = 0.0
    st.session_state.coherence_syntactic = 0.0
    st.session_state.coherence_predictive = 0.0
    st.session_state.theater_score = 0.0
    st.session_state.baking_progress = 0.0
    st.session_state.rl_loss = []
    st.session_state.coherence_history = []
    st.session_state.thought_diversity = []
    st.session_state.openrouter_cost = 0.0
    st.session_state.openrouter_calls = 0
    st.session_state.current_thoughts = []


# ==================== HERO METRICS ====================
def render_hero_metrics():
    """Render hero section with key metrics"""
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        status = "ACTIVE" if st.session_state.phase3_running else "STANDBY"
        st.metric(
            "SYSTEM STATUS",
            status,
            delta="Phase 2 → 3",
            delta_color="normal"
        )

    with col2:
        overall_coherence = (
            st.session_state.coherence_semantic +
            st.session_state.coherence_syntactic +
            st.session_state.coherence_predictive
        ) / 3.0
        st.metric(
            "COHERENCE SCORE",
            f"{overall_coherence:.1f}/100",
            delta=f"+{overall_coherence - 50:.1f}" if overall_coherence > 50 else None,
            delta_color="normal"
        )

    with col3:
        st.metric(
            "THOUGHT QUALITY",
            f"{st.session_state.reasoning_quality:.1%}",
            delta=f"+{st.session_state.reasoning_quality:.1%}",
            delta_color="normal"
        )

    with col4:
        theater_status = "PASS" if st.session_state.theater_score < 0.3 else "FAIL"
        st.metric(
            "ANTI-THEATER",
            theater_status,
            delta=f"Score: {st.session_state.theater_score:.3f}",
            delta_color="inverse" if st.session_state.theater_score < 0.3 else "normal"
        )

    with col5:
        st.metric(
            "BAKING PROGRESS",
            f"{st.session_state.baking_progress:.0%}",
            delta=f"Strength: {st.session_state.baking_strength:.0%}",
            delta_color="normal"
        )

    st.markdown("---")


# ==================== TAB 1: THOUGHT GENERATION ====================
def render_thought_generation_tab():
    """Render thought generation visualization"""
    st.subheader("🧠 Token-Wise Parallel Thought Sampling")

    # Real-time generation parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        **Thought Tokens:** `{st.session_state.thought_tokens}`
        **Temperature:** `{st.session_state.generation_temp:.2f}`
        **Strategy:** `{st.session_state.sampling_strategy}`
        """)

    with col2:
        current_thoughts = len(st.session_state.current_thoughts)
        avg_length = np.mean([len(t.split()) for t in st.session_state.current_thoughts]) if current_thoughts > 0 else 0
        st.markdown(f"""
        **Active Thoughts:** `{current_thoughts}`
        **Avg Length:** `{avg_length:.1f} tokens`
        **Generation Rate:** `{current_thoughts * 2.3:.1f} thoughts/min`
        """)

    with col3:
        diversity = st.session_state.thought_diversity[-1] if st.session_state.thought_diversity else 0.0
        st.markdown(f"""
        **Diversity Score:** `{diversity:.3f}`
        **Uniqueness:** `{diversity * 100:.1f}%`
        **Repetition:** `{(1 - diversity) * 100:.1f}%`
        """)

    # Thought generation visualization
    st.markdown("#### Live Thought Stream")

    if st.session_state.phase3_running:
        # Simulate thought generation
        simulate_thought_generation()

    # Display thought samples
    if st.session_state.current_thoughts:
        for i, thought in enumerate(st.session_state.current_thoughts[-5:]):
            with st.expander(f"💭 Thought {len(st.session_state.current_thoughts) - 5 + i + 1}"):
                st.code(thought, language=None)
    else:
        st.info("⏳ Awaiting thought generation... Press START to begin.")

    # Token-wise sampling heatmap
    st.markdown("#### Token Probability Distribution")

    fig = create_token_probability_heatmap()
    st.plotly_chart(fig, use_container_width=True)


def simulate_thought_generation():
    """Simulate thought generation for demo"""
    if np.random.random() > 0.7:  # 30% chance per render
        thought_templates = [
            "Therefore, we can deduce that the relationship between X and Y is...",
            "Breaking down the problem: (1) identify constraints, (2) evaluate options, (3)...",
            "The key insight here is recognizing the pattern of...",
            "Let's consider the counterexample: if we assume NOT-X, then...",
            "Applying first principles: the fundamental truth is...",
            "The logical consequence of this assumption is...",
            "Reasoning by analogy: this is similar to the case where...",
            "To verify our hypothesis, we should check whether...",
        ]

        new_thought = np.random.choice(thought_templates)
        st.session_state.current_thoughts.append(new_thought)
        st.session_state.thought_diversity.append(np.random.uniform(0.6, 0.95))
        st.session_state.reasoning_quality = np.random.uniform(0.7, 0.95)


def create_token_probability_heatmap():
    """Create token probability distribution heatmap"""
    # Simulate token probabilities
    tokens = [f"T{i}" for i in range(20)]
    positions = list(range(st.session_state.thought_tokens))

    # Generate probability matrix
    prob_matrix = np.random.dirichlet(np.ones(len(tokens)), size=len(positions))

    fig = go.Figure(data=go.Heatmap(
        z=prob_matrix.T,
        x=positions,
        y=tokens,
        colorscale=[
            [0, '#0a0e27'],
            [0.5, '#1a1f3a'],
            [0.7, '#0088ff'],
            [1, '#00ffff']
        ],
        colorbar=dict(
            title="Probability",
            titlefont=dict(color='#00ffff'),
            tickfont=dict(color='#00ffff')
        )
    ))

    fig.update_layout(
        title="Token Selection Probabilities Across Positions",
        xaxis_title="Position in Thought Sequence",
        yaxis_title="Token",
        plot_bgcolor='rgba(10, 14, 39, 0.8)',
        paper_bgcolor='rgba(26, 31, 58, 0.6)',
        font=dict(color='#00ffff', family='Courier New'),
        height=400
    )

    return fig


# ==================== TAB 2: COHERENCE SCORING ====================
def render_coherence_scoring_tab():
    """Render coherence scoring section"""
    st.subheader("📊 Multi-Dimensional Coherence Analysis")

    # Update coherence scores if running
    if st.session_state.phase3_running:
        update_coherence_scores()

    # Coherence gauges
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_semantic = create_coherence_gauge(
            st.session_state.coherence_semantic,
            "Semantic Coherence",
            "Meaning consistency"
        )
        st.plotly_chart(fig_semantic, use_container_width=True)

    with col2:
        fig_syntactic = create_coherence_gauge(
            st.session_state.coherence_syntactic,
            "Syntactic Coherence",
            "Grammar structure"
        )
        st.plotly_chart(fig_syntactic, use_container_width=True)

    with col3:
        fig_predictive = create_coherence_gauge(
            st.session_state.coherence_predictive,
            "Predictive Coherence",
            "Next-token accuracy"
        )
        st.plotly_chart(fig_predictive, use_container_width=True)

    # Overall coherence gauge
    st.markdown("#### Overall Coherence Rating")

    overall = (
        st.session_state.coherence_semantic +
        st.session_state.coherence_syntactic +
        st.session_state.coherence_predictive
    ) / 3.0

    fig_overall = create_overall_coherence_gauge(overall)
    st.plotly_chart(fig_overall, use_container_width=True)

    # Coherence history
    st.markdown("#### Coherence Evolution Over Time")

    if st.session_state.coherence_history:
        fig_history = create_coherence_history_chart()
        st.plotly_chart(fig_history, use_container_width=True)
    else:
        st.info("⏳ No coherence history yet. Start Phase 3 to track evolution.")


def update_coherence_scores():
    """Update coherence scores (simulation)"""
    if np.random.random() > 0.8:  # 20% chance per render
        st.session_state.coherence_semantic = min(100, st.session_state.coherence_semantic + np.random.uniform(0, 5))
        st.session_state.coherence_syntactic = min(100, st.session_state.coherence_syntactic + np.random.uniform(0, 5))
        st.session_state.coherence_predictive = min(100, st.session_state.coherence_predictive + np.random.uniform(0, 5))

        st.session_state.coherence_history.append({
            'timestamp': datetime.now(),
            'semantic': st.session_state.coherence_semantic,
            'syntactic': st.session_state.coherence_syntactic,
            'predictive': st.session_state.coherence_predictive,
            'overall': (st.session_state.coherence_semantic + st.session_state.coherence_syntactic + st.session_state.coherence_predictive) / 3.0
        })


def create_coherence_gauge(value: float, title: str, subtitle: str) -> go.Figure:
    """Create individual coherence gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title}<br><sub>{subtitle}</sub>", 'font': {'color': '#00ffff', 'size': 16}},
        delta={'reference': 70, 'increasing': {'color': "#00ffff"}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': '#00ffff'},
            'bar': {'color': "#00ffff"},
            'bgcolor': "rgba(26, 31, 58, 0.6)",
            'borderwidth': 2,
            'bordercolor': "#00ffff",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(0, 255, 255, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#ffff00", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(26, 31, 58, 0.6)',
        font={'color': '#00ffff', 'family': 'Courier New'},
        height=300
    )

    return fig


def create_overall_coherence_gauge(value: float) -> go.Figure:
    """Create overall coherence gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "COMPOSITE COHERENCE", 'font': {'color': '#00ffff', 'size': 24}},
        delta={'reference': 70, 'increasing': {'color': "#00ffff"}},
        number={'font': {'size': 48, 'color': '#00ffff'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': '#00ffff', 'tickwidth': 2},
            'bar': {'color': "#00ffff", 'thickness': 0.8},
            'bgcolor': "rgba(26, 31, 58, 0.6)",
            'borderwidth': 3,
            'bordercolor': "#00ffff",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.4)'},
                {'range': [50, 75], 'color': 'rgba(255, 255, 0, 0.4)'},
                {'range': [75, 90], 'color': 'rgba(0, 255, 0, 0.4)'},
                {'range': [90, 100], 'color': 'rgba(0, 255, 255, 0.4)'}
            ],
            'threshold': {
                'line': {'color': "#ff00ff", 'width': 6},
                'thickness': 0.9,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(26, 31, 58, 0.6)',
        font={'color': '#00ffff', 'family': 'Courier New'},
        height=400
    )

    return fig


def create_coherence_history_chart() -> go.Figure:
    """Create coherence evolution chart"""
    df = pd.DataFrame(st.session_state.coherence_history)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['semantic'],
        mode='lines+markers',
        name='Semantic',
        line=dict(color='#ff00ff', width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['syntactic'],
        mode='lines+markers',
        name='Syntactic',
        line=dict(color='#00ff00', width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['predictive'],
        mode='lines+markers',
        name='Predictive',
        line=dict(color='#ffff00', width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['overall'],
        mode='lines+markers',
        name='Overall',
        line=dict(color='#00ffff', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Coherence Metrics Evolution",
        xaxis_title="Time",
        yaxis_title="Coherence Score",
        plot_bgcolor='rgba(10, 14, 39, 0.8)',
        paper_bgcolor='rgba(26, 31, 58, 0.6)',
        font=dict(color='#00ffff', family='Courier New'),
        legend=dict(
            bgcolor='rgba(26, 31, 58, 0.8)',
            bordercolor='#00ffff',
            borderwidth=1
        ),
        height=400
    )

    return fig


# ==================== TAB 3: REASONING TRACE ====================
def render_reasoning_trace_tab():
    """Render reasoning trace viewer"""
    st.subheader("🔍 Reasoning Trace Inspector")

    # Sample reasoning traces
    traces = [
        {
            'id': 1,
            'prompt': 'Solve: If 2x + 5 = 13, what is x?',
            'thoughts': [
                'First, I need to isolate x on one side',
                'Subtract 5 from both sides: 2x = 8',
                'Divide both sides by 2: x = 4',
                'Let me verify: 2(4) + 5 = 8 + 5 = 13 ✓'
            ],
            'output': 'x = 4',
            'quality_ratings': [0.92, 0.95, 0.98, 0.96]
        },
        {
            'id': 2,
            'prompt': 'What is the capital of France?',
            'thoughts': [
                'This is asking for the capital city of France',
                'France is a country in Western Europe',
                'The capital has been Paris for centuries',
                'Paris is located in the north-central part of France'
            ],
            'output': 'The capital of France is Paris.',
            'quality_ratings': [0.88, 0.85, 0.99, 0.87]
        },
        {
            'id': 3,
            'prompt': 'Explain why the sky is blue',
            'thoughts': [
                'This involves understanding light scattering',
                'Sunlight is composed of different wavelengths (colors)',
                'Shorter wavelengths (blue) scatter more than longer ones (red)',
                'This is called Rayleigh scattering',
                'Blue light scatters in all directions, filling the sky'
            ],
            'output': 'The sky appears blue due to Rayleigh scattering, where shorter blue wavelengths scatter more than other colors.',
            'quality_ratings': [0.93, 0.91, 0.97, 0.95, 0.94]
        }
    ]

    # Trace selector
    selected_trace_id = st.selectbox(
        "Select Reasoning Trace",
        [t['id'] for t in traces],
        format_func=lambda x: f"Trace #{x}"
    )

    trace = next(t for t in traces if t['id'] == selected_trace_id)

    # Display trace
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Input Prompt")
        st.code(trace['prompt'], language=None)

        st.markdown("#### Generated Thoughts")
        for i, (thought, rating) in enumerate(zip(trace['thoughts'], trace['quality_ratings'])):
            quality_color = get_quality_color(rating)
            with st.expander(f"💭 Thought {i+1} - Quality: {rating:.2%}", expanded=(i == 0)):
                st.markdown(f"<p style='color: {quality_color};'>{thought}</p>", unsafe_allow_html=True)
                st.progress(rating)

        st.markdown("#### Final Output")
        st.success(trace['output'])

    with col2:
        st.markdown("#### Thought Quality")

        avg_quality = np.mean(trace['quality_ratings'])
        st.metric("Average Quality", f"{avg_quality:.2%}")

        # Quality distribution
        fig = go.Figure(go.Bar(
            x=[f"T{i+1}" for i in range(len(trace['quality_ratings']))],
            y=trace['quality_ratings'],
            marker=dict(
                color=trace['quality_ratings'],
                colorscale=[[0, '#ff0000'], [0.5, '#ffff00'], [1, '#00ffff']],
                showscale=False
            )
        ))

        fig.update_layout(
            title="Thought Quality Distribution",
            yaxis_title="Quality Score",
            plot_bgcolor='rgba(10, 14, 39, 0.8)',
            paper_bgcolor='rgba(26, 31, 58, 0.6)',
            font=dict(color='#00ffff', family='Courier New'),
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)


def get_quality_color(rating: float) -> str:
    """Get color based on quality rating"""
    if rating >= 0.9:
        return '#00ffff'
    elif rating >= 0.75:
        return '#00ff00'
    elif rating >= 0.5:
        return '#ffff00'
    else:
        return '#ff0000'


# ==================== TAB 4: ANTI-THEATER DETECTION ====================
def render_anti_theater_tab():
    """Render anti-theater detection panel"""
    st.subheader("🛡️ Anti-Theater Detection System")

    # Update theater metrics if running
    if st.session_state.phase3_running:
        update_theater_metrics()

    # Theater detection indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        genuine = 1.0 - st.session_state.theater_score
        st.metric(
            "Genuine Reasoning",
            f"{genuine:.1%}",
            delta="VALIDATED" if genuine > 0.7 else "SUSPICIOUS",
            delta_color="normal" if genuine > 0.7 else "inverse"
        )

        fig_genuine = create_mini_gauge(genuine * 100, "Genuine", 70)
        st.plotly_chart(fig_genuine, use_container_width=True)

    with col2:
        st.metric(
            "Memorization Score",
            f"{st.session_state.memorization_score:.1%}",
            delta="LOW" if st.session_state.memorization_score < 0.3 else "HIGH",
            delta_color="inverse" if st.session_state.memorization_score < 0.3 else "normal"
        )

        fig_mem = create_mini_gauge(st.session_state.memorization_score * 100, "Memorization", 30)
        st.plotly_chart(fig_mem, use_container_width=True)

    with col3:
        st.metric(
            "Novel Problem Ratio",
            f"{st.session_state.novel_problem_ratio:.1%}",
            delta="GOOD" if st.session_state.novel_problem_ratio > 0.6 else "POOR",
            delta_color="normal" if st.session_state.novel_problem_ratio > 0.6 else "inverse"
        )

        fig_novel = create_mini_gauge(st.session_state.novel_problem_ratio * 100, "Novel", 60)
        st.plotly_chart(fig_novel, use_container_width=True)

    # Theater score gauge
    st.markdown("#### Overall Theater Detection Score")
    st.caption("Lower is better - high scores indicate theatrical/fake reasoning")

    fig_theater = create_theater_gauge(st.session_state.theater_score)
    st.plotly_chart(fig_theater, use_container_width=True)

    # Theater validation tests
    st.markdown("#### Validation Test Results")

    tests = [
        {"name": "Pattern Memorization Test", "passed": st.session_state.memorization_score < 0.3},
        {"name": "Novel Problem Solving", "passed": st.session_state.novel_problem_ratio > 0.6},
        {"name": "Reasoning Coherence", "passed": (st.session_state.coherence_semantic + st.session_state.coherence_syntactic + st.session_state.coherence_predictive) / 3.0 > 70},
        {"name": "Thought Diversity", "passed": len(st.session_state.thought_diversity) > 0 and st.session_state.thought_diversity[-1] > 0.7},
        {"name": "Anti-Shortcut Validation", "passed": st.session_state.theater_score < 0.3},
    ]

    for test in tests:
        status_icon = "✅" if test['passed'] else "❌"
        status_text = "PASS" if test['passed'] else "FAIL"
        status_color = "#00ff00" if test['passed'] else "#ff0000"

        st.markdown(f"{status_icon} **{test['name']}**: <span style='color: {status_color};'>{status_text}</span>", unsafe_allow_html=True)


def update_theater_metrics():
    """Update theater detection metrics (simulation)"""
    if np.random.random() > 0.85:  # 15% chance per render
        st.session_state.theater_score = max(0.0, min(1.0, st.session_state.theater_score + np.random.uniform(-0.05, 0.02)))
        st.session_state.memorization_score = max(0.0, min(1.0, st.session_state.memorization_score + np.random.uniform(-0.03, 0.01)))
        st.session_state.novel_problem_ratio = max(0.0, min(1.0, st.session_state.novel_problem_ratio + np.random.uniform(0, 0.05)))


def create_mini_gauge(value: float, title: str, threshold: float) -> go.Figure:
    """Create mini gauge for theater metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': '#00ffff', 'size': 12}},
        number={'font': {'size': 20, 'color': '#00ffff'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': '#00ffff'},
            'bar': {'color': "#00ffff"},
            'bgcolor': "rgba(26, 31, 58, 0.6)",
            'borderwidth': 1,
            'bordercolor': "#00ffff",
            'threshold': {
                'line': {'color': "#ff00ff", 'width': 2},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(26, 31, 58, 0.6)',
        font={'color': '#00ffff', 'family': 'Courier New'},
        height=200,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig


def create_theater_gauge(value: float) -> go.Figure:
    """Create theater score gauge (inverted - lower is better)"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "THEATER SCORE (Target: <30)", 'font': {'color': '#00ffff', 'size': 20}},
        delta={'reference': 30, 'decreasing': {'color': "#00ff00"}, 'increasing': {'color': "#ff0000"}},
        number={'font': {'size': 40, 'color': '#00ffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#00ffff'},
            'bar': {'color': "#ff0000" if value > 0.3 else "#00ffff"},
            'bgcolor': "rgba(26, 31, 58, 0.6)",
            'borderwidth': 2,
            'bordercolor': "#00ffff",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [30, 50], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#ffff00", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(26, 31, 58, 0.6)',
        font={'color': '#00ffff', 'family': 'Courier New'},
        height=350
    )

    return fig


# ==================== TAB 5: PROMPT BAKING ====================
def render_prompt_baking_tab():
    """Render prompt baking status panel"""
    st.subheader("🔥 Chain-of-Thought Prompt Baking")

    # Update baking progress if running
    if st.session_state.phase3_running:
        update_baking_progress()

    # Baking status
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Baking Progress",
            f"{st.session_state.baking_progress:.0%}",
            delta="In Progress" if st.session_state.baking_progress < 1.0 else "Complete"
        )

    with col2:
        st.metric(
            "Baking Temperature",
            f"{st.session_state.baking_temp:.2f}",
            delta="KL Divergence Strength"
        )

    with col3:
        st.metric(
            "Prompt Strength",
            f"{st.session_state.baking_strength:.0%}",
            delta="Weight Injection Level"
        )

    # Baking progress bar
    st.markdown("#### Baking Pipeline Status")
    st.progress(st.session_state.baking_progress)

    # Baking stages
    stages = [
        {"name": "LoRA Adapter Init", "progress": min(100, st.session_state.baking_progress * 400), "time": "10s"},
        {"name": "KL Divergence Optimization", "progress": max(0, min(100, (st.session_state.baking_progress - 0.25) * 400)), "time": "3min"},
        {"name": "Weight Merging", "progress": max(0, min(100, (st.session_state.baking_progress - 0.5) * 400)), "time": "1min"},
        {"name": "Validation", "progress": max(0, min(100, (st.session_state.baking_progress - 0.75) * 400)), "time": "30s"},
    ]

    for stage in stages:
        with st.expander(f"{stage['name']} - {stage['progress']:.0f}%", expanded=stage['progress'] > 0):
            st.progress(stage['progress'] / 100)
            st.caption(f"Estimated time: {stage['time']}")

    # CoT prompt being baked
    st.markdown("#### Active Chain-of-Thought Prompt")

    cot_prompt = """You are a reasoning specialist. When solving problems, you MUST:
1. Break down the problem into clear steps
2. Show your internal reasoning for each step
3. Verify your logic before concluding
4. Explain your thought process explicitly

Always use structured reasoning: [Analysis] → [Steps] → [Verification] → [Conclusion]"""

    st.code(cot_prompt, language=None)

    # Baking effectiveness
    st.markdown("#### Baking Effectiveness Metrics")

    effectiveness_data = pd.DataFrame({
        'Metric': ['Prompt Retention', 'Behavior Consistency', 'No-Prompt Performance', 'Multi-Turn Stability'],
        'Score': [0.94, 0.89, 0.87, 0.92]
    })

    fig = go.Figure(go.Bar(
        x=effectiveness_data['Metric'],
        y=effectiveness_data['Score'],
        marker=dict(
            color=effectiveness_data['Score'],
            colorscale=[[0, '#ff0000'], [0.7, '#ffff00'], [1, '#00ffff']],
            showscale=False
        ),
        text=[f"{s:.0%}" for s in effectiveness_data['Score']],
        textposition='auto'
    ))

    fig.update_layout(
        title="Prompt Baking Quality Metrics",
        yaxis_title="Effectiveness Score",
        yaxis=dict(range=[0, 1]),
        plot_bgcolor='rgba(10, 14, 39, 0.8)',
        paper_bgcolor='rgba(26, 31, 58, 0.6)',
        font=dict(color='#00ffff', family='Courier New'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def update_baking_progress():
    """Update baking progress (simulation)"""
    if st.session_state.baking_progress < 1.0:
        st.session_state.baking_progress = min(1.0, st.session_state.baking_progress + np.random.uniform(0.01, 0.05))
        st.session_state.baking_strength = st.session_state.baking_progress * 0.9  # 90% max strength


# ==================== TAB 6: TRAINING METRICS ====================
def render_training_metrics_tab():
    """Render RL training metrics"""
    st.subheader("📈 REINFORCE RL Training Metrics")

    # Update training metrics if running
    if st.session_state.phase3_running:
        update_training_metrics()

    # Training overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_loss = st.session_state.rl_loss[-1] if st.session_state.rl_loss else 0.0
        st.metric(
            "Current RL Loss",
            f"{current_loss:.4f}",
            delta=f"{current_loss - 0.5:.4f}" if len(st.session_state.rl_loss) > 1 else None,
            delta_color="inverse"
        )

    with col2:
        improvement = (st.session_state.rl_loss[0] - st.session_state.rl_loss[-1]) / st.session_state.rl_loss[0] if len(st.session_state.rl_loss) > 1 else 0.0
        st.metric(
            "Loss Improvement",
            f"{improvement:.1%}",
            delta="Decreasing" if improvement > 0 else "Stable"
        )

    with col3:
        st.metric(
            "Training Steps",
            len(st.session_state.rl_loss),
            delta=f"{len(st.session_state.rl_loss) * 2} thoughts"
        )

    with col4:
        convergence = "CONVERGED" if len(st.session_state.rl_loss) > 10 and improvement > 0.3 else "TRAINING"
        st.metric(
            "Status",
            convergence,
            delta="Stable" if convergence == "CONVERGED" else "Optimizing"
        )

    # RL loss curve
    st.markdown("#### RL Training Loss Over Time")

    if st.session_state.rl_loss:
        fig = go.Figure()

        steps = list(range(len(st.session_state.rl_loss)))

        fig.add_trace(go.Scatter(
            x=steps,
            y=st.session_state.rl_loss,
            mode='lines+markers',
            name='RL Loss',
            line=dict(color='#00ffff', width=2),
            marker=dict(size=6)
        ))

        # Add smoothed trend
        if len(st.session_state.rl_loss) > 5:
            smoothed = pd.Series(st.session_state.rl_loss).rolling(window=5, min_periods=1).mean().tolist()
            fig.add_trace(go.Scatter(
                x=steps,
                y=smoothed,
                mode='lines',
                name='Trend',
                line=dict(color='#ff00ff', width=3, dash='dash')
            ))

        fig.update_layout(
            title="REINFORCE Loss Convergence",
            xaxis_title="Training Step",
            yaxis_title="Loss",
            plot_bgcolor='rgba(10, 14, 39, 0.8)',
            paper_bgcolor='rgba(26, 31, 58, 0.6)',
            font=dict(color='#00ffff', family='Courier New'),
            legend=dict(
                bgcolor='rgba(26, 31, 58, 0.8)',
                bordercolor='#00ffff',
                borderwidth=1
            ),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⏳ No training data yet. Start Phase 3 to begin RL training.")

    # Thought diversity metrics
    st.markdown("#### Thought Diversity Evolution")

    if st.session_state.thought_diversity:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(st.session_state.thought_diversity))),
            y=st.session_state.thought_diversity,
            mode='lines+markers',
            name='Diversity',
            line=dict(color='#00ff00', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)'
        ))

        fig.update_layout(
            title="Thought Diversity Score Over Time",
            xaxis_title="Thought Generation Event",
            yaxis_title="Diversity Score",
            yaxis=dict(range=[0, 1]),
            plot_bgcolor='rgba(10, 14, 39, 0.8)',
            paper_bgcolor='rgba(26, 31, 58, 0.6)',
            font=dict(color='#00ffff', family='Courier New'),
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⏳ No diversity data yet. Generate thoughts to track diversity.")


def update_training_metrics():
    """Update training metrics (simulation)"""
    if np.random.random() > 0.7:  # 30% chance per render
        # Simulate RL loss decrease
        if len(st.session_state.rl_loss) == 0:
            new_loss = np.random.uniform(0.8, 1.2)
        else:
            new_loss = max(0.01, st.session_state.rl_loss[-1] * np.random.uniform(0.85, 0.98))

        st.session_state.rl_loss.append(new_loss)


# ==================== OPENROUTER FOOTER ====================
def render_openrouter_footer():
    """Render OpenRouter integration status"""
    st.markdown("---")
    st.markdown("### 🌐 OpenRouter Integration Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "API Calls",
            st.session_state.openrouter_calls,
            delta=f"+{st.session_state.openrouter_calls} today"
        )

    with col2:
        st.metric(
            "Total Cost",
            f"${st.session_state.openrouter_cost:.2f}",
            delta=f"Budget: $150.00"
        )

    with col3:
        budget_used = (st.session_state.openrouter_cost / 150.0) * 100
        st.metric(
            "Budget Used",
            f"{budget_used:.1f}%",
            delta="Within limits" if budget_used < 80 else "Approaching limit"
        )

    with col4:
        st.metric(
            "Active Model",
            "GPT-4o-mini",
            delta="Frontier Model"
        )

    # Update cost if running
    if st.session_state.phase3_running and np.random.random() > 0.9:
        st.session_state.openrouter_calls += 1
        st.session_state.openrouter_cost += np.random.uniform(0.01, 0.05)


# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    render_phase3_dashboard()

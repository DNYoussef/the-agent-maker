"""
Phase 7: Self-Guided Experts - Streamlit UI Dashboard

Model-driven expert discovery with Transformer-squared SVF training and NSGA-II ADAS.
Futuristic command center theme with real-time visualization.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Color scheme for futuristic command center
COLORS = {
    "primary": "#00FFFF",  # Cyan
    "secondary": "#9D4EDD",  # Purple
    "accent": "#7209B7",  # Deep purple
    "success": "#06FFA5",  # Bright green
    "warning": "#FFD60A",  # Yellow
    "danger": "#FF006E",  # Pink
    "background": "#0A0E27",  # Dark blue-black
    "surface": "#1A1F3A",  # Lighter dark
    "text": "#E0E7FF",  # Light blue-white
}


def render_phase7_dashboard() -> None:
    """Main dashboard for Phase 7 Self-Guided Experts"""

    # Apply custom CSS for futuristic theme
    apply_custom_css()

    # Hero section
    render_hero_section()

    # Sidebar controls
    with st.sidebar:
        st.header("Expert Discovery Controls")
        render_config_panel()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Overview",
            "Expert Discovery",
            "SVF Training",
            "ADAS Search",
            "Architecture",
            "Cost & Time",
            "Results",
        ]
    )

    with tab1:
        render_overview_tab()

    with tab2:
        render_expert_discovery_tab()

    with tab3:
        render_svf_training_tab()

    with tab4:
        render_adas_search_tab()

    with tab5:
        render_architecture_tab()

    with tab6:
        render_cost_time_tab()

    with tab7:
        render_results_tab()


def apply_custom_css() -> None:
    """Apply futuristic command center CSS theme"""
    st.markdown(
        f"""
    <style>
        /* Global background */
        .stApp {{
            background: linear-gradient(135deg, {COLORS['background']} 0%, #0D1B2A 100%);
            color: {COLORS['text']};
        }}

        /* Headers with glow effect */
        h1, h2, h3 {{
            color: {COLORS['primary']};
            text-shadow: 0 0 10px {COLORS['primary']}40;
            font-family: 'Courier New', monospace;
        }}

        /* Metric cards with border glow */
        [data-testid="stMetricValue"] {{
            color: {COLORS['primary']};
            font-size: 2rem;
            text-shadow: 0 0 8px {COLORS['primary']}60;
        }}

        /* Progress bars */
        .stProgress > div > div {{
            background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        }}

        /* Buttons with neon effect */
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['secondary']} 0%, {COLORS['accent']} 100%);
            color: white;
            border: 2px solid {COLORS['primary']};
            box-shadow: 0 0 15px {COLORS['primary']}40;
            transition: all 0.3s;
        }}

        .stButton > button:hover {{
            box-shadow: 0 0 25px {COLORS['primary']}80;
            transform: translateY(-2px);
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {COLORS['surface']};
            border-radius: 8px;
            padding: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border: 1px solid {COLORS['primary']}40;
            color: {COLORS['primary']};
            border-radius: 4px;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {COLORS['primary']}20 0%, {COLORS['secondary']}20 100%);
            border: 2px solid {COLORS['primary']};
            box-shadow: 0 0 10px {COLORS['primary']}40;
        }}

        /* Info/Success/Warning boxes */
        .stAlert {{
            background-color: {COLORS['surface']};
            border-left: 4px solid {COLORS['primary']};
            border-radius: 4px;
        }}

        /* Dataframe styling */
        .dataframe {{
            background-color: {COLORS['surface']};
            color: {COLORS['text']};
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {COLORS['surface']};
            color: {COLORS['primary']};
            border: 1px solid {COLORS['primary']}40;
        }}

        /* Code blocks */
        code {{
            color: {COLORS['success']};
            background-color: {COLORS['surface']};
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_hero_section() -> None:
    """Render hero section with phase title and progress"""
    st.markdown(
        f"""
    <div style='text-align: center; padding: 2rem 0; border-bottom: 2px solid {COLORS['primary']}40;'>
        <h1 style='font-size: 3rem; margin: 0;'>
            PHASE 7: SELF-GUIDED EXPERTS
        </h1>
        <p style='color: {COLORS['secondary']}; font-size: 1.2rem; margin-top: 0.5rem;'>
            Model-Driven Expert Discovery + Transformer² SVF + NSGA-II ADAS
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Overall progress
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        stage1_complete = st.session_state.get("stage1_complete", False)
        status_icon = "✓" if stage1_complete else "⟳"
        st.markdown(
            f"""
        <div style='text-align: center; padding: 1rem; background: {COLORS['surface']}; border-radius: 8px; border: 1px solid {COLORS['primary']}40;'>
            <div style='font-size: 2rem;'>{status_icon}</div>
            <div style='color: {COLORS['primary']};'>Stage 1</div>
            <div style='font-size: 0.8rem; color: {COLORS['text']}80;'>Self-Analysis</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div style='text-align: center; padding: 1rem; background: {COLORS['surface']}; border-radius: 8px; border: 1px solid {COLORS['primary']}40;'>
            <div style='font-size: 2rem;'>→</div>
            <div style='color: {COLORS['secondary']};'>Pipeline</div>
            <div style='font-size: 0.8rem; color: {COLORS['text']}80;'>Flow</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        stage2_complete = st.session_state.get("stage2_complete", False)
        status_icon = "✓" if stage2_complete else "⏸" if stage1_complete else "○"
        st.markdown(
            f"""
        <div style='text-align: center; padding: 1rem; background: {COLORS['surface']}; border-radius: 8px; border: 1px solid {COLORS['primary']}40;'>
            <div style='font-size: 2rem;'>{status_icon}</div>
            <div style='color: {COLORS['primary']};'>Stage 2</div>
            <div style='font-size: 0.8rem; color: {COLORS['text']}80;'>SVF Training</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div style='text-align: center; padding: 1rem; background: {COLORS['surface']}; border-radius: 8px; border: 1px solid {COLORS['primary']}40;'>
            <div style='font-size: 2rem;'>→</div>
            <div style='color: {COLORS['secondary']};'>Pipeline</div>
            <div style='font-size: 0.8rem; color: {COLORS['text']}80;'>Flow</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        stage3_complete = st.session_state.get("stage3_complete", False)
        status_icon = "✓" if stage3_complete else "⏸" if stage2_complete else "○"
        st.markdown(
            f"""
        <div style='text-align: center; padding: 1rem; background: {COLORS['surface']}; border-radius: 8px; border: 1px solid {COLORS['primary']}40;'>
            <div style='font-size: 2rem;'>{status_icon}</div>
            <div style='color: {COLORS['primary']};'>Stage 3</div>
            <div style='font-size: 0.8rem; color: {COLORS['text']}80;'>ADAS Search</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Progress bar
    total_progress = calculate_total_progress()
    st.progress(total_progress)
    st.markdown(
        f"<div style='text-align: center; color: {COLORS['primary']};'>Overall Progress: {total_progress:.1%}</div>",
        unsafe_allow_html=True,
    )


def calculate_total_progress() -> float:
    """Calculate overall progress across all stages"""
    stage1_prog = st.session_state.get("stage1_progress", 0.0)
    stage2_prog = st.session_state.get("stage2_progress", 0.0)
    stage3_prog = st.session_state.get("stage3_progress", 0.0)

    # Weights: Stage 1 (10%), Stage 2 (40%), Stage 3 (50%)
    return float(0.1 * stage1_prog + 0.4 * stage2_prog + 0.5 * stage3_prog)


def render_config_panel() -> None:
    """Render configuration controls in sidebar"""
    st.subheader("Expert Configuration")

    # Expert count range (model determines)
    st.markdown(
        f"<div style='color: {COLORS['secondary']};'>Model-Determined Expert Count:</div>",
        unsafe_allow_html=True,
    )
    min_experts = st.slider("Minimum Experts", 3, 10, 3, help="Minimum N for self-analysis")
    max_experts = st.slider("Maximum Experts", 3, 10, 10, help="Maximum N for self-analysis")

    st.markdown(
        f"<div style='padding: 0.5rem; background: {COLORS['surface']}; border-radius: 4px; margin: 0.5rem 0;'>Model will analyze capabilities and select N ∈ [{min_experts}, {max_experts}]</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # SVF Training settings
    st.subheader("SVF Training (Stage 2)")

    svf_algorithm = st.selectbox(
        "Training Algorithm",
        ["REINFORCE (Primary)", "MuonGrokfast (Fallback)"],
        help="REINFORCE policy gradient with MuonGrokfast fallback",
    )

    svf_epochs = st.slider("Training Epochs", 1, 10, 5, help="Transformer² SVF training epochs")

    svf_lr = st.number_input(
        "Learning Rate",
        min_value=1e-5,
        max_value=1e-2,
        value=3e-4,
        format="%.5f",
        help="SVF training learning rate",
    )

    st.divider()

    # ADAS settings
    st.subheader("ADAS Search (Stage 3)")

    st.markdown(
        f"<div style='color: {COLORS['secondary']};'>NSGA-II Configuration:</div>",
        unsafe_allow_html=True,
    )

    adas_population = st.slider(
        "Population Size", 20, 100, 50, help="NSGA-II population per generation"
    )
    adas_generations = st.slider("Generations", 50, 200, 100, help="NSGA-II evolution generations")

    total_evaluations = adas_population * adas_generations
    st.metric("Total Evaluations", f"{total_evaluations:,}", help="Population × Generations")

    st.divider()

    # Frontier models
    st.subheader("Frontier Models")

    models = st.multiselect(
        "Select Models",
        ["GPT-4o-mini", "Claude-3.5 Haiku", "Gemini 2.0 Flash", "Qwen 2.5"],
        default=["GPT-4o-mini", "Claude-3.5 Haiku", "Gemini 2.0 Flash", "Qwen 2.5"],
        help="Frontier models for self-analysis and ADAS",
    )

    st.divider()

    # Action buttons
    if st.button("▶ Start Discovery", type="primary", use_container_width=True):
        st.session_state.phase7_running = True
        st.session_state.stage1_complete = False
        st.session_state.stage2_complete = False
        st.session_state.stage3_complete = False

    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.phase7_running = False

    if st.button("🔄 Reset", use_container_width=True):
        reset_session_state()


def render_overview_tab() -> None:
    """Overview of Phase 7 process"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Phase Status",
            "Running" if st.session_state.get("phase7_running") else "Ready",
            delta="Phase 6 → 7",
        )

    with col2:
        discovered_experts = st.session_state.get("discovered_experts", 0)
        st.metric(
            "Discovered Experts",
            f"{discovered_experts}",
            delta=f"Model-determined" if discovered_experts > 0 else "Pending",
        )

    with col3:
        elapsed_time = st.session_state.get("elapsed_hours", 0.0)
        st.metric("Elapsed Time", f"{elapsed_time:.1f}h", delta=f"/ 78h total")

    with col4:
        api_cost = st.session_state.get("api_cost", 0.0)
        st.metric("API Cost", f"${api_cost:.2f}", delta=f"/ $250 budget")

    # Three-stage pipeline
    st.subheader("Three-Stage Expert Discovery Pipeline")

    stages = [
        {
            "name": "Stage 1: Self-Analysis",
            "description": "Model analyzes own capabilities to determine expert count N",
            "duration": "15-30 min",
            "cost": "$20-40",
            "status": st.session_state.get("stage1_complete", False),
            "details": [
                "Model introspection via frontier models",
                "Capability clustering analysis",
                "Expert count determination (N=3-10)",
                "Specialization identification",
            ],
        },
        {
            "name": "Stage 2: Transformer² SVF Training",
            "description": "Train Singular Value Factorization routing with REINFORCE",
            "duration": "36 hours",
            "cost": "$80-120",
            "status": st.session_state.get("stage2_complete", False),
            "details": [
                "REINFORCE policy gradient (primary)",
                "MuonGrokfast fallback if needed",
                "Expert routing optimization",
                "Singular value decomposition",
            ],
        },
        {
            "name": "Stage 3: NSGA-II ADAS",
            "description": "Model-guided architecture search (100 gen × 50 pop)",
            "duration": "42 hours",
            "cost": "$50-90",
            "status": st.session_state.get("stage3_complete", False),
            "details": [
                "Multi-objective optimization",
                "5,000 architecture evaluations",
                "Pareto frontier discovery",
                "Best architecture selection",
            ],
        },
    ]

    for i, stage in enumerate(stages):
        with st.expander(
            f"{'✓' if stage['status'] else '⏸'} {stage['name']}", expanded=not stage["status"]
        ):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"**Description:** {stage['description']}")
                st.markdown("**Details:**")
                for detail in stage["details"]:
                    st.markdown(f"- {detail}")

            with col2:
                st.metric("Duration", stage["duration"])
                st.metric("Cost", stage["cost"])

            with col3:
                st.metric("Status", "Complete" if stage["status"] else "Pending")
                if i > 0 and not stages[i - 1]["status"]:
                    st.warning("Waiting for previous stage")

    # Key features
    st.subheader("Phase 7 Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
        <div style='background: {COLORS['surface']}; padding: 1rem; border-radius: 8px; border: 1px solid {COLORS['primary']}40;'>
            <h4 style='color: {COLORS['primary']};'>Model-Driven Discovery</h4>
            <ul style='color: {COLORS['text']};'>
                <li>Self-analysis determines expert count (N=3-10)</li>
                <li>Auto-discovered specializations</li>
                <li>Capability-based expert routing</li>
                <li>No manual architecture design</li>
                <li>Adaptive to model strengths</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div style='background: {COLORS['surface']}; padding: 1rem; border-radius: 8px; border: 1px solid {COLORS['primary']}40;'>
            <h4 style='color: {COLORS['secondary']};'>Advanced Techniques</h4>
            <ul style='color: {COLORS['text']};'>
                <li>Transformer² Singular Value Factorization</li>
                <li>REINFORCE policy gradient training</li>
                <li>NSGA-II multi-objective evolution</li>
                <li>Pareto-optimal architecture search</li>
                <li>5,000 total evaluations (100×50)</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_expert_discovery_tab() -> None:
    """Expert discovery visualization (Stage 1)"""
    st.subheader("Stage 1: Model Self-Analysis")

    # Progress
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Analysis Progress", f"{st.session_state.get('stage1_progress', 0.0):.0%}")

    with col2:
        discovered_n = st.session_state.get("discovered_experts", 0)
        st.metric("Discovered N", discovered_n, delta="Expert count")

    with col3:
        st.metric("Frontier Model", "GPT-4o-mini", delta="Active")

    # Capability analysis heatmap
    st.markdown("#### Model Capability Analysis")

    # Simulate capability clusters
    capabilities = [
        "Reasoning",
        "Code",
        "Math",
        "Language",
        "Memory",
        "Tool Use",
        "Creative",
        "Analytical",
    ]
    task_samples = 50

    # Generate synthetic data
    capability_scores = np.random.rand(len(capabilities), task_samples)

    # Add clustering structure
    for i in range(len(capabilities)):
        cluster_strength = np.random.rand() * 0.5 + 0.3
        cluster_center = np.random.randint(0, task_samples)
        cluster_width = task_samples // 5

        for j in range(
            max(0, cluster_center - cluster_width),
            min(task_samples, cluster_center + cluster_width),
        ):
            capability_scores[i, j] += cluster_strength

    capability_scores = np.clip(capability_scores, 0, 1)

    fig = go.Figure(
        data=go.Heatmap(
            z=capability_scores,
            x=[f"Task {i+1}" for i in range(task_samples)],
            y=capabilities,
            colorscale=[
                [0, COLORS["background"]],
                [0.5, COLORS["secondary"]],
                [1, COLORS["primary"]],
            ],
            colorbar=dict(title="Score", tickfont=dict(color=COLORS["text"])),
        )
    )

    fig.update_layout(
        title=dict(text="Capability Clustering Analysis", font=dict(color=COLORS["primary"])),
        xaxis_title="Task Samples (n=50)",
        yaxis_title="Capability Dimension",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Expert specializations
    st.markdown("#### Discovered Expert Specializations")

    if discovered_n > 0:
        expert_specs = generate_expert_specializations(discovered_n)

        for expert in expert_specs:
            with st.expander(f"Expert {expert['id']}: {expert['name']}", expanded=True):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Primary Focus:** {expert['focus']}")
                    st.markdown("**Capabilities:**")
                    for cap in expert["capabilities"]:
                        st.markdown(f"- {cap}")

                with col2:
                    st.metric("Specialization Score", f"{expert['score']:.2f}")
                    st.metric("Task Coverage", f"{expert['coverage']:.0%}")
    else:
        st.info("Expert discovery in progress... Model analyzing capabilities.")

    # Capability matrix
    if discovered_n > 0:
        st.markdown("#### Expert × Capability Matrix")

        expert_names = [f"Expert {i+1}" for i in range(discovered_n)]
        capability_matrix = np.random.rand(discovered_n, len(capabilities))

        # Normalize rows to show specialization
        for i in range(discovered_n):
            peak_idx = np.random.choice(len(capabilities))
            capability_matrix[i, peak_idx] = 0.9 + np.random.rand() * 0.1
            capability_matrix[i] = capability_matrix[i] / capability_matrix[i].sum()

        fig2 = go.Figure(
            data=go.Heatmap(
                z=capability_matrix,
                x=capabilities,
                y=expert_names,
                colorscale=[
                    [0, COLORS["background"]],
                    [0.5, COLORS["accent"]],
                    [1, COLORS["success"]],
                ],
                text=capability_matrix.round(2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Strength", tickfont=dict(color=COLORS["text"])),
            )
        )

        fig2.update_layout(
            title=dict(text="Expert Specialization Matrix", font=dict(color=COLORS["primary"])),
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            height=300 + discovered_n * 30,
        )

        st.plotly_chart(fig2, use_container_width=True)


def generate_expert_specializations(n: int) -> List[Dict]:
    """Generate synthetic expert specialization data"""
    focus_areas = [
        (
            "Code Specialist",
            "Code generation and debugging",
            ["Python", "JavaScript", "System design", "API development"],
        ),
        (
            "Math Reasoner",
            "Mathematical reasoning and proof",
            ["Algebra", "Calculus", "Logic", "Problem solving"],
        ),
        (
            "Language Expert",
            "Natural language understanding",
            ["Translation", "Summarization", "Sentiment", "Grammar"],
        ),
        (
            "Creative Writer",
            "Creative content generation",
            ["Storytelling", "Poetry", "Dialogue", "Narrative"],
        ),
        (
            "Analytical Thinker",
            "Data analysis and insights",
            ["Statistics", "Patterns", "Inference", "Prediction"],
        ),
        (
            "Tool Orchestrator",
            "Tool use and integration",
            ["API calls", "File ops", "Web search", "Execution"],
        ),
        (
            "Memory Manager",
            "Context and memory management",
            ["Recall", "Summarization", "Indexing", "Retrieval"],
        ),
        (
            "Reasoning Engine",
            "Complex multi-step reasoning",
            ["Chain-of-thought", "Planning", "Debugging", "Verification"],
        ),
    ]

    experts = []
    selected_areas = np.random.choice(len(focus_areas), size=n, replace=False)

    for i, idx in enumerate(selected_areas):
        name, focus, capabilities = focus_areas[idx]
        experts.append(
            {
                "id": i + 1,
                "name": name,
                "focus": focus,
                "capabilities": capabilities,
                "score": np.random.rand() * 0.3 + 0.7,  # 0.7-1.0
                "coverage": np.random.rand() * 0.2 + 0.15,  # 15-35%
            }
        )

    return experts


def render_svf_training_tab() -> None:
    """SVF training visualization (Stage 2)"""
    st.subheader("Stage 2: Transformer² SVF Training")

    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Training Progress", f"{st.session_state.get('stage2_progress', 0.0):.0%}")

    with col2:
        current_epoch = int(st.session_state.get("stage2_progress", 0.0) * 5)
        st.metric("Epoch", f"{current_epoch}/5")

    with col3:
        st.metric("Algorithm", "REINFORCE", delta="Active")

    with col4:
        fallback_active = st.session_state.get("svf_fallback", False)
        st.metric(
            "Fallback Status", "Active" if fallback_active else "Standby", delta="MuonGrokfast"
        )

    # Training loss curves
    st.markdown("#### Training Loss Curves")

    epochs = np.linspace(0, 5, 100)

    # REINFORCE loss (policy gradient)
    reinforce_loss = 2.5 * np.exp(-0.4 * epochs) + 0.3 + np.random.normal(0, 0.05, len(epochs))

    # Routing loss (expert assignment)
    routing_loss = 1.8 * np.exp(-0.3 * epochs) + 0.2 + np.random.normal(0, 0.03, len(epochs))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Policy Gradient Loss", "Routing Loss"))

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=reinforce_loss,
            mode="lines",
            name="REINFORCE",
            line=dict(color=COLORS["primary"], width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=routing_loss,
            mode="lines",
            name="Routing",
            line=dict(color=COLORS["secondary"], width=2),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        height=350,
        showlegend=False,
    )

    # Convert to rgba format (8-char hex not supported in Plotly)
    fig.update_xaxes(title_text="Epoch", gridcolor="rgba(0, 255, 255, 0.125)")
    fig.update_yaxes(title_text="Loss", gridcolor="rgba(0, 255, 255, 0.125)")

    st.plotly_chart(fig, use_container_width=True)

    # Singular value decomposition visualization
    st.markdown("#### Singular Value Decomposition")

    col1, col2 = st.columns(2)

    with col1:
        # Singular values
        n_experts = st.session_state.get("discovered_experts", 5)
        singular_values = np.sort(np.random.exponential(2.0, n_experts))[::-1]

        fig_sv = go.Figure(
            data=go.Bar(
                x=[f"SV {i+1}" for i in range(n_experts)],
                y=singular_values,
                marker=dict(
                    color=singular_values,
                    colorscale=[[0, COLORS["accent"]], [1, COLORS["primary"]]],
                    line=dict(color=COLORS["primary"], width=1),
                ),
            )
        )

        fig_sv.update_layout(
            title=dict(
                text="Singular Values (Expert Importance)", font=dict(color=COLORS["primary"])
            ),
            xaxis_title="Expert",
            yaxis_title="Singular Value",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            height=300,
        )

        st.plotly_chart(fig_sv, use_container_width=True)

    with col2:
        # Routing entropy
        epochs_entropy = np.arange(0, 5.1, 0.1)
        entropy = 2.0 - 0.3 * epochs_entropy + np.random.normal(0, 0.05, len(epochs_entropy))
        entropy = np.clip(entropy, 0.5, 2.0)

        fig_entropy = go.Figure(
            data=go.Scatter(
                x=epochs_entropy,
                y=entropy,
                mode="lines+markers",
                name="Routing Entropy",
                line=dict(color=COLORS["success"], width=2),
                marker=dict(size=4),
            )
        )

        fig_entropy.add_hline(
            y=1.0, line_dash="dash", line_color=COLORS["warning"], annotation_text="Optimal Range"
        )

        fig_entropy.update_layout(
            title=dict(text="Expert Routing Entropy", font=dict(color=COLORS["primary"])),
            xaxis_title="Epoch",
            yaxis_title="Entropy (nats)",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            height=300,
        )

        st.plotly_chart(fig_entropy, use_container_width=True)

    # Expert routing visualization
    st.markdown("#### Expert Routing Patterns")

    n_experts = st.session_state.get("discovered_experts", 5)
    n_tasks = 20

    # Generate routing probability matrix
    routing_probs = np.random.dirichlet(np.ones(n_experts), size=n_tasks)

    fig_routing = go.Figure(
        data=go.Heatmap(
            z=routing_probs.T,
            x=[f"Task {i+1}" for i in range(n_tasks)],
            y=[f"Expert {i+1}" for i in range(n_experts)],
            colorscale=[[0, COLORS["background"]], [0.5, COLORS["accent"]], [1, COLORS["primary"]]],
            colorbar=dict(title="Routing<br>Probability", tickfont=dict(color=COLORS["text"])),
        )
    )

    fig_routing.update_layout(
        title=dict(text="Task → Expert Routing Probabilities", font=dict(color=COLORS["primary"])),
        xaxis_title="Task Sample",
        yaxis_title="Expert",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        height=250 + n_experts * 30,
    )

    st.plotly_chart(fig_routing, use_container_width=True)


def render_adas_search_tab() -> None:
    """ADAS architecture search visualization (Stage 3)"""
    st.subheader("Stage 3: NSGA-II Architecture Search")

    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_gen = int(st.session_state.get("stage3_progress", 0.0) * 100)
        st.metric("Generation", f"{current_gen}/100")

    with col2:
        evaluations = current_gen * 50
        st.metric("Evaluations", f"{evaluations:,}/5,000")

    with col3:
        st.metric("Population Size", "50", delta="Per generation")

    with col4:
        pareto_size = min(int(current_gen * 0.3), 15)
        st.metric("Pareto Frontier", f"{pareto_size}", delta="Optimal architectures")

    # Evolution progress
    st.markdown("#### NSGA-II Evolution Progress")

    progress = st.session_state.get("stage3_progress", 0.0)
    generations = np.arange(0, int(progress * 100) + 1)

    # Fitness evolution
    best_fitness = (
        0.3 + 0.6 * (1 - np.exp(-0.05 * generations)) + np.random.normal(0, 0.02, len(generations))
    )
    avg_fitness = (
        0.2 + 0.4 * (1 - np.exp(-0.04 * generations)) + np.random.normal(0, 0.02, len(generations))
    )

    fig_evolution = go.Figure()

    fig_evolution.add_trace(
        go.Scatter(
            x=generations,
            y=best_fitness,
            mode="lines",
            name="Best Fitness",
            line=dict(color=COLORS["primary"], width=3),
        )
    )

    fig_evolution.add_trace(
        go.Scatter(
            x=generations,
            y=avg_fitness,
            mode="lines",
            name="Population Average",
            line=dict(color=COLORS["secondary"], width=2, dash="dash"),
        )
    )

    fig_evolution.update_layout(
        title=dict(text="Fitness Evolution Over Generations", font=dict(color=COLORS["primary"])),
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        height=350,
        hovermode="x unified",
    )

    st.plotly_chart(fig_evolution, use_container_width=True)

    # Pareto frontier
    st.markdown("#### Pareto Frontier (Multi-Objective Optimization)")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Generate Pareto frontier candidates
        n_candidates = min(int(progress * 100), 50)

        if n_candidates > 0:
            # Objective 1: Accuracy
            accuracy = np.random.beta(8, 2, n_candidates)

            # Objective 2: Efficiency (inverse relationship)
            efficiency = 1 - 0.7 * accuracy + np.random.normal(0, 0.05, n_candidates)
            efficiency = np.clip(efficiency, 0.2, 0.95)

            # Identify Pareto frontier
            is_pareto = np.zeros(n_candidates, dtype=bool)
            for i in range(n_candidates):
                is_dominated = False
                for j in range(n_candidates):
                    if i != j and accuracy[j] >= accuracy[i] and efficiency[j] >= efficiency[i]:
                        if accuracy[j] > accuracy[i] or efficiency[j] > efficiency[i]:
                            is_dominated = True
                            break
                is_pareto[i] = not is_dominated

            fig_pareto = go.Figure()

            # Non-Pareto candidates
            fig_pareto.add_trace(
                go.Scatter(
                    x=accuracy[~is_pareto],
                    y=efficiency[~is_pareto],
                    mode="markers",
                    name="Candidates",
                    marker=dict(
                        size=8,
                        color=COLORS["surface"],
                        line=dict(color=COLORS["secondary"], width=1),
                    ),
                )
            )

            # Pareto frontier
            pareto_x = accuracy[is_pareto]
            pareto_y = efficiency[is_pareto]
            sorted_indices = np.argsort(pareto_x)

            fig_pareto.add_trace(
                go.Scatter(
                    x=pareto_x[sorted_indices],
                    y=pareto_y[sorted_indices],
                    mode="markers+lines",
                    name="Pareto Frontier",
                    marker=dict(
                        size=12,
                        color=COLORS["primary"],
                        symbol="star",
                        line=dict(color=COLORS["success"], width=2),
                    ),
                    line=dict(color=COLORS["primary"], width=2, dash="dash"),
                )
            )

            fig_pareto.update_layout(
                title=dict(
                    text="Accuracy vs Efficiency Trade-off", font=dict(color=COLORS["primary"])
                ),
                xaxis_title="Accuracy",
                yaxis_title="Efficiency (FLOPs)",
                paper_bgcolor=COLORS["background"],
                plot_bgcolor=COLORS["surface"],
                font=dict(color=COLORS["text"]),
                height=400,
            )

            st.plotly_chart(fig_pareto, use_container_width=True)
        else:
            st.info("ADAS search not started yet...")

    with col2:
        st.markdown("**Optimization Objectives:**")
        st.markdown(
            f"""
        <div style='background: {COLORS['surface']}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <div style='color: {COLORS['primary']};'>1. Accuracy</div>
            <div style='font-size: 0.9rem; color: {COLORS['text']}80;'>Task performance</div>
        </div>
        <div style='background: {COLORS['surface']}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <div style='color: {COLORS['secondary']};'>2. Efficiency</div>
            <div style='font-size: 0.9rem; color: {COLORS['text']}80;'>FLOPs / latency</div>
        </div>
        <div style='background: {COLORS['surface']}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <div style='color: {COLORS['success']};'>3. Memory</div>
            <div style='font-size: 0.9rem; color: {COLORS['text']}80;'>Parameter count</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Architecture candidates grid
    st.markdown("#### Top Architecture Candidates")

    if pareto_size > 0:
        candidates = []
        for i in range(min(pareto_size, 6)):
            candidates.append(
                {
                    "ID": f"ARCH-{i+1:03d}",
                    "Accuracy": f"{np.random.uniform(0.85, 0.95):.3f}",
                    "Efficiency": f"{np.random.uniform(0.75, 0.90):.3f}",
                    "Memory": f"{np.random.uniform(0.70, 0.85):.3f}",
                    "Score": f"{np.random.uniform(0.80, 0.92):.3f}",
                }
            )

        df = pd.DataFrame(candidates)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.TextColumn("Architecture ID", width="small"),
                "Accuracy": st.column_config.TextColumn("Accuracy", width="small"),
                "Efficiency": st.column_config.TextColumn("Efficiency", width="small"),
                "Memory": st.column_config.TextColumn("Memory", width="small"),
                "Score": st.column_config.TextColumn("Overall Score", width="small"),
            },
        )


def render_architecture_tab() -> None:
    """Architecture visualization"""
    st.subheader("Discovered Architecture")

    n_experts = st.session_state.get("discovered_experts", 5)

    if n_experts > 0:
        # Architecture diagram
        st.markdown("#### Mixture-of-Experts Architecture")

        # Create network visualization
        fig = go.Figure()

        # Input layer
        fig.add_trace(
            go.Scatter(
                x=[0.1],
                y=[0.5],
                mode="markers+text",
                marker=dict(size=30, color=COLORS["primary"], symbol="circle"),
                text=["Input"],
                textposition="middle center",
                textfont=dict(color="white", size=10),
                name="Input",
                showlegend=False,
            )
        )

        # Routing layer
        fig.add_trace(
            go.Scatter(
                x=[0.35],
                y=[0.5],
                mode="markers+text",
                marker=dict(size=25, color=COLORS["secondary"], symbol="diamond"),
                text=["Router"],
                textposition="middle center",
                textfont=dict(color="white", size=9),
                name="Router",
                showlegend=False,
            )
        )

        # Expert layers
        expert_y = np.linspace(0.1, 0.9, n_experts)
        for i, y in enumerate(expert_y):
            fig.add_trace(
                go.Scatter(
                    x=[0.65],
                    y=[y],
                    mode="markers+text",
                    marker=dict(size=20, color=COLORS["success"], symbol="square"),
                    text=[f"E{i+1}"],
                    textposition="middle center",
                    textfont=dict(color="white", size=8),
                    name=f"Expert {i+1}",
                    showlegend=False,
                )
            )

        # Output layer
        fig.add_trace(
            go.Scatter(
                x=[0.9],
                y=[0.5],
                mode="markers+text",
                marker=dict(size=30, color=COLORS["primary"], symbol="circle"),
                text=["Output"],
                textposition="middle center",
                textfont=dict(color="white", size=10),
                name="Output",
                showlegend=False,
            )
        )

        # Connection lines
        # Input to router
        fig.add_trace(
            go.Scatter(
                x=[0.1, 0.35],
                y=[0.5, 0.5],
                mode="lines",
                line=dict(color=COLORS["primary"], width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Router to experts
        for y in expert_y:
            fig.add_trace(
                go.Scatter(
                    x=[0.35, 0.65],
                    y=[0.5, y],
                    mode="lines",
                    line=dict(color=COLORS["secondary"] + "40", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Experts to output
        for y in expert_y:
            fig.add_trace(
                go.Scatter(
                    x=[0.65, 0.9],
                    y=[y, 0.5],
                    mode="lines",
                    line=dict(color=COLORS["success"] + "40", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            title=dict(
                text=f"MoE Architecture (N={n_experts} Experts)", font=dict(color=COLORS["primary"])
            ),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1]),
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            height=400,
            hovermode=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Expert routing paths
        st.markdown("#### Expert Routing Paths")

        col1, col2 = st.columns(2)

        with col1:
            # Sankey diagram
            task_types = ["Code", "Math", "Language", "Creative"]
            source = []
            target = []
            value = []

            for i, task in enumerate(task_types):
                for j in range(n_experts):
                    source.append(i)
                    target.append(len(task_types) + j)
                    value.append(np.random.exponential(0.3))

            fig_sankey = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color=COLORS["primary"], width=0.5),
                            label=task_types + [f"Expert {i+1}" for i in range(n_experts)],
                            color=[COLORS["primary"]] * len(task_types)
                            + [COLORS["success"]] * n_experts,
                        ),
                        link=dict(
                            source=source,
                            target=target,
                            value=value,
                            color=COLORS["secondary"] + "30",
                        ),
                    )
                ]
            )

            fig_sankey.update_layout(
                title=dict(text="Task → Expert Flow", font=dict(color=COLORS["primary"])),
                paper_bgcolor=COLORS["background"],
                font=dict(color=COLORS["text"], size=10),
                height=300,
            )

            st.plotly_chart(fig_sankey, use_container_width=True)

        with col2:
            # Expert utilization
            utilization = np.random.exponential(0.2, n_experts)
            utilization = utilization / utilization.sum()

            fig_util = go.Figure(
                data=[
                    go.Pie(
                        labels=[f"Expert {i+1}" for i in range(n_experts)],
                        values=utilization,
                        marker=dict(colors=px.colors.sequential.Viridis_r[:n_experts]),
                        textinfo="label+percent",
                        textfont=dict(color="white", size=10),
                    )
                ]
            )

            fig_util.update_layout(
                title=dict(text="Expert Utilization", font=dict(color=COLORS["primary"])),
                paper_bgcolor=COLORS["background"],
                font=dict(color=COLORS["text"]),
                height=300,
            )

            st.plotly_chart(fig_util, use_container_width=True)
    else:
        st.info("Architecture will be displayed after expert discovery completes...")


def render_cost_time_tab() -> None:
    """Cost and time tracking"""
    st.subheader("Resource Usage & Budget")

    # Time tracking
    st.markdown("#### Time Breakdown")

    col1, col2 = st.columns([2, 1])

    with col1:
        stages_time = {
            "Stage 1: Self-Analysis": {
                "planned": 0.5,
                "actual": st.session_state.get("stage1_hours", 0.0),
            },
            "Stage 2: SVF Training": {
                "planned": 36,
                "actual": st.session_state.get("stage2_hours", 0.0),
            },
            "Stage 3: ADAS Search": {
                "planned": 42,
                "actual": st.session_state.get("stage3_hours", 0.0),
            },
        }

        stage_names = list(stages_time.keys())
        planned_hours = [stages_time[s]["planned"] for s in stage_names]
        actual_hours = [stages_time[s]["actual"] for s in stage_names]

        fig_time = go.Figure()

        fig_time.add_trace(
            go.Bar(name="Planned", x=stage_names, y=planned_hours, marker_color=COLORS["secondary"])
        )

        fig_time.add_trace(
            go.Bar(name="Actual", x=stage_names, y=actual_hours, marker_color=COLORS["primary"])
        )

        fig_time.update_layout(
            title=dict(text="Time Allocation (Hours)", font=dict(color=COLORS["primary"])),
            xaxis_title="Stage",
            yaxis_title="Hours",
            barmode="group",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            height=300,
        )

        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        total_planned = sum(planned_hours)
        total_actual = sum(actual_hours)

        st.metric("Total Planned", f"{total_planned:.1f}h")
        st.metric(
            "Total Actual", f"{total_actual:.1f}h", delta=f"{total_actual - total_planned:+.1f}h"
        )
        st.metric("Remaining", f"{max(0, total_planned - total_actual):.1f}h")

    # Cost tracking
    st.markdown("#### API Cost Breakdown")

    col1, col2 = st.columns([2, 1])

    with col1:
        stages_cost = {
            "Stage 1: Self-Analysis": {
                "budget": 40,
                "actual": st.session_state.get("stage1_cost", 0.0),
            },
            "Stage 2: SVF Training": {
                "budget": 120,
                "actual": st.session_state.get("stage2_cost", 0.0),
            },
            "Stage 3: ADAS Search": {
                "budget": 90,
                "actual": st.session_state.get("stage3_cost", 0.0),
            },
        }

        stage_names_cost = list(stages_cost.keys())
        budget_cost = [stages_cost[s]["budget"] for s in stage_names_cost]
        actual_cost = [stages_cost[s]["actual"] for s in stage_names_cost]

        fig_cost = go.Figure()

        fig_cost.add_trace(
            go.Bar(name="Budget", x=stage_names_cost, y=budget_cost, marker_color=COLORS["accent"])
        )

        fig_cost.add_trace(
            go.Bar(name="Actual", x=stage_names_cost, y=actual_cost, marker_color=COLORS["success"])
        )

        fig_cost.update_layout(
            title=dict(text="Cost Allocation ($)", font=dict(color=COLORS["primary"])),
            xaxis_title="Stage",
            yaxis_title="Cost ($)",
            barmode="group",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            height=300,
        )

        st.plotly_chart(fig_cost, use_container_width=True)

    with col2:
        total_budget = sum(budget_cost)
        total_spent = sum(actual_cost)

        st.metric("Total Budget", f"${total_budget:.0f}")
        st.metric("Total Spent", f"${total_spent:.2f}", delta=f"${total_spent - total_budget:+.2f}")
        st.metric("Remaining", f"${max(0, total_budget - total_spent):.2f}")

    # Frontier model usage
    st.markdown("#### Frontier Model Usage")

    models_usage = {
        "GPT-4o-mini": {"calls": np.random.randint(1000, 2000), "cost": np.random.uniform(40, 80)},
        "Claude-3.5 Haiku": {
            "calls": np.random.randint(800, 1500),
            "cost": np.random.uniform(30, 60),
        },
        "Gemini 2.0 Flash": {
            "calls": np.random.randint(500, 1000),
            "cost": np.random.uniform(20, 40),
        },
        "Qwen 2.5": {"calls": np.random.randint(300, 800), "cost": np.random.uniform(10, 30)},
    }

    models_df = pd.DataFrame(
        [
            {
                "Model": model,
                "API Calls": data["calls"],
                "Cost ($)": f"${data['cost']:.2f}",
                "Avg Cost/Call": f"${data['cost']/data['calls']*1000:.3f}",
            }
            for model, data in models_usage.items()
        ]
    )

    st.dataframe(models_df, use_container_width=True, hide_index=True)


def render_results_tab() -> None:
    """Results and final model"""
    st.subheader("Phase 7 Results")

    phase_complete = st.session_state.get("stage3_complete", False)

    if phase_complete:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Final Expert Count", st.session_state.get("discovered_experts", 5))

        with col2:
            st.metric("Best Architecture", "ARCH-042", delta="Pareto optimal")

        with col3:
            st.metric("Overall Score", "0.917", delta="+0.152 vs baseline")

        with col4:
            st.metric("Phase Status", "COMPLETE", delta="Ready for Phase 8")

        # Performance comparison
        st.markdown("#### Performance vs Baseline")

        metrics = ["Accuracy", "Efficiency", "Memory", "Routing Speed", "Overall"]
        baseline = [0.765, 0.682, 0.701, 0.650, 0.699]
        phase7 = [0.891, 0.823, 0.779, 0.945, 0.860]

        fig_comparison = go.Figure()

        fig_comparison.add_trace(
            go.Bar(name="Baseline", x=metrics, y=baseline, marker_color=COLORS["secondary"])
        )

        fig_comparison.add_trace(
            go.Bar(name="Phase 7 MoE", x=metrics, y=phase7, marker_color=COLORS["primary"])
        )

        fig_comparison.update_layout(
            title=dict(
                text="Baseline vs Expert-Enhanced Model", font=dict(color=COLORS["primary"])
            ),
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            barmode="group",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            height=350,
        )

        st.plotly_chart(fig_comparison, use_container_width=True)

        # Export information
        st.markdown("#### Output Artifacts")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
            <div style='background: {COLORS['surface']}; padding: 1rem; border-radius: 8px; border: 1px solid {COLORS['primary']}40;'>
                <h4 style='color: {COLORS['primary']};'>Model Export</h4>
                <ul style='color: {COLORS['text']};'>
                    <li>Expert-enhanced model (50.2 MB)</li>
                    <li>{st.session_state.get('discovered_experts', 5)} specialized experts</li>
                    <li>SVF routing weights</li>
                    <li>Architecture config (ARCH-042)</li>
                    <li>Ready for Phase 8 compression</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div style='background: {COLORS['surface']}; padding: 1rem; border-radius: 8px; border: 1px solid {COLORS['secondary']}40;'>
                <h4 style='color: {COLORS['secondary']};'>Metadata</h4>
                <ul style='color: {COLORS['text']};'>
                    <li>Expert specializations map</li>
                    <li>Routing statistics</li>
                    <li>Pareto frontier archive (15 archs)</li>
                    <li>Training logs (SVF + ADAS)</li>
                    <li>Cost breakdown report</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Success message
        st.success(
            f"""
        ✓ **Phase 7 Complete!**

        - {st.session_state.get('discovered_experts', 5)} experts discovered via self-analysis
        - SVF training complete (36 hours)
        - ADAS optimization complete (5,000 evaluations)
        - Best architecture selected (ARCH-042)
        - Ready for Phase 8 final compression
        """
        )

    else:
        st.info("Results will be displayed when Phase 7 completes all three stages.")

        # Progress checklist
        st.markdown("#### Completion Checklist")

        checklist = [
            ("Stage 1: Self-Analysis", st.session_state.get("stage1_complete", False)),
            ("Stage 2: SVF Training", st.session_state.get("stage2_complete", False)),
            ("Stage 3: ADAS Search", st.session_state.get("stage3_complete", False)),
            ("Expert routing validated", False),
            ("Architecture optimized", False),
            ("Phase 8 handoff prepared", False),
        ]

        for item, complete in checklist:
            icon = "✓" if complete else "○"
            color = COLORS["success"] if complete else COLORS["text"] + "60"
            st.markdown(f"<div style='color: {color};'>{icon} {item}</div>", unsafe_allow_html=True)


def reset_session_state() -> None:
    """Reset all session state variables"""
    keys_to_reset = [
        "phase7_running",
        "stage1_complete",
        "stage2_complete",
        "stage3_complete",
        "stage1_progress",
        "stage2_progress",
        "stage3_progress",
        "discovered_experts",
        "elapsed_hours",
        "api_cost",
        "svf_fallback",
        "stage1_hours",
        "stage2_hours",
        "stage3_hours",
        "stage1_cost",
        "stage2_cost",
        "stage3_cost",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


# Initialize session state
if "phase7_running" not in st.session_state:
    st.session_state.phase7_running = False

if "stage1_complete" not in st.session_state:
    st.session_state.stage1_complete = False

if "stage2_complete" not in st.session_state:
    st.session_state.stage2_complete = False

if "stage3_complete" not in st.session_state:
    st.session_state.stage3_complete = False

if "stage1_progress" not in st.session_state:
    st.session_state.stage1_progress = 0.0

if "stage2_progress" not in st.session_state:
    st.session_state.stage2_progress = 0.0

if "stage3_progress" not in st.session_state:
    st.session_state.stage3_progress = 0.0

if "discovered_experts" not in st.session_state:
    st.session_state.discovered_experts = 0

if "elapsed_hours" not in st.session_state:
    st.session_state.elapsed_hours = 0.0

if "api_cost" not in st.session_state:
    st.session_state.api_cost = 0.0


# Main entry point
if __name__ == "__main__":
    render_phase7_dashboard()

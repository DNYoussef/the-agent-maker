"""
Phase 5: Curriculum Learning - Comprehensive Streamlit Dashboard

7-stage adaptive curriculum with frontier models, edge-of-chaos assessment,
tool use training, eudaimonia baking, self-modeling, and dream consolidation.

Futuristic theme with dark background and cyan accents.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Custom Plotly Theme (Dark with Cyan Accents)
CUSTOM_PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#1B2838",
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            size=12,
            color="#E0E1DD",
        ),
        title=dict(font=dict(size=18, color="#00F5D4", family="Inter"), x=0.5, xanchor="center"),
        xaxis=dict(
            gridcolor="#2E3F4F", zerolinecolor="#2E3F4F", color="#E0E1DD", linecolor="#2E3F4F"
        ),
        yaxis=dict(
            gridcolor="#2E3F4F", zerolinecolor="#2E3F4F", color="#E0E1DD", linecolor="#2E3F4F"
        ),
        colorway=["#00F5D4", "#FF006E", "#8338EC", "#FB5607", "#FFBE0B"],
        hovermode="closest",
        margin=dict(l=60, r=40, t=60, b=60),
    )
)


def create_gradient_metric(
    label: str,
    value: str,
    delta: Optional[str] = None,
    gradient_start: str = "#00F5D4",
    gradient_end: str = "#8338EC",
):
    """Create a metric card with gradient background"""
    delta_html = (
        f'<div style="color: #00F5D4; font-size: 14px; margin-top: 5px;">{delta}</div>'
        if delta
        else ""
    )

    html = f"""
    <div style="
        background: linear-gradient(135deg, {gradient_start}22 0%, {gradient_end}22 100%);
        border: 1px solid {gradient_start}44;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 245, 212, 0.1);
        transition: all 0.3s ease;
    ">
        <div style="color: #8B9DAF; font-size: 12px; text-transform: uppercase;
                    letter-spacing: 1px; margin-bottom: 8px;">
            {label}
        </div>
        <div style="color: #FFFFFF; font-size: 32px; font-weight: 700;
                    font-family: 'Inter', sans-serif;">
            {value}
        </div>
        {delta_html}
    </div>
    """
    return html


def create_curriculum_stages_visual(current_stage: int = 3) -> go.Figure:
    """Create visual representation of 7-stage curriculum with progress"""
    stages = [
        {
            "name": "Foundation",
            "level": 1,
            "questions": 2000,
            "difficulty": 2.5,
            "status": "complete",
        },
        {"name": "Basics", "level": 2, "questions": 2000, "difficulty": 3.5, "status": "complete"},
        {
            "name": "Intermediate",
            "level": 3,
            "questions": 2000,
            "difficulty": 5.0,
            "status": "in_progress",
        },
        {"name": "Advanced", "level": 4, "questions": 2000, "difficulty": 6.5, "status": "pending"},
        {"name": "Expert", "level": 5, "questions": 2000, "difficulty": 7.5, "status": "pending"},
        {"name": "Master", "level": 6, "questions": 2000, "difficulty": 8.5, "status": "pending"},
        {
            "name": "Frontier",
            "level": 7,
            "questions": 2000,
            "difficulty": 10.0,
            "status": "pending",
        },
    ]

    fig = go.Figure()

    # Add connecting path
    x_positions = list(range(len(stages)))
    y_difficulties = [s["difficulty"] for s in stages]

    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=y_difficulties,
            mode="lines",
            line=dict(color="#00F5D4", width=4, shape="spline"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add stage nodes with different colors based on status
    for i, stage in enumerate(stages):
        status_colors = {"complete": "#00F5D4", "in_progress": "#FF006E", "pending": "#4A5568"}
        color = status_colors[cast(str, stage["status"])]

        # Larger marker for in_progress
        marker_size = 35 if cast(str, stage["status"]) == "in_progress" else 25

        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[stage["difficulty"]],
                mode="markers+text",
                marker=dict(
                    size=marker_size,
                    color=color,
                    line=dict(color="#FFFFFF", width=3)
                    if cast(str, stage["status"]) == "in_progress"
                    else dict(color="#2E3F4F", width=2),
                    symbol="circle",
                ),
                text=[f"L{stage['level']}"],
                textfont=dict(size=14, color="#FFFFFF", family="Inter"),
                textposition="middle center",
                name=stage["name"],
                hovertemplate=f"<b>{stage['name']}</b><br>Difficulty: {stage['difficulty']}/10<br>Questions: {stage['questions']:,}<extra></extra>",
                showlegend=False,
            )
        )

        # Add stage labels below
        completion = (
            "100%"
            if cast(str, stage["status"]) == "complete"
            else "75%"
            if cast(str, stage["status"]) == "in_progress"
            else "0%"
        )
        fig.add_annotation(
            x=i,
            y=cast(float, stage["difficulty"]) - 0.8,
            text=f"<b>{stage['name']}</b><br>{completion}",
            showarrow=False,
            font=dict(size=10, color=color),
            xanchor="center",
        )

    # Add optimal difficulty zone (edge of chaos)
    fig.add_hrect(
        y0=7.0,
        y1=8.0,
        fillcolor="rgba(0, 245, 212, 0.1)",
        line_width=0,
        annotation_text="Edge-of-Chaos Zone",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#00F5D4"),
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="7-Stage Adaptive Curriculum Progress",
        xaxis=dict(
            tickvals=x_positions, ticktext=[s["name"] for s in stages], title="Curriculum Stage"
        ),
        yaxis=dict(title="Difficulty Level", range=[1, 11]),
        height=400,
        showlegend=False,
    )

    return fig


def create_edge_of_chaos_gauge(current_difficulty: float = 7.5, accuracy: float = 0.76) -> go.Figure:
    """Create gauge for edge-of-chaos assessment"""
    # Determine if in optimal zone
    in_zone = 7.0 <= current_difficulty <= 8.5 and 0.70 <= accuracy <= 0.80

    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_difficulty,
            delta={"reference": 7.5, "increasing": {"color": "#00F5D4"}},
            gauge={
                "axis": {"range": [1, 10], "tickwidth": 1, "tickcolor": "#2E3F4F"},
                "bar": {"color": "#00F5D4" if in_zone else "#FF006E", "thickness": 0.75},
                "bgcolor": "#1B2838",
                "borderwidth": 2,
                "bordercolor": "#00F5D4" if in_zone else "#FF006E",
                "steps": [
                    {"range": [1, 6.5], "color": "#4A5568"},
                    {"range": [6.5, 7.0], "color": "#8338EC"},
                    {"range": [7.0, 8.5], "color": "rgba(0, 245, 212, 0.3)"},
                    {"range": [8.5, 10], "color": "#FF006E"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 4},
                    "thickness": 0.75,
                    "value": 7.5,
                },
            },
            number={"suffix": "/10", "font": {"size": 28, "color": "#FFFFFF"}},
            title={"text": "Current Difficulty", "font": {"size": 16, "color": "#00F5D4"}},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE, height=300, margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def create_accuracy_threshold_indicator(accuracy: float = 0.76) -> go.Figure:
    """Create accuracy threshold indicator with 75% target"""
    fig = go.Figure()

    # Create horizontal bar showing accuracy
    fig.add_trace(
        go.Bar(
            x=[accuracy * 100],
            y=["Accuracy"],
            orientation="h",
            marker=dict(
                color="#00F5D4" if accuracy >= 0.75 else "#FF006E",
                line=dict(color="#FFFFFF", width=2),
            ),
            text=[f"{accuracy*100:.1f}%"],
            textposition="inside",
            textfont=dict(size=18, color="#FFFFFF"),
            hovertemplate=f"<b>Current Accuracy</b><br>{accuracy*100:.1f}%<extra></extra>",
        )
    )

    # Add target line
    fig.add_vline(
        x=75,
        line_dash="dash",
        line_color="#00F5D4",
        line_width=3,
        annotation_text="Target: 75%",
        annotation_position="top right",
        annotation_font=dict(color="#00F5D4", size=12),
    )

    # Add acceptable range
    fig.add_vrect(
        x0=70,
        x1=80,
        fillcolor="rgba(0, 245, 212, 0.1)",
        line_width=0,
        annotation_text="Optimal Zone",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#8B9DAF"),
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Accuracy vs 75% Threshold",
        xaxis=dict(title="Accuracy (%)", range=[0, 100]),
        yaxis=dict(visible=False),
        height=200,
        showlegend=False,
    )

    return fig


def create_adaptive_curriculum_chart(level_data: pd.DataFrame) -> go.Figure:
    """Create chart showing 20,000 questions across 10 difficulty levels"""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Questions Distribution", "Performance by Level"),
        vertical_spacing=0.15,
        row_heights=[0.4, 0.6],
    )

    # Top: Questions distribution (bar chart)
    colors = [
        "#00F5D4" if status == "complete" else "#FF006E" if status == "in_progress" else "#4A5568"
        for status in level_data["Status"]
    ]

    fig.add_trace(
        go.Bar(
            x=level_data["Level"],
            y=level_data["Questions"],
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            text=level_data["Questions"],
            textposition="outside",
            name="Questions",
            hovertemplate="<b>Level %{x}</b><br>Questions: %{y:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Bottom: Performance metrics (line chart with multiple metrics)
    fig.add_trace(
        go.Scatter(
            x=level_data["Level"],
            y=level_data["Accuracy"] * 100,
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#00F5D4", width=3),
            marker=dict(size=10, line=dict(color="#FFFFFF", width=2)),
            hovertemplate="<b>Level %{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=level_data["Level"],
            y=level_data["Difficulty"] * 10,
            mode="lines+markers",
            name="Difficulty",
            line=dict(color="#8338EC", width=3, dash="dot"),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="<b>Level %{x}</b><br>Difficulty: %{y:.1f}/10<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add 75% threshold line
    fig.add_hline(
        y=75,
        line_dash="dash",
        line_color="#FF006E",
        line_width=2,
        row=2,
        col=1,
        annotation_text="75% Threshold",
        annotation_position="right",
    )

    fig.update_xaxes(title_text="Difficulty Level", row=2, col=1)
    fig.update_yaxes(title_text="Questions", row=1, col=1)
    fig.update_yaxes(title_text="Percentage / Score", row=2, col=1)

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        height=600,
        showlegend=True,
        legend=dict(bgcolor="rgba(27, 40, 56, 0.8)", bordercolor="#2E3F4F", borderwidth=1),
    )

    return fig


def create_tool_use_metrics() -> go.Figure:
    """Create tool use training metrics visualization"""
    tools = ["Code Exec", "Validation", "Debug", "Test Gen", "Refactor"]
    proficiency = [0.85, 0.78, 0.72, 0.68, 0.65]
    success_rate = [0.92, 0.87, 0.81, 0.75, 0.70]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=tools,
            y=[p * 100 for p in proficiency],
            name="Proficiency",
            marker=dict(color="#00F5D4", line=dict(color="#FFFFFF", width=1)),
            text=[f"{p*100:.1f}%" for p in proficiency],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Proficiency: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=tools,
            y=[s * 100 for s in success_rate],
            name="Success Rate",
            marker=dict(color="#8338EC", line=dict(color="#FFFFFF", width=1)),
            text=[f"{s*100:.1f}%" for s in success_rate],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Success Rate: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Tool Use Training Progress",
        xaxis_title="Tool Type",
        yaxis=dict(title="Percentage (%)", range=[0, 100]),
        height=350,
        barmode="group",
    )

    return fig


def create_eudaimonia_radar() -> go.Figure:
    """Create radar chart for 4-rule moral system"""
    categories = ["Benevolence", "Non-Harm", "Respect", "Autonomy"]
    values = [0.89, 0.94, 0.87, 0.82]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(0, 245, 212, 0.2)",
            line=dict(color="#00F5D4", width=3),
            marker=dict(size=10, color="#00F5D4", line=dict(color="#FFFFFF", width=2)),
            name="Compliance",
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>",
        )
    )

    # Add target circle at 0.80
    fig.add_trace(
        go.Scatterpolar(
            r=[0.80] * len(categories),
            theta=categories,
            mode="lines",
            line=dict(color="#FF006E", width=2, dash="dash"),
            name="Target (0.80)",
            showlegend=True,
        )
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1], gridcolor="#2E3F4F", tickfont=dict(color="#E0E1DD")
            ),
            angularaxis=dict(
                gridcolor="#2E3F4F", linecolor="#2E3F4F", tickfont=dict(color="#E0E1DD")
            ),
            bgcolor="#1B2838",
        ),
        title="Eudaimonia 4-Rule Moral System",
        height=400,
        showlegend=True,
        legend=dict(bgcolor="rgba(27, 40, 56, 0.8)", bordercolor="#2E3F4F", borderwidth=1),
    )

    return fig


def create_self_modeling_chart() -> go.Figure:
    """Create self-modeling temperature prediction chart"""
    temperatures = np.linspace(0.0, 2.0, 21)
    actual_performance = 0.85 * np.exp(-((temperatures - 0.7) ** 2) / 0.5) + 0.15
    predicted_performance = actual_performance + np.random.normal(0, 0.03, len(temperatures))

    fig = go.Figure()

    # Actual performance
    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=actual_performance,
            mode="lines",
            name="Actual Performance",
            line=dict(color="#00F5D4", width=3),
            fill="tonexty",
            fillcolor="rgba(0, 245, 212, 0.1)",
            hovertemplate="<b>Temp: %{x:.2f}</b><br>Actual: %{y:.3f}<extra></extra>",
        )
    )

    # Predicted performance
    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=predicted_performance,
            mode="markers",
            name="Self-Prediction",
            marker=dict(
                size=8, color="#8338EC", symbol="diamond", line=dict(color="#FFFFFF", width=1)
            ),
            hovertemplate="<b>Temp: %{x:.2f}</b><br>Predicted: %{y:.3f}<extra></extra>",
        )
    )

    # Highlight optimal range
    fig.add_vrect(
        x0=0.6,
        x1=0.9,
        fillcolor="rgba(0, 245, 212, 0.1)",
        line_width=0,
        annotation_text="Optimal Range",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#8B9DAF"),
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Self-Modeling: Temperature Range Prediction",
        xaxis_title="Temperature",
        yaxis_title="Performance Score",
        height=350,
        hovermode="x unified",
    )

    return fig


def create_dream_consolidation_metrics() -> go.Figure:
    """Create dream consolidation visualization with autoencoder quality"""
    epochs_per_level = 3
    levels_completed = 3
    total_epochs = epochs_per_level * 10  # 10 levels

    epochs = np.arange(1, levels_completed * epochs_per_level + 1)
    reconstruction_quality = 0.95 - 0.02 * np.random.rand(len(epochs))
    forgetting_metric = 1.0 - (0.05 * np.arange(len(epochs)) + 0.02 * np.random.rand(len(epochs)))

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Autoencoder Reconstruction Quality", "Catastrophic Forgetting Prevention"),
        vertical_spacing=0.15,
    )

    # Reconstruction quality
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=reconstruction_quality,
            mode="lines+markers",
            name="Reconstruction",
            line=dict(color="#00F5D4", width=3),
            marker=dict(size=8, line=dict(color="#FFFFFF", width=2)),
            hovertemplate="<b>Epoch %{x}</b><br>Quality: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add target line
    fig.add_hline(
        y=0.90,
        line_dash="dash",
        line_color="#FF006E",
        line_width=2,
        row=1,
        col=1,
        annotation_text="Target: 0.90",
        annotation_position="right",
    )

    # Forgetting metric
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=forgetting_metric,
            mode="lines+markers",
            name="Retention",
            line=dict(color="#8338EC", width=3),
            marker=dict(size=8, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
            fill="tonexty",
            fillcolor="rgba(131, 56, 236, 0.1)",
            hovertemplate="<b>Epoch %{x}</b><br>Retention: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add critical threshold
    fig.add_hline(
        y=0.85,
        line_dash="dash",
        line_color="#FF006E",
        line_width=2,
        row=2,
        col=1,
        annotation_text="Critical: 0.85",
        annotation_position="right",
    )

    fig.update_xaxes(title_text="Training Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Quality Score", row=1, col=1)
    fig.update_yaxes(title_text="Retention Score", row=2, col=1)

    fig.update_layout(template=CUSTOM_PLOTLY_TEMPLATE, height=550, showlegend=False)

    return fig


def create_frontier_models_usage() -> go.Figure:
    """Create visualization of frontier model usage and costs"""
    models = ["GPT-4o-mini", "Claude-3.5\nHaiku", "Gemini 2.0\nFlash", "Qwen 2.5"]
    api_calls = [4500, 3800, 4200, 3500]
    costs = [285, 190, 165, 95]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("API Usage Distribution", "Cost Breakdown"),
        specs=[[{"type": "bar"}, {"type": "pie"}]],
    )

    # Bar chart for API calls
    colors_bar = ["#00F5D4", "#8338EC", "#FF006E", "#FB5607"]
    fig.add_trace(
        go.Bar(
            x=models,
            y=api_calls,
            marker=dict(color=colors_bar, line=dict(color="#FFFFFF", width=1)),
            text=[f"{c:,}" for c in api_calls],
            textposition="outside",
            name="API Calls",
            hovertemplate="<b>%{x}</b><br>Calls: %{y:,}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Pie chart for costs
    fig.add_trace(
        go.Pie(
            labels=models,
            values=costs,
            marker=dict(colors=colors_bar, line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color="#FFFFFF"),
            hovertemplate="<b>%{label}</b><br>Cost: $%{value}<br>%{percent}<extra></extra>",
            hole=0.4,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_yaxes(title_text="API Calls", row=1, col=1)

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        height=350,
        showlegend=False,
        annotations=[
            dict(
                text=f"${sum(costs)}",
                x=0.825,
                y=0.5,
                font=dict(size=20, color="#00F5D4"),
                showarrow=False,
                xref="paper",
                yref="paper",
            )
        ],
    )

    return fig


def render_phase5_dashboard() -> None:
    """Main dashboard for Phase 5 Curriculum Learning"""
    st.title("📚 Phase 5: Curriculum Learning")
    st.markdown(
        "**7-stage adaptive curriculum** → Edge-of-Chaos → Frontier models → Tool use → Ethics baking"
    )

    # Custom CSS for enhanced styling
    st.markdown(
        """
    <style>
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1B2838;
        border-radius: 8px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #8B9DAF;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
        color: #00F5D4;
        border: 1px solid #00F5D444;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid #00F5D444;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 245, 212, 0.3);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00F5D4 0%, #8338EC 100%);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        render_config_panel()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Overview",
            "📚 Curriculum Stages",
            "🎯 Edge-of-Chaos",
            "🔧 Tool Use & Ethics",
            "🧠 Self-Modeling & Dreams",
            "💰 Frontier Models",
        ]
    )

    with tab1:
        render_overview_tab()

    with tab2:
        render_curriculum_stages_tab()

    with tab3:
        render_edge_of_chaos_tab()

    with tab4:
        render_tool_ethics_tab()

    with tab5:
        render_self_modeling_tab()

    with tab6:
        render_frontier_models_tab()


def render_config_panel() -> None:
    """Render configuration controls in sidebar"""
    st.markdown("### 🎛️ Curriculum Settings")

    current_stage = st.slider(
        "Current Stage", min_value=1, max_value=7, value=3, help="7-stage adaptive curriculum"
    )

    st.session_state.current_stage = current_stage

    accuracy_threshold = st.slider(
        "Accuracy Threshold",
        min_value=0.60,
        max_value=0.90,
        value=0.75,
        step=0.05,
        help="Target accuracy for progression (75% optimal)",
    )

    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
                border: 1px solid #00F5D444; border-radius: 8px; padding: 12px; text-align: center;">
        <div style="color: #8B9DAF; font-size: 11px; text-transform: uppercase;">Threshold</div>
        <div style="color: #00F5D4; font-size: 28px; font-weight: 700;">{accuracy_threshold*100:.0f}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🎯 Training Mode")

    training_mode = st.selectbox(
        "Mode",
        ["Adaptive", "Fixed Difficulty", "Manual"],
        help="How difficulty adjusts during training",
    )

    if training_mode == "Adaptive":
        st.info("📊 Auto-adjusts difficulty based on 75% accuracy target")

    enable_dream = st.checkbox(
        "Enable Dream Consolidation", value=True, help="3 epochs per level with T=1.2 replay"
    )

    if enable_dream:
        st.success("✓ Prevents catastrophic forgetting")

    st.markdown("---")
    st.markdown("### 🤖 Frontier Models")

    models = st.multiselect(
        "Active Models",
        ["GPT-4o-mini", "Claude-3.5 Haiku", "Gemini 2.0 Flash", "Qwen 2.5"],
        default=["GPT-4o-mini", "Claude-3.5 Haiku"],
        help="Select frontier models for data generation",
    )

    budget_limit = st.number_input(
        "Budget Limit ($)",
        min_value=100,
        max_value=1000,
        value=700,
        step=50,
        help="OpenRouter API budget ($600-800 recommended)",
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Start", type="primary", use_container_width=True):
            st.session_state.training_running = True
            st.rerun()

    with col2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.training_running = False
            st.rerun()


def render_overview_tab() -> None:
    """Overview tab with hero metrics and key stats"""
    # Hero section
    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #1B2838 0%, #0D1B2A 100%);
                border: 2px solid #00F5D4; border-radius: 16px; padding: 30px; margin-bottom: 30px;
                box-shadow: 0 8px 24px rgba(0, 245, 212, 0.2);">
        <h2 style="color: #00F5D4; margin-top: 0; font-size: 32px;">Phase 5: Curriculum Learning</h2>
        <p style="color: #E0E1DD; font-size: 16px; line-height: 1.6;">
            7-stage adaptive curriculum with frontier models (GPT-4o-mini, Claude-3.5 Haiku, Gemini 2.0 Flash, Qwen 2.5).
            Edge-of-chaos assessment finds optimal difficulty (75% accuracy). Dream consolidation prevents catastrophic forgetting.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            create_gradient_metric(
                "Current Stage",
                f"{st.session_state.get('current_stage', 3)}/7",
                "Intermediate",
                gradient_start="#00F5D4",
                gradient_end="#8338EC",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            create_gradient_metric(
                "Questions",
                "6,000/20,000",
                "30% Complete",
                gradient_start="#8338EC",
                gradient_end="#FF006E",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            create_gradient_metric(
                "Accuracy",
                "76.2%",
                "Above 75% threshold",
                gradient_start="#FF006E",
                gradient_end="#FB5607",
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            create_gradient_metric(
                "API Cost",
                "$427",
                "$273 remaining",
                gradient_start="#FB5607",
                gradient_end="#FFBE0B",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Progress overview
    st.markdown("### 📊 Training Progress")

    progress_col1, progress_col2 = st.columns([3, 1])

    with progress_col1:
        overall_progress = 0.30
        st.progress(overall_progress)
        st.caption(f"Overall Progress: {overall_progress*100:.1f}%")

    with progress_col2:
        time_remaining = "18-36 hrs"
        st.metric("Time Remaining", time_remaining)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div style="background: linear-gradient(135deg, #00F5D411 0%, #8338EC11 100%);
                    border: 1px solid #00F5D422; border-radius: 12px; padding: 20px;">
            <h4 style="color: #00F5D4; margin-top: 0;">🎯 Adaptive Curriculum Features</h4>
            <ul style="color: #E0E1DD; line-height: 1.8;">
                <li>✅ 7 stages × 2,000 questions each</li>
                <li>✅ 10 difficulty levels (1-10 scale)</li>
                <li>✅ Edge-of-Chaos assessment (75% accuracy)</li>
                <li>✅ Tool use training (code execution)</li>
                <li>✅ Eudaimonia 4-rule moral system</li>
                <li>✅ Self-modeling (temperature prediction)</li>
                <li>✅ Dream consolidation (T=1.2 replay)</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div style="background: linear-gradient(135deg, #8338EC11 0%, #FF006E11 100%);
                    border: 1px solid #8338EC22; border-radius: 12px; padding: 20px;">
            <h4 style="color: #8338EC; margin-top: 0;">🤖 Frontier Model Integration</h4>
            <ul style="color: #E0E1DD; line-height: 1.8;">
                <li>🔷 <b>GPT-4o-mini</b> - 4,500 calls ($285)</li>
                <li>🔶 <b>Claude-3.5 Haiku</b> - 3,800 calls ($190)</li>
                <li>🔵 <b>Gemini 2.0 Flash</b> - 4,200 calls ($165)</li>
                <li>🔴 <b>Qwen 2.5</b> - 3,500 calls ($95)</li>
                <li>💰 Total budget: <b style="color: #00F5D4;">$600-800</b></li>
                <li>⏱️ Training time: <b style="color: #00F5D4;">120-240 hours</b></li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Timeline
    st.markdown("### ⏱️ Expected Timeline")

    timeline_data = pd.DataFrame(
        {
            "Stage": [f"Stage {i}" for i in range(1, 8)],
            "Duration (hours)": [12, 14, 16, 18, 20, 24, 30],
            "Cost ($)": [60, 70, 80, 95, 110, 130, 155],
        }
    )

    fig_timeline = go.Figure()

    fig_timeline.add_trace(
        go.Bar(
            x=timeline_data["Stage"],
            y=timeline_data["Duration (hours)"],
            name="Time",
            marker=dict(color="#00F5D4", line=dict(color="#FFFFFF", width=1)),
            text=timeline_data["Duration (hours)"],
            textposition="outside",
            yaxis="y",
            hovertemplate="<b>%{x}</b><br>Time: %{y} hours<extra></extra>",
        )
    )

    fig_timeline.add_trace(
        go.Scatter(
            x=timeline_data["Stage"],
            y=timeline_data["Cost ($)"],
            name="Cost",
            mode="lines+markers",
            line=dict(color="#FF006E", width=3),
            marker=dict(size=10, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Cost: $%{y}<extra></extra>",
        )
    )

    fig_timeline.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Training Timeline & Cost Projection",
        xaxis_title="Curriculum Stage",
        yaxis=dict(title="Duration (hours)", side="left"),
        yaxis2=dict(title="Cost ($)", side="right", overlaying="y"),
        height=350,
        hovermode="x unified",
    )

    st.plotly_chart(fig_timeline, use_container_width=True)


def render_curriculum_stages_tab() -> None:
    """Curriculum stages tab with 7-stage visualization"""
    st.markdown("### 📚 7-Stage Adaptive Curriculum")

    # Curriculum stages visualization
    fig_stages = create_curriculum_stages_visual(
        current_stage=st.session_state.get("current_stage", 3)
    )
    st.plotly_chart(fig_stages, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed level breakdown
    st.markdown("### 📊 20,000 Questions Across 10 Levels")

    level_data = pd.DataFrame(
        {
            "Level": list(range(1, 11)),
            "Questions": [2000] * 10,
            "Difficulty": [i / 10 for i in range(1, 11)],
            "Accuracy": [0.92, 0.89, 0.85, 0.81, 0.77, 0.73, 0.68, 0.62, 0.55, 0.48],
            "Status": ["complete", "complete", "in_progress"] + ["pending"] * 7,
        }
    )

    fig_levels = create_adaptive_curriculum_chart(level_data)
    st.plotly_chart(fig_levels, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stage details table
    st.markdown("### 📋 Stage Breakdown")

    stage_details = pd.DataFrame(
        {
            "Stage": [
                "1. Foundation",
                "2. Basics",
                "3. Intermediate",
                "4. Advanced",
                "5. Expert",
                "6. Master",
                "7. Frontier",
            ],
            "Levels": ["1-2", "2-3", "3-4", "4-5", "5-6", "6-8", "8-10"],
            "Questions": ["2,000", "2,000", "2,000", "2,000", "2,000", "3,000", "4,000"],
            "Difficulty": ["2.5/10", "3.5/10", "5.0/10", "6.5/10", "7.5/10", "8.5/10", "10.0/10"],
            "Duration": ["12h", "14h", "16h", "18h", "20h", "24h", "30h"],
            "Cost": ["$60", "$70", "$80", "$95", "$110", "$130", "$155"],
            "Status": [
                "✓ Complete",
                "✓ Complete",
                "⏳ 75%",
                "Pending",
                "Pending",
                "Pending",
                "Pending",
            ],
        }
    )

    st.dataframe(stage_details, use_container_width=True, hide_index=True)


def render_edge_of_chaos_tab() -> None:
    """Edge-of-chaos assessment tab"""
    st.markdown("### 🎯 Edge-of-Chaos Assessment")

    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #00F5D411 0%, #8338EC11 100%);
                border-left: 4px solid #00F5D4; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
        <h4 style="color: #00F5D4; margin-top: 0;">📖 What is Edge-of-Chaos?</h4>
        <p style="color: #E0E1DD; line-height: 1.6;">
            Optimal learning occurs at the "edge of chaos" - not too easy (boring), not too hard (frustrating).
            Target: <b style="color: #00F5D4;">75% accuracy</b> at difficulty level <b style="color: #00F5D4;">7.0-8.5/10</b>.
        </p>
        <p style="color: #E0E1DD; line-height: 1.6;">
            If accuracy >80%, increase difficulty. If accuracy <70%, decrease difficulty.
            This adaptive approach maximizes learning efficiency.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎚️ Difficulty Level")
        current_difficulty = 7.5
        fig_gauge = create_edge_of_chaos_gauge(current_difficulty, 0.762)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown("#### 🎯 Accuracy vs Threshold")
        fig_accuracy = create_accuracy_threshold_indicator(0.762)
        st.plotly_chart(fig_accuracy, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Optimal zone metrics
    st.markdown("### 📊 Current Zone Status")

    zone_col1, zone_col2, zone_col3 = st.columns(3)

    with zone_col1:
        st.markdown(
            create_gradient_metric(
                "Zone Status",
                "OPTIMAL",
                "In edge-of-chaos",
                gradient_start="#00F5D4",
                gradient_end="#8338EC",
            ),
            unsafe_allow_html=True,
        )

    with zone_col2:
        st.markdown(
            create_gradient_metric(
                "Difficulty",
                "7.5/10",
                "Target: 7.0-8.5",
                gradient_start="#8338EC",
                gradient_end="#FF006E",
            ),
            unsafe_allow_html=True,
        )

    with zone_col3:
        st.markdown(
            create_gradient_metric(
                "Accuracy",
                "76.2%",
                "Target: 70-80%",
                gradient_start="#FF006E",
                gradient_end="#00F5D4",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Adjustment history
    st.markdown("### 📈 Difficulty Adjustment History")

    adjustment_data = pd.DataFrame(
        {
            "Iteration": list(range(1, 11)),
            "Difficulty": [5.0, 5.5, 6.0, 6.5, 7.0, 7.2, 7.5, 7.5, 7.3, 7.5],
            "Accuracy": [0.92, 0.88, 0.84, 0.79, 0.76, 0.75, 0.73, 0.76, 0.78, 0.76],
            "Adjustment": [
                "↑ +0.5",
                "↑ +0.5",
                "↑ +0.5",
                "↑ +0.5",
                "↑ +0.5",
                "↑ +0.2",
                "↑ +0.3",
                "→ 0",
                "↓ -0.2",
                "↑ +0.2",
            ],
        }
    )

    fig_adjustment = go.Figure()

    fig_adjustment.add_trace(
        go.Scatter(
            x=adjustment_data["Iteration"],
            y=adjustment_data["Difficulty"],
            mode="lines+markers",
            name="Difficulty",
            line=dict(color="#00F5D4", width=3),
            marker=dict(size=10, line=dict(color="#FFFFFF", width=2)),
            yaxis="y",
            hovertemplate="<b>Iteration %{x}</b><br>Difficulty: %{y:.1f}<extra></extra>",
        )
    )

    fig_adjustment.add_trace(
        go.Scatter(
            x=adjustment_data["Iteration"],
            y=adjustment_data["Accuracy"] * 10,
            mode="lines+markers",
            name="Accuracy (scaled)",
            line=dict(color="#8338EC", width=3, dash="dot"),
            marker=dict(size=8, symbol="diamond"),
            yaxis="y",
            hovertemplate="<b>Iteration %{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>",
        )
    )

    # Add optimal zones
    fig_adjustment.add_hrect(
        y0=7.0,
        y1=8.5,
        fillcolor="rgba(0, 245, 212, 0.1)",
        line_width=0,
        annotation_text="Optimal Difficulty",
        annotation_position="top left",
    )

    fig_adjustment.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Adaptive Difficulty Adjustment",
        xaxis_title="Training Iteration",
        yaxis_title="Difficulty / Accuracy (%)",
        height=350,
        hovermode="x unified",
    )

    st.plotly_chart(fig_adjustment, use_container_width=True)


def render_tool_ethics_tab() -> None:
    """Tool use training and ethics baking tab"""
    st.markdown("### 🔧 Tool Use Training")

    fig_tools = create_tool_use_metrics()
    st.plotly_chart(fig_tools, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tool use details
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            create_gradient_metric(
                "Code Execution",
                "85%",
                "Proficiency",
                gradient_start="#00F5D4",
                gradient_end="#8338EC",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            create_gradient_metric(
                "Validation",
                "92%",
                "Success Rate",
                gradient_start="#8338EC",
                gradient_end="#FF006E",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            create_gradient_metric(
                "Total Tools", "5/5", "All Active", gradient_start="#FF006E", gradient_end="#FB5607"
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Eudaimonia section
    st.markdown("### 🌟 Eudaimonia 4-Rule Moral System")

    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #8338EC11 0%, #FF006E11 100%);
                border-left: 4px solid #8338EC; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
        <h4 style="color: #8338EC; margin-top: 0;">📖 4-Rule Moral Framework</h4>
        <ol style="color: #E0E1DD; line-height: 1.8;">
            <li><b style="color: #00F5D4;">Benevolence</b>: Act to promote well-being and flourishing</li>
            <li><b style="color: #00F5D4;">Non-Harm</b>: Avoid actions that cause suffering or damage</li>
            <li><b style="color: #00F5D4;">Respect</b>: Honor autonomy and dignity of all agents</li>
            <li><b style="color: #00F5D4;">Autonomy</b>: Preserve freedom of choice and self-determination</li>
        </ol>
        <p style="color: #8B9DAF; font-size: 13px; margin-top: 15px;">
            Based on eudaimonic ethics - virtue-based morality focused on human flourishing.
            Baked into model weights during curriculum training.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_eudaimonia = create_eudaimonia_radar()
        st.plotly_chart(fig_eudaimonia, use_container_width=True)

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            create_gradient_metric(
                "Ethics Score",
                "88%",
                "Above 80% target",
                gradient_start="#8338EC",
                gradient_end="#FF006E",
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            create_gradient_metric(
                "Baking Progress",
                "75%",
                "Stage 3/7",
                gradient_start="#FF006E",
                gradient_end="#00F5D4",
            ),
            unsafe_allow_html=True,
        )


def render_self_modeling_tab() -> None:
    """Self-modeling and dream consolidation tab"""
    st.markdown("### 🧠 Self-Modeling: Temperature Range Prediction")

    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #00F5D411 0%, #8338EC11 100%);
                border-left: 4px solid #00F5D4; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
        <h4 style="color: #00F5D4; margin-top: 0;">📖 Self-Modeling Training</h4>
        <p style="color: #E0E1DD; line-height: 1.6;">
            Model learns to predict its own performance at different temperature settings (0.0-2.0).
            This meta-cognitive ability enables the model to understand its own capabilities and limitations.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig_self_model = create_self_modeling_chart()
    st.plotly_chart(fig_self_model, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Self-modeling metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            create_gradient_metric(
                "Prediction MAE",
                "0.028",
                "Mean Absolute Error",
                gradient_start="#00F5D4",
                gradient_end="#8338EC",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            create_gradient_metric(
                "Optimal Temp", "0.7", "Predicted", gradient_start="#8338EC", gradient_end="#FF006E"
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            create_gradient_metric(
                "Self-Awareness",
                "92%",
                "Correlation",
                gradient_start="#FF006E",
                gradient_end="#00F5D4",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Dream consolidation section
    st.markdown("### 💭 Dream Consolidation (T=1.2 Replay)")

    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #8338EC11 0%, #FF006E11 100%);
                border-left: 4px solid #8338EC; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
        <h4 style="color: #8338EC; margin-top: 0;">📖 Dream-Based Learning</h4>
        <p style="color: #E0E1DD; line-height: 1.6;">
            After each curriculum level, model "dreams" by replaying experiences at high temperature (T=1.2).
            Uses full autoencoder (encoder + decoder) to reconstruct knowledge and prevent catastrophic forgetting.
        </p>
        <p style="color: #E0E1DD; line-height: 1.6;">
            <b style="color: #00F5D4;">3 epochs per level × 10 levels = 30 total dream epochs</b>
        </p>
        <p style="color: #8B9DAF; font-size: 13px; margin-top: 10px;">
            Based on "Dreaming is All You Need" paper - dream consolidation improves generalization and prevents forgetting.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig_dream = create_dream_consolidation_metrics()
    st.plotly_chart(fig_dream, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dream metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            create_gradient_metric(
                "Reconstruction",
                "94.8%",
                "Autoencoder quality",
                gradient_start="#00F5D4",
                gradient_end="#8338EC",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            create_gradient_metric(
                "Retention",
                "91.2%",
                "No forgetting",
                gradient_start="#8338EC",
                gradient_end="#FF006E",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            create_gradient_metric(
                "Dream Epochs",
                "9/30",
                "3 levels complete",
                gradient_start="#FF006E",
                gradient_end="#00F5D4",
            ),
            unsafe_allow_html=True,
        )


def render_frontier_models_tab() -> None:
    """Frontier models usage and cost tracking tab"""
    st.markdown("### 🤖 Frontier Model Integration")

    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
                border: 2px solid #00F5D4; border-radius: 12px; padding: 20px; margin-bottom: 20px;
                box-shadow: 0 4px 16px rgba(0, 245, 212, 0.2);">
        <h4 style="color: #00F5D4; margin-top: 0;">🌟 4 Frontier Models via OpenRouter</h4>
        <p style="color: #E0E1DD; line-height: 1.6;">
            Phase 5 uses multiple frontier models for data generation and validation:
        </p>
        <ul style="color: #E0E1DD; line-height: 1.8;">
            <li><b style="color: #00F5D4;">GPT-4o-mini</b> - Fast, cost-effective, high quality</li>
            <li><b style="color: #8338EC;">Claude-3.5 Haiku</b> - Reasoning specialist</li>
            <li><b style="color: #FF006E;">Gemini 2.0 Flash</b> - Long context, multimodal</li>
            <li><b style="color: #FB5607;">Qwen 2.5</b> - Multilingual, math specialist</li>
        </ul>
        <p style="color: #8B9DAF; font-size: 13px; margin-top: 15px;">
            Total budget: <b style="color: #00F5D4;">$600-800</b> | Training time: <b style="color: #00F5D4;">120-240 hours</b>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig_usage = create_frontier_models_usage()
    st.plotly_chart(fig_usage, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cost breakdown
    st.markdown("### 💰 Cost & Usage Breakdown")

    cost_data = pd.DataFrame(
        {
            "Model": ["GPT-4o-mini", "Claude-3.5 Haiku", "Gemini 2.0 Flash", "Qwen 2.5"],
            "API Calls": ["4,500", "3,800", "4,200", "3,500"],
            "Cost/Call": ["$0.063", "$0.050", "$0.039", "$0.027"],
            "Total Cost": ["$285", "$190", "$165", "$95"],
            "Avg Latency": ["340ms", "420ms", "380ms", "290ms"],
            "Success Rate": ["98.2%", "97.5%", "96.8%", "99.1%"],
        }
    )

    st.dataframe(cost_data, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Budget tracking
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            create_gradient_metric(
                "Total Spent",
                "$427",
                "61% of budget",
                gradient_start="#00F5D4",
                gradient_end="#8338EC",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            create_gradient_metric(
                "Remaining",
                "$273",
                "39% available",
                gradient_start="#8338EC",
                gradient_end="#FF006E",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            create_gradient_metric(
                "Total Calls",
                "16,000",
                "API requests",
                gradient_start="#FF006E",
                gradient_end="#FB5607",
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            create_gradient_metric(
                "Avg Cost/Call",
                "$0.045",
                "Blended rate",
                gradient_start="#FB5607",
                gradient_end="#FFBE0B",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Cost projection
    st.markdown("### 📈 Cost Projection by Stage")

    projection_data = pd.DataFrame(
        {
            "Stage": [f"Stage {i}" for i in range(1, 8)],
            "Estimated Calls": [1800, 2000, 2200, 2400, 2600, 3000, 4000],
            "Projected Cost": [60, 70, 80, 95, 110, 130, 155],
        }
    )

    fig_projection = go.Figure()

    fig_projection.add_trace(
        go.Bar(
            x=projection_data["Stage"],
            y=projection_data["Projected Cost"],
            marker=dict(
                color=projection_data["Projected Cost"],
                colorscale=[[0, "#00F5D4"], [0.5, "#8338EC"], [1, "#FF006E"]],
                line=dict(color="#FFFFFF", width=1),
                showscale=False,
            ),
            text=[f"${c}" for c in projection_data["Projected Cost"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Cost: $%{y}<br>Calls: %{customdata:,}<extra></extra>",
            customdata=projection_data["Estimated Calls"],
        )
    )

    # Add cumulative cost line
    cumulative_cost = projection_data["Projected Cost"].cumsum()
    fig_projection.add_trace(
        go.Scatter(
            x=projection_data["Stage"],
            y=cumulative_cost,
            mode="lines+markers",
            name="Cumulative",
            line=dict(color="#00F5D4", width=3, dash="dot"),
            marker=dict(size=10, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Cumulative: $%{y}<extra></extra>",
        )
    )

    fig_projection.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Cost Projection for Remaining Stages",
        xaxis_title="Stage",
        yaxis=dict(title="Stage Cost ($)", side="left"),
        yaxis2=dict(title="Cumulative Cost ($)", side="right", overlaying="y"),
        height=350,
        showlegend=True,
    )

    st.plotly_chart(fig_projection, use_container_width=True)


# Initialize session state
if "current_stage" not in st.session_state:
    st.session_state.current_stage = 3

if "training_running" not in st.session_state:
    st.session_state.training_running = False


# Main entry point
if __name__ == "__main__":
    render_phase5_dashboard()

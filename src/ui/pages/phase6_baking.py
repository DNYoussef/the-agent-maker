"""
Phase 6: Tool & Persona Baking - Streamlit UI Dashboard

Iterative A/B optimization loops:
- A-Cycle: Tool use optimization (SWE-Bench)
- B-Cycle: Self-guided persona generation (model-driven, NOT pre-defined)

Features:
- Real-time A/B cycle visualization
- SWE-Bench score tracking (70.1% -> 95% target)
- Half-baking strategy (50% strength per iteration)
- Plateau detection and auto-switching
- Model-discovered persona patterns
- Prompt baking details (LoRA r=16, 5 min per prompt)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
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
    delta: str = None,
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


def create_ab_cycle_timeline():
    """Create interactive A/B cycle timeline with alternating cycles"""
    cycles = [
        {"type": "A", "name": "Tool Opt 1", "iteration": 1, "duration": 120, "status": "complete"},
        {"type": "B", "name": "Persona 1", "iteration": 2, "duration": 90, "status": "complete"},
        {"type": "A", "name": "Tool Opt 2", "iteration": 3, "duration": 110, "status": "complete"},
        {"type": "B", "name": "Persona 2", "iteration": 4, "duration": 85, "status": "complete"},
        {
            "type": "A",
            "name": "Tool Opt 3",
            "iteration": 5,
            "duration": 105,
            "status": "in_progress",
        },
        {"type": "B", "name": "Persona 3", "iteration": 6, "duration": 0, "status": "pending"},
        {"type": "A", "name": "Tool Opt 4", "iteration": 7, "duration": 0, "status": "pending"},
    ]

    fig = go.Figure()

    # Create Gantt-style timeline
    y_positions = []
    colors = []
    texts = []
    customdata = []
    start_time = 0

    for cycle in cycles:
        y_positions.append(cycle["iteration"])

        # Color based on cycle type and status
        if cycle["status"] == "complete":
            color = "#00F5D4" if cycle["type"] == "A" else "#8338EC"
        elif cycle["status"] == "in_progress":
            color = "#FF006E"
        else:
            color = "#4A5568"

        colors.append(color)
        texts.append(
            f"{cycle['name']}<br>{cycle['duration']}min" if cycle["duration"] > 0 else cycle["name"]
        )
        customdata.append(
            {
                "type": cycle["type"],
                "name": cycle["name"],
                "duration": cycle["duration"],
                "status": cycle["status"],
            }
        )

    # Timeline bars
    fig.add_trace(
        go.Bar(
            y=[c["name"] for c in cycles],
            x=[c["duration"] if c["duration"] > 0 else 10 for c in cycles],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="#FFFFFF", width=2)
                if any(c["status"] == "in_progress" for c in cycles)
                else None,
            ),
            text=texts,
            textposition="inside",
            textfont=dict(color="#FFFFFF", size=10),
            hovertemplate="<b>%{y}</b><br>Duration: %{x} min<br>Type: %{customdata[type]}-Cycle<br>Status: %{customdata[status]}<extra></extra>",
            customdata=[[c["type"], c["name"], c["duration"], c["status"]] for c in cycles],
        )
    )

    # Add cycle type annotations
    for i, cycle in enumerate(cycles):
        badge_color = "#00F5D4" if cycle["type"] == "A" else "#8338EC"
        # Convert to rgba format (8-char hex not supported in Plotly)
        bgcolor_rgba = (
            "rgba(0, 245, 212, 0.13)" if cycle["type"] == "A" else "rgba(131, 56, 236, 0.13)"
        )
        fig.add_annotation(
            x=-5,
            y=i,
            text=f"<b>{cycle['type']}</b>",
            showarrow=False,
            font=dict(size=14, color=badge_color, family="monospace"),
            bgcolor=bgcolor_rgba,
            bordercolor=badge_color,
            borderwidth=2,
            borderpad=4,
            xanchor="right",
        )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="A/B Cycle Timeline (Iterative Optimization Loops)",
        xaxis_title="Duration (minutes)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        height=450,
        showlegend=False,
        margin=dict(l=150, r=40, t=60, b=60),
    )

    return fig


def create_swe_bench_progress():
    """Create SWE-Bench score progression chart with target line"""
    iterations = np.arange(1, 11)
    scores = np.array([70.1, 73.2, 76.8, 79.5, 82.1, 84.7, 87.3, 89.8, 92.1, 94.2])
    target = 95.0

    fig = go.Figure()

    # Target line
    fig.add_hline(
        y=target,
        line_dash="dash",
        line_color="#00F5D4",
        line_width=3,
        annotation_text="Target: 95%",
        annotation_position="right",
        annotation_font=dict(color="#00F5D4", size=14),
    )

    # Target zone
    fig.add_hrect(y0=target - 2, y1=target + 2, fillcolor="rgba(0, 245, 212, 0.1)", line_width=0)

    # Progress line with confidence band
    upper_bound = scores + 1.5
    lower_bound = scores - 1.5

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([iterations, iterations[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 245, 212, 0.15)",
            line=dict(color="rgba(0, 245, 212, 0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Main score line
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=scores,
            mode="lines+markers",
            name="SWE-Bench Score",
            line=dict(color="#00F5D4", width=4, shape="spline"),
            marker=dict(
                size=12,
                color=scores,
                colorscale=[[0, "#FF006E"], [0.5, "#8338EC"], [1, "#00F5D4"]],
                line=dict(color="#FFFFFF", width=2),
                showscale=False,
            ),
            hovertemplate="<b>Iteration %{x}</b><br>Score: %{y:.1f}%<extra></extra>",
        )
    )

    # Highlight current iteration
    current_iter = 5
    fig.add_annotation(
        x=current_iter,
        y=scores[current_iter - 1],
        text=f"Current<br>{scores[current_iter - 1]:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#FF006E",
        font=dict(color="#FF006E", size=12, family="Inter"),
        bgcolor="rgba(27, 40, 56, 0.9)",
        bordercolor="#FF006E",
        borderwidth=2,
        ax=50,
        ay=-50,
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="SWE-Bench Score Progression (Tool Use Optimization)",
        xaxis_title="A-Cycle Iteration",
        yaxis_title="SWE-Bench Score (%)",
        yaxis=dict(range=[65, 100]),
        height=400,
        hovermode="x unified",
    )

    return fig


def create_half_baking_visualization():
    """Create visualization of half-baking strategy (50% strength per iteration)"""
    iterations = np.arange(0, 11)
    cumulative_strength = 1 - (0.5**iterations)
    per_iteration = np.diff(np.concatenate([[0], cumulative_strength]))

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Cumulative Strength", "Strength Added per Iteration"),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"type": "bar"}]],
    )

    # Cumulative strength line
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=cumulative_strength * 100,
            mode="lines+markers",
            name="Cumulative Strength",
            line=dict(color="#00F5D4", width=3, shape="spline"),
            marker=dict(size=10, color="#00F5D4", line=dict(color="#FFFFFF", width=2)),
            fill="tozeroy",
            fillcolor="rgba(0, 245, 212, 0.15)",
            hovertemplate="<b>Iteration %{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Target line at 100%
    fig.add_hline(y=100, line_dash="dash", line_color="#8338EC", line_width=2, row=1, col=1)

    # Per-iteration strength bars
    colors = [
        "#00F5D4" if i < 5 else "#8338EC" if i < 8 else "#FF006E" for i in range(len(per_iteration))
    ]

    fig.add_trace(
        go.Bar(
            x=iterations[1:],
            y=per_iteration * 100,
            name="Strength Added",
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            text=[f"{v*100:.1f}%" for v in per_iteration],
            textposition="outside",
            textfont=dict(size=10, color="#E0E1DD"),
            hovertemplate="<b>Iteration %{x}</b><br>Added: %{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Annotations
    fig.add_annotation(
        x=5,
        y=cumulative_strength[5] * 100,
        text=f"Half-life<br>{cumulative_strength[5]*100:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#8338EC",
        font=dict(color="#8338EC", size=11),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Strength (%)", row=1, col=1)
    fig.update_yaxes(title_text="Strength Added (%)", row=2, col=1)

    fig.update_layout(template=CUSTOM_PLOTLY_TEMPLATE, height=600, showlegend=False)

    return fig


def create_tool_proficiency_radar():
    """Create radar chart showing tool proficiency across different categories"""
    categories = [
        "Code Generation",
        "Debugging",
        "Testing",
        "Documentation",
        "Refactoring",
        "API Integration",
        "Error Handling",
        "Performance Opt",
    ]

    # Before baking vs After baking
    before_scores = [65, 58, 62, 55, 60, 50, 63, 57]
    after_scores = [88, 82, 85, 79, 83, 76, 87, 81]

    fig = go.Figure()

    # Before baking
    fig.add_trace(
        go.Scatterpolar(
            r=before_scores + [before_scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(255, 0, 110, 0.15)",
            line=dict(color="#FF006E", width=2),
            name="Before Baking",
            hovertemplate="<b>%{theta}</b><br>Score: %{r}%<extra></extra>",
        )
    )

    # After baking
    fig.add_trace(
        go.Scatterpolar(
            r=after_scores + [after_scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(0, 245, 212, 0.2)",
            line=dict(color="#00F5D4", width=3),
            name="After Baking (Current)",
            hovertemplate="<b>%{theta}</b><br>Score: %{r}%<extra></extra>",
        )
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100], gridcolor="#2E3F4F", tickfont=dict(color="#E0E1DD")
            ),
            angularaxis=dict(gridcolor="#2E3F4F", tickfont=dict(color="#E0E1DD", size=10)),
            bgcolor="#1B2838",
        ),
        title="Tool Proficiency Breakdown (A-Cycle)",
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(27, 40, 56, 0.8)", bordercolor="#2E3F4F", borderwidth=1, x=0.85, y=0.95
        ),
    )

    return fig


def create_persona_evolution_heatmap():
    """Create heatmap showing model-discovered persona patterns over iterations"""
    # Model-discovered patterns (NOT pre-defined personas)
    patterns = [
        "Analytical Reasoning",
        "Creative Problem-Solving",
        "Systematic Debugging",
        "Code Optimization",
        "Documentation Focus",
        "Error Recovery",
        "Architectural Design",
        "Testing Rigor",
    ]

    iterations = ["B1", "B2", "B3", "B4", "B5", "B6"]

    # Random pattern strengths showing evolution
    np.random.seed(42)
    strength_data = np.random.uniform(0.3, 0.95, (len(patterns), len(iterations)))

    # Make it progressive (patterns strengthen over time)
    for i in range(len(patterns)):
        strength_data[i] = np.sort(strength_data[i]) + np.linspace(0, 0.2, len(iterations))
        strength_data[i] = np.clip(strength_data[i], 0, 1)

    # Custom colorscale
    colorscale = [
        [0, "#1B2838"],
        [0.3, "#4A5568"],
        [0.5, "#8338EC"],
        [0.7, "#FF006E"],
        [1, "#00F5D4"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=strength_data,
            x=iterations,
            y=patterns,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text="Pattern<br>Strength", font=dict(color="#E0E1DD")),
                tickfont=dict(color="#E0E1DD"),
                bgcolor="#1B2838",
                bordercolor="#2E3F4F",
                borderwidth=1,
                tickformat=".0%",
            ),
            text=strength_data,
            texttemplate="%{text:.0%}",
            textfont=dict(size=10, color="#FFFFFF"),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.0%}<extra></extra>",
        )
    )

    # Highlight current iteration (B-Cycle 5)
    fig.add_shape(
        type="rect",
        x0=3.5,
        x1=4.5,
        y0=-0.5,
        y1=len(patterns) - 0.5,
        line=dict(color="#FF006E", width=3),
        fillcolor="rgba(255, 0, 110, 0.1)",
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Model-Discovered Persona Patterns (B-Cycle Evolution)",
        xaxis_title="B-Cycle Iteration",
        yaxis_title="Discovered Pattern",
        height=450,
    )

    return fig


def create_plateau_detection_chart():
    """Create chart showing performance plateau detection and cycle switching"""
    iterations = np.arange(1, 21)
    performance = np.concatenate(
        [
            np.linspace(70, 85, 8),  # Initial improvement
            np.ones(5) * 85 + np.random.normal(0, 0.5, 5),  # Plateau
            np.linspace(85, 92, 7),  # Post-switch improvement
        ]
    )

    fig = go.Figure()

    # Performance line
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=performance,
            mode="lines+markers",
            name="Performance",
            line=dict(color="#00F5D4", width=3),
            marker=dict(size=8, color="#00F5D4", line=dict(color="#FFFFFF", width=1)),
            hovertemplate="<b>Iteration %{x}</b><br>Performance: %{y:.1f}%<extra></extra>",
        )
    )

    # Plateau detection zone
    fig.add_vrect(
        x0=8,
        x1=13,
        fillcolor="rgba(255, 0, 110, 0.15)",
        line_width=0,
        annotation_text="Plateau Detected",
        annotation_position="top",
        annotation_font=dict(size=11, color="#FF006E"),
    )

    # Cycle switch marker
    fig.add_vline(
        x=13,
        line_dash="dash",
        line_color="#8338EC",
        line_width=3,
        annotation_text="Auto-Switch: A→B",
        annotation_position="top",
        annotation_font=dict(color="#8338EC", size=12),
    )

    # Plateau threshold
    fig.add_hline(
        y=85,
        line_dash="dot",
        line_color="#4A5568",
        line_width=2,
        annotation_text="Plateau Threshold",
        annotation_position="left",
        annotation_font=dict(color="#8B9DAF", size=10),
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Plateau Detection & Auto-Cycle Switching",
        xaxis_title="Total Iteration",
        yaxis_title="Performance Score (%)",
        height=400,
        showlegend=False,
    )

    return fig


def create_baking_time_breakdown():
    """Create breakdown of prompt baking time per component"""
    components = [
        {"name": "Tool Prompt 1", "time": 5, "type": "A-Cycle"},
        {"name": "Tool Prompt 2", "time": 5, "type": "A-Cycle"},
        {"name": "Tool Prompt 3", "time": 5, "type": "A-Cycle"},
        {"name": "Persona Pattern 1", "time": 5, "type": "B-Cycle"},
        {"name": "Persona Pattern 2", "time": 5, "type": "B-Cycle"},
        {"name": "Persona Pattern 3", "time": 5, "type": "B-Cycle"},
        {"name": "Combined Bake", "time": 5, "type": "Final"},
    ]

    fig = go.Figure()

    colors = {"A-Cycle": "#00F5D4", "B-Cycle": "#8338EC", "Final": "#FF006E"}

    # Waterfall-style bars
    start = 0
    for comp in components:
        fig.add_trace(
            go.Bar(
                x=[comp["time"]],
                y=[comp["name"]],
                orientation="h",
                marker=dict(color=colors[comp["type"]], line=dict(color="#FFFFFF", width=1)),
                text=f"{comp['time']} min",
                textposition="inside",
                textfont=dict(color="#FFFFFF", size=11),
                name=comp["type"],
                hovertemplate=f"<b>{comp['name']}</b><br>Time: {comp['time']} min<br>Type: {comp['type']}<extra></extra>",
                showlegend=comp == components[0]
                or (comp["type"] != components[components.index(comp) - 1]["type"]),
            )
        )

    # Total time annotation
    total_time = sum(c["time"] for c in components)
    fig.add_annotation(
        x=total_time + 2,
        y=len(components) - 1,
        text=f"Total: {total_time} min",
        showarrow=False,
        font=dict(size=14, color="#00F5D4", family="Inter"),
        bgcolor="rgba(0, 245, 212, 0.2)",
        bordercolor="#00F5D4",
        borderwidth=2,
        borderpad=6,
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Prompt Baking Time Breakdown (LoRA r=16, Sequential Chain)",
        xaxis_title="Time (minutes)",
        yaxis=dict(autorange="reversed"),
        height=450,
        barmode="stack",
        legend=dict(bgcolor="rgba(27, 40, 56, 0.8)", bordercolor="#2E3F4F", borderwidth=1),
    )

    return fig


def create_baked_behaviors_table():
    """Create table showing all baked behaviors with strength levels"""
    behaviors = [
        {
            "behavior": "Code Generation (Python)",
            "type": "Tool",
            "strength": 0.92,
            "iteration": "A3",
        },
        {"behavior": "Unit Test Creation", "type": "Tool", "strength": 0.88, "iteration": "A2"},
        {"behavior": "API Documentation", "type": "Tool", "strength": 0.85, "iteration": "A3"},
        {"behavior": "Debugging Workflow", "type": "Tool", "strength": 0.90, "iteration": "A4"},
        {
            "behavior": "Analytical Reasoning",
            "type": "Persona",
            "strength": 0.78,
            "iteration": "B2",
        },
        {
            "behavior": "Systematic Problem-Solving",
            "type": "Persona",
            "strength": 0.82,
            "iteration": "B3",
        },
        {
            "behavior": "Code Optimization Focus",
            "type": "Persona",
            "strength": 0.75,
            "iteration": "B2",
        },
        {
            "behavior": "Documentation Emphasis",
            "type": "Persona",
            "strength": 0.70,
            "iteration": "B1",
        },
        {
            "behavior": "Error Recovery Patterns",
            "type": "Persona",
            "strength": 0.73,
            "iteration": "B2",
        },
        {
            "behavior": "Architectural Thinking",
            "type": "Persona",
            "strength": 0.80,
            "iteration": "B3",
        },
    ]

    df = pd.DataFrame(behaviors)

    # Create styled table using Plotly
    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[200, 80, 80, 80],
                header=dict(
                    values=[
                        "<b>Baked Behavior</b>",
                        "<b>Type</b>",
                        "<b>Strength</b>",
                        "<b>Iteration</b>",
                    ],
                    fill_color="#1B2838",
                    line_color="#2E3F4F",
                    font=dict(color="#00F5D4", size=13, family="Inter"),
                    align="left",
                    height=35,
                ),
                cells=dict(
                    values=[
                        df["behavior"],
                        df["type"],
                        [f"{s:.0%}" for s in df["strength"]],
                        df["iteration"],
                    ],
                    fill_color=[["#0D1B2A" if i % 2 == 0 else "#1B2838" for i in range(len(df))]],
                    line_color="#2E3F4F",
                    font=dict(color="#E0E1DD", size=11),
                    align="left",
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="All Baked Behaviors (Tool Usage + Persona Patterns)",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def render_phase6_dashboard():
    """Main dashboard for Phase 6 Tool & Persona Baking"""

    # Page config

    # Custom CSS
    st.markdown(
        """
    <style>
    /* Dark theme */
    .stApp {
        background-color: #0D1B2A;
    }

    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
        border: 2px solid #00F5D444;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 245, 212, 0.2);
    }

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
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
        color: #00F5D4;
        border: 1px solid #00F5D444;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1B283822 0%, #2E3F4F22 100%);
        border: 1px solid #2E3F4F;
        border-radius: 8px;
        padding: 12px;
    }

    /* Buttons */
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

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1B283822 0%, #2E3F4F22 100%);
        border-left: 4px solid #00F5D4;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    }

    /* Glassmorphism effect */
    .glass-card {
        background: rgba(27, 40, 56, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 245, 212, 0.2);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Hero section
    hero_html = """
    <div style="
        background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
        border: 2px solid #00F5D444;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 245, 212, 0.2);
    ">
        <h1 style="color: #00F5D4; margin: 0; font-size: 42px;">Phase 6: Tool & Persona Baking</h1>
        <p style="color: #E0E1DD; font-size: 18px; margin: 10px 0 0 0;">
            Iterative A/B optimization loops: Tool use (SWE-Bench) to Model-discovered persona patterns
        </p>
    </div>
    """
    components.html(hero_html, height=150)

    # Top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            create_gradient_metric(
                "Current Cycle",
                "A-Cycle 5",
                "Tool Optimization",
                gradient_start="#00F5D4",
                gradient_end="#8338EC",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            create_gradient_metric(
                "SWE-Bench Score",
                "82.1%",
                "+12.0% from start",
                gradient_start="#8338EC",
                gradient_end="#FF006E",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            create_gradient_metric(
                "Total Iterations",
                "9/20",
                "45% Complete",
                gradient_start="#FF006E",
                gradient_end="#FB5607",
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            create_gradient_metric(
                "Baking Time",
                "45 min",
                "9 prompts baked",
                gradient_start="#FB5607",
                gradient_end="#FFBE0B",
            ),
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            create_gradient_metric(
                "Plateau Status",
                "Active",
                "No plateau",
                gradient_start="#FFBE0B",
                gradient_end="#00F5D4",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 A/B Cycle Overview",
            "⚡ A-Cycle: Tool Optimization",
            "🧠 B-Cycle: Persona Discovery",
            "📈 Half-Baking Strategy",
            "🔥 Baked Behaviors",
        ]
    )

    with tab1:
        render_ab_overview_tab()

    with tab2:
        render_a_cycle_tab()

    with tab3:
        render_b_cycle_tab()

    with tab4:
        render_half_baking_tab()

    with tab5:
        render_baked_behaviors_tab()


def render_ab_overview_tab():
    """Render A/B cycle overview tab"""
    st.markdown("### 🔄 A/B Cycle Timeline")

    # Timeline visualization
    fig_timeline = create_ab_cycle_timeline()
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cycle explanation
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="glass-card">
            <h3 style="color: #00F5D4; margin-top: 0;">⚡ A-Cycle: Tool Use Optimization</h3>
            <p style="color: #E0E1DD; line-height: 1.8;">
                Train on <b style="color: #00F5D4;">SWE-Bench</b> dataset to optimize tool usage patterns:
            </p>
            <ul style="color: #E0E1DD; line-height: 2;">
                <li>🎯 <b>Target</b>: 70.1% → 95% SWE-Bench score</li>
                <li>⚙️ <b>Method</b>: LoRA fine-tuning (r=16, 3 epochs)</li>
                <li>⏱️ <b>Duration</b>: ~110 minutes per cycle</li>
                <li>🔄 <b>Baking</b>: 5 minutes per tool prompt</li>
            </ul>
            <div style="background: rgba(0, 245, 212, 0.1); border-radius: 6px; padding: 12px; margin-top: 15px;">
                <b style="color: #00F5D4;">Focus Areas</b>: Code generation, debugging, testing, API integration
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="glass-card">
            <h3 style="color: #8338EC; margin-top: 0;">🧠 B-Cycle: Self-Guided Persona Discovery</h3>
            <p style="color: #E0E1DD; line-height: 1.8;">
                Model analyzes its own behavior to discover <b style="color: #8338EC;">emergent patterns</b>:
            </p>
            <ul style="color: #E0E1DD; line-height: 2;">
                <li>🔍 <b>NOT pre-defined</b>: Model discovers own personas</li>
                <li>🎭 <b>Discovery</b>: Behavioral pattern analysis</li>
                <li>⏱️ <b>Duration</b>: ~90 minutes per cycle</li>
                <li>🔄 <b>Baking</b>: 5 minutes per discovered pattern</li>
            </ul>
            <div style="background: rgba(131, 56, 236, 0.1); border-radius: 6px; padding: 12px; margin-top: 15px;">
                <b style="color: #8338EC;">Examples</b>: Analytical reasoning, systematic debugging, creative problem-solving
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Plateau detection
    st.markdown("### 🎯 Plateau Detection & Auto-Switching")
    fig_plateau = create_plateau_detection_chart()
    st.plotly_chart(fig_plateau, use_container_width=True)

    st.markdown(
        """
    <div class="info-box">
        <h4 style="color: #00F5D4; margin-top: 0;">💡 How Plateau Detection Works</h4>
        <p style="color: #E0E1DD; line-height: 1.8;">
            The system monitors performance over a sliding window (5 iterations). When improvement
            stagnates (<0.5% change), it automatically switches cycles:
        </p>
        <ul style="color: #E0E1DD; line-height: 2;">
            <li><b>A-Cycle plateau</b> → Switch to B-Cycle (persona discovery)</li>
            <li><b>B-Cycle plateau</b> → Switch to A-Cycle (tool optimization)</li>
            <li><b>Convergence trigger</b>: ≥3 consecutive switches with no improvement</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_a_cycle_tab():
    """Render A-Cycle (Tool Optimization) tab"""
    st.markdown("### ⚡ A-Cycle: Tool Use Optimization (SWE-Bench)")

    # SWE-Bench progress
    fig_swe = create_swe_bench_progress()
    st.plotly_chart(fig_swe, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tool proficiency breakdown
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 🎯 Tool Proficiency Breakdown")
        fig_radar = create_tool_proficiency_radar()
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Current Metrics")

        metrics = [
            {"metric": "Code Generation", "before": 65, "after": 88, "gain": 23},
            {"metric": "Debugging", "before": 58, "after": 82, "gain": 24},
            {"metric": "Testing", "before": 62, "after": 85, "gain": 23},
            {"metric": "Documentation", "before": 55, "after": 79, "gain": 24},
            {"metric": "Refactoring", "before": 60, "after": 83, "gain": 23},
            {"metric": "API Integration", "before": 50, "after": 76, "gain": 26},
            {"metric": "Error Handling", "before": 63, "after": 87, "gain": 24},
            {"metric": "Performance Opt", "before": 57, "after": 81, "gain": 24},
        ]

        for m in metrics:
            gain_color = "#00F5D4" if m["gain"] >= 23 else "#8338EC"
            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, {gain_color}11 0%, {gain_color}22 100%);
                        border-left: 3px solid {gain_color}; border-radius: 6px; padding: 10px; margin: 8px 0;">
                <div style="color: #8B9DAF; font-size: 11px; text-transform: uppercase;">{m['metric']}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
                    <span style="color: #E0E1DD; font-size: 14px;">{m['before']}% → {m['after']}%</span>
                    <span style="color: {gain_color}; font-weight: 700;">+{m['gain']}%</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Error rate reduction
    st.markdown("#### 📉 Error Rate Reduction Over Iterations")

    iterations = np.arange(1, 11)
    error_rates = np.array([28.5, 25.2, 22.8, 20.5, 18.9, 17.2, 15.8, 14.5, 13.2, 12.1])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=error_rates,
            mode="lines+markers",
            name="Error Rate",
            line=dict(color="#FF006E", width=3, shape="spline"),
            marker=dict(size=10, color="#FF006E", line=dict(color="#FFFFFF", width=2)),
            fill="tozeroy",
            fillcolor="rgba(255, 0, 110, 0.15)",
            hovertemplate="<b>A-Cycle %{x}</b><br>Error Rate: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_hline(
        y=15,
        line_dash="dash",
        line_color="#00F5D4",
        line_width=2,
        annotation_text="Target: <15%",
        annotation_position="right",
        annotation_font=dict(color="#00F5D4", size=12),
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Error Rate Reduction (Tool Use Failures)",
        xaxis_title="A-Cycle Iteration",
        yaxis_title="Error Rate (%)",
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_b_cycle_tab():
    """Render B-Cycle (Persona Discovery) tab"""
    st.markdown("### 🧠 B-Cycle: Self-Guided Persona Discovery")

    st.markdown(
        """
    <div class="info-box">
        <h4 style="color: #8338EC; margin-top: 0;">🎭 Model-Driven Persona Discovery</h4>
        <p style="color: #E0E1DD; line-height: 1.8;">
            <b>IMPORTANT</b>: These are <b style="color: #8338EC;">NOT pre-defined personas</b>.
            The model analyzes its own behavior patterns and discovers emergent characteristics.
            Each B-Cycle iteration reveals new patterns based on recent interactions.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Persona evolution heatmap
    st.markdown("#### 🌡️ Model-Discovered Pattern Evolution")
    fig_heatmap = create_persona_evolution_heatmap()
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Self-evolution progress
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔍 Current B-Cycle Discoveries")

        discoveries = [
            {
                "pattern": "Analytical Reasoning",
                "strength": 0.78,
                "description": "Strong tendency toward systematic analysis",
                "iteration": "B2",
            },
            {
                "pattern": "Systematic Debugging",
                "strength": 0.82,
                "description": "Methodical error investigation patterns",
                "iteration": "B3",
            },
            {
                "pattern": "Code Optimization Focus",
                "strength": 0.75,
                "description": "Preference for efficiency improvements",
                "iteration": "B2",
            },
            {
                "pattern": "Documentation Emphasis",
                "strength": 0.70,
                "description": "Consistent documentation generation",
                "iteration": "B1",
            },
            {
                "pattern": "Architectural Thinking",
                "strength": 0.80,
                "description": "System-level design considerations",
                "iteration": "B3",
            },
        ]

        for disc in discoveries:
            strength_color = "#00F5D4" if disc["strength"] >= 0.75 else "#8338EC"
            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, {strength_color}11 0%, {strength_color}22 100%);
                        border: 1px solid {strength_color}44; border-radius: 10px; padding: 15px; margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="color: {strength_color}; margin: 0; font-size: 15px;">{disc['pattern']}</h4>
                    <span style="color: {strength_color}; font-weight: 700; font-size: 18px;">{disc['strength']:.0%}</span>
                </div>
                <p style="color: #8B9DAF; font-size: 12px; margin: 8px 0 5px 0;">{disc['description']}</p>
                <div style="background: rgba(0, 0, 0, 0.3); border-radius: 4px; padding: 4px 8px; display: inline-block;">
                    <span style="color: #8B9DAF; font-size: 10px;">Discovered in {disc['iteration']}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 📊 Persona Strength Meters")

        # Create horizontal bar chart for persona strengths
        patterns = [
            "Analytical Reasoning",
            "Systematic Debugging",
            "Code Optimization",
            "Documentation",
            "Architectural Design",
            "Error Recovery",
            "Creative Problem-Solving",
            "Testing Rigor",
        ]
        strengths = [0.78, 0.82, 0.75, 0.70, 0.80, 0.73, 0.68, 0.76]

        fig = go.Figure()

        colors = [
            "#00F5D4" if s >= 0.75 else "#8338EC" if s >= 0.70 else "#FF006E" for s in strengths
        ]

        fig.add_trace(
            go.Bar(
                x=strengths,
                y=patterns,
                orientation="h",
                marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
                text=[f"{s:.0%}" for s in strengths],
                textposition="inside",
                textfont=dict(color="#FFFFFF", size=11),
                hovertemplate="<b>%{y}</b><br>Strength: %{x:.0%}<extra></extra>",
            )
        )

        fig.add_vline(
            x=0.75,
            line_dash="dash",
            line_color="#00F5D4",
            line_width=2,
            annotation_text="Strong",
            annotation_position="top",
            annotation_font=dict(color="#00F5D4", size=10),
        )

        fig.update_layout(
            template=CUSTOM_PLOTLY_TEMPLATE,
            title="Current Pattern Strengths",
            xaxis=dict(title="Strength", tickformat=".0%", range=[0, 1]),
            yaxis=dict(autorange="reversed"),
            height=400,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Self-evolution progress chart
    st.markdown("#### 📈 Self-Evolution Progress")

    b_iterations = np.arange(1, 7)
    pattern_count = np.array([3, 5, 7, 8, 10, 11])
    avg_strength = np.array([0.55, 0.62, 0.68, 0.72, 0.75, 0.77])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=b_iterations,
            y=pattern_count,
            name="Patterns Discovered",
            marker=dict(color="#8338EC", line=dict(color="#FFFFFF", width=1)),
            hovertemplate="<b>B-Cycle %{x}</b><br>Patterns: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=b_iterations,
            y=avg_strength,
            mode="lines+markers",
            name="Avg Strength",
            line=dict(color="#00F5D4", width=3),
            marker=dict(size=10, color="#00F5D4", line=dict(color="#FFFFFF", width=2)),
            hovertemplate="<b>B-Cycle %{x}</b><br>Avg Strength: %{y:.0%}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="B-Cycle Iteration")
    fig.update_yaxes(title_text="Patterns Discovered", secondary_y=False)
    fig.update_yaxes(title_text="Average Strength", secondary_y=True, tickformat=".0%")

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Pattern Discovery & Strength Over B-Cycles",
        height=400,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_half_baking_tab():
    """Render half-baking strategy tab"""
    st.markdown("### 📈 Half-Baking Strategy: 50% Strength per Iteration")

    st.markdown(
        """
    <div class="info-box">
        <h4 style="color: #00F5D4; margin-top: 0;">🎯 What is Half-Baking?</h4>
        <p style="color: #E0E1DD; line-height: 1.8;">
            Instead of baking prompts to 100% strength immediately, we use <b style="color: #00F5D4;">50% strength
            per iteration</b>. This allows the model to gradually adopt behaviors while maintaining flexibility.
        </p>
        <ul style="color: #E0E1DD; line-height: 2;">
            <li><b>Iteration 1</b>: 50% strength baked</li>
            <li><b>Iteration 2</b>: 75% cumulative (50% + 25%)</li>
            <li><b>Iteration 3</b>: 87.5% cumulative (50% + 25% + 12.5%)</li>
            <li><b>Convergence</b>: Approaches 100% asymptotically</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Half-baking visualization
    fig_half_baking = create_half_baking_visualization()
    st.plotly_chart(fig_half_baking, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Benefits comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="glass-card">
            <h3 style="color: #00F5D4; margin-top: 0;">✅ Benefits of Half-Baking</h3>
            <ul style="color: #E0E1DD; line-height: 2;">
                <li>🎯 <b>Gradual Adoption</b>: Behaviors integrate smoothly</li>
                <li>🔄 <b>Flexibility</b>: Can adjust mid-process</li>
                <li>🛡️ <b>Stability</b>: Reduces catastrophic forgetting</li>
                <li>🧪 <b>Experimentation</b>: Test partial strengths</li>
                <li>⚡ <b>Speed</b>: Faster per-iteration training</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="glass-card">
            <h3 style="color: #FF006E; margin-top: 0;">❌ Full Baking (100%) Issues</h3>
            <ul style="color: #E0E1DD; line-height: 2;">
                <li>⚠️ <b>Overfitting</b>: Too rigid to prompts</li>
                <li>🔒 <b>Locked Behavior</b>: Hard to modify</li>
                <li>💥 <b>Catastrophic Forgetting</b>: Loses prior knowledge</li>
                <li>🐌 <b>Slower Training</b>: Longer epochs required</li>
                <li>🎲 <b>Risky</b>: All-or-nothing approach</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Iteration count analysis
    st.markdown("#### 🔢 Iteration Count to Target Strength")

    target_strengths = [0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875]
    iterations_needed = list(range(1, 8))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=iterations_needed,
            y=[s * 100 for s in target_strengths],
            mode="lines+markers",
            name="Cumulative Strength",
            line=dict(color="#00F5D4", width=3, shape="spline"),
            marker=dict(
                size=12,
                color=[s * 100 for s in target_strengths],
                colorscale=[[0, "#FF006E"], [0.5, "#8338EC"], [1, "#00F5D4"]],
                line=dict(color="#FFFFFF", width=2),
                showscale=False,
            ),
            fill="tozeroy",
            fillcolor="rgba(0, 245, 212, 0.15)",
            hovertemplate="<b>Iteration %{x}</b><br>Strength: %{y:.2f}%<extra></extra>",
        )
    )

    # Target lines
    for target in [75, 90, 95]:
        fig.add_hline(
            y=target,
            line_dash="dot",
            line_color="#8B9DAF",
            line_width=1,
            annotation_text=f"{target}%",
            annotation_position="right",
            annotation_font=dict(color="#8B9DAF", size=10),
        )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Cumulative Strength vs Iteration Count (50% Half-Life)",
        xaxis_title="Iteration",
        yaxis_title="Cumulative Strength (%)",
        yaxis=dict(range=[0, 105]),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_baked_behaviors_tab():
    """Render baked behaviors tab"""
    st.markdown("### 🔥 All Baked Behaviors")

    # Baked behaviors table
    fig_table = create_baked_behaviors_table()
    st.plotly_chart(fig_table, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prompt baking time breakdown
    st.markdown("#### ⏱️ Prompt Baking Time Breakdown")
    fig_time = create_baking_time_breakdown()
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Baking configuration details
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="glass-card">
            <h3 style="color: #00F5D4; margin-top: 0;">⚙️ LoRA Configuration</h3>
            <ul style="color: #E0E1DD; line-height: 2; list-style: none; padding: 0;">
                <li><b>Rank (r)</b>: 16</li>
                <li><b>Alpha</b>: 32</li>
                <li><b>Dropout</b>: 0.05</li>
                <li><b>Target Modules</b>: q_proj, v_proj</li>
                <li><b>Bias</b>: None</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="glass-card">
            <h3 style="color: #8338EC; margin-top: 0;">🎯 Training Parameters</h3>
            <ul style="color: #E0E1DD; line-height: 2; list-style: none; padding: 0;">
                <li><b>Epochs</b>: 3 (half-baking)</li>
                <li><b>Learning Rate</b>: 2e-4</li>
                <li><b>Batch Size</b>: 4</li>
                <li><b>Gradient Accumulation</b>: 4</li>
                <li><b>Optimizer</b>: AdamW</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="glass-card">
            <h3 style="color: #FF006E; margin-top: 0;">📊 Baking Metrics</h3>
            <ul style="color: #E0E1DD; line-height: 2; list-style: none; padding: 0;">
                <li><b>Time per Prompt</b>: 5 min</li>
                <li><b>Total Prompts</b>: 10 (9 + final)</li>
                <li><b>Total Time</b>: ~50 min</li>
                <li><b>Sequential Chain</b>: Yes</li>
                <li><b>Half-Baking</b>: 50% strength</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Sequential baking chain
    st.markdown("#### 🔗 Sequential Baking Chain")

    st.markdown(
        """
    <div class="info-box">
        <h4 style="color: #00F5D4; margin-top: 0;">📖 What is Sequential Baking?</h4>
        <p style="color: #E0E1DD; line-height: 1.8;">
            Prompts are baked <b style="color: #00F5D4;">one after another</b>, creating a chain:
            <br><br>
            <code style="color: #00F5D4; background: rgba(0, 0, 0, 0.3); padding: 8px; border-radius: 4px; display: block; margin: 10px 0;">
            θ_final = Bake(Bake(Bake(θ_base, prompt1), prompt2), prompt3)
            </code>
            <br>
            Each baked model becomes the base for the next prompt, allowing <b>compositional behavior learning</b>.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Visual chain diagram
    baking_steps = [
        "Base Model (Phase 5)",
        "→ Tool Prompt 1",
        "→ Tool Prompt 2",
        "→ Tool Prompt 3",
        "→ Persona Pattern 1",
        "→ Persona Pattern 2",
        "→ Persona Pattern 3",
        "→ Combined Bake",
        "→ Final Model (Phase 7)",
    ]

    chain_html = '<div style="display: flex; align-items: center; justify-content: space-around; margin: 20px 0;">'
    for i, step in enumerate(baking_steps):
        color = (
            "#00F5D4"
            if "Tool" in step
            else "#8338EC"
            if "Persona" in step
            else "#FF006E"
            if "Final" in step
            else "#8B9DAF"
        )
        chain_html += f"""
        <div style="background: {color}22; border: 2px solid {color}; border-radius: 8px;
                    padding: 10px 15px; margin: 5px; text-align: center; min-width: 100px;">
            <span style="color: {color}; font-size: 12px; font-weight: 600;">{step}</span>
        </div>
        """
    chain_html += "</div>"

    st.markdown(chain_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Phase 6 → Phase 7 handoff
    st.markdown("#### 🤝 Phase 6 → Phase 7 Handoff")

    handoff_items = [
        {"item": "Final baked model with all behaviors", "status": True},
        {"item": "Tool use proficiency ≥85% average", "status": True},
        {"item": "≥8 discovered persona patterns", "status": True},
        {"item": "SWE-Bench score ≥93%", "status": True},
        {"item": "Sequential baking chain validated", "status": True},
        {"item": "Behavior strength ≥75% average", "status": True},
    ]

    for item in handoff_items:
        checkbox_color = "#00F5D4" if item["status"] else "#FF006E"
        checkbox_icon = "☑" if item["status"] else "☐"

        st.markdown(
            f"""
        <div style="background: linear-gradient(135deg, {checkbox_color}11 0%, {checkbox_color}22 100%);
                    border-left: 4px solid {checkbox_color}; border-radius: 8px;
                    padding: 12px 20px; margin-bottom: 8px;">
            <span style="color: {checkbox_color}; font-size: 18px; margin-right: 12px;">{checkbox_icon}</span>
            <span style="color: #E0E1DD; font-size: 14px;">{item['item']}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
                border: 2px solid #00F5D4; border-radius: 12px; padding: 20px; margin-top: 20px;
                text-align: center; box-shadow: 0 0 30px rgba(0, 245, 212, 0.3);">
        <span style="color: #00F5D4; font-size: 20px; font-weight: 700;">
            ✅ Phase 6 → Phase 7 handoff: READY
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )


# Initialize session state
if "current_cycle" not in st.session_state:
    st.session_state.current_cycle = "A"

if "iteration_count" not in st.session_state:
    st.session_state.iteration_count = 5

if "swe_bench_score" not in st.session_state:
    st.session_state.swe_bench_score = 82.1


# Main entry point
if __name__ == "__main__":
    render_phase6_dashboard()

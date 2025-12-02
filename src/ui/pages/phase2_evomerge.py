"""
Phase 2: EvoMerge - Evolutionary Optimization Dashboard

Comprehensive Streamlit dashboard for Phase 2 (EvoMerge) - 50 generations of
evolutionary model merging with 6 merge techniques (Linear, SLERP, TIES, DARE,
FrankenMerge, DFS).

Features:
- Real-time evolution progress (1-50 generations)
- Population fitness tracking and visualization
- Merge technique performance comparison
- Evolutionary tree visualization (lineage tracking)
- Fitness landscape contour plots
- Champion model selection and analysis
- Futuristic command center theme (dark #0D1B2A, cyan #00F5D4)
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

# ============================================================================
# THEME & STYLING
# ============================================================================

CUSTOM_PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#1B2838",
        font=dict(
            family="Space Grotesk, Inter, -apple-system, sans-serif", size=12, color="#E0E1DD"
        ),
        title=dict(
            font=dict(size=18, color="#00F5D4", family="Space Grotesk"), x=0.5, xanchor="center"
        ),
        xaxis=dict(
            gridcolor="#2E3F4F", zerolinecolor="#2E3F4F", color="#E0E1DD", linecolor="#2E3F4F"
        ),
        yaxis=dict(
            gridcolor="#2E3F4F", zerolinecolor="#2E3F4F", color="#E0E1DD", linecolor="#2E3F4F"
        ),
        colorway=["#00F5D4", "#FF006E", "#8338EC", "#FB5607", "#FFBE0B", "#06FFA5"],
        hovermode="closest",
        margin=dict(l=60, r=40, t=60, b=60),
    )
)


class MergeTechnique(Enum):
    """6 merge techniques used in Phase 2"""

    LINEAR = "Linear Interpolation"
    SLERP = "SLERP"
    TIES = "TIES"
    DARE = "DARE"
    FRANKENMERGE = "FrankenMerge"
    DFS = "DFS"


@dataclass
class ModelIndividual:
    """Individual model in the population"""

    id: str
    generation: int
    fitness: float
    parent1_id: Optional[str]
    parent2_id: Optional[str]
    merge_technique: Optional[str]
    position_x: float  # For visualization
    position_y: float


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_gradient_metric(
    label: str,
    value: str,
    delta: str = None,
    gradient_start: str = "#00F5D4",
    gradient_end: str = "#8338EC",
) -> str:
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
                    font-family: 'Space Grotesk', sans-serif;">
            {value}
        </div>
        {delta_html}
    </div>
    """
    return html


def generate_mock_evolution_data(current_gen: int = 25, pop_size: int = 20) -> Dict:
    """Generate mock evolution data for visualization"""
    np.random.seed(42)

    # Fitness improvement over generations
    generations = np.arange(1, current_gen + 1)
    base_fitness = 0.45
    fitness_trend = base_fitness + 0.15 * (1 - np.exp(-generations / 15))
    fitness_noise = np.random.normal(0, 0.01, len(generations))
    best_fitness_history = fitness_trend + fitness_noise
    avg_fitness_history = best_fitness_history - 0.08 - np.random.uniform(0, 0.02, len(generations))

    # Current population
    population = []
    for i in range(pop_size):
        fitness = np.random.uniform(0.50, 0.68)
        population.append(
            ModelIndividual(
                id=f"model_gen{current_gen}_{i}",
                generation=current_gen,
                fitness=fitness,
                parent1_id=f"model_gen{current_gen-1}_{np.random.randint(0, pop_size)}"
                if current_gen > 1
                else None,
                parent2_id=f"model_gen{current_gen-1}_{np.random.randint(0, pop_size)}"
                if current_gen > 1
                else None,
                merge_technique=np.random.choice([t.value for t in MergeTechnique]),
                position_x=np.random.uniform(-1, 1),
                position_y=np.random.uniform(-1, 1),
            )
        )

    # Merge technique stats
    technique_stats = {}
    for technique in MergeTechnique:
        technique_stats[technique.value] = {
            "usage_count": np.random.randint(50, 200),
            "success_rate": np.random.uniform(0.65, 0.92),
            "avg_fitness_contribution": np.random.uniform(0.03, 0.08),
            "color": {
                "Linear Interpolation": "#00F5D4",
                "SLERP": "#8338EC",
                "TIES": "#FF006E",
                "DARE": "#FB5607",
                "FrankenMerge": "#FFBE0B",
                "DFS": "#06FFA5",
            }[technique.value],
        }

    # Champion model
    champion = max(population, key=lambda x: x.fitness)

    return {
        "current_generation": current_gen,
        "population_size": pop_size,
        "best_fitness_history": best_fitness_history.tolist(),
        "avg_fitness_history": avg_fitness_history.tolist(),
        "current_population": population,
        "technique_stats": technique_stats,
        "champion": champion,
        "total_evaluations": current_gen * pop_size,
        "fitness_improvement": (best_fitness_history[-1] - best_fitness_history[0])
        / best_fitness_history[0],
    }


# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================


def create_evolution_progress_chart(data: Dict) -> go.Figure:
    """Create fitness improvement chart over generations"""
    generations = list(range(1, data["current_generation"] + 1))

    fig = go.Figure()

    # Average fitness (filled area)
    fig.add_trace(
        go.Scatter(
            x=generations,
            y=data["avg_fitness_history"],
            mode="lines",
            name="Population Average",
            line=dict(color="#8338EC", width=2, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(131, 56, 236, 0.1)",
            hovertemplate="<b>Gen %{x}</b><br>Avg Fitness: %{y:.3f}<extra></extra>",
        )
    )

    # Best fitness (glowing line)
    fig.add_trace(
        go.Scatter(
            x=generations,
            y=data["best_fitness_history"],
            mode="lines+markers",
            name="Best Fitness",
            line=dict(color="#00F5D4", width=3, shape="spline"),
            marker=dict(
                size=8, color="#00F5D4", line=dict(color="#FFFFFF", width=2), symbol="circle"
            ),
            hovertemplate="<b>Gen %{x}</b><br>Best Fitness: %{y:.3f}<extra></extra>",
        )
    )

    # Current generation marker
    current_gen = data["current_generation"]
    current_best = data["best_fitness_history"][-1]

    fig.add_annotation(
        x=current_gen,
        y=current_best,
        text=f"Current<br>Gen {current_gen}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#00F5D4",
        arrowwidth=2,
        ax=-60,
        ay=-40,
        font=dict(color="#00F5D4", size=12, family="Space Grotesk"),
        bgcolor="rgba(0, 245, 212, 0.1)",
        bordercolor="#00F5D4",
        borderwidth=2,
        borderpad=4,
    )

    # Fitness improvement annotation
    improvement = data["fitness_improvement"] * 100
    fig.add_annotation(
        x=1,
        y=data["best_fitness_history"][0],
        text=f"+{improvement:.1f}% improvement",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#8338EC",
        ax=60,
        ay=40,
        font=dict(color="#8338EC", size=11),
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Fitness Evolution Over Generations",
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        height=400,
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(27, 40, 56, 0.8)", bordercolor="#2E3F4F", borderwidth=1, x=0.02, y=0.98
        ),
    )

    return fig


def create_merge_techniques_panel(technique_stats: Dict) -> go.Figure:
    """Create merge techniques performance comparison"""
    techniques = list(technique_stats.keys())
    usage_counts = [technique_stats[t]["usage_count"] for t in techniques]
    success_rates = [technique_stats[t]["success_rate"] * 100 for t in techniques]
    fitness_contributions = [
        technique_stats[t]["avg_fitness_contribution"] * 100 for t in techniques
    ]
    colors = [technique_stats[t]["color"] for t in techniques]

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Usage Count", "Success Rate (%)", "Avg Fitness Contribution (%)"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
    )

    # Usage count
    fig.add_trace(
        go.Bar(
            x=techniques,
            y=usage_counts,
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            text=usage_counts,
            textposition="outside",
            textfont=dict(size=10, color="#E0E1DD"),
            hovertemplate="<b>%{x}</b><br>Used: %{y} times<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Success rate
    fig.add_trace(
        go.Bar(
            x=techniques,
            y=success_rates,
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            text=[f"{sr:.1f}%" for sr in success_rates],
            textposition="outside",
            textfont=dict(size=10, color="#E0E1DD"),
            hovertemplate="<b>%{x}</b><br>Success: %{y:.1f}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Fitness contribution
    fig.add_trace(
        go.Bar(
            x=techniques,
            y=fitness_contributions,
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            text=[f"+{fc:.2f}%" for fc in fitness_contributions],
            textposition="outside",
            textfont=dict(size=10, color="#E0E1DD"),
            hovertemplate="<b>%{x}</b><br>Contribution: +%{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # Target line for success rate
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#FF006E",
        line_width=2,
        annotation_text="Target: 80%",
        annotation_position="right",
        row=1,
        col=2,
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        height=400,
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=100),
    )

    return fig


def create_population_grid(population: List[ModelIndividual]) -> go.Figure:
    """Create visual grid of current population colored by fitness"""
    # Sort by fitness for grid layout
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

    # Create grid layout
    grid_size = int(np.ceil(np.sqrt(len(sorted_pop))))
    x_coords = []
    y_coords = []
    fitness_values = []
    hover_texts = []
    colors = []

    for i, individual in enumerate(sorted_pop):
        row = i // grid_size
        col = i % grid_size
        x_coords.append(col)
        y_coords.append(grid_size - 1 - row)  # Flip y-axis
        fitness_values.append(individual.fitness)

        hover_text = f"<b>{individual.id}</b><br>"
        hover_text += f"Fitness: {individual.fitness:.4f}<br>"
        hover_text += f"Gen: {individual.generation}<br>"
        if individual.merge_technique:
            hover_text += f"Technique: {individual.merge_technique}"
        hover_texts.append(hover_text)

        # Color by fitness (cyan to magenta gradient)
        if individual.fitness > 0.65:
            colors.append("#00F5D4")  # High fitness - cyan
        elif individual.fitness > 0.60:
            colors.append("#8338EC")  # Medium-high - purple
        elif individual.fitness > 0.55:
            colors.append("#FF006E")  # Medium - magenta
        else:
            colors.append("#4A5568")  # Low - gray

    fig = go.Figure()

    # Grid cells
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            marker=dict(
                size=40, color=colors, line=dict(color="#FFFFFF", width=2), symbol="square"
            ),
            text=[f"{f:.3f}" for f in fitness_values],
            textfont=dict(size=9, color="#FFFFFF", family="Space Grotesk"),
            textposition="middle center",
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Highlight champion (top-left, brightest glow)
    champion_idx = 0
    fig.add_trace(
        go.Scatter(
            x=[x_coords[champion_idx]],
            y=[y_coords[champion_idx]],
            mode="markers",
            marker=dict(
                size=50,
                color="rgba(0, 245, 212, 0)",
                line=dict(color="#00F5D4", width=4),
                symbol="square",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Current Population Grid (Sorted by Fitness)",
        xaxis=dict(visible=False, range=[-0.5, grid_size - 0.5]),
        yaxis=dict(visible=False, range=[-0.5, grid_size - 0.5]),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_evolutionary_tree_2d(current_gen: int = 25, pop_size: int = 20) -> go.Figure:
    """Create 2D evolutionary tree showing model lineage"""
    np.random.seed(42)

    # Generate tree data (simplified for visualization)
    generations = list(range(1, current_gen + 1))

    fig = go.Figure()

    # Draw connections between generations (simplified)
    for gen in range(1, current_gen):
        # Sample connections
        for _ in range(pop_size // 2):
            parent_x = gen
            child_x = gen + 1
            parent_y = np.random.uniform(0, pop_size)
            child_y = np.random.uniform(0, pop_size)

            # Color by merge technique
            technique_colors = ["#00F5D4", "#8338EC", "#FF006E", "#FB5607", "#FFBE0B", "#06FFA5"]
            color = np.random.choice(technique_colors)

            fig.add_trace(
                go.Scatter(
                    x=[parent_x, child_x],
                    y=[parent_y, child_y],
                    mode="lines",
                    line=dict(color=color, width=1, shape="spline"),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Draw nodes for each generation
    for gen in generations:
        y_positions = np.linspace(0, pop_size, pop_size)
        fitness_values = np.random.uniform(0.45 + gen * 0.01, 0.65 + gen * 0.01, pop_size)

        fig.add_trace(
            go.Scatter(
                x=[gen] * pop_size,
                y=y_positions,
                mode="markers",
                marker=dict(
                    size=8,
                    color=fitness_values,
                    colorscale=[[0, "#4A5568"], [0.5, "#8338EC"], [1, "#00F5D4"]],
                    line=dict(color="#FFFFFF", width=1),
                    showscale=False,
                ),
                hovertemplate=f"<b>Gen {gen}</b><br>Fitness: %{{marker.color:.3f}}<extra></extra>",
                showlegend=False,
            )
        )

    # Highlight current generation
    fig.add_vline(
        x=current_gen,
        line_dash="dash",
        line_color="#00F5D4",
        line_width=2,
        annotation_text=f"Current Gen {current_gen}",
        annotation_position="top",
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Evolutionary Tree (Model Lineage)",
        xaxis_title="Generation",
        yaxis_title="Population Index",
        height=600,
        hovermode="closest",
    )

    return fig


def create_fitness_landscape_3d(population: List[ModelIndividual]) -> go.Figure:
    """Create 3D fitness landscape with current population positions"""
    # Create mesh grid for fitness landscape
    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)

    # Generate fitness landscape (peaks and valleys)
    Z = (
        0.6
        + 0.1 * np.sin(3 * X) * np.cos(3 * Y)
        + 0.05 * np.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / 0.2)
        + 0.03 * np.exp(-((X + 0.4) ** 2 + (Y + 0.4) ** 2) / 0.15)
    )

    fig = go.Figure()

    # Surface plot (fitness landscape)
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=[[0, "#1B2838"], [0.3, "#4A5568"], [0.6, "#8338EC"], [1, "#00F5D4"]],
            opacity=0.7,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="#00F5D4", project=dict(z=True))
            ),
            name="Fitness Landscape",
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Fitness: %{z:.3f}<extra></extra>",
        )
    )

    # Current population positions
    pop_x = [ind.position_x for ind in population]
    pop_y = [ind.position_y for ind in population]
    pop_z = [ind.fitness for ind in population]

    fig.add_trace(
        go.Scatter3d(
            x=pop_x,
            y=pop_y,
            z=pop_z,
            mode="markers",
            marker=dict(
                size=8,
                color=pop_z,
                colorscale=[[0, "#FF006E"], [0.5, "#8338EC"], [1, "#00F5D4"]],
                line=dict(color="#FFFFFF", width=2),
                symbol="circle",
                showscale=True,
                colorbar=dict(
                    title="Fitness", titlefont=dict(color="#E0E1DD"), tickfont=dict(color="#E0E1DD")
                ),
            ),
            name="Current Population",
            hovertemplate="<b>%{text}</b><br>Fitness: %{z:.4f}<extra></extra>",
            text=[ind.id for ind in population],
        )
    )

    # Highlight champion
    champion = max(population, key=lambda x: x.fitness)
    fig.add_trace(
        go.Scatter3d(
            x=[champion.position_x],
            y=[champion.position_y],
            z=[champion.fitness],
            mode="markers",
            marker=dict(
                size=12, color="#00F5D4", line=dict(color="#FFFFFF", width=3), symbol="diamond"
            ),
            name="Champion",
            hovertemplate=f"<b>Champion</b><br>Fitness: {champion.fitness:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Fitness Landscape (3D)",
        scene=dict(
            xaxis=dict(title="Dimension X", gridcolor="#2E3F4F", color="#E0E1DD"),
            yaxis=dict(title="Dimension Y", gridcolor="#2E3F4F", color="#E0E1DD"),
            zaxis=dict(title="Fitness", gridcolor="#2E3F4F", color="#E0E1DD"),
            bgcolor="#0D1B2A",
        ),
        height=700,
        showlegend=True,
        legend=dict(bgcolor="rgba(27, 40, 56, 0.8)", bordercolor="#2E3F4F", borderwidth=1),
    )

    return fig


def create_fitness_landscape_contour(population: List[ModelIndividual]) -> go.Figure:
    """Create 2D contour plot of fitness landscape"""
    # Create mesh grid
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Generate fitness landscape
    Z = (
        0.6
        + 0.1 * np.sin(3 * X) * np.cos(3 * Y)
        + 0.05 * np.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / 0.2)
        + 0.03 * np.exp(-((X + 0.4) ** 2 + (Y + 0.4) ** 2) / 0.15)
    )

    fig = go.Figure()

    # Contour plot
    fig.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=Z,
            colorscale=[[0, "#1B2838"], [0.3, "#4A5568"], [0.6, "#8338EC"], [1, "#00F5D4"]],
            contours=dict(
                coloring="heatmap", showlabels=True, labelfont=dict(size=10, color="#FFFFFF")
            ),
            colorbar=dict(
                title="Fitness", titlefont=dict(color="#E0E1DD"), tickfont=dict(color="#E0E1DD")
            ),
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Fitness: %{z:.3f}<extra></extra>",
        )
    )

    # Current population
    pop_x = [ind.position_x for ind in population]
    pop_y = [ind.position_y for ind in population]
    pop_fitness = [ind.fitness for ind in population]

    fig.add_trace(
        go.Scatter(
            x=pop_x,
            y=pop_y,
            mode="markers",
            marker=dict(
                size=10, color="#FFFFFF", line=dict(color="#00F5D4", width=2), symbol="circle"
            ),
            name="Population",
            hovertemplate="<b>%{text}</b><br>Fitness: %{customdata:.4f}<extra></extra>",
            text=[ind.id for ind in population],
            customdata=pop_fitness,
        )
    )

    # Champion marker
    champion = max(population, key=lambda x: x.fitness)
    fig.add_trace(
        go.Scatter(
            x=[champion.position_x],
            y=[champion.position_y],
            mode="markers+text",
            marker=dict(
                size=16, color="#00F5D4", line=dict(color="#FFFFFF", width=3), symbol="star"
            ),
            text=["Champion"],
            textposition="top center",
            textfont=dict(size=12, color="#00F5D4", family="Space Grotesk"),
            name="Champion",
            hovertemplate=f"<b>Champion</b><br>Fitness: {champion.fitness:.4f}<extra></extra>",
        )
    )

    # Pareto frontier (approximate)
    pareto_x = np.linspace(-0.8, 0.8, 20)
    pareto_y = 0.1 * np.sin(2 * pareto_x) + 0.3

    fig.add_trace(
        go.Scatter(
            x=pareto_x,
            y=pareto_y,
            mode="lines",
            line=dict(color="#FF006E", width=2, dash="dash"),
            name="Pareto Frontier",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Fitness Landscape (Contour Map)",
        xaxis_title="Dimension X",
        yaxis_title="Dimension Y",
        height=600,
        hovermode="closest",
    )

    return fig


# ============================================================================
# DASHBOARD SECTIONS
# ============================================================================


def render_hero_section(data: Dict):
    """Hero section with phase title and evolution progress"""
    hero_html = """
    <div style="
        background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
        border: 2px solid #00F5D4;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 0 40px rgba(0, 245, 212, 0.2);
    ">
        <h1 style="
            color: #00F5D4;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 42px;
            font-weight: 700;
            margin: 0 0 10px 0;
            text-shadow: 0 0 20px rgba(0, 245, 212, 0.5);
        ">
            PHASE 2: EvoMerge
        </h1>
        <p style="
            color: #8B9DAF;
            font-size: 16px;
            margin: 0;
            letter-spacing: 1px;
        ">
            Evolutionary Optimization | 50 Generations | 6 Merge Techniques
        </p>
    </div>
    """
    components.html(hero_html, height=150)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            create_gradient_metric(
                "Current Generation",
                f"{data['current_generation']}/50",
                f"{(data['current_generation']/50)*100:.0f}% Complete",
                gradient_start="#00F5D4",
                gradient_end="#8338EC",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            create_gradient_metric(
                "Population Size",
                str(data["population_size"]),
                f"{data['total_evaluations']} total evals",
                gradient_start="#8338EC",
                gradient_end="#FF006E",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        best_fitness = data["best_fitness_history"][-1]
        st.markdown(
            create_gradient_metric(
                "Best Fitness",
                f"{best_fitness:.4f}",
                f"+{data['fitness_improvement']*100:.1f}% gain",
                gradient_start="#FF006E",
                gradient_end="#FB5607",
            ),
            unsafe_allow_html=True,
        )

    with col4:
        champion_fitness = data["champion"].fitness
        st.markdown(
            create_gradient_metric(
                "Champion Model",
                f"{champion_fitness:.4f}",
                "Ready for Phase 3",
                gradient_start="#FB5607",
                gradient_end="#00F5D4",
            ),
            unsafe_allow_html=True,
        )


def render_evolution_progress_section(data: Dict):
    """Evolution progress section"""
    st.markdown("### Evolution Progress")

    # Fitness chart
    fig_evolution = create_evolution_progress_chart(data)
    st.plotly_chart(fig_evolution, use_container_width=True)

    # Progress stats
    col1, col2, col3 = st.columns(3)

    with col1:
        initial_fitness = data["best_fitness_history"][0]
        st.markdown(
            f"""
        <div style="background: #1B283822; border: 1px solid #2E3F4F; border-radius: 8px; padding: 16px;">
            <div style="color: #8B9DAF; font-size: 12px; text-transform: uppercase; margin-bottom: 8px;">
                Initial Fitness (Gen 1)
            </div>
            <div style="color: #8338EC; font-size: 24px; font-weight: 700;">
                {initial_fitness:.4f}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        current_fitness = data["best_fitness_history"][-1]
        st.markdown(
            f"""
        <div style="background: #1B283822; border: 1px solid #2E3F4F; border-radius: 8px; padding: 16px;">
            <div style="color: #8B9DAF; font-size: 12px; text-transform: uppercase; margin-bottom: 8px;">
                Current Best (Gen {data['current_generation']})
            </div>
            <div style="color: #00F5D4; font-size: 24px; font-weight: 700;">
                {current_fitness:.4f}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        improvement_percent = data["fitness_improvement"] * 100
        st.markdown(
            f"""
        <div style="background: #1B283822; border: 1px solid #2E3F4F; border-radius: 8px; padding: 16px;">
            <div style="color: #8B9DAF; font-size: 12px; text-transform: uppercase; margin-bottom: 8px;">
                Total Improvement
            </div>
            <div style="color: #00F5D4; font-size: 24px; font-weight: 700;">
                +{improvement_percent:.1f}%
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_merge_techniques_section(technique_stats: Dict):
    """Merge techniques performance panel"""
    st.markdown("### Merge Techniques Performance")

    # Techniques comparison chart
    fig_techniques = create_merge_techniques_panel(technique_stats)
    st.plotly_chart(fig_techniques, use_container_width=True)

    # Detailed technique cards
    st.markdown("#### Technique Details")

    techniques = list(technique_stats.keys())
    cols = st.columns(3)

    for i, technique in enumerate(techniques):
        stats = technique_stats[technique]
        col_idx = i % 3

        with cols[col_idx]:
            st.markdown(
                f"""
            <div style="
                background: linear-gradient(135deg, {stats['color']}11 0%, {stats['color']}22 100%);
                border: 2px solid {stats['color']};
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 4px 12px {stats['color']}44;
            ">
                <h4 style="color: {stats['color']}; margin: 0 0 12px 0; font-family: 'Space Grotesk';">
                    {technique}
                </h4>
                <div style="color: #E0E1DD; font-size: 13px; line-height: 1.8;">
                    <div>Uses: <b>{stats['usage_count']}</b></div>
                    <div>Success: <b>{stats['success_rate']*100:.1f}%</b></div>
                    <div>Contribution: <b>+{stats['avg_fitness_contribution']*100:.2f}%</b></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_population_section(population: List[ModelIndividual]):
    """Population grid and stats"""
    st.markdown("### Current Population")

    # Population grid
    fig_grid = create_population_grid(population)
    st.plotly_chart(fig_grid, use_container_width=True)

    # Population statistics
    st.markdown("#### Population Statistics")

    fitness_values = [ind.fitness for ind in population]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Max Fitness", f"{max(fitness_values):.4f}")

    with col2:
        st.metric("Mean Fitness", f"{np.mean(fitness_values):.4f}")

    with col3:
        st.metric("Min Fitness", f"{min(fitness_values):.4f}")

    with col4:
        st.metric("Std Dev", f"{np.std(fitness_values):.4f}")


def render_evolutionary_tree_section(current_gen: int, pop_size: int):
    """Evolutionary tree visualization"""
    st.markdown("### Evolutionary Tree")

    # 2D tree
    fig_tree = create_evolutionary_tree_2d(current_gen, pop_size)
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown(
        """
    <div style="background: #1B283822; border-left: 4px solid #00F5D4; border-radius: 8px; padding: 16px;">
        <p style="color: #E0E1DD; margin: 0; line-height: 1.6;">
            Each node represents a model in the population. Lines show parent-child relationships.
            Color indicates merge technique used. Fitness increases over generations (left to right).
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_fitness_landscape_section(population: List[ModelIndividual]):
    """Fitness landscape visualization"""
    st.markdown("### Fitness Landscape")

    # Tab selection
    tab1, tab2 = st.tabs(["3D Surface", "2D Contour"])

    with tab1:
        fig_3d = create_fitness_landscape_3d(population)
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab2:
        fig_contour = create_fitness_landscape_contour(population)
        st.plotly_chart(fig_contour, use_container_width=True)

    st.markdown(
        """
    <div style="background: #1B283822; border-left: 4px solid #8338EC; border-radius: 8px; padding: 16px;">
        <p style="color: #E0E1DD; margin: 0; line-height: 1.6;">
            <b>Fitness Landscape:</b> Visualization of the search space. Peaks represent high-fitness regions.
            White dots show current population positions. Star marks champion model. Dashed line indicates Pareto frontier.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_champion_model_card(champion: ModelIndividual, technique_stats: Dict):
    """Champion model details card"""
    st.markdown("### Champion Model")

    champion_html = f"""
    <div style="
        background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
        border: 3px solid #00F5D4;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 0 40px rgba(0, 245, 212, 0.3);
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="
                display: inline-block;
                background: #00F5D4;
                color: #0D1B2A;
                font-size: 48px;
                font-weight: 700;
                padding: 20px 40px;
                border-radius: 12px;
                font-family: 'Space Grotesk', sans-serif;
                box-shadow: 0 0 30px rgba(0, 245, 212, 0.5);
            ">
                {champion.fitness:.4f}
            </div>
            <div style="color: #8B9DAF; font-size: 14px; margin-top: 10px; text-transform: uppercase; letter-spacing: 1px;">
                Champion Fitness Score
            </div>
        </div>

        <div style="
            background: #0D1B2A;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        ">
            <div style="color: #E0E1DD; font-size: 14px; line-height: 2;">
                <div><span style="color: #8B9DAF;">Model ID:</span> <b>{champion.id}</b></div>
                <div><span style="color: #8B9DAF;">Generation:</span> <b>{champion.generation}</b></div>
                <div><span style="color: #8B9DAF;">Merge Technique:</span> <b style="color: #00F5D4;">{champion.merge_technique}</b></div>
                <div><span style="color: #8B9DAF;">Position:</span> X={champion.position_x:.3f}, Y={champion.position_y:.3f}</div>
            </div>
        </div>

        <div style="
            background: #00F5D422;
            border: 2px solid #00F5D4;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        ">
            <div style="color: #00F5D4; font-size: 16px; font-weight: 700; font-family: 'Space Grotesk';">
                READY FOR PHASE 3 (Quiet-STaR)
            </div>
        </div>
    </div>
    """
    components.html(champion_html, height=320)

    # Fitness breakdown
    st.markdown("#### Fitness Breakdown")

    # Mock fitness components
    components = {
        "Perplexity": 0.25,
        "GSM8K Accuracy": 0.20,
        "MMLU": 0.18,
        "BBH": 0.15,
        "HumanEval": 0.10,
        "Diversity": 0.12,
    }

    component_names = list(components.keys())
    component_values = list(components.values())

    fig_breakdown = go.Figure()

    fig_breakdown.add_trace(
        go.Bar(
            x=component_names,
            y=component_values,
            marker=dict(
                color=["#00F5D4", "#8338EC", "#FF006E", "#FB5607", "#FFBE0B", "#06FFA5"],
                line=dict(color="#FFFFFF", width=1),
            ),
            text=[f"{v:.2f}" for v in component_values],
            textposition="outside",
            textfont=dict(size=11, color="#E0E1DD"),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
        )
    )

    fig_breakdown.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title="Champion Model Fitness Components",
        yaxis_title="Component Score",
        height=350,
        showlegend=False,
    )

    st.plotly_chart(fig_breakdown, use_container_width=True)


# ============================================================================
# MAIN DASHBOARD
# ============================================================================


def render_phase2_dashboard():
    """Main Phase 2 EvoMerge dashboard"""

    # Custom CSS
    st.markdown(
        """
    <style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
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
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
        color: #00F5D4;
        border: 1px solid #00F5D444;
        box-shadow: 0 0 20px rgba(0, 245, 212, 0.2);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        transition: all 0.3s ease;
        border: 1px solid #00F5D444;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 245, 212, 0.3);
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1B283822 0%, #2E3F4F22 100%);
        border: 1px solid #2E3F4F;
        border-radius: 8px;
        padding: 12px;
    }

    /* Glassmorphism effect */
    .glass-card {
        background: rgba(27, 40, 56, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 245, 212, 0.2);
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Evolution Controls")

        current_gen = st.slider(
            "Current Generation",
            min_value=1,
            max_value=50,
            value=25,
            help="Adjust to simulate different generation stages",
        )

        pop_size = st.slider(
            "Population Size",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Number of models in each generation",
        )

        st.markdown("---")
        st.markdown("### Phase 2 Settings")

        mutation_rate = st.slider(
            "Mutation Rate",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Probability of random parameter changes",
        )

        crossover_rate = st.slider(
            "Crossover Rate",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Probability of combining parent models",
        )

        elitism_count = st.number_input(
            "Elite Models (preserved)",
            min_value=1,
            max_value=10,
            value=2,
            help="Top N models automatically moved to next generation",
        )

        st.markdown("---")

        if st.button("Run Next Generation", type="primary", use_container_width=True):
            st.session_state.evolution_running = True
            st.rerun()

        if st.button("Reset Evolution", use_container_width=True):
            st.session_state.evolution_running = False
            st.rerun()

    # Generate mock data
    data = generate_mock_evolution_data(current_gen, pop_size)

    # Hero section
    render_hero_section(data)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Evolution Progress",
            "Merge Techniques",
            "Population & Tree",
            "Fitness Landscape",
            "Champion Model",
        ]
    )

    with tab1:
        render_evolution_progress_section(data)

    with tab2:
        render_merge_techniques_section(data["technique_stats"])

    with tab3:
        col1, col2 = st.columns([1, 1])

        with col1:
            render_population_section(data["current_population"])

        with col2:
            render_evolutionary_tree_section(current_gen, pop_size)

    with tab4:
        render_fitness_landscape_section(data["current_population"])

    with tab5:
        render_champion_model_card(data["champion"], data["technique_stats"])


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize session state
    if "evolution_running" not in st.session_state:
        st.session_state.evolution_running = False

    render_phase2_dashboard()

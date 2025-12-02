"""
3D Evolutionary Merge Tree Visualization for Phase 2 (EvoMerge)

This module provides an interactive 3D visualization of the evolutionary merge tree
showing 50 generations of model evolution with 6 different merge techniques.

Features:
- 3D scatter plot with generation (X), fitness (Y), and diversity (Z) axes
- Color-coded merge techniques (Linear, SLERP, TIES, DARE, FrankenMerge, DFS)
- Interactive lineage highlighting
- Hover information for each model
- Sample data generation for demonstration
"""

import random
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# MERGE TECHNIQUE CONFIGURATION
# ============================================================================

MERGE_TECHNIQUES = {
    "Linear": {"color": "#4287f5", "symbol": "circle"},
    "SLERP": {"color": "#39FF14", "symbol": "diamond"},
    "TIES": {"color": "#FFB703", "symbol": "square"},
    "DARE": {"color": "#B565D8", "symbol": "cross"},
    "FrankenMerge": {"color": "#FF006E", "symbol": "triangle-up"},
    "DFS": {"color": "#00F5D4", "symbol": "star"},
}


# ============================================================================
# DATA GENERATION
# ============================================================================


def generate_evolution_tree_data(
    generations: int = 50, models_per_gen: int = 8, initial_models: int = 3, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate sample evolutionary tree data for demonstration.

    Args:
        generations: Number of generations (default: 50)
        models_per_gen: Models per generation (default: 8)
        initial_models: Initial Phase 1 models (default: 3)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (nodes_df, edges_df):
            - nodes_df: DataFrame with columns [id, generation, fitness, diversity,
                        technique, parents, size]
            - edges_df: DataFrame with columns [parent_id, child_id, technique]
    """
    np.random.seed(seed)
    random.seed(seed)

    nodes = []
    edges = []

    # Generation 0: Phase 1 models (3 initial models)
    for i in range(initial_models):
        nodes.append(
            {
                "id": f"gen0_model{i}",
                "generation": 0,
                "fitness": np.random.uniform(0.65, 0.75),
                "diversity": np.random.uniform(0.4, 0.6),
                "technique": "Phase 1 (Cognate)",
                "parents": [],
                "size": 25_000_000,  # 25M parameters
            }
        )

    # Generations 1-50: Evolutionary merging
    best_fitness_history = [max(n["fitness"] for n in nodes)]

    for gen in range(1, generations + 1):
        gen_models = []

        # Get previous generation models
        prev_gen = [n for n in nodes if cast(str, n["generation"]) == gen - 1]

        # Generate models for this generation
        for m in range(models_per_gen):
            # Select merge technique
            technique = random.choice(list(MERGE_TECHNIQUES.keys()))

            # Select 2 parents from previous generation (binary pairing)
            if len(prev_gen) >= 2:
                parents = random.sample(prev_gen, 2)
            else:
                parents = random.sample(nodes, 2)

            parent_ids = [p["id"] for p in parents]

            # Calculate child fitness (inherit from parents with mutation)
            parent_fitness_avg = np.mean([p["fitness"] for p in parents])

            # Fitness improvement based on technique (TIES is most effective)
            technique_bonus = {
                "Linear": 0.01,
                "SLERP": 0.015,
                "TIES": 0.025,
                "DARE": 0.02,
                "FrankenMerge": 0.008,
                "DFS": 0.018,
            }

            # Add evolutionary improvement with some randomness
            fitness_mutation = np.random.normal(technique_bonus[technique], 0.01)

            # Add generational improvement (decreasing over time)
            gen_improvement = 0.003 * (1 - gen / generations)

            child_fitness = min(
                parent_fitness_avg + fitness_mutation + gen_improvement, 0.95  # Cap at 95% fitness
            )

            # Diversity metric (how different from parents)
            diversity_base = np.mean([p["diversity"] for p in parents])
            diversity = max(0.1, min(0.9, diversity_base + np.random.normal(0, 0.1)))

            # Model size (slightly compressed over generations)
            size = 25_000_000 * (1 - 0.001 * gen)

            model = {
                "id": f"gen{gen}_model{m}",
                "generation": gen,
                "fitness": child_fitness,
                "diversity": diversity,
                "technique": technique,
                "parents": parent_ids,
                "size": int(size),
            }

            nodes.append(model)
            gen_models.append(model)

            # Create edges
            for parent_id in parent_ids:
                edges.append(
                    {
                        "parent_id": parent_id,
                        "child_id": model["id"],
                        "technique": technique,
                    }
                )

        # Track best fitness
        best_fitness_history.append(max(m["fitness"] for m in gen_models))

    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)

    return nodes_df, edges_df


# ============================================================================
# 3D VISUALIZATION
# ============================================================================


def create_3d_merge_tree(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    highlight_lineage: Optional[str] = None,
    height: int = 800,
) -> go.Figure:
    """
    Create interactive 3D Plotly visualization of merge tree.

    Args:
        nodes_df: DataFrame with node data
        edges_df: DataFrame with edge data
        highlight_lineage: Model ID to highlight lineage (optional)
        height: Figure height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Dark theme background
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        title={
            "text": "3D Evolutionary Merge Tree (Phase 2: EvoMerge)",
            "font": {"size": 24, "color": "#00F5D4", "family": "Space Grotesk"},
            "x": 0.5,
            "xanchor": "center",
        },
        scene=dict(
            xaxis=dict(
                title="Generation",
                backgroundcolor="#0D1B2A",
                gridcolor="#2D3748",
                showbackground=True,
                zerolinecolor="#2D3748",
                range=[0, nodes_df["generation"].max()],
            ),
            yaxis=dict(
                title="Fitness Score",
                backgroundcolor="#0D1B2A",
                gridcolor="#2D3748",
                showbackground=True,
                zerolinecolor="#2D3748",
                range=[0, 1],
            ),
            zaxis=dict(
                title="Model Diversity",
                backgroundcolor="#0D1B2A",
                gridcolor="#2D3748",
                showbackground=True,
                zerolinecolor="#2D3748",
                range=[0, 1],
            ),
            bgcolor="#0D1B2A",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(27, 38, 59, 0.8)",
            bordercolor="#00F5D4",
            borderwidth=1,
            font=dict(color="#E0E1DD"),
        ),
        hovermode="closest",
    )

    # Plot edges (parent-child connections)
    for technique in MERGE_TECHNIQUES.keys():
        technique_edges = edges_df[cast(str, edges_df["technique"]) == technique]

        if len(technique_edges) == 0:
            continue

        # Prepare line data
        x_lines = []
        y_lines = []
        z_lines = []

        for _, edge in technique_edges.iterrows():
            parent = nodes_df[cast(str, nodes_df["id"]) == edge["parent_id"]].iloc[0]
            child = nodes_df[cast(str, nodes_df["id"]) == edge["child_id"]].iloc[0]

            # Add line segment (parent -> child)
            x_lines.extend([parent["generation"], child["generation"], None])
            y_lines.extend([parent["fitness"], child["fitness"], None])
            z_lines.extend([parent["diversity"], child["diversity"], None])

        # Plot technique-specific edges
        fig.add_trace(
            go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode="lines",
                line=dict(
                    color=MERGE_TECHNIQUES[technique]["color"],
                    width=2,
                ),
                opacity=0.3,
                name=f"{technique} merges",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # Plot nodes (models) by technique
    for technique, config in MERGE_TECHNIQUES.items():
        technique_nodes = nodes_df[cast(str, nodes_df["technique"]) == technique]

        if len(technique_nodes) == 0:
            continue

        # Size proportional to fitness
        sizes = technique_nodes["fitness"] * 10 + 5

        # Hover text
        hover_text = []
        for _, node in technique_nodes.iterrows():
            parents_str = ", ".join(node["parents"]) if node["parents"] else "None (Phase 1)"
            text = (
                f"<b>{node['id']}</b><br>"
                f"Generation: {node['generation']}<br>"
                f"Fitness: {node['fitness']:.3f}<br>"
                f"Diversity: {node['diversity']:.3f}<br>"
                f"Technique: {technique}<br>"
                f"Parents: {parents_str}<br>"
                f"Size: {node['size']:,} params"
            )
            hover_text.append(text)

        fig.add_trace(
            go.Scatter3d(
                x=technique_nodes["generation"],
                y=technique_nodes["fitness"],
                z=technique_nodes["diversity"],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=config["color"],
                    symbol=config["symbol"],
                    line=dict(color="#E0E1DD", width=1),
                    opacity=0.9,
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name=technique,
                showlegend=True,
            )
        )

    # Special handling for Phase 1 models (larger, distinctive)
    phase1_nodes = nodes_df[cast(str, nodes_df["technique"]) == "Phase 1 (Cognate)"]
    if len(phase1_nodes) > 0:
        hover_text = []
        for _, node in phase1_nodes.iterrows():
            text = (
                f"<b>{node['id']}</b><br>"
                f"<b>Phase 1 (Cognate) Model</b><br>"
                f"Generation: 0<br>"
                f"Fitness: {node['fitness']:.3f}<br>"
                f"Diversity: {node['diversity']:.3f}<br>"
                f"Size: 25M params (TRM x Titans-MAG)"
            )
            hover_text.append(text)

        fig.add_trace(
            go.Scatter3d(
                x=phase1_nodes["generation"],
                y=phase1_nodes["fitness"],
                z=phase1_nodes["diversity"],
                mode="markers",
                marker=dict(
                    size=20,
                    color="#F72585",  # Magenta accent
                    symbol="diamond",
                    line=dict(color="#E0E1DD", width=2),
                    opacity=1.0,
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="Phase 1 (Cognate)",
                showlegend=True,
            )
        )

    # Highlight lineage if specified
    if highlight_lineage and highlight_lineage in nodes_df["id"].values:
        lineage_nodes = _get_lineage_nodes(nodes_df, edges_df, highlight_lineage)
        lineage_df = nodes_df[nodes_df["id"].isin(lineage_nodes)]

        # Highlight nodes
        fig.add_trace(
            go.Scatter3d(
                x=lineage_df["generation"],
                y=lineage_df["fitness"],
                z=lineage_df["diversity"],
                mode="markers",
                marker=dict(
                    size=15,
                    color="#FFB703",  # Amber highlight
                    symbol="circle",
                    line=dict(color="#FFFFFF", width=3),
                    opacity=1.0,
                ),
                name="Highlighted Lineage",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    return fig


def _get_lineage_nodes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, node_id: str) -> List[str]:
    """
    Get all ancestor nodes (lineage) for a given node.

    Args:
        nodes_df: Nodes DataFrame
        edges_df: Edges DataFrame
        node_id: Target node ID

    Returns:
        List of node IDs in lineage (including target)
    """
    lineage = {node_id}
    to_process = [node_id]

    while to_process:
        current = to_process.pop()

        # Find parents
        parent_edges = edges_df[cast(str, edges_df["child_id"]) == current]

        for _, edge in parent_edges.iterrows():
            parent_id = edge["parent_id"]
            if parent_id not in lineage:
                lineage.add(parent_id)
                to_process.append(parent_id)

    return list(lineage)


# ============================================================================
# STREAMLIT COMPONENT
# ============================================================================


def render_phase2_3d_visualization(
    generations: int = 50, models_per_gen: int = 8, height: int = 800, show_controls: bool = True
) -> None:
    """
    Render the complete Phase 2 3D visualization component in Streamlit.

    Args:
        generations: Number of generations to simulate
        models_per_gen: Models per generation
        height: Figure height in pixels
        show_controls: Show interactive controls
    """
    st.markdown("---")
    st.markdown(
        '<h2 class="section-header">3D Merge Tree Visualization</h2>', unsafe_allow_html=True
    )

    # Generate or load data
    if "merge_tree_data" not in st.session_state:
        with st.spinner("Generating evolutionary tree data..."):
            nodes_df, edges_df = generate_evolution_tree_data(
                generations=generations, models_per_gen=models_per_gen
            )
            st.session_state.merge_tree_data = (nodes_df, edges_df)

    nodes_df, edges_df = st.session_state.merge_tree_data

    # Controls
    if show_controls:
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            # Model selection for lineage highlighting
            model_ids = ["None"] + sorted(nodes_df["id"].tolist())
            selected_model = st.selectbox(
                "Highlight Lineage (click model)", model_ids, index=0, key="lineage_select"
            )

        with col2:
            # Generation range filter
            max_gen = int(nodes_df["generation"].max())
            gen_range = st.slider(
                "Generation Range", 0, max_gen, (0, max_gen), key="gen_range_slider"
            )

        with col3:
            # Regenerate button
            if st.button("Regenerate Tree", key="regen_tree"):
                del st.session_state.merge_tree_data
                st.rerun()
    else:
        selected_model = None
        gen_range = (0, int(nodes_df["generation"].max()))

    # Filter data by generation range
    filtered_nodes = nodes_df[
        (cast(int, nodes_df["generation"]) >= gen_range[0])
        & (cast(int, nodes_df["generation"]) <= gen_range[1])
    ]
    filtered_edges = edges_df[
        edges_df["child_id"].isin(filtered_nodes["id"])
        & edges_df["parent_id"].isin(filtered_nodes["id"])
    ]

    # Create and display figure
    highlight = selected_model if selected_model != "None" else None

    fig = create_3d_merge_tree(
        filtered_nodes, filtered_edges, highlight_lineage=highlight, height=height
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.markdown("---")
    st.markdown("### Evolution Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        initial_fitness = nodes_df[cast(str, nodes_df["generation"]) == 0]["fitness"].mean()
        st.metric("Initial Avg Fitness", f"{initial_fitness:.3f}", delta=None)

    with col2:
        final_fitness = nodes_df[cast(str, nodes_df["generation"]) == nodes_df["generation"].max()][
            "fitness"
        ].mean()
        st.metric(
            "Final Avg Fitness",
            f"{final_fitness:.3f}",
            delta=f"+{(final_fitness - initial_fitness):.3f}",
        )

    with col3:
        best_fitness = nodes_df["fitness"].max()
        best_model = nodes_df[cast(str, nodes_df["fitness"]) == best_fitness].iloc[0]
        st.metric(
            "Best Fitness", f"{best_fitness:.3f}", delta=f'Gen {int(best_model["generation"])}'
        )

    with col4:
        improvement = ((final_fitness - initial_fitness) / initial_fitness) * 100
        st.metric("Total Improvement", f"{improvement:.1f}%", delta=f"{generations} generations")

    # Technique breakdown
    with st.expander("View Merge Technique Breakdown"):
        technique_stats = (
            nodes_df[cast(str, nodes_df["technique"]) != "Phase 1 (Cognate)"]
            .groupby("technique")
            .agg({"fitness": ["count", "mean", "max"], "id": "count"})
            .round(3)
        )

        st.dataframe(technique_stats, use_container_width=True)


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directories to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from ui.design_system import get_custom_css

    st.set_page_config(
        page_title="3D Merge Tree Visualization",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Render component
    st.markdown('<h1 class="main-header">Phase 2: EvoMerge</h1>', unsafe_allow_html=True)

    render_phase2_3d_visualization(generations=50, models_per_gen=8, height=800, show_controls=True)

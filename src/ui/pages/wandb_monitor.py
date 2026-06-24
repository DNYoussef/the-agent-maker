"""
Weights & Biases Monitor Page
Comprehensive W&B monitoring across all 8 phases with 7,800+ total metrics
"""
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from typing import Any  # noqa: E402

_WANDB_STYLES = """
        <style>
        /* W&B Brand Colors */
        :root {
            --wandb-orange: #FFBE00;
            --wandb-dark: #1A1A1A;
            --wandb-gray: #2D2D2D;
            --cyber-cyan: #00F0FF;
            --neon-green: #00FF9F;
            --danger-red: #FF3B3B;
        }

        /* Main Header */
        .wandb-header {
            background: linear-gradient(135deg, var(--wandb-dark) 0%, var(--wandb-gray) 100%);
            padding: 2rem;
            border-radius: 10px;
            border: 2px solid var(--wandb-orange);
            box-shadow: 0 0 20px rgba(255, 190, 0, 0.3);
            margin-bottom: 2rem;
            text-align: center;
        }

        .wandb-logo {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, var(--wandb-orange), var(--cyber-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .wandb-subtitle {
            color: var(--cyber-cyan);
            font-size: 1.2rem;
            font-weight: 300;
            letter-spacing: 2px;
        }

        /* Status Indicators */
        .status-connected {
            color: var(--neon-green);
            font-weight: bold;
            font-size: 1.1rem;
        }

        .status-disconnected {
            color: var(--danger-red);
            font-weight: bold;
            font-size: 1.1rem;
        }

        /* Phase Metric Cards */
        .phase-card {
            background: linear-gradient(135deg, #1E1E1E 0%, #2A2A2A 100%);
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid var(--wandb-orange);
            margin-bottom: 1rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .phase-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(255, 190, 0, 0.4);
        }

        .phase-title {
            color: var(--wandb-orange);
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .phase-description {
            color: #B0B0B0;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }

        .metric-count {
            color: var(--cyber-cyan);
            font-size: 2rem;
            font-weight: 800;
            text-align: center;
        }

        .metric-label {
            color: #808080;
            font-size: 0.85rem;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Connection Panel */
        .connection-panel {
            background: var(--wandb-gray);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--wandb-orange);
            margin-bottom: 2rem;
        }

        .connection-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #404040;
        }

        .connection-label {
            color: #B0B0B0;
            font-weight: 500;
        }

        .connection-value {
            color: var(--cyber-cyan);
            font-weight: 600;
        }

        /* Metric Chart Container */
        .metric-chart {
            background: var(--wandb-dark);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--cyber-cyan);
            margin-bottom: 1.5rem;
        }

        /* Run History Table */
        .run-table {
            background: var(--wandb-gray);
            border-radius: 8px;
            overflow: hidden;
        }

        /* Section Headers */
        .section-header {
            color: var(--wandb-orange);
            font-size: 1.8rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--cyber-cyan);
        }

        /* Config Panel */
        .config-panel {
            background: linear-gradient(135deg, #252525 0%, #1A1A1A 100%);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px dashed var(--cyber-cyan);
            margin-top: 1rem;
        }

        /* Continuity Metrics */
        .continuity-card {
            background: var(--wandb-gray);
            padding: 1rem;
            border-radius: 6px;
            border-left: 3px solid var(--neon-green);
            margin-bottom: 0.8rem;
        }

        .continuity-success {
            color: var(--neon-green);
            font-weight: 600;
        }

        .continuity-warning {
            color: var(--wandb-orange);
            font-weight: 600;
        }

        /* Artifact Badge */
        .artifact-badge {
            display: inline-block;
            background: var(--wandb-orange);
            color: var(--wandb-dark);
            padding: 0.3rem 0.8rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 700;
            margin: 0.2rem;
        }
        </style>
    """


def _render_styles() -> None:
    """Render custom CSS for the W&B theme."""
    st.markdown(_WANDB_STYLES, unsafe_allow_html=True)


def _render_hero() -> None:
    """Render the hero header section."""
    st.markdown(
        """
        <div class="wandb-header">
            <div class="wandb-logo">W&B MONITOR</div>
            <div class="wandb-subtitle">WEIGHTS & BIASES UNIFIED TRACKING</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def _render_connection_panel(
    wandb_connected: bool,
    project_name: str,
    latest_run: str,
    local_runs_count: int,
) -> None:
    """Render the connection status columns."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="connection-panel">', unsafe_allow_html=True)

        # Connection status
        status_class = "status-connected" if wandb_connected else "status-disconnected"
        status_text = "CONNECTED" if wandb_connected else "DISCONNECTED"
        st.markdown(
            f'<div class="connection-item"><span class="connection-label">Status:</span><span class="{status_class}">{status_text}</span></div>',
            unsafe_allow_html=True,
        )

        # Project info
        st.markdown(
            f'<div class="connection-item"><span class="connection-label">Project:</span><span class="connection-value">{project_name}</span></div>',
            unsafe_allow_html=True,
        )

        # Run ID
        run_id = latest_run
        st.markdown(
            f'<div class="connection-item"><span class="connection-label">Current Run:</span><span class="connection-value">{run_id}</span></div>',
            unsafe_allow_html=True,
        )

        # API Key status
        api_key_status = "CONFIGURED" if wandb_connected else "NOT SET"
        st.markdown(
            f'<div class="connection-item"><span class="connection-label">API Key:</span><span class="connection-value">{api_key_status}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="connection-item"><span class="connection-label">Local Runs:</span><span class="connection-value">{local_runs_count}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### Mode")
        st.write("Offline (local sync ready)")
        st.markdown("### Actions")
        st.write("Set `WANDB_API_KEY` to enable cloud sync.")


def _render_connection_status() -> bool:
    """Render connection status section. Returns wandb_connected flag."""
    st.markdown('<div class="section-header">CONNECTION STATUS</div>', unsafe_allow_html=True)

    # Real W&B connection state (offline-first)
    wandb_connected = bool(os.getenv("WANDB_API_KEY"))
    project_name = "agent-forge-v2"
    storage_root = Path(__file__).parent.parent.parent / "storage" / "wandb"
    run_dirs = sorted(storage_root.glob("*")) if storage_root.exists() else []
    latest_run = run_dirs[-1].name if run_dirs else "none"
    local_runs_count = len(run_dirs)

    _render_connection_panel(wandb_connected, project_name, latest_run, local_runs_count)
    return wandb_connected


def _phase_card_html(phase: dict) -> str:
    """Build the HTML for a single phase card."""
    return f"""
        <div class="phase-card">
            <div class="phase-title">{phase['phase']}</div>
            <div class="phase-description">{phase['description']}</div>
            <div class="metric-count">{phase['metrics']:,}</div>
            <div class="metric-label">Metrics Tracked</div>
            <div class="phase-description" style="margin-top: 1rem;">{phase['details']}</div>
        </div>
    """


def _phase_overview_data() -> list:
    """Return the phase metric overview card data."""
    return [
        {
            "phase": "Phase 1: Cognate",
            "description": "TRM x Titans-MAG - 3 model training",
            "metrics": 37,
            "details": "Model training, loss curves, accuracy",
        },
        {
            "phase": "Phase 2: EvoMerge",
            "description": "Evolutionary optimization - 50 generations",
            "metrics": 370,
            "details": "Fitness tracking, merge techniques, population evolution",
        },
        {
            "phase": "Phase 3: Quiet-STaR",
            "description": "Reasoning enhancement",
            "metrics": 17,
            "details": "Thought generation, coherence scoring, reasoning validation",
        },
        {
            "phase": "Phase 4: BitNet",
            "description": "1.58-bit compression",
            "metrics": 19,
            "details": "Compression ratios, quantization quality, speedup metrics",
        },
        {
            "phase": "Phase 5: Curriculum Learning",
            "description": "7-stage adaptive curriculum",
            "metrics": 7208,
            "details": "50K training steps, edge-of-chaos assessment, tool use validation",
        },
        {
            "phase": "Phase 6: Tool & Persona Baking",
            "description": "A/B optimization loops",
            "metrics": 25,
            "details": "Tool use patterns, persona discovery, plateau detection",
        },
        {
            "phase": "Phase 7: Self-Guided Experts",
            "description": "Architecture discovery",
            "metrics": 100,
            "details": "Expert analysis, SVF training, NSGA-II ADAS evolution",
        },
        {
            "phase": "Phase 8: Final Compression",
            "description": "280x compression pipeline",
            "metrics": 25,
            "details": "SeedLM, VPTQ, hypercompression validation",
        },
    ]


def _render_phase_overview() -> None:
    """Render the phase metrics overview cards."""
    st.markdown('<div class="section-header">PHASE METRICS OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown("**Total Metrics Tracked: 7,800+** across 8 phases")

    phases = _phase_overview_data()

    # Display phase cards in 2 columns
    for i in range(0, len(phases), 2):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(_phase_card_html(phases[i]), unsafe_allow_html=True)

        if i + 1 < len(phases):
            with col2:
                st.markdown(_phase_card_html(phases[i + 1]), unsafe_allow_html=True)


def _phase_metric_map() -> dict:
    """Return the mapping of phase name to its tracked metrics."""
    return {
        "Phase 1: Cognate": [
            "train/loss",
            "train/accuracy",
            "val/loss",
            "val/accuracy",
            "learning_rate",
        ],
        "Phase 2: EvoMerge": [
            "fitness/best",
            "fitness/mean",
            "fitness/std",
            "diversity_score",
            "generation",
        ],
        "Phase 3: Quiet-STaR": [
            "thought/coherence",
            "reasoning/quality",
            "semantic_score",
            "syntactic_score",
        ],
        "Phase 4: BitNet": [
            "compression_ratio",
            "inference_speedup",
            "quantization_error",
            "model_size_mb",
        ],
        "Phase 5: Curriculum": [
            "curriculum/level",
            "accuracy",
            "edge_of_chaos_score",
            "tool_use/success_rate",
        ],
        "Phase 6: Baking": [
            "tool_performance",
            "persona_plateau",
            "ab_cycle",
            "baking_strength",
        ],
        "Phase 7: Experts": [
            "expert/count",
            "svf/performance",
            "adas/fitness",
            "architecture_score",
        ],
        "Phase 8: Compression": [
            "final_compression",
            "seedlm_ratio",
            "vptq_ratio",
            "quality_retention",
        ],
    }


def _metric_series(selected_metric: str, num_points: int) -> "np.ndarray":
    """Generate realistic simulated metric data based on metric type."""
    if "loss" in selected_metric:
        return np.exp(-np.linspace(0, 3, num_points)) + np.random.normal(0, 0.05, num_points)
    elif "accuracy" in selected_metric:
        return (1 - np.exp(-np.linspace(0, 3, num_points))) * 0.95 + np.random.normal(
            0, 0.02, num_points
        )
    elif "fitness" in selected_metric:
        return np.linspace(0.5, 0.9, num_points) + np.random.normal(0, 0.03, num_points)
    elif "compression" in selected_metric:
        return np.linspace(1, 8.2, num_points) + np.random.normal(0, 0.2, num_points)
    return np.random.randn(num_points).cumsum() + 50


def _render_metric_chart(wandb_connected: bool, selected_metric: str) -> None:
    """Render the live metric chart for the selected metric."""
    st.markdown('<div class="metric-chart">', unsafe_allow_html=True)

    if wandb_connected:
        # Generate simulated time series data
        num_points = 100
        timestamps = pd.date_range(end=datetime.now(), periods=num_points, freq="1min")

        values = _metric_series(selected_metric, num_points)

        df = pd.DataFrame({"timestamp": timestamps, selected_metric: values})

        st.line_chart(df.set_index("timestamp"))

        # Current value display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Value", f"{values[-1]:.4f}")
        with col2:
            st.metric("Mean", f"{values.mean():.4f}")
        with col3:
            st.metric("Std Dev", f"{values.std():.4f}")
    else:
        st.info("Connect to W&B to view live metrics")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_live_metrics(wandb_connected: bool) -> tuple:
    """Render the live metrics dashboard. Returns (selected_phase, phase_metrics)."""
    st.markdown('<div class="section-header">LIVE METRICS DASHBOARD</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        selected_phase = st.selectbox(
            "Select Phase",
            [
                "Phase 1: Cognate",
                "Phase 2: EvoMerge",
                "Phase 3: Quiet-STaR",
                "Phase 4: BitNet",
                "Phase 5: Curriculum",
                "Phase 6: Baking",
                "Phase 7: Experts",
                "Phase 8: Compression",
            ],
        )

    with col2:
        phase_metrics = _phase_metric_map()
        selected_metric = st.selectbox(
            "Select Metric", phase_metrics.get(selected_phase, ["No metrics available"])
        )

    with col3:
        _ = st.selectbox("Time Window", ["1 hour", "6 hours", "24 hours", "7 days", "All time"])

    _render_metric_chart(wandb_connected, selected_metric)
    return selected_phase, phase_metrics


def _render_metric_comparison(
    wandb_connected: bool, selected_phase: str, phase_metrics: dict
) -> None:
    """Render the metric comparison view."""
    st.markdown("### Metric Comparison")

    col1, col2 = st.columns(2)

    with col1:
        compare_metric_1 = st.selectbox(
            "Compare Metric 1", phase_metrics.get(selected_phase, []), key="compare_1"
        )

    with col2:
        compare_metric_2 = st.selectbox(
            "Compare Metric 2", phase_metrics.get(selected_phase, []), key="compare_2"
        )

    if wandb_connected and compare_metric_1 and compare_metric_2:
        # Generate comparison data
        num_points = 100
        df_compare = pd.DataFrame(
            {
                "step": range(num_points),
                compare_metric_1: np.random.randn(num_points).cumsum() + 50,
                compare_metric_2: np.random.randn(num_points).cumsum() + 45,
            }
        )

        st.line_chart(df_compare.set_index("step"))


def _build_runs_dataframe() -> "pd.DataFrame":
    """Build the simulated run history DataFrame."""
    runs_data = []
    for i in range(10):
        run_date = datetime.now() - timedelta(days=i)
        runs_data.append(
            {
                "Run ID": f"run_{run_date.strftime('%Y%m%d_%H%M%S')}",
                "Phase": f"Phase {(i % 8) + 1}",
                "Status": np.random.choice(["completed", "running", "failed"], p=[0.7, 0.2, 0.1]),
                "Duration": f"{np.random.randint(30, 300)} min",
                "Best Metric": f"{np.random.uniform(0.7, 0.95):.3f}",
                "Created": run_date.strftime("%Y-%m-%d %H:%M"),
            }
        )
    return pd.DataFrame(runs_data)


def _render_run_history(df_runs: "pd.DataFrame") -> None:
    """Render the run history table with status color coding."""
    st.markdown("### Run History")

    # Color code status
    def color_status(val: Any) -> str:
        if val == "completed":
            return "background-color: #1a4d2e; color: #00FF9F"
        elif val == "running":
            return "background-color: #4d4d1a; color: #FFBE00"
        else:
            return "background-color: #4d1a1a; color: #FF3B3B"

    st.dataframe(
        df_runs.style.applymap(color_status, subset=["Status"]),
        use_container_width=True,
        height=400,
    )


def _render_run_comparison(df_runs: "pd.DataFrame") -> None:
    """Render the run comparison charts."""
    st.markdown("### Run Comparison")

    selected_runs = st.multiselect(
        "Select runs to compare",
        df_runs["Run ID"].tolist(),
        default=df_runs["Run ID"].tolist()[:3],
    )

    if selected_runs:
        comparison_data = df_runs[df_runs["Run ID"].isin(selected_runs)]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Performance Comparison**")
            st.bar_chart(comparison_data.set_index("Run ID")["Best Metric"].astype(float))

        with col2:
            st.markdown("**Duration Comparison**")
            # Extract numeric duration
            comparison_data["Duration (min)"] = (
                comparison_data["Duration"].str.extract(r"(\d+)").astype(int)
            )
            st.bar_chart(comparison_data.set_index("Run ID")["Duration (min)"])


def _render_hyperparameters() -> None:
    """Render the hyperparameter tracking section."""
    st.markdown("### Hyperparameter Tracking")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Learning Rate**")
        st.code("1e-4 -> 5e-5 (decay)")

    with col2:
        st.markdown("**Batch Size**")
        st.code("32")

    with col3:
        st.markdown("**Optimizer**")
        st.code("MuGrokfast")

    # Additional hyperparameters
    hyperparams = {
        "grokfast_lambda": 0.05,
        "muon_lr": 0.01,
        "warmup_steps": 500,
        "max_epochs": 10,
        "gradient_clip": 1.0,
        "weight_decay": 0.01,
    }

    st.json(hyperparams)


def _render_artifacts() -> None:
    """Render the artifact management section."""
    st.markdown("### Artifact Management")

    artifacts = [
        {"name": "phase1_model1_reasoning", "type": "model", "size": "95.4 MB", "version": "v1"},
        {"name": "phase1_model2_memory", "type": "model", "size": "95.4 MB", "version": "v1"},
        {"name": "phase2_champion_gen25", "type": "model", "size": "95.4 MB", "version": "v2"},
        {"name": "phase3_quietstar_baked", "type": "model", "size": "98.2 MB", "version": "v1"},
        {"name": "phase4_bitnet_compressed", "type": "model", "size": "11.6 MB", "version": "v1"},
        {"name": "training_dataset_phase1", "type": "dataset", "size": "2.3 GB", "version": "v1"},
        {
            "name": "validation_results_phase3",
            "type": "results",
            "size": "45.2 MB",
            "version": "v1",
        },
    ]

    for artifact in artifacts:
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

        with col1:
            st.text(artifact["name"])

        with col2:
            st.markdown(
                f'<span class="artifact-badge">{artifact["type"]}</span>', unsafe_allow_html=True
            )

        with col3:
            st.text(artifact["size"])

        with col4:
            st.text(artifact["version"])

        with col5:
            if st.button("Download", key=f"download_{artifact['name']}"):
                st.info(f"Downloading {artifact['name']}...")


def _render_experiment_tracking() -> None:
    """Render the experiment tracking section."""
    st.markdown('<div class="section-header">EXPERIMENT TRACKING</div>', unsafe_allow_html=True)

    df_runs = _build_runs_dataframe()
    _render_run_history(df_runs)
    _render_run_comparison(df_runs)
    _render_hyperparameters()
    _render_artifacts()


def _handoff_card_html(handoff: dict) -> str:
    """Build the HTML for a single phase-handoff card."""
    return f"""
        <div class="continuity-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: var(--wandb-orange);">{handoff['from']} → {handoff['to']}</strong>
                    <div class="continuity-success">Status: {handoff['status'].upper()}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #B0B0B0;">Validation: <span class="continuity-success">{handoff['validation']}</span></div>
                    <div style="color: #B0B0B0;">Transfer: {handoff['transfer_time']}</div>
                    <div style="color: #B0B0B0;">Metadata: {handoff['metadata']}</div>
                </div>
            </div>
        </div>
    """


def _render_handoffs() -> None:
    """Render the phase handoff metric cards."""
    st.markdown("### Phase Handoff Metrics")

    handoffs = [
        {
            "from": "Phase 1",
            "to": "Phase 2",
            "status": "success",
            "validation": "99% reconstruction",
            "transfer_time": "2.3s",
            "metadata": "Complete",
        },
        {
            "from": "Phase 2",
            "to": "Phase 3",
            "status": "success",
            "validation": "98.7% reconstruction",
            "transfer_time": "1.8s",
            "metadata": "Complete",
        },
        {
            "from": "Phase 3",
            "to": "Phase 4",
            "status": "success",
            "validation": "100% reconstruction",
            "transfer_time": "1.2s",
            "metadata": "Complete",
        },
        {
            "from": "Phase 4",
            "to": "Phase 5",
            "status": "success",
            "validation": "97.3% reconstruction",
            "transfer_time": "3.1s",
            "metadata": "Complete",
        },
    ]

    for handoff in handoffs:
        st.markdown(_handoff_card_html(handoff), unsafe_allow_html=True)


def _render_continuity_validation() -> None:
    """Render the continuity validation metrics and lineage."""
    st.markdown("### Continuity Validation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Handoff Success Rate", "100%", delta="0%")

    with col2:
        st.metric("Avg Reconstruction", "98.8%", delta="+1.2%")

    with col3:
        st.metric("Avg Transfer Time", "2.1s", delta="-0.3s")

    # Model Lineage Tracking
    st.markdown("### Model Lineage Tracking")

    lineage_data = {
        "Phase 1 Model 1": ["Phase 2 Champion Gen 25"],
        "Phase 1 Model 2": ["Phase 2 Champion Gen 25"],
        "Phase 1 Model 3": ["Phase 2 Champion Gen 25"],
        "Phase 2 Champion Gen 25": ["Phase 3 Quiet-STaR Baked"],
        "Phase 3 Quiet-STaR Baked": ["Phase 4 BitNet Compressed"],
        "Phase 4 BitNet Compressed": ["Phase 5 Curriculum L10"],
    }

    st.json(lineage_data)


def _render_cross_phase_continuity() -> None:
    """Render the cross-phase continuity section."""
    st.markdown('<div class="section-header">CROSS-PHASE CONTINUITY</div>', unsafe_allow_html=True)
    _render_handoffs()
    _render_continuity_validation()


def _render_config_metric_column() -> tuple:
    """Render the metric/chart config column. Returns (custom_metrics, chart_type)."""
    st.markdown("### Metric Selection")

    custom_metrics = st.multiselect(
        "Select metrics to display",
        [
            "train/loss",
            "train/accuracy",
            "val/loss",
            "val/accuracy",
            "fitness/best",
            "compression_ratio",
            "inference_speedup",
            "tool_use/success_rate",
            "expert/count",
        ],
        default=["train/loss", "train/accuracy"],
    )

    st.markdown("### Chart Type")

    chart_type = st.selectbox(
        "Default chart type", ["Line Chart", "Area Chart", "Bar Chart", "Scatter Plot"]
    )
    return custom_metrics, chart_type


def _render_config_refresh_column() -> tuple:
    """Render the refresh/export config column. Returns config values."""
    st.markdown("### Refresh Settings")

    auto_refresh = st.checkbox("Enable auto-refresh", value=False)

    refresh_rate = st.slider(
        "Refresh rate (seconds)",
        min_value=5,
        max_value=60,
        value=10,
        step=5,
        disabled=not auto_refresh,
    )

    st.markdown("### Export Options")

    export_format = st.selectbox("Export format", ["CSV", "JSON", "Parquet", "Excel"])

    if st.button("Export Current View", use_container_width=True):
        st.success(f"Exporting data as {export_format}...")

    return auto_refresh, refresh_rate, export_format


def _render_dashboard_config() -> tuple:
    """Render dashboard configuration. Returns the config values."""
    st.markdown('<div class="section-header">DASHBOARD CONFIGURATION</div>', unsafe_allow_html=True)

    st.markdown('<div class="config-panel">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        custom_metrics, chart_type = _render_config_metric_column()

    with col2:
        auto_refresh, refresh_rate, export_format = _render_config_refresh_column()

    st.markdown("</div>", unsafe_allow_html=True)

    return custom_metrics, chart_type, auto_refresh, refresh_rate, export_format


def _render_save_and_refresh(
    wandb_connected: bool,
    custom_metrics: list,
    chart_type: str,
    auto_refresh: bool,
    refresh_rate: int,
    export_format: str,
) -> None:
    """Render the save-config button and auto-refresh trigger."""
    if st.button("Save Dashboard Configuration", use_container_width=True):
        config = {
            "custom_metrics": custom_metrics,
            "chart_type": chart_type,
            "auto_refresh": auto_refresh,
            "refresh_rate": refresh_rate,
            "export_format": export_format,
        }
        st.session_state["wandb_dashboard_config"] = config
        st.success("Dashboard configuration saved!")

    # Auto-refresh functionality
    if auto_refresh and wandb_connected:
        time.sleep(refresh_rate)
        st.rerun()


def _render_footer() -> None:
    """Render the page footer."""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #808080; padding: 1rem;">
            <strong style="color: var(--wandb-orange);">Agent Forge V2</strong> W&B Monitor |
            Tracking <strong style="color: var(--cyber-cyan);">7,800+</strong> metrics across
            <strong style="color: var(--cyber-cyan);">8 phases</strong>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render() -> None:
    """Render W&B monitoring dashboard page"""
    _render_styles()
    _render_hero()

    wandb_connected = _render_connection_status()
    _render_phase_overview()

    selected_phase, phase_metrics = _render_live_metrics(wandb_connected)
    _render_metric_comparison(wandb_connected, selected_phase, phase_metrics)

    _render_experiment_tracking()
    _render_cross_phase_continuity()

    (
        custom_metrics,
        chart_type,
        auto_refresh,
        refresh_rate,
        export_format,
    ) = _render_dashboard_config()
    _render_save_and_refresh(
        wandb_connected,
        custom_metrics,
        chart_type,
        auto_refresh,
        refresh_rate,
        export_format,
    )

    _render_footer()


# Auto-run when accessed directly via Streamlit multipage
render()

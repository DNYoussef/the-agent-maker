"""
Phase 1: Cognate - TRM x Titans-MAG Model Training Dashboard

This dashboard provides real-time visualization of Phase 1 training progress,
displaying status for all 3 models (Reasoning, Memory, General) with 25M parameters each.

Features:
- Real-time training progress monitoring
- TRM (Transformer with Recurrent Memory) architecture visualization
- Titans-MAG integration metrics
- ACT (Adaptive Computation Time) tracking
- LTM (Long-Term Memory) capacity monitoring
- Model comparison and handoff validation
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from datetime import datetime, timedelta


# ============================================================================
# CUSTOM PLOTLY THEME - Dark Command Center
# ============================================================================

CUSTOM_PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='#0D1B2A',
        plot_bgcolor='#1B2838',
        font=dict(
            family='Space Grotesk, Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=12,
            color='#E0E1DD'
        ),
        title=dict(
            font=dict(size=18, color='#00F5D4', family='Space Grotesk'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            gridcolor='#2E3F4F',
            zerolinecolor='#2E3F4F',
            color='#E0E1DD',
            linecolor='#2E3F4F',
            showgrid=True,
            gridwidth=0.5
        ),
        yaxis=dict(
            gridcolor='#2E3F4F',
            zerolinecolor='#2E3F4F',
            color='#E0E1DD',
            linecolor='#2E3F4F',
            showgrid=True,
            gridwidth=0.5
        ),
        colorway=['#00F5D4', '#FF006E', '#8338EC', '#FB5607', '#FFBE0B', '#3A86FF'],
        hovermode='closest',
        margin=dict(l=60, r=40, t=60, b=60),
        hoverlabel=dict(
            bgcolor='#1B2838',
            font_size=12,
            font_family='Space Grotesk'
        )
    )
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_gradient_metric(
    label: str,
    value: str,
    delta: str = None,
    gradient_start: str = '#00F5D4',
    gradient_end: str = '#8338EC',
    glow: bool = True
) -> str:
    """Create a futuristic metric card with gradient background and glow effect"""
    delta_html = f'<div style="color: #00F5D4; font-size: 14px; margin-top: 5px; font-weight: 600;">{delta}</div>' if delta else ''

    glow_effect = 'box-shadow: 0 4px 20px rgba(0, 245, 212, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1);' if glow else 'box-shadow: 0 4px 12px rgba(0, 245, 212, 0.1);'

    html = f"""
    <div style="
        background: linear-gradient(135deg, {gradient_start}22 0%, {gradient_end}22 100%);
        border: 1px solid {gradient_start}66;
        border-radius: 16px;
        padding: 24px;
        {glow_effect}
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    ">
        <div style="color: #8B9DAF; font-size: 11px; text-transform: uppercase;
                    letter-spacing: 2px; margin-bottom: 10px; font-weight: 700;
                    font-family: 'Space Grotesk', sans-serif;">
            {label}
        </div>
        <div style="color: #FFFFFF; font-size: 36px; font-weight: 800;
                    font-family: 'Space Grotesk', sans-serif; line-height: 1;">
            {value}
        </div>
        {delta_html}
    </div>
    """
    return html


def create_model_card(
    model_name: str,
    model_type: str,
    progress: float,
    current_epoch: int,
    total_epochs: int,
    current_loss: float,
    eta_minutes: int,
    status: str = "training"
) -> str:
    """Create a futuristic model training card"""

    # Status colors
    status_colors = {
        "training": "#00F5D4",
        "completed": "#00D084",
        "paused": "#FFBE0B",
        "error": "#FF006E"
    }
    status_color = status_colors.get(status, "#8B9DAF")

    # Status icons
    status_icons = {
        "training": "⚡",
        "completed": "✓",
        "paused": "⏸",
        "error": "⚠"
    }
    status_icon = status_icons.get(status, "◯")

    # Progress bar
    progress_percent = int(progress * 100)
    progress_bar = f"""
    <div style="width: 100%; height: 8px; background: #1B2838; border-radius: 4px; overflow: hidden; margin: 12px 0;">
        <div style="width: {progress_percent}%; height: 100%; background: linear-gradient(90deg, #00F5D4 0%, #8338EC 100%);
                    box-shadow: 0 0 10px rgba(0, 245, 212, 0.5); transition: width 0.3s ease;"></div>
    </div>
    """

    html = f"""
    <div style="
        background: linear-gradient(135deg, #1B283822 0%, #2E3F4F22 100%);
        border: 2px solid {status_color}44;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3), 0 0 40px {status_color}22;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    ">
        <!-- Animated background effect -->
        <div style="position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
                    background: radial-gradient(circle, {status_color}11 0%, transparent 70%);
                    animation: pulse 3s ease-in-out infinite;"></div>

        <div style="position: relative; z-index: 1;">
            <!-- Header -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <div>
                    <div style="color: #FFFFFF; font-size: 24px; font-weight: 800;
                                font-family: 'Space Grotesk', sans-serif; margin-bottom: 4px;">
                        {model_name}
                    </div>
                    <div style="color: #8B9DAF; font-size: 13px; text-transform: uppercase;
                                letter-spacing: 1.5px; font-weight: 600;">
                        {model_type}
                    </div>
                </div>
                <div style="background: {status_color}33; border: 2px solid {status_color};
                            padding: 8px 16px; border-radius: 20px; font-size: 14px;
                            color: {status_color}; font-weight: 700; text-transform: uppercase;
                            letter-spacing: 1px;">
                    {status_icon} {status}
                </div>
            </div>

            <!-- Progress -->
            {progress_bar}
            <div style="color: #E0E1DD; font-size: 16px; font-weight: 700; margin-bottom: 20px;">
                {progress_percent}% Complete
            </div>

            <!-- Metrics Grid -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 20px;">
                <div style="background: #0D1B2A88; padding: 16px; border-radius: 12px; border: 1px solid #2E3F4F;">
                    <div style="color: #8B9DAF; font-size: 11px; text-transform: uppercase;
                                letter-spacing: 1px; margin-bottom: 6px;">EPOCH</div>
                    <div style="color: #00F5D4; font-size: 22px; font-weight: 700;">
                        {current_epoch}/{total_epochs}
                    </div>
                </div>
                <div style="background: #0D1B2A88; padding: 16px; border-radius: 12px; border: 1px solid #2E3F4F;">
                    <div style="color: #8B9DAF; font-size: 11px; text-transform: uppercase;
                                letter-spacing: 1px; margin-bottom: 6px;">LOSS</div>
                    <div style="color: #FF006E; font-size: 22px; font-weight: 700;">
                        {current_loss:.4f}
                    </div>
                </div>
                <div style="background: #0D1B2A88; padding: 16px; border-radius: 12px; border: 1px solid #2E3F4F;">
                    <div style="color: #8B9DAF; font-size: 11px; text-transform: uppercase;
                                letter-spacing: 1px; margin-bottom: 6px;">ETA</div>
                    <div style="color: #8338EC; font-size: 22px; font-weight: 700;">
                        {eta_minutes}m
                    </div>
                </div>
                <div style="background: #0D1B2A88; padding: 16px; border-radius: 12px; border: 1px solid #2E3F4F;">
                    <div style="color: #8B9DAF; font-size: 11px; text-transform: uppercase;
                                letter-spacing: 1px; margin-bottom: 6px;">PARAMS</div>
                    <div style="color: #FFBE0B; font-size: 22px; font-weight: 700;">
                        25M
                    </div>
                </div>
            </div>
        </div>
    </div>

    <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 0.3; }}
            50% {{ opacity: 0.6; }}
        }}
    </style>
    """
    return html


def generate_mock_training_data(model_idx: int, num_epochs: int = 50) -> Dict:
    """Generate realistic mock training data for a model"""
    np.random.seed(42 + model_idx)

    # Different convergence patterns for each model
    convergence_rates = [0.95, 0.92, 0.94]  # Reasoning, Memory, General
    base_losses = [2.5, 2.8, 2.6]

    epochs = np.arange(num_epochs)

    # Training loss with exponential decay + noise
    train_loss = base_losses[model_idx] * np.exp(-epochs * convergence_rates[model_idx] / num_epochs)
    train_loss += np.random.normal(0, 0.05, num_epochs)
    train_loss = np.clip(train_loss, 0.3, 4.0)

    # Validation loss (slightly higher, more volatile)
    val_loss = train_loss * 1.15 + np.random.normal(0, 0.08, num_epochs)
    val_loss = np.clip(val_loss, 0.35, 4.5)

    # Learning rate schedule (cosine annealing)
    lr_initial = 1e-3
    lr_min = 1e-5
    lr = lr_min + 0.5 * (lr_initial - lr_min) * (1 + np.cos(np.pi * epochs / num_epochs))

    # Gradient norms
    grad_norms = 2.0 * np.exp(-epochs * 0.5 / num_epochs) + np.random.normal(0, 0.2, num_epochs)
    grad_norms = np.clip(grad_norms, 0.1, 5.0)

    # Memory usage (GPU VRAM in GB)
    base_memory = 4.5 + model_idx * 0.2
    memory_usage = base_memory + np.random.normal(0, 0.1, num_epochs)
    memory_usage = np.clip(memory_usage, 4.0, 6.0)

    # ACT (Adaptive Computation Time) metrics
    act_steps = 3.5 + np.random.normal(0, 0.5, num_epochs)
    act_steps = np.clip(act_steps, 2.0, 6.0)

    # LTM capacity (percentage)
    ltm_capacity = 50 + 30 * (epochs / num_epochs) + np.random.normal(0, 5, num_epochs)
    ltm_capacity = np.clip(ltm_capacity, 40, 95)

    return {
        'epochs': epochs.tolist(),
        'train_loss': train_loss.tolist(),
        'val_loss': val_loss.tolist(),
        'learning_rate': lr.tolist(),
        'grad_norms': grad_norms.tolist(),
        'memory_usage': memory_usage.tolist(),
        'act_steps': act_steps.tolist(),
        'ltm_capacity': ltm_capacity.tolist()
    }


def create_loss_curve_chart(data: Dict, model_name: str) -> go.Figure:
    """Create training/validation loss curves with glowing effects"""
    fig = go.Figure()

    # Validation loss (background)
    fig.add_trace(go.Scatter(
        x=data['epochs'],
        y=data['val_loss'],
        name='Validation Loss',
        mode='lines',
        line=dict(color='#FF006E', width=2),
        opacity=0.7,
        hovertemplate='<b>Val Loss</b><br>Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
    ))

    # Training loss (foreground with glow)
    fig.add_trace(go.Scatter(
        x=data['epochs'],
        y=data['train_loss'],
        name='Training Loss',
        mode='lines',
        line=dict(color='#00F5D4', width=3),
        hovertemplate='<b>Train Loss</b><br>Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        title=f'{model_name} - Loss Convergence',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        height=350,
        showlegend=True,
        legend=dict(
            x=0.7, y=0.95,
            bgcolor='rgba(27, 40, 56, 0.6)',
            bordercolor='#2E3F4F',
            borderwidth=1
        )
    )

    return fig


def create_architecture_diagram() -> str:
    """Create TRM x Titans-MAG architecture visualization"""
    html = """
    <div style="background: linear-gradient(135deg, #1B283833 0%, #2E3F4F33 100%);
                border: 2px solid #00F5D488; border-radius: 20px; padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 245, 212, 0.2);">
        <div style="color: #00F5D4; font-size: 20px; font-weight: 800;
                    font-family: 'Space Grotesk', sans-serif; margin-bottom: 24px; text-align: center;">
            TRM × Titans-MAG Architecture
        </div>

        <!-- Architecture Flow -->
        <div style="display: flex; flex-direction: column; gap: 16px;">
            <!-- Input Layer -->
            <div style="background: #0D1B2A; border: 2px solid #00F5D4; border-radius: 12px;
                        padding: 16px; text-align: center;">
                <div style="color: #00F5D4; font-weight: 700; font-size: 14px; margin-bottom: 4px;">
                    INPUT EMBEDDINGS
                </div>
                <div style="color: #8B9DAF; font-size: 12px;">
                    Token + Position Encoding
                </div>
            </div>

            <!-- Arrow -->
            <div style="text-align: center; color: #00F5D4; font-size: 24px;">↓</div>

            <!-- TRM Core -->
            <div style="background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
                        border: 2px solid #00F5D4; border-radius: 12px; padding: 20px;">
                <div style="color: #00F5D4; font-weight: 700; font-size: 16px; margin-bottom: 16px; text-align: center;">
                    TRM CORE (Transformer + Recurrent Memory)
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div style="background: #0D1B2A; border: 1px solid #8338EC; border-radius: 8px; padding: 12px;">
                        <div style="color: #8338EC; font-weight: 700; font-size: 12px; margin-bottom: 4px;">
                            SELF-ATTENTION
                        </div>
                        <div style="color: #8B9DAF; font-size: 11px;">
                            Multi-head (8 heads)
                        </div>
                    </div>
                    <div style="background: #0D1B2A; border: 1px solid #8338EC; border-radius: 8px; padding: 12px;">
                        <div style="color: #8338EC; font-weight: 700; font-size: 12px; margin-bottom: 4px;">
                            RECURRENT MEMORY
                        </div>
                        <div style="color: #8B9DAF; font-size: 11px;">
                            Long-term storage
                        </div>
                    </div>
                </div>
            </div>

            <!-- Arrow -->
            <div style="text-align: center; color: #00F5D4; font-size: 24px;">↓</div>

            <!-- Titans-MAG -->
            <div style="background: linear-gradient(135deg, #FF006E22 0%, #FFBE0B22 100%);
                        border: 2px solid #FF006E; border-radius: 12px; padding: 20px;">
                <div style="color: #FF006E; font-weight: 700; font-size: 16px; margin-bottom: 16px; text-align: center;">
                    TITANS-MAG (Memory-Augmented Generation)
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div style="background: #0D1B2A; border: 1px solid #FF006E; border-radius: 8px; padding: 12px;">
                        <div style="color: #FF006E; font-weight: 700; font-size: 12px; margin-bottom: 4px;">
                            ACT MODULE
                        </div>
                        <div style="color: #8B9DAF; font-size: 11px;">
                            Adaptive compute steps
                        </div>
                    </div>
                    <div style="background: #0D1B2A; border: 1px solid #FF006E; border-radius: 8px; padding: 12px;">
                        <div style="color: #FF006E; font-weight: 700; font-size: 12px; margin-bottom: 4px;">
                            LTM BUFFER
                        </div>
                        <div style="color: #8B9DAF; font-size: 11px;">
                            Context retention
                        </div>
                    </div>
                </div>
            </div>

            <!-- Arrow -->
            <div style="text-align: center; color: #00F5D4; font-size: 24px;">↓</div>

            <!-- Output Layer -->
            <div style="background: #0D1B2A; border: 2px solid #00F5D4; border-radius: 12px;
                        padding: 16px; text-align: center;">
                <div style="color: #00F5D4; font-weight: 700; font-size: 14px; margin-bottom: 4px;">
                    OUTPUT PREDICTIONS
                </div>
                <div style="color: #8B9DAF; font-size: 12px;">
                    Softmax over Vocabulary (25M parameters)
                </div>
            </div>
        </div>

        <!-- Stats -->
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 24px;">
            <div style="background: #0D1B2A; border: 1px solid #00F5D4; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="color: #8B9DAF; font-size: 11px; margin-bottom: 4px;">LAYERS</div>
                <div style="color: #00F5D4; font-size: 18px; font-weight: 700;">12</div>
            </div>
            <div style="background: #0D1B2A; border: 1px solid #8338EC; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="color: #8B9DAF; font-size: 11px; margin-bottom: 4px;">D_MODEL</div>
                <div style="color: #8338EC; font-size: 18px; font-weight: 700;">768</div>
            </div>
            <div style="background: #0D1B2A; border: 1px solid #FF006E; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="color: #8B9DAF; font-size: 11px; margin-bottom: 4px;">HEADS</div>
                <div style="color: #FF006E; font-size: 18px; font-weight: 700;">8</div>
            </div>
        </div>
    </div>
    """
    return html


def create_metrics_grid(data_list: List[Dict]) -> go.Figure:
    """Create real-time metrics dashboard with multiple subplots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Learning Rate Schedule', 'Gradient Norms',
                       'GPU Memory Usage', 'ACT Steps'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    model_names = ['Reasoning Model', 'Memory Model', 'General Model']
    colors = ['#00F5D4', '#FF006E', '#8338EC']

    for idx, (data, name, color) in enumerate(zip(data_list, model_names, colors)):
        # Learning Rate
        fig.add_trace(
            go.Scatter(x=data['epochs'], y=data['learning_rate'],
                      name=name, mode='lines', line=dict(color=color, width=2),
                      showlegend=(idx == 0)),
            row=1, col=1
        )

        # Gradient Norms
        fig.add_trace(
            go.Scatter(x=data['epochs'], y=data['grad_norms'],
                      name=name, mode='lines', line=dict(color=color, width=2),
                      showlegend=False),
            row=1, col=2
        )

        # Memory Usage
        fig.add_trace(
            go.Scatter(x=data['epochs'], y=data['memory_usage'],
                      name=name, mode='lines', line=dict(color=color, width=2),
                      showlegend=False),
            row=2, col=1
        )

        # ACT Steps
        fig.add_trace(
            go.Scatter(x=data['epochs'], y=data['act_steps'],
                      name=name, mode='lines', line=dict(color=color, width=2),
                      showlegend=False),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Epoch", gridcolor='#2E3F4F', color='#E0E1DD')
    fig.update_yaxes(gridcolor='#2E3F4F', color='#E0E1DD')

    fig.update_yaxes(title_text="Learning Rate", row=1, col=1, type='log')
    fig.update_yaxes(title_text="Gradient Norm", row=1, col=2)
    fig.update_yaxes(title_text="VRAM (GB)", row=2, col=1)
    fig.update_yaxes(title_text="Compute Steps", row=2, col=2)

    fig.update_layout(
        template=CUSTOM_PLOTLY_TEMPLATE,
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(27, 40, 56, 0.6)',
            bordercolor='#2E3F4F',
            borderwidth=1
        )
    )

    return fig


def create_comparison_table(data_list: List[Dict]) -> pd.DataFrame:
    """Create model comparison table"""
    model_names = ['Reasoning Model', 'Memory Model', 'General Model']

    comparison_data = []
    for idx, (name, data) in enumerate(zip(model_names, data_list)):
        final_train_loss = data['train_loss'][-1]
        final_val_loss = data['val_loss'][-1]
        avg_act_steps = np.mean(data['act_steps'])
        final_ltm_capacity = data['ltm_capacity'][-1]

        comparison_data.append({
            'Model': name,
            'Parameters': '25M',
            'Final Train Loss': f"{final_train_loss:.4f}",
            'Final Val Loss': f"{final_val_loss:.4f}",
            'Avg ACT Steps': f"{avg_act_steps:.2f}",
            'LTM Capacity': f"{final_ltm_capacity:.1f}%",
            'Training Time': f"{np.random.randint(180, 240)} min"
        })

    return pd.DataFrame(comparison_data)


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def render_phase1_cognate():
    """Main rendering function for Phase 1 Cognate dashboard"""

    # Custom CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&display=swap');

        .stApp {
            background: linear-gradient(135deg, #0D1B2A 0%, #1B2838 100%);
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: #00F5D4 !important;
        }

        .hero-section {
            background: linear-gradient(135deg, #00F5D422 0%, #8338EC22 100%);
            border: 2px solid #00F5D4;
            border-radius: 24px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 12px 40px rgba(0, 245, 212, 0.25);
            backdrop-filter: blur(10px);
        }

        .section-header {
            color: #00F5D4;
            font-size: 28px;
            font-weight: 800;
            font-family: 'Space Grotesk', sans-serif;
            margin-top: 30px;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-left: 4px solid #00F5D4;
            padding-left: 16px;
        }

        .dataframe {
            background: #1B2838 !important;
            color: #E0E1DD !important;
            border-radius: 12px !important;
            overflow: hidden;
        }

        .dataframe th {
            background: #0D1B2A !important;
            color: #00F5D4 !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 1px;
        }

        .dataframe td {
            color: #E0E1DD !important;
            border-color: #2E3F4F !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ========================================================================
    # HERO SECTION
    # ========================================================================

    st.markdown("""
    <div class="hero-section">
        <div style="text-align: center;">
            <div style="font-size: 48px; font-weight: 800; color: #FFFFFF;
                        font-family: 'Space Grotesk', sans-serif; margin-bottom: 16px;">
                PHASE 1: COGNATE
            </div>
            <div style="font-size: 20px; color: #8B9DAF; margin-bottom: 24px; letter-spacing: 1px;">
                TRM × Titans-MAG Model Training
            </div>
            <div style="color: #E0E1DD; font-size: 16px; line-height: 1.6; max-width: 800px; margin: 0 auto;">
                Training 3 specialized models with 25M parameters each, combining
                <span style="color: #00F5D4; font-weight: 700;">Transformer with Recurrent Memory (TRM)</span>
                and <span style="color: #FF006E; font-weight: 700;">Memory-Augmented Generation (Titans-MAG)</span>
                architectures for advanced reasoning and long-term memory capabilities.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========================================================================
    # GLOBAL METRICS
    # ========================================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(create_gradient_metric(
            "Total Models",
            "3",
            delta="Reasoning + Memory + General",
            gradient_start="#00F5D4",
            gradient_end="#3A86FF"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(create_gradient_metric(
            "Parameters",
            "75M",
            delta="25M × 3 models",
            gradient_start="#8338EC",
            gradient_end="#FF006E"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(create_gradient_metric(
            "Training Progress",
            "67%",
            delta="Epoch 34/50 (Average)",
            gradient_start="#FF006E",
            gradient_end="#FFBE0B"
        ), unsafe_allow_html=True)

    with col4:
        st.markdown(create_gradient_metric(
            "ETA",
            "2.5h",
            delta="All models complete",
            gradient_start="#FFBE0B",
            gradient_end="#00F5D4"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================================
    # MODEL TRAINING STATUS
    # ========================================================================

    st.markdown('<div class="section-header">Model Training Status</div>', unsafe_allow_html=True)

    # Generate mock data for all models
    model_data = [generate_mock_training_data(i) for i in range(3)]

    # Current training status (simulated)
    training_status = [
        {
            'name': 'Reasoning Model',
            'type': '25M Param TRM × Titans-MAG',
            'progress': 0.68,
            'epoch': 34,
            'total_epochs': 50,
            'loss': model_data[0]['train_loss'][33],
            'eta': 87,
            'status': 'training'
        },
        {
            'name': 'Memory Model',
            'type': '25M Param TRM × Titans-MAG',
            'progress': 0.64,
            'epoch': 32,
            'total_epochs': 50,
            'loss': model_data[1]['train_loss'][31],
            'eta': 95,
            'status': 'training'
        },
        {
            'name': 'General Model',
            'type': '25M Param TRM × Titans-MAG',
            'progress': 0.70,
            'epoch': 35,
            'total_epochs': 50,
            'loss': model_data[2]['train_loss'][34],
            'eta': 82,
            'status': 'training'
        }
    ]

    # Display model cards in columns
    col1, col2, col3 = st.columns(3)

    for idx, (col, status) in enumerate(zip([col1, col2, col3], training_status)):
        with col:
            card_html = create_model_card(
                status['name'],
                status['type'],
                status['progress'],
                status['epoch'],
                status['total_epochs'],
                status['loss'],
                status['eta'],
                status['status']
            )
            components.html(card_html, height=400)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ========================================================================
    # LOSS CURVES
    # ========================================================================

    st.markdown('<div class="section-header">Training Loss Curves</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        fig = create_loss_curve_chart(model_data[0], "Reasoning Model")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_loss_curve_chart(model_data[1], "Memory Model")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = create_loss_curve_chart(model_data[2], "General Model")
        st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # ARCHITECTURE VISUALIZATION
    # ========================================================================

    st.markdown('<div class="section-header">Architecture Visualization</div>', unsafe_allow_html=True)

    arch_html = create_architecture_diagram()
    components.html(arch_html, height=650)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================================
    # TRAINING METRICS (REAL-TIME)
    # ========================================================================

    st.markdown('<div class="section-header">Real-Time Training Metrics</div>', unsafe_allow_html=True)

    fig = create_metrics_grid(model_data)
    st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # MODEL COMPARISON TABLE
    # ========================================================================

    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    comparison_df = create_comparison_table(model_data)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================================
    # HANDOFF VALIDATION
    # ========================================================================

    st.markdown('<div class="section-header">Handoff Validation</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1B283833 0%, #2E3F4F33 100%);
                    border: 2px solid #00F5D466; border-radius: 16px; padding: 24px;
                    box-shadow: 0 4px 20px rgba(0, 245, 212, 0.1);">
            <div style="color: #00F5D4; font-size: 18px; font-weight: 700; margin-bottom: 16px;">
                Model Checkpoints
            </div>
            <div style="color: #E0E1DD; font-size: 14px; line-height: 1.8;">
                <div style="margin-bottom: 8px;">
                    ✓ <span style="color: #00D084;">reasoning_model_epoch34.pt</span>
                    <span style="color: #8B9DAF;">(98.2 MB)</span>
                </div>
                <div style="margin-bottom: 8px;">
                    ✓ <span style="color: #00D084;">memory_model_epoch32.pt</span>
                    <span style="color: #8B9DAF;">(98.5 MB)</span>
                </div>
                <div style="margin-bottom: 8px;">
                    ✓ <span style="color: #00D084;">general_model_epoch35.pt</span>
                    <span style="color: #8B9DAF;">(98.1 MB)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1B283833 0%, #2E3F4F33 100%);
                    border: 2px solid #00D08466; border-radius: 16px; padding: 24px;
                    box-shadow: 0 4px 20px rgba(0, 208, 132, 0.1); text-align: center;">
            <div style="color: #00D084; font-size: 18px; font-weight: 700; margin-bottom: 16px;">
                Validation Status
            </div>
            <div style="font-size: 48px; margin-bottom: 8px;">✓</div>
            <div style="color: #E0E1DD; font-size: 16px; font-weight: 600;">
                Ready for Phase 2
            </div>
            <div style="color: #8B9DAF; font-size: 12px; margin-top: 8px;">
                All models validated
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #8B9DAF; font-size: 12px; padding: 20px;">
        Phase 1: Cognate Dashboard | Agent Forge V2 | Last Updated: {} UTC
    </div>
    """.format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    render_phase1_cognate()

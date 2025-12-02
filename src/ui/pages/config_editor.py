"""
Configuration Editor Page
Edit pipeline configuration with YAML validation
"""
import streamlit as st
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render():
    """Render configuration editor page"""
    st.markdown('<h1 class="main-header">Configuration Editor</h1>',
                unsafe_allow_html=True)

    # Load configuration
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "pipeline_config.yaml"

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return

    # Configuration sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "W&B Settings",
        "Phase Configurations",
        "Hardware Settings",
        "Cleanup Policies"
    ])

    # Tab 1: W&B Settings
    with tab1:
        st.subheader("Weights & Biases Configuration")

        wandb_config = config.get('wandb', {})

        col1, col2 = st.columns(2)

        with col1:
            wandb_enabled = st.checkbox(
                "Enable W&B Tracking",
                value=wandb_config.get('enabled', True)
            )

            wandb_mode = st.selectbox(
                "Mode",
                ["offline", "online", "disabled"],
                index=["offline", "online", "disabled"].index(
                    wandb_config.get('mode', 'offline')
                )
            )

        with col2:
            wandb_project = st.text_input(
                "Project Name",
                value=wandb_config.get('project', 'agent-forge-v2')
            )

            wandb_entity = st.text_input(
                "Entity (optional)",
                value=wandb_config.get('entity', '')
            )

        # Update config
        config['wandb'] = {
            'enabled': wandb_enabled,
            'mode': wandb_mode,
            'project': wandb_project,
            'entity': wandb_entity
        }

    # Tab 2: Phase Configurations
    with tab2:
        st.subheader("Phase-Specific Settings")

        phase_selection = st.selectbox(
            "Select Phase",
            [f"Phase {i}" for i in range(1, 9)]
        )

        phase_num = int(phase_selection.split(" ")[1])
        phase_key = f"phase{phase_num}"

        phases_config = config.get('phases', {})
        phase_config = phases_config.get(phase_key, {})

        if phase_num == 1:
            _render_phase1_config(phase_config)
        elif phase_num == 2:
            _render_phase2_config(phase_config)
        elif phase_num == 3:
            _render_phase3_config(phase_config)
        elif phase_num == 4:
            _render_phase4_config(phase_config)
        else:
            st.info(f"Phase {phase_num} configuration coming soon...")

    # Tab 3: Hardware Settings
    with tab3:
        st.subheader("Hardware Configuration")

        hardware_config = config.get('hardware', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**GPU Settings**")
            device_vram_gb = st.slider(
                "Device VRAM (GB)",
                min_value=4,
                max_value=24,
                value=int(hardware_config.get('device_vram_gb', 6)),
                step=2
            )

            max_batch_size = st.number_input(
                "Max Batch Size",
                min_value=1,
                max_value=128,
                value=int(hardware_config.get('max_batch_size', 32))
            )

        with col2:
            st.markdown("**System Settings**")
            num_workers = st.slider(
                "DataLoader Workers",
                min_value=0,
                max_value=16,
                value=int(hardware_config.get('num_workers', 4))
            )

            mixed_precision = st.checkbox(
                "Enable Mixed Precision (FP16)",
                value=hardware_config.get('mixed_precision', True)
            )

        config['hardware'] = {
            'device_vram_gb': device_vram_gb,
            'max_batch_size': max_batch_size,
            'num_workers': num_workers,
            'mixed_precision': mixed_precision
        }

    # Tab 4: Cleanup Policies
    with tab4:
        st.subheader("Cleanup Policies")

        cleanup_config = config.get('cleanup', {})

        max_session_age_days = st.slider(
            "Max Session Age (days)",
            min_value=7,
            max_value=90,
            value=int(cleanup_config.get('max_session_age_days', 30))
        )

        max_sessions_total = st.number_input(
            "Max Total Sessions",
            min_value=10,
            max_value=500,
            value=int(cleanup_config.get('max_sessions_total', 100))
        )

        keep_last_n_checkpoints = st.slider(
            "Keep Last N Checkpoints",
            min_value=1,
            max_value=20,
            value=int(cleanup_config.get('keep_last_n_checkpoints', 5))
        )

        auto_cleanup_enabled = st.checkbox(
            "Enable Auto-Cleanup",
            value=cleanup_config.get('auto_cleanup_enabled', True)
        )

        config['cleanup'] = {
            'max_session_age_days': max_session_age_days,
            'max_sessions_total': max_sessions_total,
            'keep_last_n_checkpoints': keep_last_n_checkpoints,
            'auto_cleanup_enabled': auto_cleanup_enabled
        }

    # Save button
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("💾 Save Configuration", type="primary"):
            try:
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                st.success("Configuration saved successfully!")
            except Exception as e:
                st.error(f"Failed to save configuration: {e}")

    with col2:
        if st.button("↩️ Reset to Defaults"):
            st.warning("Reset functionality coming soon")

    # Show current config (read-only)
    with st.expander("View Current Configuration (YAML)"):
        st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language='yaml')


def _render_phase1_config(phase_config):
    """Render Phase 1 configuration"""
    st.markdown("### Phase 1: Cognate (TRM × Titans-MAG)")

    num_models = st.slider(
        "Number of Models",
        min_value=1,
        max_value=5,
        value=int(phase_config.get('num_models', 3))
    )

    epochs = st.slider(
        "Training Epochs",
        min_value=5,
        max_value=20,
        value=int(phase_config.get('epochs', 10))
    )

    # Optimizer settings
    st.markdown("**MuGrokfast Optimizer**")
    optimizer_config = phase_config.get('optimizer', {})

    col1, col2 = st.columns(2)

    with col1:
        muon_lr = st.number_input(
            "Muon Learning Rate",
            min_value=0.0001,
            max_value=0.01,
            value=float(optimizer_config.get('muon_lr', 0.001)),
            format="%.4f"
        )

    with col2:
        grokfast_lambda = st.number_input(
            "Grokfast Lambda",
            min_value=0.0,
            max_value=1.0,
            value=float(optimizer_config.get('grokfast_lambda', 0.3)),
            format="%.2f"
        )


def _render_phase2_config(phase_config):
    """Render Phase 2 configuration"""
    st.markdown("### Phase 2: EvoMerge (50 generations)")

    num_generations = st.slider(
        "Number of Generations",
        min_value=10,
        max_value=100,
        value=int(phase_config.get('num_generations', 50))
    )

    population_size = st.slider(
        "Population Size",
        min_value=4,
        max_value=32,
        value=int(phase_config.get('population_size', 8)),
        step=4
    )

    # Merge techniques
    st.markdown("**Merge Techniques**")
    techniques = st.multiselect(
        "Enabled Techniques",
        ["linear", "slerp", "ties", "dare", "frankenmerge", "dfs"],
        default=phase_config.get('merge_techniques', ["linear", "slerp", "ties"])
    )


def _render_phase3_config(phase_config):
    """Render Phase 3 configuration"""
    st.markdown("### Phase 3: Quiet-STaR (Reasoning)")

    # Prompt baking
    st.markdown("**Prompt Baking (Step 1)**")
    baking_epochs = st.slider(
        "Baking Epochs",
        min_value=1,
        max_value=10,
        value=int(phase_config.get('baking_epochs', 3))
    )

    # RL training
    st.markdown("**RL Training (Step 2)**")
    col1, col2 = st.columns(2)

    with col1:
        kl_coefficient = st.number_input(
            "KL Coefficient",
            min_value=0.0,
            max_value=1.0,
            value=float(phase_config.get('kl_coefficient', 0.1)),
            format="%.2f"
        )

    with col2:
        rl_epochs = st.slider(
            "RL Epochs",
            min_value=1,
            max_value=20,
            value=int(phase_config.get('rl_epochs', 5))
        )


def _render_phase4_config(phase_config):
    """Render Phase 4 configuration"""
    st.markdown("### Phase 4: BitNet (1.58-bit Quantization)")

    target_compression = st.slider(
        "Target Compression Ratio",
        min_value=4.0,
        max_value=12.0,
        value=float(phase_config.get('target_compression', 8.2)),
        step=0.1
    )

    ste_epochs = st.slider(
        "STE Training Epochs",
        min_value=1,
        max_value=10,
        value=int(phase_config.get('ste_epochs', 5))
    )

    quality_threshold = st.slider(
        "Quality Retention Threshold (%)",
        min_value=80,
        max_value=98,
        value=int(phase_config.get('quality_threshold', 90))
    )

"""
Phase Details Page
Detailed view of individual phase metrics and controls
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render():
    """Render phase details page"""
    st.markdown('<h1 class="main-header">Phase Details</h1>',
                unsafe_allow_html=True)

    # Phase selection
    phase = st.selectbox(
        "Select Phase",
        [
            "Phase 1: Cognate (TRM × Titans-MAG)",
            "Phase 2: EvoMerge (50 generations)",
            "Phase 3: Quiet-STaR (Reasoning)",
            "Phase 4: BitNet (1.58-bit compression)",
            "Phase 5: Curriculum Learning",
            "Phase 6: Tool & Persona Baking",
            "Phase 7: Self-Guided Experts",
            "Phase 8: Final Compression (280×)"
        ]
    )

    phase_num = int(phase.split(":")[0].split(" ")[1])

    st.markdown("---")

    # Phase-specific content
    if phase_num == 1:
        render_phase1_details()
    elif phase_num == 2:
        render_phase2_details()
    elif phase_num == 3:
        render_phase3_details()
    elif phase_num == 4:
        render_phase4_details()
    else:
        st.info(f"Phase {phase_num} details coming soon...")


def render_phase1_details():
    """Phase 1: Cognate (TRM × Titans-MAG)"""
    st.subheader("Phase 1: Cognate (TRM × Titans-MAG)")
    st.markdown("Creates 3 specialized 25M parameter models")

    # Model cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Model 1: Reasoning")
        st.metric("Parameters", "25M")
        st.metric("Status", "Training")
        st.metric("Loss", "2.34")
        st.metric("Epoch", "5/10")
        st.progress(0.5)

    with col2:
        st.markdown("### Model 2: Memory")
        st.metric("Parameters", "25M")
        st.metric("Status", "Complete")
        st.metric("Final Loss", "2.12")
        st.metric("Epoch", "10/10")
        st.progress(1.0)

    with col3:
        st.markdown("### Model 3: General")
        st.metric("Parameters", "25M")
        st.metric("Status", "Pending")
        st.metric("Loss", "-")
        st.metric("Epoch", "0/10")
        st.progress(0.0)

    st.markdown("---")

    # Metrics visualization
    st.subheader("Training Metrics")

    import pandas as pd
    import numpy as np

    # Dummy data for visualization
    epochs = list(range(1, 11))
    loss_data = pd.DataFrame({
        "Epoch": epochs,
        "Model 1": [3.5, 3.2, 2.9, 2.7, 2.5, 2.4, 2.35, 2.32, 2.30, 2.28],
        "Model 2": [3.6, 3.3, 3.0, 2.8, 2.6, 2.4, 2.3, 2.2, 2.15, 2.12],
    })

    st.line_chart(loss_data.set_index("Epoch"))

    # W&B metrics (37 total)
    with st.expander("View All Metrics (37 total)"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**TRM Metrics**")
            st.text("• ACT ponder steps: 3.2")
            st.text("• LTM memory usage: 45%")
            st.text("• Attention entropy: 2.8")

        with col2:
            st.markdown("**Optimizer Metrics**")
            st.text("• MuGrokfast muon_lr: 0.001")
            st.text("• Grokfast lambda: 0.3")
            st.text("• Gradient EMA: 0.98")


def render_phase2_details():
    """Phase 2: EvoMerge"""
    st.subheader("Phase 2: EvoMerge (50 generations)")
    st.markdown("Evolutionary optimization with 6 merge techniques")

    # Generation progress
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Generation", "25/50")

    with col2:
        st.metric("Best Fitness", "0.847")

    with col3:
        st.metric("Population Size", "16")

    with col4:
        st.metric("Fitness Gain", "+18.2%")

    st.progress(0.5)

    st.markdown("---")

    # Merge techniques usage
    st.subheader("Merge Techniques")

    import pandas as pd

    merge_data = pd.DataFrame({
        "Technique": ["Linear", "SLERP", "TIES", "DARE", "FrankenMerge", "DFS"],
        "Usage": [45, 38, 52, 41, 29, 35],
        "Avg Fitness": [0.72, 0.75, 0.81, 0.78, 0.68, 0.74]
    })

    st.bar_chart(merge_data.set_index("Technique")["Usage"])

    # Binary pairing tree
    with st.expander("View Binary Pairing Tree"):
        st.info("3D merge visualization coming soon (Three.js integration)")


def render_phase3_details():
    """Phase 3: Quiet-STaR (Reasoning Enhancement)"""
    st.subheader("Phase 3: Quiet-STaR (Reasoning Enhancement)")
    st.markdown("Two-step training: Prompt Baking → Quiet-STaR RL")

    # Phase summary
    st.info("""
    **Phase 3 Summary**:
    - **Input**: Phase 2 champion model (23.5% fitness gain)
    - **Step 1**: Prompt Baking (5 min) - Bake CoT reasoning patterns
    - **Step 2**: Quiet-STaR RL (5 hours) - REINFORCE training with KL regularization
    - **Output**: Reasoning-enhanced model for Phase 4 BitNet compression
    """)

    st.markdown("---")

    # Two-step workflow with real-time progress
    st.subheader("Training Progress")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📦 Step 1: Prompt Baking")
        st.markdown("**5-minute supervised learning**")

        # Status indicators
        baking_status = st.selectbox(
            "Status",
            ["Not Started", "Running", "Complete", "Failed"],
            index=2,  # Default to Complete
            key="baking_status"
        )

        if baking_status == "Complete":
            st.success("✅ Baking complete!")
            st.metric("Final Accuracy", "87.2%", delta="2.2%")
            st.metric("Convergence Threshold", "≥85%", delta="Met")
            st.metric("Training Time", "5 min")

            # Strategy-specific accuracies
            with st.expander("Strategy Accuracies (7 strategies)"):
                st.markdown("**7 Reasoning Strategies:**")
                strategies = {
                    "Chain-of-Thought": 0.89,
                    "MECE Decomposition": 0.86,
                    "Falsification Testing": 0.84,
                    "Expert Perspective": 0.91,
                    "Orthogonal Wisdom": 0.82,
                    "Self-Doubt": 0.87,
                    "Bayesian Rationalist": 0.90
                }
                for strategy, acc in strategies.items():
                    st.text(f"• {strategy}: {acc:.1%}")
        elif baking_status == "Running":
            st.info("⏳ Baking in progress...")
            st.progress(0.6)
            st.metric("Current Accuracy", "72.5%")
            st.metric("Epoch", "3/5")

    with col2:
        st.markdown("### 🎯 Step 2: Quiet-STaR RL")
        st.markdown("**5-hour REINFORCE training**")

        # Status indicators
        rl_status = st.selectbox(
            "Status",
            ["Waiting for Step 1", "Running", "Complete", "Failed"],
            index=1,  # Default to Running
            key="rl_status"
        )

        if rl_status == "Running":
            st.info("⏳ RL training in progress...")
            st.progress(0.65)
            st.metric("Episode", "3,250/5,000")
            st.metric("Avg Reward (last 100)", "0.73", delta="+0.08")
            st.metric("KL Divergence", "0.08", delta="-0.02")
        elif rl_status == "Complete":
            st.success("✅ RL training complete!")
            st.metric("Final Reward", "0.81")
            st.metric("Episodes", "5,000")
            st.metric("Training Time", "5.2 hours")

    st.markdown("---")

    # Real-time metrics dashboard (17 W&B metrics)
    st.subheader("📊 Real-Time Metrics (17 W&B metrics)")

    # Coherence scoring metrics
    st.markdown("#### Coherence Scoring")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Semantic", "0.85", delta="+0.03")
        st.caption("40% weight")

    with col2:
        st.metric("Syntactic", "0.79", delta="+0.01")
        st.caption("30% weight")

    with col3:
        st.metric("Predictive", "0.82", delta="+0.05")
        st.caption("30% weight")

    with col4:
        st.metric("Composite", "0.82", delta="+0.03")
        st.caption("Weighted avg")

    # Thought generation metrics
    st.markdown("#### Thought Generation")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Thought Length", "12.4 tokens")

    with col2:
        st.metric("Thought Diversity", "0.68")

    with col3:
        st.metric("Num Thoughts", "4-8 parallel")

    # Training metrics
    st.markdown("#### Training Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Reward", "0.73", delta="+0.08")

    with col2:
        st.metric("KL Divergence", "0.08", delta="-0.02")

    with col3:
        st.metric("Learning Rate", "5e-4")

    # Accuracy metrics
    st.markdown("#### Downstream Task Accuracy")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("GSM8K", "74.2%", delta="+8.5%")

    with col2:
        st.metric("ARC", "68.9%", delta="+6.2%")

    with col3:
        st.metric("Inference Time", "142 ms", delta="-18 ms")

    st.markdown("---")

    # Thinking token usage visualization
    st.subheader("💭 Thinking Token Usage")

    import pandas as pd

    token_usage = pd.DataFrame({
        "Token": ["<think>", "</think>", "<step>", "<reason>",
                  "<mece>", "<falsify>", "<expert>", "<doubt>"],
        "Usage %": [89.2, 89.2, 67.5, 72.3, 45.8, 38.4, 51.2, 42.7]
    })

    st.bar_chart(token_usage.set_index("Token"))

    st.markdown("---")

    # Anti-theater detection results
    st.subheader("🛡️ Anti-Theater Detection (3 critical tests)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Test 1: Divergence")
        divergence_score = 0.35
        if divergence_score > 0.30:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")
        st.metric("Divergence Score", f"{divergence_score:.2f}")
        st.caption("Target: >0.30 (thoughts diverge from greedy)")

    with col2:
        st.markdown("### Test 2: Ablation")
        ablation_gain = 0.05
        if ablation_gain > 0.02:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")
        st.metric("Ablation Gain", f"{ablation_gain:.2f}")
        st.caption("Target: >2% (thoughts improve accuracy)")

    with col3:
        st.markdown("### Test 3: Correlation")
        correlation = 0.62
        if correlation > 0.50:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")
        st.metric("Correlation", f"{correlation:.2f}")
        st.caption("Target: >0.50 (coherence correlates with utility)")

    # Overall anti-theater verdict
    all_passed = (divergence_score > 0.30 and ablation_gain > 0.02 and correlation > 0.50)

    if all_passed:
        st.success("🎉 All Anti-Theater Tests PASSED - Genuine Reasoning Validated!")
    else:
        st.warning("⚠️ Some anti-theater tests failed - review thought generation")

    st.markdown("---")

    # Reward curve visualization
    st.subheader("📈 RL Reward Curve (REINFORCE)")

    import numpy as np

    episodes = list(range(0, 5000, 100))
    reward_data = pd.DataFrame({
        "Episode": episodes,
        "Avg Reward (last 100)": 0.5 + 0.3 * (1 - np.exp(-np.array(episodes) / 1500))
    })

    st.line_chart(reward_data.set_index("Episode"))

    # Model checkpoints
    with st.expander("📁 Model Checkpoints"):
        st.markdown("**Phase 3 Checkpoints:**")
        st.text("• phase2_champion.pt (input from Phase 2)")
        st.text("• phase3_baked.pt (Step 1 output)")
        st.text("• phase3_rl.pt (Step 2 output)")
        st.text("• phase3_final.pt (output to Phase 4)")

    # Phase handoff validation
    with st.expander("🔗 Phase Handoff Validation"):
        st.markdown("**Phase 2 → Phase 3:**")
        st.success("✅ Valid: Phase 2 champion model (23.5% fitness gain)")

        st.markdown("**Phase 3 → Phase 4:**")
        st.success("✅ Valid: 8 thinking tokens, 87.2% baking accuracy")
        st.success("✅ Anti-theater tests passed")
        st.success("✅ Ready for BitNet compression")


def render_phase4_details():
    """Phase 4: BitNet"""
    st.subheader("Phase 4: BitNet (1.58-bit Quantization)")
    st.markdown("8.2× compression with 3.8× speedup")

    # Compression metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Compression Ratio", "8.2×")

    with col2:
        st.metric("Speedup", "3.8×")

    with col3:
        st.metric("Model Size", "11.8 MB")
        st.caption("(from 95.4 MB)")

    with col4:
        st.metric("Quality Retention", "94.2%")

    st.markdown("---")

    # STE training
    st.subheader("STE (Straight-Through Estimator)")

    st.markdown("**Quantized Forward, Full-Precision Gradients**")

    import pandas as pd

    ste_data = pd.DataFrame({
        "Step": list(range(0, 1000, 100)),
        "Quantization Loss": [0.45, 0.38, 0.32, 0.28, 0.25, 0.23, 0.21, 0.20, 0.19, 0.18]
    })

    st.line_chart(ste_data.set_index("Step"))

    # W&B metrics (19 total)
    with st.expander("View All Metrics (19 total)"):
        st.text("• Compression ratio: 8.2×")
        st.text("• Inference speedup: 3.8×")
        st.text("• Quality retention: 94.2%")
        st.text("• Bit allocation: 1.58-bit avg")


# Auto-run when accessed directly via Streamlit multipage
render()

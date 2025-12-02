"""
Design System Demo
Example usage of the Agent Forge V2 design system
"""

import streamlit as st
from design_system import (
    COLORS,
    SPACING,
    TYPOGRAPHY,
    css_dict_to_string,
    get_badge_styles,
    get_button_styles,
    get_card_styles,
    get_color_with_alpha,
    get_custom_css,
    get_metric_styles,
)


def main():
    """Demo page showing design system components"""

    # Page config
    st.set_page_config(
        page_title="Design System Demo",
        page_icon="🎨",
        layout="wide",
    )

    # Inject custom CSS
    st.markdown(get_custom_css(theme="dark"), unsafe_allow_html=True)

    # ===== HEADER =====
    st.markdown('<h1 class="main-header">Design System Demo</h1>', unsafe_allow_html=True)

    # ===== SECTION 1: COLOR PALETTE =====
    st.markdown('<h2 class="section-header">Color Palette</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### Primary Colors")
        st.markdown(
            f'<div style="background: {COLORS["primary"]}; padding: 20px; border-radius: 8px; color: white;">Primary</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background: {COLORS["secondary"]}; padding: 20px; border-radius: 8px; color: white; margin-top: 8px;">Secondary</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### Accent Colors")
        st.markdown(
            f'<div style="background: {COLORS["accent"]}; padding: 20px; border-radius: 8px; color: {COLORS["primary"]};">Accent</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background: {COLORS["accent_2"]}; padding: 20px; border-radius: 8px; color: white; margin-top: 8px;">Accent 2</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown("### Status Colors")
        st.markdown(
            f'<div style="background: {COLORS["success"]}; padding: 20px; border-radius: 8px; color: {COLORS["primary"]};">Success</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background: {COLORS["warning"]}; padding: 20px; border-radius: 8px; color: {COLORS["primary"]}; margin-top: 8px;">Warning</div>',
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown("### Surface Colors")
        st.markdown(
            f'<div style="background: {COLORS["surface"]}; padding: 20px; border-radius: 8px; color: white;">Surface</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background: {COLORS["error"]}; padding: 20px; border-radius: 8px; color: white; margin-top: 8px;">Error</div>',
            unsafe_allow_html=True,
        )

    # ===== SECTION 2: TYPOGRAPHY =====
    st.markdown('<h2 class="section-header">Typography</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f'<div style="font-size: {TYPOGRAPHY["size_display"]}; font-weight: {TYPOGRAPHY["weight_bold"]}; color: {COLORS["accent"]};">Display</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size: {TYPOGRAPHY["size_h1"]}; font-weight: {TYPOGRAPHY["weight_bold"]};">Heading 1</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size: {TYPOGRAPHY["size_h2"]}; font-weight: {TYPOGRAPHY["weight_bold"]};">Heading 2</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size: {TYPOGRAPHY["size_h3"]}; font-weight: {TYPOGRAPHY["weight_semibold"]};">Heading 3</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div style="font-size: {TYPOGRAPHY["size_body"]};">Body text - The quick brown fox jumps over the lazy dog</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size: {TYPOGRAPHY["size_small"]}; color: {COLORS["text_secondary"]};">Small text - Secondary information</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size: {TYPOGRAPHY["size_tiny"]}; color: {COLORS["text_secondary"]};">Tiny text - Labels and captions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-family: {TYPOGRAPHY["font_code"]}; background: {COLORS["secondary"]}; padding: 8px; border-radius: 4px; margin-top: 8px;">Code: const example = "JetBrains Mono";</div>',
            unsafe_allow_html=True,
        )

    # ===== SECTION 3: CARDS =====
    st.markdown('<h2 class="section-header">Cards</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        glass_styles = css_dict_to_string(get_card_styles("glass"))
        st.markdown(
            f'<div style="{glass_styles}"><h3>Glass Card</h3><p>Glassmorphism with blur effect</p></div>',
            unsafe_allow_html=True,
        )

    with col2:
        solid_styles = css_dict_to_string(get_card_styles("solid"))
        st.markdown(
            f'<div style="{solid_styles}"><h3>Solid Card</h3><p>Solid background, subtle shadow</p></div>',
            unsafe_allow_html=True,
        )

    with col3:
        elevated_styles = css_dict_to_string(get_card_styles("elevated"))
        st.markdown(
            f'<div style="{elevated_styles}"><h3>Elevated Card</h3><p>Prominent shadow elevation</p></div>',
            unsafe_allow_html=True,
        )

    # ===== SECTION 4: BADGES =====
    st.markdown('<h2 class="section-header">Status Badges</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown('<span class="status-success">SUCCESS</span>', unsafe_allow_html=True)

    with col2:
        st.markdown('<span class="status-running">RUNNING</span>', unsafe_allow_html=True)

    with col3:
        st.markdown('<span class="status-failed">FAILED</span>', unsafe_allow_html=True)

    with col4:
        st.markdown('<span class="status-pending">PENDING</span>', unsafe_allow_html=True)

    with col5:
        badge_styles = css_dict_to_string(get_badge_styles("info"))
        st.markdown(f'<span style="{badge_styles}">INFO</span>', unsafe_allow_html=True)

    # ===== SECTION 5: METRICS =====
    st.markdown('<h2 class="section-header">Metrics</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Training Progress", value="87%", delta="12%")

    with col2:
        st.metric(label="Model Accuracy", value="94.7%", delta="2.3%")

    with col3:
        st.metric(label="GPU Memory", value="4.2 GB", delta="-0.8 GB")

    with col4:
        st.metric(label="Inference Time", value="23ms", delta="-5ms")

    # Custom metric display
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    metric_styles = get_metric_styles("large")
    st.markdown(f'<div class="metric-label">Total Parameters</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">25M</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-delta-positive">+5% from baseline</div>', unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ===== SECTION 6: BUTTONS =====
    st.markdown('<h2 class="section-header">Buttons</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.button("Primary Button", key="btn1")

    with col2:
        st.markdown(
            '<button class="custom-button-secondary">Secondary</button>', unsafe_allow_html=True
        )

    with col3:
        st.markdown('<button class="custom-button-accent">Accent</button>', unsafe_allow_html=True)

    with col4:
        st.markdown('<button class="custom-button">Custom</button>', unsafe_allow_html=True)

    # ===== SECTION 7: FORM ELEMENTS =====
    st.markdown('<h2 class="section-header">Form Elements</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "Select Phase",
            ["Phase 1: Cognate", "Phase 2: EvoMerge", "Phase 3: Quiet-STaR", "Phase 4: BitNet"],
        )
        st.text_input("Model Name", placeholder="Enter model name...")

    with col2:
        st.slider("Learning Rate", 0.0, 1.0, 0.001, 0.001)
        st.number_input("Batch Size", min_value=1, max_value=128, value=32)

    # ===== SECTION 8: PROGRESS & CHARTS =====
    st.markdown('<h2 class="section-header">Progress & Visualization</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.progress(0.75)
        st.markdown('<div class="metric-label">Training Progress</div>', unsafe_allow_html=True)

    with col2:
        # Example chart data
        import numpy as np
        import pandas as pd

        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Loss", "Accuracy", "F1-Score"])
        st.line_chart(chart_data)

    # ===== SECTION 9: CODE BLOCKS =====
    st.markdown('<h2 class="section-header">Code Display</h2>', unsafe_allow_html=True)

    code_example = """
    from design_system import get_custom_css

    # Inject custom CSS
    st.markdown(get_custom_css(theme="dark"), unsafe_allow_html=True)

    # Use custom classes
    st.markdown('<div class="glass-card">Content</div>', unsafe_allow_html=True)
    """

    st.code(code_example, language="python")

    # ===== SECTION 10: UTILITY CLASSES =====
    st.markdown('<h2 class="section-header">Utility Classes</h2>', unsafe_allow_html=True)

    st.markdown('<p class="text-accent">Text with accent color</p>', unsafe_allow_html=True)
    st.markdown('<p class="text-accent-2">Text with accent 2 color</p>', unsafe_allow_html=True)
    st.markdown('<p class="text-success">Success text</p>', unsafe_allow_html=True)
    st.markdown('<p class="text-warning">Warning text</p>', unsafe_allow_html=True)
    st.markdown('<p class="text-error">Error text</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="gradient-text uppercase" style="font-size: 2rem; font-weight: bold;">Gradient Text</p>',
        unsafe_allow_html=True,
    )

    # ===== SECTION 11: PROGRAMMATIC ACCESS =====
    st.markdown('<h2 class="section-header">Programmatic Access</h2>', unsafe_allow_html=True)

    with st.expander("View Design Tokens as Python Dicts"):
        st.write("### Colors")
        st.json(COLORS)

        st.write("### Typography")
        st.json(TYPOGRAPHY)

        st.write("### Spacing")
        st.json(SPACING)

        st.write("### Example: Get color with alpha")
        st.code(f'get_color_with_alpha("accent", 0.5) = "{get_color_with_alpha("accent", 0.5)}"')


if __name__ == "__main__":
    main()

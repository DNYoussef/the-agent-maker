"""
Agent Forge V2 - Design System
Comprehensive design system for the Streamlit UI with futuristic command center theme
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, cast

# ============================================================================
# COLOR PALETTE - Futuristic Command Center Theme
# ============================================================================

COLORS = {
    # Primary Colors
    "primary": "#0D1B2A",  # Deep navy - Primary background
    "secondary": "#1B263B",  # Slate - Secondary surfaces
    "accent": "#00F5D4",  # Electric cyan - Primary accent
    "accent_2": "#F72585",  # Magenta - Secondary accent
    "surface": "#415A77",  # Dark slate - Elevated surfaces
    # Text Colors
    "text_primary": "#E0E1DD",  # Off-white - Primary text
    "text_secondary": "#778DA9",  # Muted blue-gray - Secondary text
    "text_disabled": "#4A5568",  # Dark gray - Disabled text
    "text_inverse": "#0D1B2A",  # Dark on light backgrounds
    # Status Colors
    "success": "#39FF14",  # Neon green - Success states
    "warning": "#FFB703",  # Amber - Warning states
    "error": "#FF006E",  # Hot pink - Error states
    "info": "#00B4D8",  # Bright cyan - Info states
    # Semantic Colors
    "running": "#FFB703",  # Amber - Running/in-progress
    "completed": "#39FF14",  # Neon green - Completed
    "failed": "#FF006E",  # Hot pink - Failed
    "pending": "#778DA9",  # Muted - Pending/idle
    # UI Elements
    "border": "#2D3748",  # Subtle border color
    "border_accent": "#00F5D4",  # Accent border for focus/hover
    "shadow": "rgba(0, 0, 0, 0.4)",  # Drop shadow
    "overlay": "rgba(13, 27, 42, 0.85)",  # Modal overlay
    # Gradients (CSS gradient strings)
    "gradient_primary": "linear-gradient(135deg, #0D1B2A 0%, #1B263B 100%)",
    "gradient_accent": "linear-gradient(135deg, #00F5D4 0%, #00B4D8 100%)",
    "gradient_accent_2": "linear-gradient(135deg, #F72585 0%, #FF006E 100%)",
    "gradient_surface": "linear-gradient(135deg, #1B263B 0%, #415A77 100%)",
    # Glassmorphism
    "glass_bg": "rgba(27, 38, 59, 0.6)",
    "glass_border": "rgba(224, 225, 221, 0.1)",
}

# Light mode color overrides (optional)
COLORS_LIGHT = {
    "primary": "#F8F9FA",
    "secondary": "#E9ECEF",
    "surface": "#FFFFFF",
    "text_primary": "#212529",
    "text_secondary": "#6C757D",
    "text_disabled": "#ADB5BD",
    "border": "#DEE2E6",
    "glass_bg": "rgba(255, 255, 255, 0.6)",
    "glass_border": "rgba(0, 0, 0, 0.1)",
}


# ============================================================================
# TYPOGRAPHY SCALE
# ============================================================================

TYPOGRAPHY = {
    # Font Families
    "font_display": "'Space Grotesk', 'Helvetica Neue', sans-serif",
    "font_heading": "'Space Grotesk', 'Helvetica Neue', sans-serif",
    "font_body": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    "font_code": "'JetBrains Mono', 'Fira Code', 'Monaco', monospace",
    # Font Sizes
    "size_display": "3rem",  # 48px - Hero text
    "size_h1": "2.5rem",  # 40px - Page titles
    "size_h2": "2rem",  # 32px - Section headers
    "size_h3": "1.5rem",  # 24px - Subsections
    "size_h4": "1.25rem",  # 20px - Card titles
    "size_body": "1rem",  # 16px - Body text
    "size_small": "0.875rem",  # 14px - Small text
    "size_tiny": "0.75rem",  # 12px - Captions, labels
    # Font Weights
    "weight_light": "300",
    "weight_normal": "400",
    "weight_medium": "500",
    "weight_semibold": "600",
    "weight_bold": "700",
    "weight_black": "900",
    # Line Heights
    "line_tight": "1.2",
    "line_normal": "1.5",
    "line_relaxed": "1.75",
    "line_loose": "2.0",
    # Letter Spacing
    "tracking_tight": "-0.025em",
    "tracking_normal": "0",
    "tracking_wide": "0.025em",
    "tracking_wider": "0.05em",
}


# ============================================================================
# SPACING SCALE (8px base unit)
# ============================================================================

SPACING = {
    "xs": "4px",  # 0.5 * base
    "sm": "8px",  # 1 * base
    "md": "16px",  # 2 * base
    "lg": "24px",  # 3 * base
    "xl": "32px",  # 4 * base
    "xxl": "48px",  # 6 * base
    "xxxl": "64px",  # 8 * base
}


# ============================================================================
# BORDER RADIUS
# ============================================================================

RADII = {
    "none": "0",
    "sm": "4px",
    "md": "8px",
    "lg": "12px",
    "xl": "16px",
    "xxl": "24px",
    "full": "9999px",  # Pill shape
}


# ============================================================================
# SHADOWS
# ============================================================================

SHADOWS = {
    "sm": "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
    "md": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
    "lg": "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
    "xl": "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
    "glow_cyan": "0 0 20px rgba(0, 245, 212, 0.4), 0 0 40px rgba(0, 245, 212, 0.2)",
    "glow_magenta": "0 0 20px rgba(247, 37, 133, 0.4), 0 0 40px rgba(247, 37, 133, 0.2)",
    "glow_success": "0 0 20px rgba(57, 255, 20, 0.4), 0 0 40px rgba(57, 255, 20, 0.2)",
    "inner": "inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)",
}


# ============================================================================
# COMPONENT STYLES
# ============================================================================


def get_card_styles(variant: Literal["glass", "solid", "elevated"] = "glass") -> Dict[str, str]:
    """Get card component styles"""
    base = {
        "border-radius": RADII["lg"],
        "padding": SPACING["lg"],
        "margin-bottom": SPACING["md"],
    }

    if variant == "glass":
        return {
            **base,
            "background": COLORS["glass_bg"],
            "backdrop-filter": "blur(10px)",
            "border": f"1px solid {COLORS['glass_border']}",
            "box-shadow": SHADOWS["md"],
        }
    elif variant == "solid":
        return {
            **base,
            "background": COLORS["secondary"],
            "border": f"1px solid {COLORS['border']}",
            "box-shadow": SHADOWS["sm"],
        }
    else:  # elevated
        return {
            **base,
            "background": COLORS["surface"],
            "border": f"1px solid {COLORS['border']}",
            "box-shadow": SHADOWS["lg"],
        }


def get_button_styles(
    variant: Literal["primary", "secondary", "accent", "ghost"] = "primary"
) -> Dict[str, str]:
    """Get button component styles"""
    base = {
        "border-radius": RADII["md"],
        "padding": f"{SPACING['sm']} {SPACING['lg']}",
        "font-family": TYPOGRAPHY["font_body"],
        "font-size": TYPOGRAPHY["size_body"],
        "font-weight": TYPOGRAPHY["weight_semibold"],
        "cursor": "pointer",
        "transition": "all 0.3s ease",
        "border": "none",
        "text-transform": "uppercase",
        "letter-spacing": TYPOGRAPHY["tracking_wide"],
    }

    if variant == "primary":
        return {
            **base,
            "background": COLORS["gradient_accent"],
            "color": COLORS["primary"],
            "box-shadow": SHADOWS["md"],
        }
    elif variant == "secondary":
        return {
            **base,
            "background": COLORS["secondary"],
            "color": COLORS["text_primary"],
            "border": f"1px solid {COLORS['border_accent']}",
        }
    elif variant == "accent":
        return {
            **base,
            "background": COLORS["gradient_accent_2"],
            "color": COLORS["text_primary"],
            "box-shadow": SHADOWS["glow_magenta"],
        }
    else:  # ghost
        return {
            **base,
            "background": "transparent",
            "color": COLORS["accent"],
            "border": f"1px solid {COLORS['accent']}",
        }


def get_badge_styles(
    status: Literal["success", "warning", "error", "info", "pending"] = "info"
) -> Dict[str, str]:
    """Get badge component styles"""
    color_map = {
        "success": COLORS["success"],
        "warning": COLORS["warning"],
        "error": COLORS["error"],
        "info": COLORS["info"],
        "pending": COLORS["pending"],
    }

    return {
        "display": "inline-block",
        "padding": f"{SPACING['xs']} {SPACING['sm']}",
        "border-radius": RADII["full"],
        "background": f"rgba({_hex_to_rgb(color_map[status])}, 0.2)",
        "color": color_map[status],
        "font-size": TYPOGRAPHY["size_tiny"],
        "font-weight": TYPOGRAPHY["weight_semibold"],
        "text-transform": "uppercase",
        "letter-spacing": TYPOGRAPHY["tracking_wider"],
        "border": f"1px solid {color_map[status]}",
    }


def get_metric_styles(size: Literal["small", "medium", "large"] = "medium") -> Dict[str, Dict[str, str]]:
    """Get metric display styles"""
    sizes = {
        "small": (TYPOGRAPHY["size_h3"], TYPOGRAPHY["size_tiny"]),
        "medium": (TYPOGRAPHY["size_h2"], TYPOGRAPHY["size_small"]),
        "large": (TYPOGRAPHY["size_display"], TYPOGRAPHY["size_body"]),
    }

    value_size, label_size = sizes[size]

    return {
        "value": {
            "font-size": value_size,
            "font-weight": TYPOGRAPHY["weight_bold"],
            "color": COLORS["accent"],
            "line-height": TYPOGRAPHY["line_tight"],
            "font-family": TYPOGRAPHY["font_heading"],
        },
        "label": {
            "font-size": label_size,
            "color": COLORS["text_secondary"],
            "text-transform": "uppercase",
            "letter-spacing": TYPOGRAPHY["tracking_wide"],
            "font-weight": TYPOGRAPHY["weight_medium"],
        },
        "delta_positive": {
            "color": COLORS["success"],
            "font-size": label_size,
        },
        "delta_negative": {
            "color": COLORS["error"],
            "font-size": label_size,
        },
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB string (for rgba usage)"""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"{r}, {g}, {b}"


def get_color_with_alpha(color_name: str, alpha: float = 1.0) -> str:
    """Get color with custom alpha value"""
    if color_name not in COLORS:
        raise ValueError(f"Color '{color_name}' not found in palette")

    hex_color = COLORS[color_name]
    rgb = _hex_to_rgb(hex_color)
    return f"rgba({rgb}, {alpha})"


def css_dict_to_string(styles: Dict[str, str]) -> str:
    """Convert Python dict of CSS properties to CSS string"""
    return "; ".join(f"{k}: {v}" for k, v in styles.items())


# ============================================================================
# COMPLETE CSS GENERATION
# ============================================================================


def get_custom_css(theme: Literal["dark", "light"] = "dark") -> str:
    """
    Generate complete CSS for Streamlit injection

    Args:
        theme: Color theme to use ("dark" or "light")

    Returns:
        Complete CSS string for st.markdown(unsafe_allow_html=True)
    """

    # Use light mode colors if specified
    colors = COLORS if theme == "dark" else {**COLORS, **COLORS_LIGHT}

    css = f"""
    <style>
    /* ===== FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ===== ROOT VARIABLES ===== */
    :root {{
        /* Colors */
        --color-primary: {colors['primary']};
        --color-secondary: {colors['secondary']};
        --color-accent: {colors['accent']};
        --color-accent-2: {colors['accent_2']};
        --color-surface: {colors['surface']};
        --color-text-primary: {colors['text_primary']};
        --color-text-secondary: {colors['text_secondary']};
        --color-success: {colors['success']};
        --color-warning: {colors['warning']};
        --color-error: {colors['error']};
        --color-info: {colors['info']};
        --color-border: {colors['border']};

        /* Spacing */
        --spacing-xs: {SPACING['xs']};
        --spacing-sm: {SPACING['sm']};
        --spacing-md: {SPACING['md']};
        --spacing-lg: {SPACING['lg']};
        --spacing-xl: {SPACING['xl']};
        --spacing-xxl: {SPACING['xxl']};

        /* Typography */
        --font-display: {TYPOGRAPHY['font_display']};
        --font-heading: {TYPOGRAPHY['font_heading']};
        --font-body: {TYPOGRAPHY['font_body']};
        --font-code: {TYPOGRAPHY['font_code']};
    }}

    /* ===== GLOBAL STYLES ===== */
    body {{
        font-family: var(--font-body);
        background: {colors['primary']};
        color: var(--color-text-primary);
    }}

    /* Streamlit container */
    .stApp {{
        background: {colors['gradient_primary']};
    }}

    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {{
        font-family: var(--font-heading);
        color: var(--color-text-primary);
        font-weight: {TYPOGRAPHY['weight_bold']};
        letter-spacing: {TYPOGRAPHY['tracking_tight']};
    }}

    h1 {{
        font-size: {TYPOGRAPHY['size_h1']};
        background: {colors['gradient_accent']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    h2 {{
        font-size: {TYPOGRAPHY['size_h2']};
        color: var(--color-accent);
    }}

    h3 {{
        font-size: {TYPOGRAPHY['size_h3']};
    }}

    p, .stMarkdown {{
        font-size: {TYPOGRAPHY['size_body']};
        line-height: {TYPOGRAPHY['line_normal']};
        color: var(--color-text-primary);
    }}

    code {{
        font-family: var(--font-code);
        background: {colors['secondary']};
        padding: 2px 6px;
        border-radius: {RADII['sm']};
        color: var(--color-accent);
        font-size: {TYPOGRAPHY['size_small']};
    }}

    /* ===== CUSTOM CLASSES ===== */

    /* Headers */
    .main-header {{
        font-family: var(--font-display);
        font-size: {TYPOGRAPHY['size_display']};
        font-weight: {TYPOGRAPHY['weight_black']};
        background: {colors['gradient_accent']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: var(--spacing-xl);
        text-align: center;
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['tracking_wide']};
    }}

    .section-header {{
        font-size: {TYPOGRAPHY['size_h2']};
        color: var(--color-accent);
        border-bottom: 2px solid var(--color-accent);
        padding-bottom: var(--spacing-sm);
        margin-bottom: var(--spacing-lg);
        font-weight: {TYPOGRAPHY['weight_bold']};
    }}

    /* Cards */
    .metric-card {{
        background: {colors['glass_bg']};
        backdrop-filter: blur(10px);
        border: 1px solid {colors['glass_border']};
        border-radius: {RADII['lg']};
        padding: var(--spacing-lg);
        margin: var(--spacing-md) 0;
        box-shadow: {SHADOWS['md']};
        transition: all 0.3s ease;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: {SHADOWS['glow_cyan']};
        border-color: var(--color-accent);
    }}

    .glass-card {{
        background: {colors['glass_bg']};
        backdrop-filter: blur(10px);
        border: 1px solid {colors['glass_border']};
        border-radius: {RADII['lg']};
        padding: var(--spacing-lg);
        box-shadow: {SHADOWS['md']};
    }}

    .solid-card {{
        background: var(--color-secondary);
        border: 1px solid var(--color-border);
        border-radius: {RADII['lg']};
        padding: var(--spacing-lg);
        box-shadow: {SHADOWS['sm']};
    }}

    .elevated-card {{
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: {RADII['lg']};
        padding: var(--spacing-lg);
        box-shadow: {SHADOWS['lg']};
    }}

    /* Status Badges */
    .status-success {{
        display: inline-block;
        padding: {SPACING['xs']} {SPACING['sm']};
        border-radius: {RADII['full']};
        background: rgba(57, 255, 20, 0.2);
        color: var(--color-success);
        font-size: {TYPOGRAPHY['size_tiny']};
        font-weight: {TYPOGRAPHY['weight_semibold']};
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['tracking_wider']};
        border: 1px solid var(--color-success);
        box-shadow: {SHADOWS['glow_success']};
    }}

    .status-running {{
        display: inline-block;
        padding: {SPACING['xs']} {SPACING['sm']};
        border-radius: {RADII['full']};
        background: rgba(255, 183, 3, 0.2);
        color: var(--color-warning);
        font-size: {TYPOGRAPHY['size_tiny']};
        font-weight: {TYPOGRAPHY['weight_semibold']};
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['tracking_wider']};
        border: 1px solid var(--color-warning);
        animation: pulse 2s ease-in-out infinite;
    }}

    .status-failed {{
        display: inline-block;
        padding: {SPACING['xs']} {SPACING['sm']};
        border-radius: {RADII['full']};
        background: rgba(255, 0, 110, 0.2);
        color: var(--color-error);
        font-size: {TYPOGRAPHY['size_tiny']};
        font-weight: {TYPOGRAPHY['weight_semibold']};
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['tracking_wider']};
        border: 1px solid var(--color-error);
    }}

    .status-pending {{
        display: inline-block;
        padding: {SPACING['xs']} {SPACING['sm']};
        border-radius: {RADII['full']};
        background: rgba(119, 141, 169, 0.2);
        color: var(--color-text-secondary);
        font-size: {TYPOGRAPHY['size_tiny']};
        font-weight: {TYPOGRAPHY['weight_semibold']};
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['tracking_wider']};
        border: 1px solid var(--color-text-secondary);
    }}

    /* Metrics */
    .metric-value {{
        font-size: {TYPOGRAPHY['size_h2']};
        font-weight: {TYPOGRAPHY['weight_bold']};
        color: var(--color-accent);
        line-height: {TYPOGRAPHY['line_tight']};
        font-family: var(--font-heading);
    }}

    .metric-label {{
        font-size: {TYPOGRAPHY['size_small']};
        color: var(--color-text-secondary);
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['tracking_wide']};
        font-weight: {TYPOGRAPHY['weight_medium']};
        margin-bottom: var(--spacing-xs);
    }}

    .metric-delta-positive {{
        color: var(--color-success);
        font-size: {TYPOGRAPHY['size_small']};
        font-weight: {TYPOGRAPHY['weight_semibold']};
    }}

    .metric-delta-negative {{
        color: var(--color-error);
        font-size: {TYPOGRAPHY['size_small']};
        font-weight: {TYPOGRAPHY['weight_semibold']};
    }}

    /* Buttons */
    .custom-button {{
        display: inline-block;
        padding: {SPACING['sm']} {SPACING['lg']};
        border-radius: {RADII['md']};
        background: {colors['gradient_accent']};
        color: var(--color-primary);
        font-family: var(--font-body);
        font-size: {TYPOGRAPHY['size_body']};
        font-weight: {TYPOGRAPHY['weight_semibold']};
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['tracking_wide']};
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        box-shadow: {SHADOWS['md']};
        text-decoration: none;
        text-align: center;
    }}

    .custom-button:hover {{
        box-shadow: {SHADOWS['glow_cyan']};
        transform: translateY(-2px);
    }}

    .custom-button-secondary {{
        background: var(--color-secondary);
        color: var(--color-text-primary);
        border: 1px solid var(--color-accent);
    }}

    .custom-button-accent {{
        background: {colors['gradient_accent_2']};
        box-shadow: {SHADOWS['glow_magenta']};
    }}

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {{
        background: {colors['gradient_surface']};
        border-right: 1px solid var(--color-border);
    }}

    .css-1d391kg h1, [data-testid="stSidebar"] h1 {{
        color: var(--color-accent);
        font-family: var(--font-heading);
    }}

    /* Streamlit specific overrides */
    .stButton > button {{
        background: {colors['gradient_accent']};
        color: var(--color-primary);
        border: none;
        border-radius: {RADII['md']};
        padding: {SPACING['sm']} {SPACING['lg']};
        font-weight: {TYPOGRAPHY['weight_semibold']};
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['tracking_wide']};
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        box-shadow: {SHADOWS['glow_cyan']};
        transform: translateY(-2px);
    }}

    .stMetric {{
        background: {colors['glass_bg']};
        backdrop-filter: blur(10px);
        border: 1px solid {colors['glass_border']};
        border-radius: {RADII['md']};
        padding: var(--spacing-md);
    }}

    .stMetric label {{
        color: var(--color-text-secondary) !important;
        font-size: {TYPOGRAPHY['size_small']} !important;
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['tracking_wide']};
    }}

    .stMetric [data-testid="stMetricValue"] {{
        color: var(--color-accent) !important;
        font-size: {TYPOGRAPHY['size_h2']} !important;
        font-weight: {TYPOGRAPHY['weight_bold']} !important;
    }}

    /* Progress bars */
    .stProgress > div > div {{
        background: {colors['gradient_accent']};
        border-radius: {RADII['full']};
    }}

    /* Selectbox, text input, etc. */
    .stSelectbox, .stTextInput, .stNumberInput {{
        background: var(--color-secondary);
        border-radius: {RADII['md']};
    }}

    .stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {{
        background: var(--color-secondary);
        border: 1px solid var(--color-border);
        border-radius: {RADII['md']};
        color: var(--color-text-primary);
    }}

    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stNumberInput > div > div:focus-within {{
        border-color: var(--color-accent);
        box-shadow: 0 0 0 1px var(--color-accent);
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background: var(--color-secondary);
        border: 1px solid var(--color-border);
        border-radius: {RADII['md']};
        color: var(--color-text-primary);
        font-weight: {TYPOGRAPHY['weight_semibold']};
    }}

    .streamlit-expanderHeader:hover {{
        border-color: var(--color-accent);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--color-secondary);
        border-radius: {RADII['md']};
        padding: var(--spacing-xs);
        gap: var(--spacing-xs);
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: var(--color-text-secondary);
        border-radius: {RADII['sm']};
        padding: {SPACING['sm']} {SPACING['md']};
        font-weight: {TYPOGRAPHY['weight_medium']};
    }}

    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {colors['gradient_accent']};
        color: var(--color-primary);
    }}

    /* Table */
    .stTable {{
        background: var(--color-secondary);
        border-radius: {RADII['md']};
        overflow: hidden;
    }}

    .stTable th {{
        background: var(--color-surface);
        color: var(--color-accent);
        font-weight: {TYPOGRAPHY['weight_semibold']};
        text-transform: uppercase;
        font-size: {TYPOGRAPHY['size_small']};
        letter-spacing: {TYPOGRAPHY['tracking_wide']};
    }}

    .stTable td {{
        color: var(--color-text-primary);
        border-color: var(--color-border);
    }}

    /* Dataframe */
    .stDataFrame {{
        background: var(--color-secondary);
        border-radius: {RADII['md']};
    }}

    /* Code blocks */
    .stCodeBlock {{
        background: var(--color-secondary);
        border: 1px solid var(--color-border);
        border-radius: {RADII['md']};
    }}

    .stCodeBlock code {{
        color: var(--color-text-primary);
        font-family: var(--font-code);
    }}

    /* Animations */
    @keyframes pulse {{
        0%, 100% {{
            opacity: 1;
        }}
        50% {{
            opacity: 0.6;
        }}
    }}

    @keyframes glow {{
        0%, 100% {{
            box-shadow: 0 0 5px var(--color-accent);
        }}
        50% {{
            box-shadow: 0 0 20px var(--color-accent), 0 0 40px var(--color-accent);
        }}
    }}

    .pulse {{
        animation: pulse 2s ease-in-out infinite;
    }}

    .glow {{
        animation: glow 2s ease-in-out infinite;
    }}

    /* Utility classes */
    .text-center {{
        text-align: center;
    }}

    .text-accent {{
        color: var(--color-accent);
    }}

    .text-accent-2 {{
        color: var(--color-accent-2);
    }}

    .text-success {{
        color: var(--color-success);
    }}

    .text-warning {{
        color: var(--color-warning);
    }}

    .text-error {{
        color: var(--color-error);
    }}

    .font-code {{
        font-family: var(--font-code);
    }}

    .font-heading {{
        font-family: var(--font-heading);
    }}

    .uppercase {{
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['tracking_wide']};
    }}

    .gradient-text {{
        background: {colors['gradient_accent']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}

    ::-webkit-scrollbar-track {{
        background: var(--color-primary);
    }}

    ::-webkit-scrollbar-thumb {{
        background: var(--color-surface);
        border-radius: {RADII['full']};
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--color-accent);
    }}

    /* Loading spinner */
    .stSpinner > div {{
        border-color: var(--color-accent) transparent transparent transparent;
    }}
    </style>
    """

    return css


# ============================================================================
# EXPORT ALL FOR PROGRAMMATIC ACCESS
# ============================================================================

__all__ = [
    "COLORS",
    "COLORS_LIGHT",
    "TYPOGRAPHY",
    "SPACING",
    "RADII",
    "SHADOWS",
    "get_card_styles",
    "get_button_styles",
    "get_badge_styles",
    "get_metric_styles",
    "get_color_with_alpha",
    "css_dict_to_string",
    "get_custom_css",
]

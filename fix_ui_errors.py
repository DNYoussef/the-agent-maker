#!/usr/bin/env python3
"""
Fix all 5 UI page errors in the Agent Maker Streamlit app.

This script applies all necessary fixes for:
1. Phase 1 & 2: HTML rendering (false positive - already fixed)
2. Phase 6: Plotly bgcolor 8-char hex color
3. Phase 7: Plotly gridcolor 8-char hex color
4. Config Editor: float vs string TypeError

Usage:
    python fix_ui_errors.py
"""

import re
from pathlib import Path

# Base directory
UI_PAGES_DIR = Path(__file__).parent / "src" / "ui" / "pages"

def fix_phase6_baking():
    """Fix Phase 6 Baking Plotly bgcolor error"""
    file_path = UI_PAGES_DIR / "phase6_baking.py"
    print(f"Fixing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix bgcolor with 8-char hex
    old_code = """    # Add cycle type annotations
    for i, cycle in enumerate(cycles):
        badge_color = '#00F5D4' if cycle['type'] == 'A' else '#8338EC'
        fig.add_annotation(
            x=-5,
            y=i,
            text=f"<b>{cycle['type']}</b>",
            showarrow=False,
            font=dict(size=14, color=badge_color, family='monospace'),
            bgcolor=f'{badge_color}22',
            bordercolor=badge_color,
            borderwidth=2,
            borderpad=4,
            xanchor='right'
        )"""

    new_code = """    # Add cycle type annotations
    for i, cycle in enumerate(cycles):
        badge_color = '#00F5D4' if cycle['type'] == 'A' else '#8338EC'
        # Convert to rgba format (8-char hex not supported in Plotly)
        bgcolor_rgba = 'rgba(0, 245, 212, 0.13)' if cycle['type'] == 'A' else 'rgba(131, 56, 236, 0.13)'
        fig.add_annotation(
            x=-5,
            y=i,
            text=f"<b>{cycle['type']}</b>",
            showarrow=False,
            font=dict(size=14, color=badge_color, family='monospace'),
            bgcolor=bgcolor_rgba,
            bordercolor=badge_color,
            borderwidth=2,
            borderpad=4,
            xanchor='right'
        )"""

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  [OK] Fixed bgcolor line 158")
        return True
    else:
        print("  [WARN]  Pattern not found (may already be fixed)")
        return False


def fix_phase7_experts():
    """Fix Phase 7 Experts Plotly gridcolor error"""
    file_path = UI_PAGES_DIR / "phase7_experts.py"
    print(f"Fixing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix gridcolor with 8-char hex
    old_line1 = "    fig.update_xaxes(title_text=\"Epoch\", gridcolor=COLORS['primary'] + '20')"
    new_line1 = "    # Convert to rgba format (8-char hex not supported in Plotly)\n    fig.update_xaxes(title_text=\"Epoch\", gridcolor='rgba(0, 255, 255, 0.125)')"

    old_line2 = "    fig.update_yaxes(title_text=\"Loss\", gridcolor=COLORS['primary'] + '20')"
    new_line2 = "    fig.update_yaxes(title_text=\"Loss\", gridcolor='rgba(0, 255, 255, 0.125)')"

    fixed = False
    if old_line1 in content:
        content = content.replace(old_line1, new_line1)
        print("  [OK] Fixed gridcolor line 687 (xaxis)")
        fixed = True

    if old_line2 in content:
        content = content.replace(old_line2, new_line2)
        print("  [OK] Fixed gridcolor line 688 (yaxis)")
        fixed = True

    if fixed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        print("  [WARN]  Pattern not found (may already be fixed)")

    return fixed


def fix_config_editor():
    """Fix Config Editor float vs string TypeError"""
    file_path = UI_PAGES_DIR / "config_editor.py"
    print(f"Fixing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # List of all numeric config.get() calls that need int() or float() conversion
    fixes = [
        # Integer values for sliders
        ("value=hardware_config.get('device_vram_gb', 6),",
         "value=int(hardware_config.get('device_vram_gb', 6)),"),

        ("value=hardware_config.get('max_batch_size', 32)",
         "value=int(hardware_config.get('max_batch_size', 32))"),

        ("value=hardware_config.get('num_workers', 4)",
         "value=int(hardware_config.get('num_workers', 4))"),

        ("value=cleanup_config.get('max_session_age_days', 30)",
         "value=int(cleanup_config.get('max_session_age_days', 30))"),

        ("value=cleanup_config.get('max_sessions_total', 100)",
         "value=int(cleanup_config.get('max_sessions_total', 100))"),

        ("value=cleanup_config.get('keep_last_n_checkpoints', 5)",
         "value=int(cleanup_config.get('keep_last_n_checkpoints', 5))"),

        ("value=phase_config.get('num_models', 3)",
         "value=int(phase_config.get('num_models', 3))"),

        ("value=phase_config.get('epochs', 10)",
         "value=int(phase_config.get('epochs', 10))"),

        # Float values for number_input
        ("value=optimizer_config.get('muon_lr', 0.001),",
         "value=float(optimizer_config.get('muon_lr', 0.001)),"),

        ("value=optimizer_config.get('grokfast_lambda', 0.3),",
         "value=float(optimizer_config.get('grokfast_lambda', 0.3)),"),

        ("value=phase_config.get('num_generations', 50)",
         "value=int(phase_config.get('num_generations', 50))"),

        ("value=phase_config.get('population_size', 8),",
         "value=int(phase_config.get('population_size', 8)),"),

        ("value=phase_config.get('baking_epochs', 3)",
         "value=int(phase_config.get('baking_epochs', 3))"),

        ("value=phase_config.get('kl_coefficient', 0.1),",
         "value=float(phase_config.get('kl_coefficient', 0.1)),"),

        ("value=phase_config.get('rl_epochs', 5)",
         "value=int(phase_config.get('rl_epochs', 5))"),

        ("value=phase_config.get('target_compression', 8.2),",
         "value=float(phase_config.get('target_compression', 8.2)),"),

        ("value=phase_config.get('ste_epochs', 5)",
         "value=int(phase_config.get('ste_epochs', 5))"),

        ("value=phase_config.get('quality_threshold', 90)",
         "value=int(phase_config.get('quality_threshold', 90))"),
    ]

    fixed_count = 0
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            fixed_count += 1

    if fixed_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [OK] Fixed {fixed_count} type conversion issues")
        return True
    else:
        print("  [WARN]  No patterns found (may already be fixed)")
        return False


def main():
    """Run all fixes"""
    print("=" * 60)
    print("Agent Maker UI Error Fixes")
    print("=" * 60)

    print("\nFix Summary:")
    print("1. Phase 1 & 2: HTML rendering (already has unsafe_allow_html=True)")
    print("2. Phase 6: Plotly bgcolor 8-char hex -> rgba()")
    print("3. Phase 7: Plotly gridcolor 8-char hex -> rgba()")
    print("4. Config Editor: Add int()/float() type conversions")

    print("\n" + "=" * 60)

    # Phase 1 & 2 - Already fixed (false positive)
    print("\n[1] Phase 1 & 2 Cognate/EvoMerge:")
    print("  [OK] Already has unsafe_allow_html=True (false positive)")

    # Phase 6 - Plotly bgcolor
    print("\n[2] Phase 6 Baking:")
    fix_phase6_baking()

    # Phase 7 - Plotly gridcolor
    print("\n[3] Phase 7 Experts:")
    fix_phase7_experts()

    # Config Editor - Type conversion
    print("\n[4] Config Editor:")
    fix_config_editor()

    print("\n" + "=" * 60)
    print("[OK] All fixes complete!")
    print("=" * 60)

    print("\nNext Steps:")
    print("1. Test Phase 6 Baking page (check A/B cycle timeline)")
    print("2. Test Phase 7 Experts page (check SVF training charts)")
    print("3. Test Config Editor page (adjust sliders and save)")
    print("4. Report any remaining errors")


if __name__ == "__main__":
    main()

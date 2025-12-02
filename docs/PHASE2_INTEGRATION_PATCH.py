"""
Simple integration patch for Phase 2 3D visualization

This script shows the exact code changes needed to integrate the 3D visualization
into phase_details.py. You can either:

1. Apply changes manually (copy the NEW CODE section)
2. Run this script to auto-patch (requires file write permissions)
"""

OLD_CODE = """    # Binary pairing tree
    with st.expander("View Binary Pairing Tree"):
        st.info("3D merge visualization coming soon (Three.js integration)")"""

NEW_CODE = """    # 3D Merge Tree Visualization (INTEGRATED)
    try:
        from ui.components.merge_tree_3d import render_phase2_3d_visualization
        render_phase2_3d_visualization(
            generations=50,
            models_per_gen=8,
            height=800,
            show_controls=True
        )
    except ImportError as e:
        st.warning(f"3D visualization unavailable: {e}")
        with st.expander("View Binary Pairing Tree"):
            st.info("Install plotly to enable 3D merge visualization: pip install plotly")"""


def show_diff():
    """Display the code diff"""
    print("=" * 80)
    print("PHASE 2 INTEGRATION PATCH")
    print("=" * 80)
    print("\nFile: src/ui/pages/phase_details.py")
    print("Lines: 154-156 (in render_phase2_details function)")
    print("\n" + "-" * 80)
    print("OLD CODE (REMOVE):")
    print("-" * 80)
    print(OLD_CODE)
    print("\n" + "-" * 80)
    print("NEW CODE (REPLACE WITH):")
    print("-" * 80)
    print(NEW_CODE)
    print("\n" + "=" * 80)
    print("\nINSTRUCTIONS:")
    print("1. Open: src/ui/pages/phase_details.py")
    print("2. Find the render_phase2_details() function (line 117)")
    print("3. Locate lines 154-156 (the OLD CODE above)")
    print("4. Replace with the NEW CODE above")
    print("5. Save the file")
    print("6. Test with: streamlit run src/ui/app.py")
    print("\nDEPENDENCIES:")
    print("pip install plotly pandas numpy")
    print("\n" + "=" * 80)


def apply_patch(file_path):
    """Apply the patch automatically"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if OLD_CODE in content:
            new_content = content.replace(OLD_CODE, NEW_CODE)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✅ Successfully patched {file_path}")
            print("Changes applied. Test with: streamlit run src/ui/app.py")
            return True
        else:
            print(f"❌ Could not find OLD CODE in {file_path}")
            print("The file may have already been patched or modified.")
            print("\nShowing diff for manual application:")
            show_diff()
            return False

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Make sure you're running from the project root directory.")
        return False
    except Exception as e:
        print(f"❌ Error applying patch: {e}")
        return False


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Default file path
    default_path = Path(__file__).parent.parent / "src/ui/pages/phase_details.py"

    if len(sys.argv) > 1 and sys.argv[1] == '--apply':
        # Auto-apply patch
        file_path = sys.argv[2] if len(sys.argv) > 2 else default_path
        apply_patch(file_path)
    else:
        # Just show diff
        show_diff()
        print("\nTo auto-apply patch:")
        print(f"python {Path(__file__).name} --apply")
        print(f"python {Path(__file__).name} --apply path/to/phase_details.py")

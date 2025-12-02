"""
Verification script for HTML rendering fixes.
Checks that all files have been properly updated.
"""

import re
from pathlib import Path

def check_file(filepath: str, checks: list) -> dict:
    """Run checks on a file and return results"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    results = {
        'file': Path(filepath).name,
        'checks': {},
        'passed': 0,
        'failed': 0
    }

    for check_name, pattern in checks:
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            results['checks'][check_name] = 'PASS'
            results['passed'] += 1
        else:
            results['checks'][check_name] = 'FAIL'
            results['failed'] += 1

    return results

def main():
    base_path = Path(r"C:\Users\17175\Desktop\the agent maker\src\ui\pages")

    files_to_check = {
        'phase1_cognate.py': [
            ('Import components', r'import streamlit\.components\.v1 as components'),
            ('Model card html', r'components\.html\(card_html, height=400\)'),
            ('Architecture html', r'components\.html\(arch_html, height=650\)')
        ],
        'phase2_evomerge.py': [
            ('Import components', r'import streamlit\.components\.v1 as components'),
            ('Hero html', r'components\.html\(hero_html, height=150\)'),
            ('Champion html', r'components\.html\(champion_html, height=320\)')
        ],
        'phase6_baking.py': [
            ('Import components', r'import streamlit\.components\.v1 as components'),
            ('Hero html', r'components\.html\(hero_html, height=150\)')
        ]
    }

    print("=" * 70)
    print("HTML RENDERING FIX VERIFICATION")
    print("=" * 70)

    all_passed = 0
    all_failed = 0

    for filename, checks in files_to_check.items():
        filepath = base_path / filename
        results = check_file(str(filepath), checks)

        print(f"\n{results['file']}:")
        print("-" * 70)
        for check_name, status in results['checks'].items():
            symbol = "OK" if status == 'PASS' else "!!"
            print(f"  [{symbol}] {check_name}: {status}")

        all_passed += results['passed']
        all_failed += results['failed']

    print("\n" + "=" * 70)
    print(f"SUMMARY: {all_passed} checks passed, {all_failed} checks failed")
    print("=" * 70)

    if all_failed == 0:
        print("\nSUCCESS! All HTML rendering fixes verified.")
        print("\nNext steps:")
        print("  1. Run: streamlit run src/ui/app.py")
        print("  2. Navigate to Phase 1, 2, and 6 pages")
        print("  3. Verify HTML renders correctly (no raw tags visible)")
        print("  4. Check that gradients, animations, and styles display")
    else:
        print("\nWARNING! Some checks failed. Review the output above.")
        print("Re-run the fix scripts if needed.")

if __name__ == "__main__":
    main()

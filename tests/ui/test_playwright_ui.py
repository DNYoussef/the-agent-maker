"""
Playwright UI Tests for Agent Forge V2
Tests all pages and takes screenshots
"""
import subprocess
import time
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def wait_for_streamlit(url: str, timeout: int = 30) -> bool:
    """Wait for Streamlit server to be ready."""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = urllib.request.urlopen(url, timeout=2)
            if response.status == 200:
                return True
        except (urllib.error.URLError, Exception):
            pass
        time.sleep(1)
    return False


def test_ui_screenshots():
    """Take screenshots of all UI pages."""
    project_root = Path(__file__).parent.parent.parent
    screenshots_dir = project_root / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    # Start Streamlit in background
    print("Starting Streamlit server...")
    streamlit_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         str(project_root / "src" / "ui" / "app.py"),
         "--server.headless", "true",
         "--server.port", "8501"],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server
        print("Waiting for server to start...")
        if not wait_for_streamlit("http://localhost:8501", timeout=30):
            raise RuntimeError("Streamlit server did not start in time")

        print("Server ready, starting Playwright...")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})

            # List of pages to test
            pages_to_test = [
                ("Pipeline Overview", "01_pipeline_overview.png"),
                ("Phase Details", "02_phase_details.png"),
                ("Phase 4: BitNet Compression", "03_phase4_bitnet.png"),
                ("Model Browser", "04_model_browser.png"),
                ("System Monitor", "05_system_monitor.png"),
                ("Configuration Editor", "06_config_editor.png"),
            ]

            results = []

            for page_name, screenshot_name in pages_to_test:
                print(f"Testing page: {page_name}")
                try:
                    # Navigate to main page
                    page.goto("http://localhost:8501", wait_until="networkidle")
                    time.sleep(2)  # Allow Streamlit to fully render

                    # Click on the navigation radio button
                    # Streamlit radio buttons are in the sidebar
                    radio_label = page.locator(f'label:has-text("{page_name}")')
                    if radio_label.count() > 0:
                        radio_label.first.click()
                        time.sleep(2)  # Wait for page to load

                    # Take screenshot
                    screenshot_path = screenshots_dir / screenshot_name
                    page.screenshot(path=str(screenshot_path), full_page=True)
                    print(f"  Screenshot saved: {screenshot_path}")
                    results.append((page_name, "PASS", str(screenshot_path)))

                except Exception as e:
                    print(f"  Error on {page_name}: {e}")
                    results.append((page_name, "FAIL", str(e)))

            browser.close()

        return results

    finally:
        # Stop Streamlit
        print("Stopping Streamlit server...")
        streamlit_proc.terminate()
        streamlit_proc.wait(timeout=5)


if __name__ == "__main__":
    results = test_ui_screenshots()
    print("\n" + "="*50)
    print("UI TEST RESULTS")
    print("="*50)
    for page_name, status, info in results:
        print(f"[{status}] {page_name}: {info}")

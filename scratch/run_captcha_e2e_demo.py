#!/usr/bin/env python3
"""
End-to-end CAPTCHA solver demo: open a local fake CAPTCHA page in Chrome
and run the solver (detect → classify tiles → click → verify).

The demo page is a 3×3 grid of the same image (a car). The solver should
detect "car" in the grid, click all 9 tiles, and hit Verify. Use this to
confirm the full pipeline works without hitting a real CAPTCHA.

Usage
-----
  python run_captcha_e2e_demo.py

Requires: selenium, ultralytics, Pillow, Chrome/ChromeDriver.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    # Path to the demo HTML (next to this script)
    demo_html = Path(__file__).resolve().parent / "captcha_demo.html"
    if not demo_html.exists():
        print("Error: captcha_demo.html not found next to this script.", file=sys.stderr)
        sys.exit(1)

    url = demo_html.as_uri()
    print(f"Opening {url}")

    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=900,700")
    driver = webdriver.Chrome(options=opts)

    try:
        driver.get(url)
        time.sleep(1.5)

        from captcha import handle_captcha

        solved = handle_captcha(driver, max_attempts=2, post_solve_wait=1.0)

        if solved:
            print("Result: CAPTCHA was solved (or none was present).")
        else:
            print("Result: Solver did not succeed. Check the browser window and logs.")
    finally:
        input("Press Enter to close the browser...")
        driver.quit()


if __name__ == "__main__":
    main()

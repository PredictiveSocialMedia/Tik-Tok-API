"""
Browser WebDriver setup and lifecycle.

Configurable for headless, window size, anti-detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def create_driver(
    headless: bool = True,
    window_width: int = 1920,
    window_height: int = 1080,
    user_agent: Optional[str] = None,
) -> WebDriver:
    """
    Create a Chrome WebDriver configured for TikTok scraping.

    Applies anti-detection tweaks (navigator.webdriver, common flags).
    """
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(f"--window-size={window_width},{window_height}")
    opts.add_argument(f"user-agent={user_agent or DEFAULT_USER_AGENT}")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-extensions")

    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(30)

    # Hide webdriver property
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )

    logger.info("WebDriver created (headless=%s)", headless)
    return driver


def load_cookies(driver: WebDriver, cookies_path: Path) -> bool:
    """Load cookies from a JSON file and add them to the driver."""
    import json
    if not cookies_path.exists():
        logger.debug("No cookies file at %s", cookies_path)
        return False
    try:
        with open(cookies_path) as f:
            cookies = json.load(f)
        if not isinstance(cookies, list):
            cookies = cookies.get("cookies", cookies) if isinstance(cookies, dict) else []
        for c in cookies:
            if isinstance(c, dict) and "name" in c and "value" in c:
                driver.add_cookie(c)
        logger.info("Loaded %d cookies from %s", len(cookies), cookies_path)
        return True
    except Exception as e:
        logger.warning("Failed to load cookies: %s", e)
        return False


def save_cookies(driver: WebDriver, cookies_path: Path) -> bool:
    """Save current cookies to a JSON file."""
    import json
    try:
        cookies_path.parent.mkdir(parents=True, exist_ok=True)
        cookies = driver.get_cookies()
        with open(cookies_path, "w") as f:
            json.dump(cookies, f, indent=2)
        logger.info("Saved %d cookies to %s", len(cookies), cookies_path)
        return True
    except Exception as e:
        logger.warning("Failed to save cookies: %s", e)
        return False

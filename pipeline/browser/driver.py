"""
Browser WebDriver setup and lifecycle.

Uses ``undetected-chromedriver`` to bypass common bot-detection checks
(navigator.webdriver, cdc_ variables, Chrome DevTools leak). Falls back
to regular ``selenium.webdriver.Chrome`` if ``undetected-chromedriver``
is unavailable.

Anti-detection layers applied automatically on creation:
    1. undetected-chromedriver patches (binary & JS)
    2. Stealth JS injections (plugins, languages, chrome.runtime, permissions)
    3. Modern user-agent string
    4. Automation-flag suppression (--disable-blink-features=AutomationControlled)

See ``pipeline.browser.stealth`` for human-interaction helpers and the
rate-limit cooldown mechanism.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from selenium.webdriver.chrome.webdriver import WebDriver

logger = logging.getLogger(__name__)

# Modern user-agent — keep in sync with a recent stable Chrome release.
# Update periodically when Chrome ships new major versions.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Whether undetected-chromedriver is installed
_UC_AVAILABLE: bool
try:
    import undetected_chromedriver as uc  # type: ignore[import-untyped]
    _UC_AVAILABLE = True
except ImportError:
    _UC_AVAILABLE = False


def create_driver(
    headless: bool = True,
    window_width: int = 1920,
    window_height: int = 1080,
    user_agent: Optional[str] = None,
) -> WebDriver:
    """
    Create a Chrome WebDriver configured for TikTok scraping.

    Strategy (in priority order):
        1. ``undetected-chromedriver`` — patches the chromedriver binary and
           runtime JS to remove the most common automation fingerprints.
        2. Falls back to ``selenium.webdriver.Chrome`` with manual stealth
           tweaks if uc is not installed.

    After creation the driver receives the full stealth-script injection
    from ``pipeline.browser.stealth.inject_stealth_scripts``.

    Args:
        headless: Run without a visible browser window.
        window_width: Initial viewport width.
        window_height: Initial viewport height.
        user_agent: Override the default user-agent string.

    Returns:
        A configured ``WebDriver`` instance.
    """
    ua = user_agent or DEFAULT_USER_AGENT

    # Prefer undetected-chromedriver, but *never* crash the pipeline if it
    # cannot download or patch Chrome (e.g. SSL issues, offline machine).
    if _UC_AVAILABLE:
        try:
            driver = _create_uc_driver(headless, window_width, window_height, ua)
        except Exception as e:
            logger.warning(
                "undetected-chromedriver failed (%s); falling back to standard "
                "Selenium. Bot detection risk is higher.",
                e,
            )
            driver = _create_standard_driver(headless, window_width, window_height, ua)
    else:
        logger.warning(
            "undetected-chromedriver not installed — falling back to "
            "standard Selenium. Bot detection risk is higher."
        )
        driver = _create_standard_driver(headless, window_width, window_height, ua)

    # Additional stealth layer on top of uc (covers edge cases)
    from pipeline.browser.stealth import inject_stealth_scripts
    inject_stealth_scripts(driver)

    logger.info(
        "WebDriver created (headless=%s, uc=%s, ua=Chrome/%s)",
        headless,
        _UC_AVAILABLE,
        ua.split("Chrome/")[1].split(" ")[0] if "Chrome/" in ua else "?",
    )
    return driver


# --- Internal constructors ---


def _create_uc_driver(
    headless: bool,
    width: int,
    height: int,
    user_agent: str,
) -> WebDriver:
    """Build a driver via ``undetected-chromedriver``."""
    options = uc.ChromeOptions()
    options.add_argument(f"--window-size={width},{height}")
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")

    driver = uc.Chrome(options=options, headless=headless, use_subprocess=True)
    driver.set_page_load_timeout(30)
    return driver


def _create_standard_driver(
    headless: bool,
    width: int,
    height: int,
    user_agent: str,
) -> WebDriver:
    """Fallback: standard ``selenium.webdriver.Chrome`` with manual stealth."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(f"--window-size={width},{height}")
    opts.add_argument(f"user-agent={user_agent}")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-extensions")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(30)

    # Manual navigator.webdriver patch (uc does this automatically)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


# --- Cookie helpers ---


def load_cookies(driver: WebDriver, cookies_path: Path) -> bool:
    """Load cookies from a JSON file and add them to the driver."""
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

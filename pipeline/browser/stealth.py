"""
Anti-detection stealth helpers.

Provides:
- human_type: Type with random per-character delays.
- human_click: Move mouse to element with slight offset, pause, click.
- human_sleep: Sleep with random jitter.
- inject_stealth_scripts: Patch navigator, plugins, permissions, chrome runtime.
- disconnect_driver / reconnect_driver: Detach chromedriver during sensitive moments.
- rate_limit_check / set_rate_limit: Cooldown file to prevent repeated runs after TikTok blocks.

See docs/anti-detection.md for full documentation.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Optional

from selenium.webdriver.common.action_chains import ActionChains  # type: ignore[import-untyped]
from selenium.webdriver.remote.webdriver import WebDriver  # type: ignore[import-untyped]

log = logging.getLogger(__name__)

# --- Rate-limit cooldown ---
DEFAULT_COOLDOWN_MINUTES = 45
COOLDOWN_FILE = Path("data/tiktok/.rate_limit_until")


def is_rate_limited(cooldown_file: Optional[Path] = None) -> bool:
    """Check if we're inside a rate-limit cooldown period. Returns True if we should NOT run."""
    path = cooldown_file or COOLDOWN_FILE
    if not path.exists():
        return False
    try:
        until = float(path.read_text().strip())
        if time.time() < until:
            remaining = int((until - time.time()) / 60)
            log.warning("[stealth] Rate-limited; ~%d minutes remaining. Skipping this run.", remaining)
            return True
        path.unlink(missing_ok=True)
    except Exception:
        path.unlink(missing_ok=True)
    return False


def set_rate_limit(minutes: int = DEFAULT_COOLDOWN_MINUTES, cooldown_file: Optional[Path] = None) -> None:
    """Write a cooldown timestamp so future runs know to wait."""
    path = cooldown_file or COOLDOWN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(time.time() + minutes * 60))
    log.info("[stealth] Rate-limit cooldown set for %d minutes.", minutes)


def clear_rate_limit(cooldown_file: Optional[Path] = None) -> None:
    """Clear the cooldown (e.g. after successful login)."""
    path = cooldown_file or COOLDOWN_FILE
    path.unlink(missing_ok=True)


# --- Human-like input ---

def human_type(element, text: str, min_delay: float = 0.05, max_delay: float = 0.18) -> None:
    """Type text one character at a time with random delays (simulates human typing)."""
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(min_delay, max_delay))
    time.sleep(random.uniform(0.3, 0.7))


def human_click(driver: WebDriver, element) -> None:
    """Move mouse to element with a small random offset, pause briefly, then click."""
    offset_x = random.randint(-3, 3)
    offset_y = random.randint(-3, 3)
    pause = random.uniform(0.1, 0.4)
    ActionChains(driver).move_to_element_with_offset(
        element, offset_x, offset_y
    ).pause(pause).click().perform()


def human_sleep(base: float, jitter: float = 0.5) -> None:
    """Sleep for ``base ± jitter`` seconds. Use instead of fixed ``time.sleep``."""
    duration = max(0.1, base + random.uniform(-jitter, jitter))
    time.sleep(duration)


# --- Stealth script injection ---

STEALTH_JS = """
// Hide webdriver flag
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

// Fake plugins array (empty = headless giveaway)
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5],
});

// Standard languages
Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en'],
});

// Chrome runtime object (missing in automation)
if (!window.chrome) window.chrome = {};
if (!window.chrome.runtime) window.chrome.runtime = {};

// Permissions API: return actual Notification.permission for 'notifications' query
const origQuery = window.navigator.permissions.query.bind(window.navigator.permissions);
window.navigator.permissions.query = (params) =>
    params.name === 'notifications'
        ? Promise.resolve({state: Notification.permission})
        : origQuery(params);

// Platform consistency
Object.defineProperty(navigator, 'platform', {get: () => 'MacIntel'});
Object.defineProperty(navigator, 'vendor', {get: () => 'Google Inc.'});
Object.defineProperty(navigator, 'maxTouchPoints', {get: () => 0});
"""


def inject_stealth_scripts(driver: WebDriver) -> None:
    """
    Inject JS that hides automation fingerprints on every new document.

    Covers: navigator.webdriver, plugins, languages, chrome.runtime,
    permissions API, platform, vendor, maxTouchPoints.
    """
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": STEALTH_JS},
        )
        log.debug("[stealth] Stealth scripts injected")
    except Exception as e:
        log.warning("[stealth] Could not inject stealth scripts: %s", e)


# --- ChromeDriver disconnect/reconnect ---

def disconnect_driver(driver: WebDriver) -> None:
    """
    Temporarily stop chromedriver service so the browser runs without
    an attached automation websocket. Useful before sensitive page loads
    or form submissions — detection scripts can't see the devtools protocol.
    """
    try:
        driver.service.stop()
        log.debug("[stealth] ChromeDriver disconnected")
    except Exception as e:
        log.debug("[stealth] Could not disconnect: %s", e)


def reconnect_driver(driver: WebDriver) -> None:
    """Re-start chromedriver service after a disconnect."""
    try:
        driver.service.start()
        log.debug("[stealth] ChromeDriver reconnected")
    except Exception as e:
        log.debug("[stealth] Could not reconnect: %s", e)

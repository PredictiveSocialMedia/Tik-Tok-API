"""
Login and session management.

Supports:
- Loading saved cookies (session reuse)
- Manual login (user logs in in browser; we save cookies after)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from selenium.webdriver.remote.webdriver import WebDriver

from pipeline.browser import create_driver, load_cookies, save_cookies
from pipeline.captcha import solve_captcha

logger = logging.getLogger(__name__)


def ensure_logged_in(
    driver: WebDriver,
    login_url: str = "https://www.tiktok.com/login",
    for_you_url: str = "https://www.tiktok.com/foryou",
    cookies_path: Optional[Path] = None,
) -> bool:
    """
    Ensure we have a logged-in session.

    1. Go to TikTok; load cookies if available.
    2. Navigate to For You; if redirected to login, wait for user to log in (or captcha).
    3. Solve captcha if it appears.
    4. Return True when we're on a page that looks logged-in (e.g. For You feed).
    """
    driver.get(login_url)
    time.sleep(2)

    if cookies_path and cookies_path.exists():
        load_cookies(driver, cookies_path)
        driver.refresh()
        time.sleep(2)

    # Try For You (logged-in users see feed)
    driver.get(for_you_url)
    time.sleep(3)

    # Check for captcha
    solved = solve_captcha(driver, max_attempts=3)
    if not solved:
        logger.warning("Captcha not solved; user may need to solve manually")
        return False

    # Heuristic: if URL contains 'login' we're probably not logged in
    current = driver.current_url
    if "login" in current.lower():
        logger.info("Still on login page; waiting for manual login (60s)...")
        time.sleep(60)
        driver.get(for_you_url)
        time.sleep(3)
        if "login" in driver.current_url.lower():
            return False

    return True


def login_with_cookies(
    cookies_path: Path,
    for_you_url: str = "https://www.tiktok.com/foryou",
    headless: bool = False,
) -> Optional[WebDriver]:
    """
    Create driver, load cookies, go to For You. Returns driver if successful.
    Use when you have a pre-saved session.
    """
    driver = create_driver(headless=headless)
    driver.get(for_you_url)
    time.sleep(2)
    if not load_cookies(driver, cookies_path):
        logger.warning("No cookies loaded")
        return None
    driver.refresh()
    time.sleep(3)
    if "login" in driver.current_url.lower():
        solve_captcha(driver, max_attempts=2)
        time.sleep(2)
    return driver

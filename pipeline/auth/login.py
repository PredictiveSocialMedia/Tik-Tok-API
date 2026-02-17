"""
Login and session management with anti-detection measures.

Template flow:
    open page → dismiss popups → click Log in → use email →
    log in with password → fill form (human typing) → submit → wait.

Anti-detection layers active during login:
    - Human-like typing with random per-character delays
    - Human-like mouse movement before each click
    - Jittered sleeps (no fixed intervals)
    - Rate-limit cooldown file to avoid rapid retries

See ``pipeline.browser.stealth`` and ``docs/anti-detection.md``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from selenium.webdriver.common.by import By  # type: ignore[import-untyped]
from selenium.webdriver.remote.webdriver import WebDriver  # type: ignore[import-untyped]
from selenium.webdriver.support import expected_conditions as EC  # type: ignore[import-untyped]
from selenium.webdriver.support.ui import WebDriverWait  # type: ignore[import-untyped]

from pipeline.browser import create_driver, load_cookies, save_cookies
from pipeline.browser.stealth import human_click, human_sleep, human_type, set_rate_limit
from pipeline.captcha import solve_captcha

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Selectors (XPath-based, matching TikTok's current login modal)
# ---------------------------------------------------------------------------
COMMUNICATION_POPUP = (
    By.XPATH,
    "//*[@id='pns-communication-service']/div/div/div/div[2]/div/button",
)
HEADER_LOGIN_BUTTON = (By.XPATH, "//*[@id='header-login-button']")
LOGIN_USE_EMAIL = (
    By.XPATH,
    "//*[@id='loginContainer']/div[1]/div/div/div/div/div[2]/div[2]",
)
LOGIN_WITH_PASSWORD_LINK = (
    By.XPATH,
    "//*[@id='loginContainer']/div[2]/div/div/div/a",
)
LOGIN_USERNAME = (
    By.XPATH,
    "//*[@id='loginContainer']/div[2]/div/div/form/div[1]/div/input",
)
LOGIN_PASSWORD = (
    By.XPATH,
    "//*[@id='loginContainer']/div[2]/div/div/form/div[2]/div/div/input",
)
LOGIN_SUBMIT = (
    By.XPATH,
    "//*[@id='loginContainer']/div[2]/div/div/form/div[4]/button",
)
LOGGED_IN_INDICATOR = (By.XPATH, "//a[contains(@href,'logout')]")

# ---------------------------------------------------------------------------
# JavaScript helpers (shadow-DOM piercing)
# ---------------------------------------------------------------------------
JS_DECLINE_COOKIES = """
var host = document.querySelector('tiktok-cookie-banner');
if (!host || !host.shadowRoot) return false;
var root = host.shadowRoot, buttons = root.querySelectorAll('button');
for (var i = 0; i < buttons.length; i++) {
    if (buttons[i].textContent.trim().indexOf('Decline optional cookies') !== -1) {
        buttons[i].click(); return true;
    }
}
return false;
"""

JS_CLICK_LOGIN_SHADOW = """
function findInShadowRoot(root, fn) {
    var el = fn(root); if (el) return el;
    var nodes = root.querySelectorAll('*');
    for (var i = 0; i < nodes.length; i++) {
        if (nodes[i].shadowRoot) {
            el = findInShadowRoot(nodes[i].shadowRoot, fn);
            if (el) return el;
        }
    }
    return null;
}
var byId = function(r) {
    return r.querySelector ? r.querySelector('[id="header-login-button"]') : null;
};
var btn = (document.getElementById && document.getElementById('header-login-button'))
           || findInShadowRoot(document.body, byId);
if (btn) { btn.click(); return true; }
var byText = function(r) {
    var b = r.querySelectorAll('button, a, [role="button"]');
    for (var j = 0; j < b.length; j++) {
        if (b[j].textContent.trim() === 'Log in') return b[j];
    }
    return null;
};
btn = findInShadowRoot(document.body, byText);
if (btn) { btn.click(); return true; }
return false;
"""


# ---------------------------------------------------------------------------
# Overlay dismissal
# ---------------------------------------------------------------------------

def dismiss_overlays(driver: WebDriver, timeout: float = 10.0) -> None:
    """
    Dismiss any blocking overlays so the login button is clickable.

    Steps:
        1. Close communication popup (if present).
        2. Decline optional cookies via shadow-DOM JS (if present).
    """
    wait = WebDriverWait(driver, 5.0)

    # 1. Communication popup
    try:
        btn = wait.until(EC.element_to_be_clickable(COMMUNICATION_POPUP))
        human_click(driver, btn)
        log.info("[login] 1. Dismissed communication popup")
        human_sleep(0.5, jitter=0.3)
    except Exception:
        log.debug("[login] 1. No communication popup")

    # 2. Cookie banner (shadow DOM)
    try:
        wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "tiktok-cookie-banner"),
            )
        )
        human_sleep(0.5, jitter=0.2)
        if driver.execute_script(JS_DECLINE_COOKIES):
            log.info("[login] 2. Dismissed cookie banner")
            human_sleep(0.5, jitter=0.2)
        else:
            btn = driver.find_element(
                By.XPATH, "//button[contains(., 'Decline optional cookies')]"
            )
            human_click(driver, btn)
            human_sleep(0.5, jitter=0.2)
            log.info("[login] 2. Dismissed cookie banner (fallback)")
    except Exception:
        log.debug("[login] 2. No cookie banner or already dismissed")


# ---------------------------------------------------------------------------
# Login-button helpers
# ---------------------------------------------------------------------------

def _click_login_btn(driver: WebDriver, wait_sec: float = 10.0) -> bool:
    """
    Click the header "Log in" button.

    Tries, in order:
        1. Selenium wait + human-like click.
        2. JavaScript ``arguments[0].click()`` on the found element.
        3. Shadow-DOM traversal JS as a last resort.

    Returns:
        ``True`` if the button was clicked successfully.
    """
    wait = WebDriverWait(driver, wait_sec)
    human_sleep(0.5, jitter=0.3)

    try:
        btn = wait.until(EC.element_to_be_clickable(HEADER_LOGIN_BUTTON))
        try:
            human_click(driver, btn)
        except Exception:
            driver.execute_script("arguments[0].click();", btn)
        log.info("[login] 3. Clicked Log in button")
        return True
    except Exception:
        pass

    if driver.execute_script(JS_CLICK_LOGIN_SHADOW):
        log.info("[login] 3. Clicked Log in button (shadow DOM)")
        return True
    return False


def is_login_button_visible(driver: WebDriver, timeout: float = 5.0) -> bool:
    """Return ``True`` if the header Log in button is present on the page."""
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(HEADER_LOGIN_BUTTON)
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main login flow
# ---------------------------------------------------------------------------

def login_with_credentials(
    driver: WebDriver,
    email: str,
    password: str,
    login_url: str = "https://www.tiktok.com/login",
    timeout: float = 15.0,
) -> bool:
    """
    Log in to TikTok via the email/password modal.

    This function works on the *current page* (no navigation needed):
        1-2. Overlays — call ``dismiss_overlays`` before this.
        3.   Click the header "Log in" button.
        4.   Select "Use phone / email / username".
        5.   Click "Log in with password".
        6.   Fill email and password with human-like typing, then submit.
        7.   Handle captcha, rate-limit, and post-login verification.

    Anti-detection measures applied automatically:
        - ``human_type`` for form inputs (random per-char delays).
        - ``human_click`` for all button clicks (random offsets + pauses).
        - ``human_sleep`` between steps (jittered intervals).
        - On rate-limit detection, writes a cooldown file so future runs
          skip login for a configurable period.

    Args:
        driver: Active WebDriver pointing at TikTok.
        email: TikTok account email / username.
        password: TikTok account password.
        login_url: Not navigated to; kept for API consistency.
        timeout: Max seconds to wait for each element.

    Returns:
        ``True`` if login succeeded, ``False`` otherwise.
    """
    wait = WebDriverWait(driver, timeout)
    log.info("[login] Starting on %s", driver.current_url)

    # --- 3. Click Log in button ---
    if not _click_login_btn(driver, wait_sec=10.0):
        log.error("[login] 3. Could not click Log in button")
        return False
    human_sleep(1.0, jitter=0.5)

    # --- 4. Use phone / email / username ---
    try:
        el = driver.find_element(*LOGIN_USE_EMAIL)
        human_click(driver, el)
        log.info("[login] 4. Clicked Use phone/email/username")
        human_sleep(0.5, jitter=0.3)
    except Exception as e:
        log.error("[login] 4. Use email option not found: %s", e)
        return False

    # --- 5. Log in with password ---
    try:
        el = driver.find_element(*LOGIN_WITH_PASSWORD_LINK)
        human_click(driver, el)
        log.info("[login] 5. Clicked Log in with password")
        human_sleep(1.0, jitter=0.5)
    except Exception as e:
        log.error("[login] 5. Log in with password link not found: %s", e)
        return False

    # --- 6. Fill form and submit ---
    try:
        username_el = wait.until(EC.presence_of_element_located(LOGIN_USERNAME))
        password_el = driver.find_element(*LOGIN_PASSWORD)
    except Exception as e:
        log.error("[login] 6. Login form not found: %s", e)
        return False

    # Human-like form filling
    human_click(driver, username_el)
    username_el.clear()
    human_type(username_el, email)

    human_click(driver, password_el)
    password_el.clear()
    human_type(password_el, password)
    log.info("[login] 6. Filled form (human-like typing)")

    # Pause before submit — mimics reading the form
    human_sleep(2.0, jitter=1.0)

    try:
        submit_el = wait.until(EC.element_to_be_clickable(LOGIN_SUBMIT))
        submit_el.click()
        log.info("[login] 6. Submitted")
    except Exception:
        from selenium.webdriver.common.keys import Keys  # type: ignore[import-untyped]
        password_el.send_keys(Keys.RETURN)
        log.info("[login] 6. Submitted (Enter)")

    # Wait while TikTok processes login
    human_sleep(5.0, jitter=1.5)

    # Solve captcha if presented
    solve_captcha(driver, max_attempts=3)
    human_sleep(3.0, jitter=1.0)

    # --- Check for rate-limit message ---
    try:
        body_text = driver.find_element(By.TAG_NAME, "body").text or ""
        if (
            "maximum number of attempts" in body_text.lower()
            or "try again later" in body_text.lower()
        ):
            log.warning(
                "[login] TikTok rate limit detected. "
                "Setting 45-min cooldown; no further attempts this run."
            )
            set_rate_limit(minutes=45)
            return False
    except Exception:
        pass

    # --- 7. Verify login success ---
    if LOGGED_IN_INDICATOR is not None:
        try:
            wait.until(EC.presence_of_element_located(LOGGED_IN_INDICATOR))
            log.info("[login] 7. Logged-in indicator found")
        except Exception:
            log.debug("[login] 7. Logged-in indicator not found (continuing)")

    try:
        driver.find_element(By.ID, "loginContainer")
        log.warning("[login] loginContainer still visible — login may have failed")
        return False
    except Exception:
        pass

    if "login" in driver.current_url.lower():
        log.warning("[login] Still on login URL")
        return False

    log.info("[login] Login successful")
    return True


# ---------------------------------------------------------------------------
# Convenience helpers (cookie-based flows)
# ---------------------------------------------------------------------------

def ensure_logged_in(
    driver: WebDriver,
    login_url: str = "https://www.tiktok.com/login",
    for_you_url: str = "https://www.tiktok.com/foryou",
    cookies_path: Optional[Path] = None,
) -> bool:
    """Load cookies if present, go to For You, solve captcha, wait for manual login if needed."""
    driver.get(login_url)
    human_sleep(2.0, jitter=0.5)
    if cookies_path and cookies_path.exists():
        load_cookies(driver, cookies_path)
        driver.refresh()
        human_sleep(2.0, jitter=0.5)
    driver.get(for_you_url)
    human_sleep(3.0, jitter=0.5)
    solve_captcha(driver, max_attempts=3)
    if "login" in driver.current_url.lower():
        log.info("[login] Still on login page; waiting 60s for manual login...")
        time.sleep(60)
        driver.get(for_you_url)
        human_sleep(3.0, jitter=0.5)
        if "login" in driver.current_url.lower():
            return False
    return True


def login_with_cookies(
    cookies_path: Path,
    for_you_url: str = "https://www.tiktok.com/foryou",
    headless: bool = False,
) -> Optional[WebDriver]:
    """Create driver, load cookies, go to For You. Returns driver if successful."""
    driver = create_driver(headless=headless)
    driver.get(for_you_url)
    human_sleep(2.0, jitter=0.5)
    if not load_cookies(driver, cookies_path):
        log.warning("[login] No cookies loaded")
        return None
    driver.refresh()
    human_sleep(3.0, jitter=0.5)
    if "login" in driver.current_url.lower():
        solve_captcha(driver, max_attempts=2)
        human_sleep(2.0, jitter=0.5)
    return driver

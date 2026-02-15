"""
Captcha detection: identify which type of captcha is present on the page.

Types: slider_puzzle, rotation, funcaptcha_* (cycle_match, quantity, dice, etc.)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


class CaptchaType(str, Enum):
    """Known captcha types we can solve."""

    SLIDER_PUZZLE = "slider_puzzle"  # Drag piece into gap
    ROTATION = "rotation"  # Align inner circle
    FUNCAPTCHA_CYCLE = "funcaptcha_cycle"
    FUNCAPTCHA_QUANTITY = "funcaptcha_quantity"
    FUNCAPTCHA_DICE = "funcaptcha_dice"
    FUNCAPTCHA_ROTATION = "funcaptcha_rotation"
    OBJECT_SELECTION = "object_selection"  # TikTok shape/grid (object_selection_captcha)
    UNKNOWN = "unknown"


# TikTok / Arkose selectors
SLIDER_PUZZLE_SELECTORS = [
    "#captcha-verify-image",
    ".captcha_verify_img_slide",
    "[class*='captcha-verify'] img[class*='slide']",
]
ROTATION_SELECTORS = [
    "img[class*='whirl-inner']",
    "img[class*='rotate-inner']",
    "#captcha-verify-image",  # Can be rotation too
]
FUNCAPTCHA_IFRAME = "iframe[src*='arkoselabs']"
OBJECT_SELECTION_SELECTORS = [
    "[class*='secsdk-captcha']",
    "[data-e2e='captcha']",
]


def _find_element(
    driver: WebDriver,
    selectors: list[str],
    container: Optional[WebElement] = None,
    timeout: float = 2.0,
) -> Optional[WebElement]:
    root = container or driver
    for sel in selectors:
        try:
            el = WebDriverWait(root, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, sel))
            )
            if el.is_displayed():
                return el
        except Exception:
            continue
    return None


def _has_slider_track(driver: WebDriver, container: Optional[WebElement] = None) -> bool:
    """Check for slider puzzle: background + piece + track."""
    root = container or driver
    try:
        bg = root.find_element(By.CSS_SELECTOR, "#captcha-verify-image")
        piece = root.find_element(By.CSS_SELECTOR, ".captcha_verify_img_slide")
        track = root.find_element(By.CSS_SELECTOR, "[class*='slidebar']")
        return bg.is_displayed() and piece.is_displayed() and track.is_displayed()
    except Exception:
        pass
    return False


def _has_rotation_slider(driver: WebDriver, container: Optional[WebElement] = None) -> bool:
    """Check for rotation captcha: outer + inner image + rotation slider."""
    root = container or driver
    try:
        outer = root.find_element(By.CSS_SELECTOR, "#captcha-verify-image")
        inner = root.find_element(By.CSS_SELECTOR, "img[class*='whirl-inner'], img[class*='rotate-inner']")
        return outer.is_displayed() and inner.is_displayed()
    except Exception:
        pass
    return False


def detect_captcha(
    driver: WebDriver,
    container: Optional[WebElement] = None,
    timeout: float = 3.0,
) -> tuple[CaptchaType, Optional[WebElement]]:
    """
    Detect which captcha type is present.

    Returns
    -------
    (CaptchaType, container_element)
        container_element is the iframe or div wrapping the captcha; use for solver scope.
    """
    # Check for Arkose iframe first
    try:
        iframe = driver.find_element(By.CSS_SELECTOR, FUNCAPTCHA_IFRAME)
        if iframe.is_displayed():
            driver.switch_to.frame(iframe)
            try:
                if _has_rotation_slider(driver):
                    return CaptchaType.ROTATION, iframe
                if _has_slider_track(driver):
                    return CaptchaType.SLIDER_PUZZLE, iframe
                # Could be funcaptcha variant; check for arrows, plus/minus, grid
                # For now default to UNKNOWN if inside iframe but not matched
                return CaptchaType.UNKNOWN, iframe
            finally:
                driver.switch_to.default_content()
    except Exception:
        pass

    # TikTok native captcha (no iframe)
    try:
        cap_container = _find_element(driver, ["[class*='secsdk-captcha']", "[class*='captcha-verify']"])
        if cap_container and cap_container.is_displayed():
            if _has_rotation_slider(driver, cap_container):
                return CaptchaType.ROTATION, cap_container
            if _has_slider_track(driver, cap_container):
                return CaptchaType.SLIDER_PUZZLE, cap_container
            # Object selection (shape/grid)
            return CaptchaType.OBJECT_SELECTION, cap_container
    except Exception:
        pass

    return CaptchaType.UNKNOWN, None


def is_captcha_visible(driver: WebDriver, timeout: float = 2.0) -> bool:
    """Quick check: is any captcha blocking the page?"""
    captcha_type, _ = detect_captcha(driver, timeout=timeout)
    return captcha_type != CaptchaType.UNKNOWN

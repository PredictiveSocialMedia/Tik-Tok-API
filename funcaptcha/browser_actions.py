"""
Shared browser actions for FunCaptcha solvers.

Provides: slider drag, arrow left/right clicks, plus/minus clicks (quantity),
and tile clicks for grid-based challenges (e.g. dice). All use Selenium
ActionChains and configurable selectors.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains

logger = logging.getLogger(__name__)

# Default selectors (Arkose/FunCaptcha style; override per site)
DEFAULT_SLIDER_HANDLE = "div[class*='slider'] div[class*='handle'], .secsdk-captcha-drag-icon"
DEFAULT_ARROW_LEFT = "button[class*='arrow'][class*='left'], [aria-label='Previous'], .arrow-left"
DEFAULT_ARROW_RIGHT = "button[class*='arrow'][class*='right'], [aria-label='Next'], .arrow-right"
DEFAULT_PLUS = "button[class*='plus'], [aria-label='Increase'], .btn-plus"
DEFAULT_MINUS = "button[class*='minus'], [aria-label='Decrease'], .btn-minus"


def _find_in_scope(driver: WebDriver, selectors: str | list[str], scope: Optional[WebElement] = None, timeout: float = 2.0) -> Optional[WebElement]:
    """Find first visible element matching any selector. *selectors* can be one string or list of alternatives."""
    if isinstance(selectors, str):
        selectors = [selectors]
    root = scope or driver
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


def slider_drag(
    driver: WebDriver,
    handle_element: Optional[WebElement] = None,
    delta_x: int = 0,
    track_length: Optional[int] = None,
    container: Optional[WebElement] = None,
    handle_selector: str = DEFAULT_SLIDER_HANDLE,
    delay: float = 0.2,
) -> bool:
    """
    Drag the slider handle by *delta_x* pixels (positive = right).

    If *handle_element* is None, looks up handle in *container* or driver using *handle_selector*.
    *track_length* is unused here but can be used by callers to compute delta from angle.
    """
    handle = handle_element or _find_in_scope(driver, handle_selector, scope=container)
    if not handle:
        logger.warning("slider_drag: handle not found")
        return False
    try:
        ActionChains(driver).click_and_hold(handle).pause(delay).move_by_offset(int(delta_x), 0).pause(delay).release().perform()
        logger.info("slider_drag: moved by %d px", delta_x)
        time.sleep(delay)
        return True
    except Exception as exc:
        logger.warning("slider_drag failed: %s", exc)
        return False


def click_arrow(
    driver: WebDriver,
    right: bool,
    container: Optional[WebElement] = None,
    left_selector: str = DEFAULT_ARROW_LEFT,
    right_selector: str = DEFAULT_ARROW_RIGHT,
    times: int = 1,
    delay: float = 0.3,
) -> int:
    """
    Click the left or right arrow *times* times. Returns number of successful clicks.
    """
    sel = right_selector if right else left_selector
    clicked = 0
    for _ in range(times):
        btn = _find_in_scope(driver, sel, scope=container)
        if not btn:
            break
        try:
            ActionChains(driver).move_to_element(btn).pause(delay).click().perform()
            clicked += 1
            time.sleep(delay)
        except Exception as exc:
            logger.warning("click_arrow failed: %s", exc)
            break
    return clicked


def click_plus_minus(
    driver: WebDriver,
    plus: bool,
    times: int = 1,
    container: Optional[WebElement] = None,
    plus_selector: str = DEFAULT_PLUS,
    minus_selector: str = DEFAULT_MINUS,
    delay: float = 0.3,
) -> int:
    """Click the + or - button *times* times. Returns number of successful clicks."""
    sel = plus_selector if plus else minus_selector
    clicked = 0
    for _ in range(times):
        btn = _find_in_scope(driver, sel, scope=container)
        if not btn:
            break
        try:
            ActionChains(driver).move_to_element(btn).pause(delay).click().perform()
            clicked += 1
            time.sleep(delay)
        except Exception as exc:
            logger.warning("click_plus_minus failed: %s", exc)
            break
    return clicked


def click_tiles(
    driver: WebDriver,
    tile_elements: list[WebElement],
    indices: list[int],
    delay: float = 0.3,
) -> int:
    """Click tiles at 0-based *indices*. Returns number of successful clicks."""
    clicked = 0
    for idx in indices:
        if 0 <= idx < len(tile_elements):
            try:
                ActionChains(driver).move_to_element(tile_elements[idx]).pause(delay).click().perform()
                clicked += 1
                time.sleep(delay)
            except Exception as exc:
                logger.warning("click_tiles failed for index %d: %s", idx, exc)
    return clicked

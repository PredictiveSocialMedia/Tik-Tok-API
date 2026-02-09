"""
Human-like slider drag for puzzle captchas.

Uses variable step size, delays, slight overshoot and correction so the
movement passes bot detection (e.g. TikTok).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains

logger = logging.getLogger(__name__)

# TikTok-style handle selector (same as funcaptcha.browser_actions)
DEFAULT_SLIDER_HANDLE = "div[class*='slider'] div[class*='handle'], .secsdk-captcha-drag-icon"


def _find_handle(
    driver: WebDriver,
    container: Optional[WebElement] = None,
    handle_selector: str = DEFAULT_SLIDER_HANDLE,
    timeout: float = 2.0,
) -> Optional[WebElement]:
    root = container or driver
    try:
        el = WebDriverWait(root, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, handle_selector))
        )
        return el if el.is_displayed() else None
    except Exception:
        return None


def slider_drag_humanized(
    driver: WebDriver,
    delta_x: int,
    container: Optional[WebElement] = None,
    handle_element: Optional[WebElement] = None,
    handle_selector: str = DEFAULT_SLIDER_HANDLE,
    *,
    step_size: int = 6,
    step_delay_ms: int = 10,
    overshoot_px: int = 5,
    overshoot_delay_ms: int = 250,
    randomize: bool = True,
) -> bool:
    """
    Drag the slider handle by delta_x pixels with human-like motion.

    Uses small steps, variable delay, a slight overshoot then correction,
    and a smooth settle so the movement is not detected as robotic.

    Parameters
    ----------
    driver : WebDriver
        Selenium WebDriver.
    delta_x : int
        Total horizontal distance to move (pixels). Positive = right.
    container : WebElement, optional
        Scope to find the handle if handle_element is None.
    handle_element : WebElement, optional
        Slider handle element. If None, looked up via handle_selector.
    handle_selector : str
        CSS selector for the handle when handle_element is None.
    step_size : int
        Pixels per step during main drag (slightly randomized if randomize).
    step_delay_ms : int
        Delay between steps (ms); slightly randomized if randomize.
    overshoot_px : int
        Extra pixels to move past target, then correct back.
    overshoot_delay_ms : int
        Pause after overshoot before correcting.
    randomize : bool
        Add small randomness to steps and delays.

    Returns
    -------
    bool
        True if the drag was performed, False if handle not found or error.
    """
    handle = handle_element or _find_handle(driver, container, handle_selector)
    if not handle:
        logger.warning("slider_drag_humanized: handle not found")
        return False

    dx = int(delta_x)
    if dx == 0:
        return True

    sign = 1 if dx > 0 else -1
    dx = abs(dx)

    try:
        chain = ActionChains(driver)
        chain.click_and_hold(handle)

        # Initial random pause (human hesitation)
        if randomize:
            time.sleep(random.uniform(0.05, 0.15))
        chain.pause(0.1)
        chain.perform()

        # Progressive drag: many small steps
        steps = dx // step_size
        remainder = dx % step_size
        if remainder:
            steps += 1

        for i in range(steps):
            step = step_size if i < steps - 1 else (step_size if remainder == 0 else remainder)
            if randomize:
                step = max(1, step + random.randint(-1, 1))
            chain = ActionChains(driver)
            chain.move_by_offset(sign * step, 0)
            chain.perform()
            delay = step_delay_ms / 1000.0
            if randomize:
                delay *= random.uniform(0.8, 1.2)
            time.sleep(max(0.001, delay))

        # Brief pause
        time.sleep(overshoot_delay_ms / 1000.0)

        # Overshoot then correct
        chain = ActionChains(driver)
        chain.move_by_offset(sign * overshoot_px, 0)
        chain.perform()
        time.sleep(0.05)
        for _ in range(3):
            chain = ActionChains(driver)
            chain.move_by_offset(-sign * 2, random.randint(-1, 1))
            chain.perform()
            time.sleep(0.02)
        chain = ActionChains(driver)
        chain.move_by_offset(-sign * (overshoot_px - 2), 0)
        chain.perform()
        time.sleep(0.2)

        # Short pause then release (human-like finish)
        time.sleep(0.25)
        chain = ActionChains(driver)
        chain.release()
        chain.perform()
        time.sleep(0.15)

        logger.info("slider_drag_humanized: moved %d px", delta_x)
        return True
    except Exception as exc:
        logger.warning("slider_drag_humanized failed: %s", exc)
        try:
            ActionChains(driver).release().perform()
        except Exception:
            pass
        return False

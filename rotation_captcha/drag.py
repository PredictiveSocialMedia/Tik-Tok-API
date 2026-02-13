"""
Human-like slider drag for rotation captchas.

Re-uses the same variable-step, overshoot+correct pattern as the slider
puzzle, but with defaults tuned for rotation (typically a longer track,
smaller overshoot, and a slower drag since the visual feedback is a spinning
circle rather than a sliding piece).
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

# TikTok rotation captcha slider handle
DEFAULT_SLIDER_HANDLE = (
    "div.secsdk-captcha-drag-icon, "
    "div[class*='slider'] div[class*='handle']"
)


def _find_handle(
    driver: WebDriver,
    container: Optional[WebElement] = None,
    handle_selector: str = DEFAULT_SLIDER_HANDLE,
    timeout: float = 3.0,
) -> Optional[WebElement]:
    root = container or driver
    try:
        el = WebDriverWait(root, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, handle_selector))
        )
        return el if el.is_displayed() else None
    except Exception:
        return None


def rotation_drag_humanized(
    driver: WebDriver,
    delta_x: int,
    container: Optional[WebElement] = None,
    handle_element: Optional[WebElement] = None,
    handle_selector: str = DEFAULT_SLIDER_HANDLE,
    *,
    step_size: int = 4,
    step_delay_ms: int = 12,
    overshoot_px: int = 3,
    overshoot_delay_ms: int = 200,
    randomize: bool = True,
) -> bool:
    """
    Drag the rotation slider handle by *delta_x* pixels with human-like motion.

    Differences from the slider-puzzle drag:
    - Smaller default step_size (4 px) for finer angular control.
    - Smaller default overshoot (3 px).
    - Includes tiny vertical jitter during the main drag.
    - Slightly longer step delay to mimic careful visual alignment.

    Parameters
    ----------
    driver : WebDriver
    delta_x : int
        Horizontal pixels to move (positive = right = clockwise rotation).
    container : WebElement, optional
        Scope to find the handle.
    handle_element : WebElement, optional
        Handle element (if already found).
    handle_selector : str
        CSS selector for the handle.
    step_size / step_delay_ms / overshoot_px / overshoot_delay_ms / randomize
        See slider_puzzle.drag for semantics.

    Returns
    -------
    bool  True if drag completed; False on error.
    """
    handle = handle_element or _find_handle(driver, container, handle_selector)
    if not handle:
        logger.warning("rotation_drag_humanized: handle not found")
        return False

    dx = int(delta_x)
    if dx == 0:
        return True

    sign = 1 if dx > 0 else -1
    dx = abs(dx)

    try:
        chain = ActionChains(driver)
        chain.click_and_hold(handle)

        # Initial human hesitation
        if randomize:
            time.sleep(random.uniform(0.06, 0.18))
        chain.pause(0.08)
        chain.perform()

        # Main drag: many small steps with tiny vertical jitter
        steps = dx // step_size
        remainder = dx % step_size
        if remainder:
            steps += 1

        for i in range(steps):
            step = step_size if i < steps - 1 else (step_size if remainder == 0 else remainder)
            if randomize:
                step = max(1, step + random.randint(-1, 1))
            jitter_y = random.randint(-1, 1) if randomize else 0
            chain = ActionChains(driver)
            chain.move_by_offset(sign * step, jitter_y)
            chain.perform()
            delay = step_delay_ms / 1000.0
            if randomize:
                delay *= random.uniform(0.7, 1.3)
            time.sleep(max(0.002, delay))

        # Short pause (human visually checking alignment)
        time.sleep(random.uniform(0.15, 0.35) if randomize else 0.25)

        # Overshoot then correct
        chain = ActionChains(driver)
        chain.move_by_offset(sign * overshoot_px, 0)
        chain.perform()
        time.sleep(overshoot_delay_ms / 1000.0)

        # Correct back in small steps
        corrected = 0
        while corrected < overshoot_px:
            step = min(2, overshoot_px - corrected)
            chain = ActionChains(driver)
            chain.move_by_offset(-sign * step, random.randint(-1, 0) if randomize else 0)
            chain.perform()
            corrected += step
            time.sleep(random.uniform(0.02, 0.04) if randomize else 0.03)

        # Final settle pause then release
        time.sleep(random.uniform(0.2, 0.4) if randomize else 0.3)
        chain = ActionChains(driver)
        chain.release()
        chain.perform()
        time.sleep(0.12)

        logger.info("rotation_drag_humanized: moved %d px", delta_x)
        return True

    except Exception as exc:
        logger.warning("rotation_drag_humanized failed: %s", exc)
        try:
            ActionChains(driver).release().perform()
        except Exception:
            pass
        return False

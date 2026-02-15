"""
Captcha router: dispatch to the appropriate solver based on detected type.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from .detector import CaptchaType, detect_captcha

logger = logging.getLogger(__name__)


def solve_captcha(
    driver: WebDriver,
    container: Optional[WebElement] = None,
    max_attempts: int = 3,
) -> bool:
    """
    Detect captcha type and invoke the appropriate solver.

    Returns
    -------
    bool
        True if captcha was solved (or not present), False if failed after max_attempts.
    """
    for attempt in range(max_attempts):
        captcha_type, cap_container = detect_captcha(driver, container)
        if captcha_type == CaptchaType.UNKNOWN:
            return True  # No captcha

        logger.info("Captcha detected: %s (attempt %d/%d)", captcha_type, attempt + 1, max_attempts)

        solved = False
        in_iframe = cap_container and cap_container.tag_name.lower() == "iframe"
        try:
            if in_iframe:
                driver.switch_to.frame(cap_container)
                scope = None
            else:
                scope = cap_container

            try:
                if captcha_type == CaptchaType.SLIDER_PUZZLE:
                    from slider_puzzle import solve_slider_puzzle
                    solved = solve_slider_puzzle(driver, container=scope, fudge_px=-6)
                elif captcha_type == CaptchaType.ROTATION:
                    from rotation_captcha import solve_rotation_captcha
                    solved = solve_rotation_captcha(driver, container=scope)
                elif captcha_type == CaptchaType.OBJECT_SELECTION:
                    try:
                        from object_selection_captcha import handle_captcha
                        solved = handle_captcha(driver, max_attempts=1)
                    except ImportError:
                        logger.warning("object_selection_captcha not installed; skipping")
                        solved = False
                elif captcha_type in (
                    CaptchaType.FUNCAPTCHA_CYCLE,
                    CaptchaType.FUNCAPTCHA_QUANTITY,
                    CaptchaType.FUNCAPTCHA_DICE,
                    CaptchaType.FUNCAPTCHA_ROTATION,
                ):
                    if captcha_type == CaptchaType.FUNCAPTCHA_ROTATION:
                        from rotation_captcha import solve_rotation_captcha
                        solved = solve_rotation_captcha(driver, container=scope)
                    else:
                        logger.warning("Funcaptcha %s not yet wired; skipping", captcha_type)
                        solved = False
                else:
                    logger.warning("No solver for captcha type: %s", captcha_type)
            finally:
                if in_iframe:
                    driver.switch_to.default_content()
        except Exception as exc:
            logger.warning("Solver failed: %s", exc)
            solved = False

        if solved:
            time.sleep(1.5)  # Let page settle
            # Re-check: sometimes "solved" still shows challenge
            captcha_type_after, _ = detect_captcha(driver, container)
            if captcha_type_after == CaptchaType.UNKNOWN:
                logger.info("Captcha cleared")
                return True
        else:
            time.sleep(2.0)  # Brief pause before retry

    logger.warning("Captcha not solved after %d attempts", max_attempts)
    return False

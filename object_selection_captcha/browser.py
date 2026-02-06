"""
Selenium helpers for CAPTCHA detection and interaction.

Handles:
- Detecting whether a CAPTCHA is currently blocking the page
- Switching into CAPTCHA iframes
- Screenshotting the CAPTCHA image element
- Clicking tiles / points inside the CAPTCHA
- Pressing verify / submit buttons
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CaptchaInfo:
    """Describes a detected CAPTCHA on the page."""

    captcha_type: str  # "grid", "click", "slider", "rotate", "shape", "unknown"
    prompt: str  # e.g. "Select all images with traffic lights"
    grid_rows: int = 0
    grid_cols: int = 0
    image_element: Optional[WebElement] = None
    container_element: Optional[WebElement] = None
    iframe: Optional[WebElement] = None
    tile_elements: list[WebElement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CAPTCHA detection
# ---------------------------------------------------------------------------

# Selectors for common CAPTCHA providers (reCAPTCHA, hCaptcha, TikTok custom)
_CAPTCHA_IFRAME_SELECTORS = [
    "iframe[src*='recaptcha']",
    "iframe[src*='hcaptcha']",
    "iframe[title*='challenge']",
    "iframe[title*='captcha']",
]

_CAPTCHA_CONTAINER_SELECTORS = [
    # TikTok's own CAPTCHA overlay
    "div[class*='captcha']",
    "div[id*='captcha']",
    "div[class*='Captcha']",
    "div[id*='Captcha']",
    # reCAPTCHA
    "div.rc-imageselect",
    "div#rc-imageselect",
    # hCaptcha
    "div.challenge-container",
    # Generic
    "div[class*='verify']",
    "div[class*='Verify']",
]


def detect_captcha(driver: webdriver.Chrome, timeout: float = 3.0) -> Optional[CaptchaInfo]:
    """
    Scan the current page for a visible CAPTCHA.

    Returns a ``CaptchaInfo`` if one is found, otherwise ``None``.
    This does **not** switch into iframes; the caller should use
    ``switch_to_captcha_frame`` before interacting.
    """
    # ── Check for CAPTCHA iframes ────────────────────────────────────
    for sel in _CAPTCHA_IFRAME_SELECTORS:
        try:
            iframes = driver.find_elements(By.CSS_SELECTOR, sel)
            for iframe in iframes:
                if iframe.is_displayed():
                    logger.info("CAPTCHA iframe detected: %s", sel)
                    return CaptchaInfo(
                        captcha_type="unknown",
                        prompt="",
                        iframe=iframe,
                    )
        except Exception:
            continue

    # ── Check for in-page CAPTCHA containers ─────────────────────────
    for sel in _CAPTCHA_CONTAINER_SELECTORS:
        try:
            containers = driver.find_elements(By.CSS_SELECTOR, sel)
            for container in containers:
                if container.is_displayed() and container.size.get("height", 0) > 50:
                    logger.info("CAPTCHA container detected: %s", sel)
                    info = _parse_captcha_container(driver, container)
                    return info
        except Exception:
            continue

    return None


def _parse_captcha_container(
    driver: webdriver.Chrome, container: WebElement
) -> CaptchaInfo:
    """Inspect a CAPTCHA container to determine type, prompt, and grid size."""
    info = CaptchaInfo(
        captcha_type="unknown",
        prompt="",
        container_element=container,
    )

    # ── Extract prompt text ──────────────────────────────────────────
    prompt_selectors = [
        # TikTok-specific: the shape CAPTCHA has a text prompt above the image
        "div[class*='captcha'] div[class*='text']",
        "div[class*='captcha'] span[class*='text']",
        "div[class*='Captcha'] div[class*='text']",
        # Generic prompt selectors
        "div[class*='prompt']",
        "div[class*='Prompt']",
        "span[class*='prompt']",
        "div.rc-imageselect-desc",
        "div.rc-imageselect-desc-no-canonical",
        "h2.prompt-text",
        "div[class*='instruction']",
    ]
    for sel in prompt_selectors:
        try:
            el = container.find_elements(By.CSS_SELECTOR, sel)
            if el and el[0].text.strip():
                info.prompt = el[0].text.strip()
                break
        except Exception:
            continue

    if not info.prompt:
        # Fall back: scan all text nodes in the container for a known pattern
        try:
            full_text = container.text or ""
            for line in full_text.split("\n"):
                line = line.strip()
                if _is_captcha_prompt(line):
                    info.prompt = line
                    break
            if not info.prompt:
                info.prompt = full_text.split("\n")[0].strip()
        except Exception:
            pass

    # ── Detect TikTok shape-matching CAPTCHA early ───────────────
    if _is_shape_prompt(info.prompt):
        info.captcha_type = "shape"
        # Find the challenge image
        _find_captcha_image(container, info)
        logger.info("Detected TikTok shape-matching CAPTCHA: %r", info.prompt)
        return info

    # ── Detect grid tiles ────────────────────────────────────────────
    tile_selectors = [
        "td.rc-imageselect-tile",          # reCAPTCHA
        "div.task-image",                    # hCaptcha
        "div[class*='captcha_verify_img']",  # TikTok grid
        "img[class*='captcha']",             # generic
    ]
    for sel in tile_selectors:
        try:
            tiles = container.find_elements(By.CSS_SELECTOR, sel)
            if tiles:
                info.tile_elements = tiles
                count = len(tiles)
                # Guess grid dimensions
                if count == 9:
                    info.grid_rows, info.grid_cols = 3, 3
                elif count == 16:
                    info.grid_rows, info.grid_cols = 4, 4
                elif count == 6:
                    info.grid_rows, info.grid_cols = 2, 3
                else:
                    # Approximate square
                    side = int(count ** 0.5)
                    info.grid_rows = side
                    info.grid_cols = (count + side - 1) // side
                info.captcha_type = "grid"
                break
        except Exception:
            continue

    # ── Detect single-image click CAPTCHA ────────────────────────────
    if info.captcha_type == "unknown":
        img_selectors = [
            "img[class*='captcha']",
            "img[class*='Captcha']",
            "canvas[class*='captcha']",
            "div[class*='captcha'] img",
        ]
        for sel in img_selectors:
            try:
                imgs = container.find_elements(By.CSS_SELECTOR, sel)
                for img in imgs:
                    if img.is_displayed() and img.size.get("height", 0) > 80:
                        info.image_element = img
                        info.captcha_type = "click"
                        break
            except Exception:
                continue
            if info.image_element:
                break

    # ── Detect slider ────────────────────────────────────────────────
    if info.captcha_type == "unknown":
        slider_sels = [
            "div[class*='slider']",
            "div[class*='Slider']",
            "div[class*='secsdk-captcha-drag']",
        ]
        for sel in slider_sels:
            try:
                sliders = container.find_elements(By.CSS_SELECTOR, sel)
                if sliders and sliders[0].is_displayed():
                    info.captcha_type = "slider"
                    break
            except Exception:
                continue

    logger.info(
        "Parsed CAPTCHA: type=%s, prompt=%r, grid=%dx%d, tiles=%d",
        info.captcha_type, info.prompt, info.grid_rows, info.grid_cols,
        len(info.tile_elements),
    )
    return info


# ---------------------------------------------------------------------------
# Frame switching
# ---------------------------------------------------------------------------


def switch_to_captcha_frame(driver: webdriver.Chrome, info: CaptchaInfo) -> bool:
    """Switch into the CAPTCHA iframe if applicable. Returns True on success."""
    if info.iframe:
        try:
            driver.switch_to.frame(info.iframe)
            return True
        except Exception as exc:
            logger.warning("Could not switch to CAPTCHA iframe: %s", exc)
            return False
    return True  # no iframe to switch to


def switch_to_default_content(driver: webdriver.Chrome) -> None:
    """Return to the top-level document after interacting with a CAPTCHA iframe."""
    try:
        driver.switch_to.default_content()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Screenshot helpers
# ---------------------------------------------------------------------------


def screenshot_element(element: WebElement) -> Image.Image:
    """Take a PNG screenshot of a specific element and return a PIL Image."""
    png_bytes = element.screenshot_as_png
    return Image.open(BytesIO(png_bytes)).convert("RGB")


def screenshot_captcha_image(
    driver: webdriver.Chrome, info: CaptchaInfo
) -> Optional[Image.Image]:
    """
    Get a screenshot of the CAPTCHA image area.

    For grid CAPTCHAs this captures the tile grid; for click CAPTCHAs the
    single challenge image.
    """
    target = info.image_element or info.container_element
    if not target:
        # Try to find the main image in the container
        if info.tile_elements:
            target = info.container_element or info.tile_elements[0]
    if not target:
        logger.warning("No element to screenshot for CAPTCHA")
        return None
    try:
        return screenshot_element(target)
    except Exception as exc:
        logger.warning("Failed to screenshot CAPTCHA element: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Click actions
# ---------------------------------------------------------------------------


def click_tiles_by_index(
    driver: webdriver.Chrome,
    info: CaptchaInfo,
    tile_indices: list[int],
    delay: float = 0.3,
) -> None:
    """Click the tile WebElements at the given 0-based indices."""
    for idx in tile_indices:
        if 0 <= idx < len(info.tile_elements):
            try:
                tile = info.tile_elements[idx]
                ActionChains(driver).move_to_element(tile).pause(delay).click().perform()
                logger.info("Clicked tile %d", idx)
                time.sleep(delay)
            except Exception as exc:
                logger.warning("Failed to click tile %d: %s", idx, exc)


def click_points_on_element(
    driver: webdriver.Chrome,
    element: WebElement,
    points: list[tuple[float, float]],
    image_size: tuple[int, int],
    delay: float = 0.4,
) -> None:
    """
    Click at pixel coordinates *points* on *element*.

    *image_size* is (width, height) of the original image that the detection
    coordinates are relative to.  The function maps them to the element's
    rendered size.
    """
    elem_size = element.size
    elem_w = elem_size.get("width", image_size[0])
    elem_h = elem_size.get("height", image_size[1])
    scale_x = elem_w / image_size[0]
    scale_y = elem_h / image_size[1]

    for px, py in points:
        # Offset from the element's top-left corner
        offset_x = int(px * scale_x) - elem_w // 2
        offset_y = int(py * scale_y) - elem_h // 2
        try:
            ActionChains(driver).move_to_element_with_offset(
                element, offset_x, offset_y
            ).pause(delay).click().perform()
            logger.info("Clicked point (%.0f, %.0f) → offset (%d, %d)", px, py, offset_x, offset_y)
            time.sleep(delay)
        except Exception as exc:
            logger.warning("Failed to click at (%.0f, %.0f): %s", px, py, exc)


# ---------------------------------------------------------------------------
# Submit / verify button
# ---------------------------------------------------------------------------


def click_verify_button(driver: webdriver.Chrome, info: CaptchaInfo) -> bool:
    """Attempt to click the CAPTCHA verify / submit button."""
    verify_selectors = [
        "button[id*='verify']",
        "button[class*='verify']",
        "button[class*='Verify']",
        "button[id*='submit']",
        "div[id='recaptcha-verify-button']",
        "div.button-submit",
        "div[class*='submit']",
        "div[class*='Submit']",
    ]
    verify_labels = [
        "Verify",
        "VERIFY",
        "Submit",
        "SUBMIT",
        "Next",
        "Skip",
        "Confirm",
    ]

    container = info.container_element

    for sel in verify_selectors:
        try:
            scope = container if container else driver
            btns = scope.find_elements(By.CSS_SELECTOR, sel)
            for btn in btns:
                if btn.is_displayed():
                    btn.click()
                    logger.info("Clicked verify button via selector: %s", sel)
                    return True
        except Exception:
            continue

    for label in verify_labels:
        try:
            btn = driver.find_element(
                By.XPATH, f"//button[normalize-space()='{label}']"
            )
            if btn.is_displayed():
                btn.click()
                logger.info("Clicked verify button via label: %s", label)
                return True
        except Exception:
            continue

    return False


# ---------------------------------------------------------------------------
# Post-solve check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TikTok shape-CAPTCHA helpers
# ---------------------------------------------------------------------------

# Phrases that indicate a shape-matching CAPTCHA
_SHAPE_PROMPT_PATTERNS = [
    "same shape",
    "same type",
    "matching shape",
    "similar shape",
    "identical shape",
    "Select 2 objects",
    "select two objects",
]


def _is_shape_prompt(prompt: str) -> bool:
    """Return True if the prompt text looks like a shape-matching CAPTCHA."""
    lower = prompt.lower()
    return any(pat.lower() in lower for pat in _SHAPE_PROMPT_PATTERNS)


def _is_captcha_prompt(text: str) -> bool:
    """Return True if a line of text looks like a CAPTCHA instruction."""
    lower = text.lower()
    keywords = ["select", "click", "choose", "pick", "identify", "verify", "same shape"]
    return any(kw in lower for kw in keywords) and len(text) > 10


def _find_captcha_image(container: WebElement, info: CaptchaInfo) -> None:
    """Locate the challenge image element inside a CAPTCHA container."""
    img_selectors = [
        "img[class*='captcha']",
        "img[class*='Captcha']",
        "img[id*='captcha']",
        "div[class*='captcha'] img",
        "div[class*='Captcha'] img",
        "canvas",
        "img",
    ]
    for sel in img_selectors:
        try:
            imgs = container.find_elements(By.CSS_SELECTOR, sel)
            for img in imgs:
                if img.is_displayed() and img.size.get("height", 0) > 80:
                    info.image_element = img
                    return
        except Exception:
            continue


def click_marker_points(
    driver: webdriver.Chrome,
    element: WebElement,
    points: list[tuple[int, int]],
    image_size: tuple[int, int],
    delay: float = 0.4,
) -> None:
    """
    Click at specific marker positions on the CAPTCHA image element.

    Like ``click_points_on_element`` but uses int pixel coords and
    human-like random jitter.
    """
    import random

    elem_size = element.size
    elem_w = elem_size.get("width", image_size[0])
    elem_h = elem_size.get("height", image_size[1])
    scale_x = elem_w / image_size[0]
    scale_y = elem_h / image_size[1]

    for px, py in points:
        # Small jitter to appear human
        jx = random.randint(-3, 3)
        jy = random.randint(-3, 3)
        offset_x = int(px * scale_x + jx) - elem_w // 2
        offset_y = int(py * scale_y + jy) - elem_h // 2
        try:
            ActionChains(driver).move_to_element_with_offset(
                element, offset_x, offset_y
            ).pause(delay + random.uniform(0.05, 0.2)).click().perform()
            logger.info(
                "Clicked marker at (%d, %d) → offset (%d, %d)", px, py, offset_x, offset_y
            )
            time.sleep(delay)
        except Exception as exc:
            logger.warning("Failed to click marker at (%d, %d): %s", px, py, exc)


# ---------------------------------------------------------------------------
# Post-solve check
# ---------------------------------------------------------------------------


def is_captcha_gone(driver: webdriver.Chrome, wait: float = 3.0) -> bool:
    """
    Return True if no CAPTCHA appears visible after waiting *wait* seconds.
    Used to confirm the solve was accepted.
    """
    time.sleep(wait)
    return detect_captcha(driver, timeout=1.0) is None

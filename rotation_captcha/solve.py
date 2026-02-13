"""
End-to-end rotation captcha solver.

Flow:
1. Extract the outer (background) and inner (fragment) images from the page.
2. Estimate the rotation angle needed to align inner with outer.
3. Convert angle → slider pixel delta using the measured track length.
4. Execute a human-like slider drag.

TikTok-specific CSS selectors are the default; override for other sites.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Callable, Optional, Tuple
from urllib.request import urlopen

from PIL import Image
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .angle import estimate_angle, angle_to_slider_delta
from .drag import rotation_drag_humanized

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default TikTok DOM selectors for the rotation captcha
# ---------------------------------------------------------------------------
DEFAULT_OUTER_SELECTOR = "#captcha-verify-image"
DEFAULT_INNER_SELECTOR = (
    "img.captcha_verify_img_slide, "
    "img[class*='whirl-inner'], "
    "img[class*='rotate-inner']"
)
DEFAULT_SLIDER_TRACK_SELECTOR = (
    "div.captcha_verify_slide--slidebar, "
    "div[class*='slider-track'], "
    "div[class*='secsdk-captcha-drag']"
)
DEFAULT_HANDLE_SELECTOR = (
    "div.secsdk-captcha-drag-icon, "
    "div[class*='slider'] div[class*='handle']"
)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image_from_src(src: str) -> Optional[Image.Image]:
    """Fetch an image from a URL and return as PIL RGB Image."""
    try:
        with urlopen(src, timeout=10) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to load image from %s: %s", src[:80], e)
        return None


def _screenshot_element(el: WebElement) -> Optional[Image.Image]:
    """Take a screenshot of an element and return a PIL Image."""
    try:
        png = el.screenshot_as_png
        return Image.open(BytesIO(png)).convert("RGB")
    except Exception as e:
        logger.warning("Screenshot failed: %s", e)
        return None


def _image_from_element(el: WebElement) -> Optional[Image.Image]:
    """Try src URL first; fall back to element screenshot."""
    try:
        src = el.get_attribute("src")
        if src:
            img = _load_image_from_src(src)
            if img:
                return img
    except Exception:
        pass
    return _screenshot_element(el)


# ---------------------------------------------------------------------------
# Locate elements and extract images
# ---------------------------------------------------------------------------

def get_track_length(
    driver: WebDriver,
    container: Optional[WebElement] = None,
    track_selector: str = DEFAULT_SLIDER_TRACK_SELECTOR,
    handle_selector: str = DEFAULT_HANDLE_SELECTOR,
) -> int:
    """
    Measure the effective slider track length in pixels.

    track_length = track_element.width - handle_element.width
    Falls back to 260 px (common TikTok default) if elements aren't found.
    """
    root = container or driver
    track_width = 0
    handle_width = 0
    try:
        track_el = root.find_element(By.CSS_SELECTOR, track_selector)
        track_width = track_el.size.get("width", 0)
    except Exception:
        logger.debug("Track element not found: %s", track_selector)

    try:
        handle_el = root.find_element(By.CSS_SELECTOR, handle_selector)
        handle_width = handle_el.size.get("width", 0)
    except Exception:
        logger.debug("Handle element not found: %s", handle_selector)

    if track_width > 0:
        effective = track_width - handle_width
        if effective > 0:
            logger.debug("Track length: %d (track=%d, handle=%d)", effective, track_width, handle_width)
            return effective

    # Fallback
    logger.debug("Using fallback track length 260 px")
    return 260


def get_captcha_images(
    driver: WebDriver,
    container: Optional[WebElement] = None,
    outer_selector: str = DEFAULT_OUTER_SELECTOR,
    inner_selector: str = DEFAULT_INNER_SELECTOR,
    timeout: float = 5.0,
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Extract the outer (background) and inner (fragment) images.

    Returns
    -------
    (outer_img, inner_img) : tuple of PIL Images or (None, None).
    """
    root = container or driver

    outer_el = None
    try:
        outer_el = WebDriverWait(root, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, outer_selector))
        )
    except Exception:
        logger.warning("Outer image element not found: %s", outer_selector)
        return None, None

    inner_el = None
    try:
        inner_el = root.find_element(By.CSS_SELECTOR, inner_selector)
    except Exception:
        logger.warning("Inner image element not found: %s", inner_selector)
        return None, None

    outer_img = _image_from_element(outer_el)
    inner_img = _image_from_element(inner_el)
    return outer_img, inner_img


# ---------------------------------------------------------------------------
# High-level solver
# ---------------------------------------------------------------------------

def solve_rotation_captcha(
    driver: WebDriver,
    container: Optional[WebElement] = None,
    *,
    method: str = "overlay",
    angle_step: int = 3,
    fudge_px: int = 0,
    inner_radius_ratio: float = 0.4,
    outer_selector: str = DEFAULT_OUTER_SELECTOR,
    inner_selector: str = DEFAULT_INNER_SELECTOR,
    track_selector: str = DEFAULT_SLIDER_TRACK_SELECTOR,
    handle_selector: str = DEFAULT_HANDLE_SELECTOR,
    get_images: Optional[
        Callable[
            [WebDriver, Optional[WebElement]],
            Tuple[Optional[Image.Image], Optional[Image.Image]],
        ]
    ] = None,
) -> bool:
    """
    Solve a rotation captcha end-to-end.

    Steps
    -----
    1. Extract outer/background and inner/fragment images from the page.
    2. Estimate the rotation angle (degrees) to align inner → outer.
    3. Measure the slider track length and convert angle → pixel delta.
    4. Humanized-drag the slider by that many pixels.

    Parameters
    ----------
    driver : WebDriver
        Selenium WebDriver.
    container : WebElement, optional
        CAPTCHA container to scope the element search.
    method : str
        Angle estimation method: "overlay" or "ring".
    angle_step : int
        Degrees between angle samples (smaller = more precise, slower).
    fudge_px : int
        Extra pixels to add/subtract from the computed delta.
    inner_radius_ratio : float
        For the overlay method: radius of the inner circle as a fraction of
        half the image size. Default 0.4 (typical for TikTok).
    outer_selector / inner_selector / track_selector / handle_selector : str
        CSS selectors for the outer image, inner image, slider track, and
        slider handle.
    get_images : callable, optional
        If provided, called as get_images(driver, container) and must return
        (outer_pil, inner_pil). Overrides selectors.

    Returns
    -------
    bool
        True if the drag completed; False on error.
    """
    # Step 1: get images
    if get_images:
        outer_img, inner_img = get_images(driver, container)
    else:
        outer_img, inner_img = get_captcha_images(
            driver, container,
            outer_selector=outer_selector,
            inner_selector=inner_selector,
        )

    if not outer_img or not inner_img:
        logger.warning("solve_rotation_captcha: could not obtain captcha images")
        return False

    # Step 2: estimate angle
    angle = estimate_angle(
        inner_image=inner_img,
        outer_image=outer_img,
        method=method,
        angle_step=angle_step,
        inner_radius_ratio=inner_radius_ratio,
    )
    logger.info("solve_rotation_captcha: estimated angle = %.1f°", angle)

    # Step 3: measure track → convert angle to pixel delta
    track_px = get_track_length(
        driver, container,
        track_selector=track_selector,
        handle_selector=handle_selector,
    )
    delta_x = angle_to_slider_delta(angle, track_px) + fudge_px
    if delta_x < 0:
        delta_x = 0
    logger.info(
        "solve_rotation_captcha: track=%d px, delta=%d px (fudge=%d)",
        track_px, delta_x, fudge_px,
    )

    # Step 4: drag
    return rotation_drag_humanized(
        driver, delta_x,
        container=container,
        handle_selector=handle_selector,
    )

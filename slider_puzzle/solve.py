"""
End-to-end puzzle slider solve: extract images from page, compute offset, humanized drag.

TikTok-style selectors are used by default; pass custom selectors or elements
for other sites.
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

from .position import get_slide_offset_as_proportion
from .drag import slider_drag_humanized

logger = logging.getLogger(__name__)

# TikTok DOM (ScrapeCreators / common variants)
DEFAULT_BG_SELECTOR = "#captcha-verify-image"
DEFAULT_PIECE_SELECTOR = ".captcha_verify_img_slide"
DEFAULT_WRAPPER_SELECTOR = ".captcha_verify_img--wrapper"


def _load_image_from_src(src: str) -> Optional[Image.Image]:
    """Fetch image from URL and return PIL Image (RGB)."""
    try:
        with urlopen(src, timeout=10) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to load image from %s: %s", src[:80], e)
        return None


def _image_from_element(el: WebElement) -> Optional[Image.Image]:
    """Screenshot element or read from src if it's an img."""
    try:
        png = el.screenshot_as_png
        return Image.open(BytesIO(png)).convert("RGB")
    except Exception as e:
        logger.warning("Screenshot failed: %s", e)
        return None


def get_captcha_images(
    driver: WebDriver,
    container: Optional[WebElement] = None,
    bg_selector: str = DEFAULT_BG_SELECTOR,
    piece_selector: str = DEFAULT_PIECE_SELECTOR,
    wrapper_selector: str = DEFAULT_WRAPPER_SELECTOR,
    timeout: float = 5.0,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[float]]:
    """
    Extract background image, piece image, and puzzle width from the page.

    Returns
    -------
    (background, piece, puzzle_width) : tuple
        background, piece are PIL Images or None if not found.
        puzzle_width is the width in px of the puzzle area (for scaling), or None.
    """
    root = container or driver
    bg_el = piece_el = wrapper_el = None
    try:
        bg_el = WebDriverWait(root, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, bg_selector))
        )
        if not bg_el.is_displayed():
            return None, None, None
    except Exception:
        logger.warning("Background element not found: %s", bg_selector)
        return None, None, None

    try:
        piece_el = root.find_element(By.CSS_SELECTOR, piece_selector)
    except Exception:
        logger.warning("Piece element not found: %s", piece_selector)

    try:
        wrapper_el = root.find_element(By.CSS_SELECTOR, wrapper_selector)
    except Exception:
        pass

    puzzle_width = None
    if wrapper_el and wrapper_el.is_displayed():
        try:
            box = wrapper_el.size
            if box:
                puzzle_width = float(box.get("width", 0))
        except Exception:
            pass
    if puzzle_width is None and bg_el:
        try:
            box = bg_el.size
            if box:
                puzzle_width = float(box.get("width", 340))
        except Exception:
            puzzle_width = 340.0

    # Prefer src URLs for full-res images (TikTok often uses img with src)
    bg_img = None
    piece_img = None
    try:
        if bg_el.get_attribute("src"):
            bg_img = _load_image_from_src(bg_el.get_attribute("src"))
        if bg_img is None:
            bg_img = _image_from_element(bg_el)
    except Exception as e:
        logger.warning("Could not get background image: %s", e)
    try:
        if piece_el and piece_el.get_attribute("src"):
            piece_img = _load_image_from_src(piece_el.get_attribute("src"))
        if piece_img is None and piece_el:
            piece_img = _image_from_element(piece_el)
    except Exception as e:
        logger.warning("Could not get piece image: %s", e)

    return (bg_img, piece_img, puzzle_width)


def solve_slider_puzzle(
    driver: WebDriver,
    container: Optional[WebElement] = None,
    *,
    fudge_px: int = -6,
    use_edges: bool = True,
    bg_selector: str = DEFAULT_BG_SELECTOR,
    piece_selector: str = DEFAULT_PIECE_SELECTOR,
    wrapper_selector: str = DEFAULT_WRAPPER_SELECTOR,
    get_images: Optional[
        Callable[[WebDriver, Optional[WebElement]], Tuple[Optional[Image.Image], Optional[Image.Image], Optional[float]]]
    ] = None,
) -> bool:
    """
    Solve a puzzle slider captcha: extract images, compute slide distance, humanized drag.

    Parameters
    ----------
    driver : WebDriver
        Selenium WebDriver.
    container : WebElement, optional
        CAPTCHA container to scope element search.
    fudge_px : int
        Pixels to add to the computed distance (TikTok often needs -6 or similar).
    use_edges : bool
        Use Canny edges in template matching (recommended).
    bg_selector, piece_selector, wrapper_selector : str
        CSS selectors for background image, piece image, and wrapper (for width).
    get_images : callable, optional
        If provided, called as get_images(driver, container) and must return
        (background_pil, piece_pil, puzzle_width). Overrides selectors.

    Returns
    -------
    bool
        True if the drag was performed successfully, False otherwise.
    """
    if get_images:
        bg_img, piece_img, puzzle_width = get_images(driver, container)
    else:
        bg_img, piece_img, puzzle_width = get_captcha_images(
            driver, container,
            bg_selector=bg_selector,
            piece_selector=piece_selector,
            wrapper_selector=wrapper_selector,
        )

    if not bg_img or not piece_img:
        logger.warning("solve_slider_puzzle: could not get background or piece image")
        return False

    proportion, confidence = get_slide_offset_as_proportion(bg_img, piece_img, use_edges=use_edges)
    if puzzle_width is None:
        puzzle_width = float(bg_img.width)
    delta_x = int(proportion * puzzle_width + fudge_px)
    if delta_x < 0:
        delta_x = 0
    logger.info("solve_slider_puzzle: proportion=%.3f confidence=%.3f delta_x=%d", proportion, confidence, delta_x)

    return slider_drag_humanized(driver, delta_x, container=container)

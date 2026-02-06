"""
CAPTCHA solver – orchestration layer.

Ties together:
- ``browser.py``:  detect CAPTCHA, screenshot, click tiles/points, verify
- ``detector.py``: YOLO object detection (COCO), tile classification
- ``tiktok_detector.py``: fine-tuned YOLO for TikTok 3D shape objects
- ``shape_solver.py``: shape-matching with YOLO tier-1 + contour tier-2

Supported CAPTCHA types
-----------------------
- **Shape** (TikTok): "Select 2 objects that are the same shape"
  - Tier 1: fine-tuned YOLOv8 (if model is trained)
  - Tier 2: classical contour / Hu-moment analysis (fallback)
- **Grid** (reCAPTCHA / hCaptcha style): "Select all images with X" on a 3×3 or 4×4 grid.
- **Click** (single-image): "Click on the X" – detect and click the target object.
- **Slider / Rotate**: logged but not auto-solved (falls back to manual).

Usage
-----
::

    from captcha.solver import handle_captcha

    # inside your scraper, after page load:
    solved = handle_captcha(driver, max_attempts=3)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from selenium import webdriver

from captcha.browser import (
    CaptchaInfo,
    click_marker_points,
    click_points_on_element,
    click_tiles_by_index,
    click_verify_button,
    detect_captcha,
    is_captcha_gone,
    screenshot_captcha_image,
    screenshot_element,
    switch_to_captcha_frame,
    switch_to_default_content,
)
from captcha.detector import (
    classify_tiles,
    find_click_points,
    resolve_prompt_labels,
)
from captcha.shape_solver import find_matching_pair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def handle_captcha(
    driver: webdriver.Chrome,
    max_attempts: int = 3,
    post_solve_wait: float = 2.0,
) -> bool:
    """
    Detect and attempt to solve any visible CAPTCHA.

    Parameters
    ----------
    driver : webdriver.Chrome
        Live browser session.
    max_attempts : int
        Maximum solve attempts before giving up.
    post_solve_wait : float
        Seconds to wait after clicking verify before checking success.

    Returns
    -------
    bool
        ``True`` if the CAPTCHA was solved (or none was present).
    """
    for attempt in range(1, max_attempts + 1):
        info = detect_captcha(driver)
        if info is None:
            logger.info("No CAPTCHA detected – page is clear")
            return True

        logger.info(
            "CAPTCHA attempt %d/%d  type=%s  prompt=%r",
            attempt, max_attempts, info.captcha_type, info.prompt,
        )

        solved = _attempt_solve(driver, info)

        if solved:
            # Switch back to main content in case we were in an iframe
            switch_to_default_content(driver)
            time.sleep(post_solve_wait)
            if is_captcha_gone(driver, wait=1.5):
                logger.info("CAPTCHA solved on attempt %d", attempt)
                return True
            else:
                logger.warning("CAPTCHA still present after attempt %d", attempt)
        else:
            switch_to_default_content(driver)
            logger.warning("Solve logic returned False on attempt %d", attempt)

    logger.error("Failed to solve CAPTCHA after %d attempts", max_attempts)
    return False


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------


def _attempt_solve(driver: webdriver.Chrome, info: CaptchaInfo) -> bool:
    """Route to the appropriate solver based on CAPTCHA type."""
    # Switch into iframe if needed
    if info.iframe and not switch_to_captcha_frame(driver, info):
        return False

    # Re-detect inside the iframe (the container may be different now)
    if info.iframe:
        inner = detect_captcha(driver, timeout=2.0)
        if inner:
            info = inner

    if info.captcha_type == "shape":
        return _solve_shape(driver, info)
    elif info.captcha_type == "grid":
        return _solve_grid(driver, info)
    elif info.captcha_type == "click":
        return _solve_click(driver, info)
    elif info.captcha_type in ("slider", "rotate"):
        logger.warning(
            "CAPTCHA type '%s' is not auto-solvable; needs manual intervention",
            info.captcha_type,
        )
        return False
    else:
        # Unknown type: try shape first (TikTok's most common), then grid, then click
        logger.warning(
            "Unknown CAPTCHA type '%s'; trying shape → grid → click", info.captcha_type
        )
        return _solve_shape(driver, info) or _solve_grid(driver, info) or _solve_click(driver, info)


# ---------------------------------------------------------------------------
# Shape solver (TikTok "select 2 objects that are the same shape")
# ---------------------------------------------------------------------------


def _solve_shape(driver: webdriver.Chrome, info: CaptchaInfo) -> bool:
    """
    Solve TikTok's shape-matching CAPTCHA.

    1. Screenshot the challenge image.
    2. Attempt Tier 1 (fine-tuned YOLO class matching) if model exists.
    3. Fall back to Tier 2 (contour/Hu-moment comparison) otherwise.
    4. Click the two matching objects.
    5. Press Confirm/Verify.
    """
    element = info.image_element or info.container_element
    if not element:
        logger.warning("Shape solver: no image element to screenshot")
        return False

    image = screenshot_element(element)
    result = find_matching_pair(image)

    if result is None:
        logger.warning("Shape solver: could not find a matching pair")
        return False

    point_a, point_b = result
    logger.info(
        "Shape solver: clicking objects at (%d,%d) and (%d,%d)",
        point_a[0], point_a[1], point_b[0], point_b[1],
    )

    # Click the centre of each matched object on the CAPTCHA image
    click_marker_points(
        driver, element,
        [point_a, point_b],
        image.size,
        delay=0.5,
    )

    time.sleep(0.5)
    click_verify_button(driver, info)
    return True


# ---------------------------------------------------------------------------
# Grid solver
# ---------------------------------------------------------------------------


def _solve_grid(driver: webdriver.Chrome, info: CaptchaInfo) -> bool:
    """
    Solve a grid-style CAPTCHA.

    1. Resolve the prompt to COCO target labels.
    2. Screenshot the grid.
    3. Run tile classification (YOLO).
    4. Click matching tiles.
    5. Press the verify button.
    """
    target_labels = resolve_prompt_labels(info.prompt)
    if not target_labels:
        logger.warning("Could not map prompt %r to any known object class", info.prompt)
        # Fall back: try to parse the bold word in the prompt
        target_labels = _extract_bold_word(info.prompt)
        if not target_labels:
            return False

    # Screenshot the grid area
    image = screenshot_captcha_image(driver, info)
    if image is None:
        return False

    rows = info.grid_rows or 3
    cols = info.grid_cols or 3

    matching_indices = classify_tiles(
        image, rows, cols, target_labels,
        confidence_threshold=0.20,
        iou_threshold=0.08,
    )

    if not matching_indices:
        logger.warning("No tiles matched target labels %s", target_labels)
        return False

    logger.info(
        "Grid solve: clicking tiles %s for labels %s", matching_indices, target_labels
    )

    # Click the tiles
    if info.tile_elements:
        click_tiles_by_index(driver, info, matching_indices)
    else:
        # No individual tile elements – click by coordinate on the image
        _click_tile_centres_on_image(driver, info, image.size, rows, cols, matching_indices)

    time.sleep(0.5)
    click_verify_button(driver, info)
    return True


def _click_tile_centres_on_image(
    driver: webdriver.Chrome,
    info: CaptchaInfo,
    image_size: tuple[int, int],
    rows: int,
    cols: int,
    indices: list[int],
) -> None:
    """Click tile centres by computing coordinates on the image element."""
    w, h = image_size
    tile_w = w / cols
    tile_h = h / rows
    points = []
    for idx in indices:
        r = idx // cols
        c = idx % cols
        cx = c * tile_w + tile_w / 2
        cy = r * tile_h + tile_h / 2
        points.append((cx, cy))

    element = info.image_element or info.container_element
    if element:
        click_points_on_element(driver, element, points, image_size)


# ---------------------------------------------------------------------------
# Click solver
# ---------------------------------------------------------------------------


def _solve_click(driver: webdriver.Chrome, info: CaptchaInfo) -> bool:
    """
    Solve a click-based CAPTCHA.

    1. Resolve the prompt to target labels.
    2. Screenshot the challenge image.
    3. Run YOLO to find target objects.
    4. Click the centre of each detection.
    5. Press verify.
    """
    target_labels = resolve_prompt_labels(info.prompt)
    if not target_labels:
        target_labels = _extract_bold_word(info.prompt)
        if not target_labels:
            return False

    element = info.image_element or info.container_element
    if not element:
        return False

    image = screenshot_element(element)
    points = find_click_points(image, target_labels, confidence_threshold=0.25)

    if not points:
        logger.warning("No click targets found for labels %s", target_labels)
        return False

    logger.info("Click solve: %d targets found for labels %s", len(points), target_labels)
    click_points_on_element(driver, element, points, image.size)

    time.sleep(0.5)
    click_verify_button(driver, info)
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_bold_word(prompt: str) -> list[str]:
    """
    Many CAPTCHA prompts put the target in bold tags or between ** markers.
    Try to extract that word and resolve it to COCO labels.
    """
    import re

    # HTML bold
    m = re.search(r"<strong>(.*?)</strong>", prompt)
    if m:
        return resolve_prompt_labels(m.group(1)) or [m.group(1).lower()]

    # Markdown bold
    m = re.search(r"\*\*(.*?)\*\*", prompt)
    if m:
        return resolve_prompt_labels(m.group(1)) or [m.group(1).lower()]

    # Try the last few words (often "…with traffic lights")
    words = prompt.lower().split()
    for n in (3, 2, 1):
        if len(words) >= n:
            tail = " ".join(words[-n:])
            labels = resolve_prompt_labels(tail)
            if labels:
                return labels

    return []

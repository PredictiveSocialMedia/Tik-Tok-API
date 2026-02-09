"""
Puzzle slider captcha solver (drag the piece into the slot).

Uses OpenCV template matching for position and human-like Selenium drag.
Designed for TikTok-style sliders; configurable selectors for other sites.

Example
-------
    from slider_puzzle import solve_slider_puzzle, get_slide_offset

    # Full solve (extract images from page, then drag)
    ok = solve_slider_puzzle(driver, container=cap_container, fudge_px=-6)

    # Position only (you have images already)
    offset_x, confidence = get_slide_offset(background_pil, piece_pil)
"""

from .position import get_slide_offset, get_slide_offset_as_proportion
from .drag import slider_drag_humanized
from .solve import get_captcha_images, solve_slider_puzzle

__all__ = [
    "get_slide_offset",
    "get_slide_offset_as_proportion",
    "slider_drag_humanized",
    "get_captcha_images",
    "solve_slider_puzzle",
]

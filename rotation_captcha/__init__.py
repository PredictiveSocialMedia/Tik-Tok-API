"""
rotation_captcha – Solver for circular rotation captchas (e.g. TikTok).

Public API
----------
- estimate_angle_overlay   : angle via pixel-difference inside the inner circle
- estimate_angle_ring      : angle via boundary-ring color matching
- estimate_angle           : unified wrapper (overlay or ring)
- angle_to_slider_delta    : convert degrees → slider pixels
- rotation_drag_humanized  : human-like slider drag
- get_captcha_images       : extract outer/inner images from the page
- get_track_length         : measure the slider track width
- solve_rotation_captcha   : end-to-end solver
"""

from .angle import (
    estimate_angle,
    estimate_angle_overlay,
    estimate_angle_ring,
    angle_to_slider_delta,
)
from .drag import rotation_drag_humanized
from .solve import (
    get_captcha_images,
    get_track_length,
    solve_rotation_captcha,
)

__all__ = [
    "estimate_angle",
    "estimate_angle_overlay",
    "estimate_angle_ring",
    "angle_to_slider_delta",
    "rotation_drag_humanized",
    "get_captcha_images",
    "get_track_length",
    "solve_rotation_captcha",
]

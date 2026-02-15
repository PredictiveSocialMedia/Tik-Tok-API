"""
Captcha detection and routing to solvers.

- detector: detect_captcha, is_captcha_visible, CaptchaType
- router: solve_captcha (dispatches to slider_puzzle, rotation_captcha, object_selection_captcha)
"""

from .detector import CaptchaType, detect_captcha, is_captcha_visible
from .router import solve_captcha

__all__ = [
    "CaptchaType",
    "detect_captcha",
    "is_captcha_visible",
    "solve_captcha",
]

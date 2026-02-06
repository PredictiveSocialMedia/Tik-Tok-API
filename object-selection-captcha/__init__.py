"""
CAPTCHA solver package.

Public API
----------
handle_captcha(driver, max_attempts=3) -> bool
    Detect and attempt to solve any visible CAPTCHA on the current page.
    Returns True if the CAPTCHA was solved (or none was present).
"""

from captcha.solver import handle_captcha  # noqa: F401

__all__ = ["handle_captcha"]

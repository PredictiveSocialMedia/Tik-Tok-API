"""
Browser WebDriver setup, cookie management, and anti-detection stealth.
"""

from .driver import create_driver, load_cookies, save_cookies
from .stealth import (
    clear_rate_limit,
    disconnect_driver,
    human_click,
    human_sleep,
    human_type,
    inject_stealth_scripts,
    is_rate_limited,
    reconnect_driver,
    set_rate_limit,
)

__all__ = [
    "create_driver",
    "load_cookies",
    "save_cookies",
    "human_type",
    "human_click",
    "human_sleep",
    "inject_stealth_scripts",
    "disconnect_driver",
    "reconnect_driver",
    "is_rate_limited",
    "set_rate_limit",
    "clear_rate_limit",
]

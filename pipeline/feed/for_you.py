"""
For You feed iterator.

Scrolls the feed and yields video URLs (or in-page references) for parsing.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Iterator, Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)

FOR_YOU_URL = "https://www.tiktok.com/foryou"

# Selectors for video links in the feed
VIDEO_LINK_SELECTORS = [
    "a[href*='/video/']",
    "[data-e2e='recommend-list-item-container'] a",
    "div[class*='DivItemContainer'] a[href*='/video/']",
]


def _scroll(driver: WebDriver, pixels: int = 800) -> None:
    driver.execute_script(f"window.scrollBy(0, {pixels});")


def _get_video_urls_on_page(driver: WebDriver) -> list[str]:
    """Extract unique video URLs currently visible on the page."""
    seen: set[str] = set()
    urls: list[str] = []
    for sel in VIDEO_LINK_SELECTORS:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            for el in els:
                try:
                    href = el.get_attribute("href")
                    if href and "/video/" in href and href not in seen:
                        # Normalize: take first part (before query string)
                        base = href.split("?")[0]
                        if base not in seen:
                            seen.add(base)
                            urls.append(base)
                except Exception:
                    continue
        except Exception:
            continue
    return urls


def iterate_for_you_feed(
    driver: WebDriver,
    for_you_url: str = FOR_YOU_URL,
    scroll_delay_sec: float = 1.5,
    max_videos: Optional[int] = None,
    delay_between_scrolls_sec: float = 1.0,
) -> Iterator[str]:
    """
    Iterate over video URLs from the For You feed.

    Yields video URLs as they appear. Scrolls the page to load more.
    """
    driver.get(for_you_url)
    time.sleep(4)  # Initial load

    yielded: set[str] = set()
    total_yielded = 0

    while True:
        urls = _get_video_urls_on_page(driver)
        for url in urls:
            if url in yielded:
                continue
            yielded.add(url)
            total_yielded += 1
            yield url
            if max_videos and total_yielded >= max_videos:
                return

        # Scroll to load more
        _scroll(driver, random.randint(600, 1000))
        time.sleep(scroll_delay_sec + random.uniform(0, 0.5))
        time.sleep(delay_between_scrolls_sec)

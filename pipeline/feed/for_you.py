"""
For You feed iterator.

Scrolls the feed and yields video URLs (or in-page references) for parsing.
Uses jittered scroll amounts and delays to avoid predictable timing patterns
that TikTok's anti-bot system could flag.
"""

from __future__ import annotations

import logging
import random
from typing import Iterator, Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from pipeline.browser.stealth import human_sleep

logger = logging.getLogger(__name__)

FOR_YOU_URL = "https://www.tiktok.com/foryou"

# Selectors for video links in the feed
VIDEO_LINK_SELECTORS = [
    "a[href*='/video/']",
    "[data-e2e='recommend-list-item-container'] a",
    "div[class*='DivItemContainer'] a[href*='/video/']",
]


def _scroll(driver: WebDriver, pixels: int = 800) -> None:
    """Scroll by a given number of pixels (randomised by caller)."""
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
    All timing is jittered via ``human_sleep`` to avoid detection.
    """
    driver.get(for_you_url)
    human_sleep(4.0, jitter=1.5)  # Initial page load

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

        # Scroll to load more — randomised distance
        _scroll(driver, random.randint(500, 1100))
        human_sleep(scroll_delay_sec, jitter=0.8)
        human_sleep(delay_between_scrolls_sec, jitter=0.5)

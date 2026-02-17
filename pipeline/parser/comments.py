"""
Comment parser: extract top N, highest-liked subset, top M replies per comment.

Uses main DOM only (TikTok loads comments async). If TikTok moves comments into
a shadow root, we'd need to pierce it via JS (same pattern as in auth/login.py).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import List, Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from pipeline.storage import Comment
from .utils import parse_count

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    from datetime import timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dismiss_overlays(driver: WebDriver) -> None:
    for label in ("Allow all", "Allow All", "Accept all cookies", "Accept All"):
        try:
            btn = driver.find_element(By.XPATH, f"//button[normalize-space()='{label}']")
            btn.click()
            time.sleep(0.5)
            return
        except Exception:
            continue


def _click_comment_icon(driver: WebDriver, timeout: int = 6) -> None:
    try:
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
        btn = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[starts-with(@aria-label,'Read or add comments')]"
                " | //button[.//span[@data-e2e='comment-icon']]",
            ))
        )
        btn.click()
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='DivCommentListContainer']"))
        )
        time.sleep(0.5)
    except Exception:
        pass


def _scroll_comments(driver: WebDriver) -> None:
    for offset in (600, 800, 400):
        driver.execute_script(f"window.scrollBy(0, {offset});")
        time.sleep(1.0)


def parse_comments_from_dom(
    driver: WebDriver,
    video_id: str,
    top_n: int = 20,
    highest_liked_k: int = 5,
    replies_per_comment_m: int = 5,
) -> List[Comment]:
    """
    Scrape comments from DOM, return top N + highest-liked subset + top M replies each.

    Returns flat list: top-level comments (sorted by likes, then top M replies each).
    """
    _dismiss_overlays(driver)
    _click_comment_icon(driver)
    _scroll_comments(driver)
    time.sleep(1)

    scraped_at = _iso_now()
    raw: List[dict] = []

    wrappers = driver.find_elements(By.CSS_SELECTOR, "div[class*='DivCommentObjectWrapper']")
    if not wrappers:
        wrappers = driver.find_elements(By.CSS_SELECTOR, "div[class*='DivCommentItemWrapper']")
    if not wrappers:
        logger.debug(
            "No comment wrappers in main DOM (comments may be in shadow DOM or not loaded yet)"
        )

    seen_text: set[str] = set()
    for wrapper in wrappers:
        if len(raw) >= top_n * 2:  # Get extra for sorting
            break
        try:
            text_el = wrapper.find_elements(By.CSS_SELECTOR, "[data-e2e='comment-level-1']")
            if not text_el:
                continue
            text = (text_el[0].text or "").strip()
            if len(text) < 2 or len(text) > 2000 or text in seen_text:
                continue
            seen_text.add(text)

            username = ""
            try:
                user_el = wrapper.find_elements(By.CSS_SELECTOR, "[data-e2e='comment-username-1'] a p")
                if user_el:
                    username = (user_el[0].text or "").strip()
            except Exception:
                pass

            like_count = 0
            try:
                like_container = wrapper.find_elements(By.CSS_SELECTOR, "div[class*='DivLikeContainer']")
                if like_container:
                    aria = like_container[0].get_attribute("aria-label") or ""
                    m = re.search(r"([\d.,]+[KkMmBb]?)\s*likes?", aria)
                    if m:
                        like_count = parse_count(m.group(1))
                    else:
                        spans = like_container[0].find_elements(By.CSS_SELECTOR, "span")
                        if spans:
                            like_count = parse_count(spans[-1].text)
            except Exception:
                pass

            reply_count = 0
            try:
                reply_els = wrapper.find_elements(
                    By.XPATH,
                    "following-sibling::div[contains(@class,'DivReplyContainer')][1]//div[contains(@class,'TUXButton-label')]",
                )
                if reply_els:
                    label = (reply_els[0].text or "").strip()
                    m = re.search(r"View\s+([\d,]+)\s+repl", label)
                    if m:
                        reply_count = int(m.group(1).replace(",", ""))
            except Exception:
                pass

            # Use text hash as comment_id if we don't have real id
            comment_id = f"{video_id}_{hash(text) % 10**10}"
            raw.append({
                "comment_id": comment_id,
                "username": username,
                "text": text,
                "likes": like_count,
                "reply_count": reply_count,
            })
        except Exception:
            continue

    # Sort by likes desc, take top N
    raw.sort(key=lambda x: x["likes"], reverse=True)
    top = raw[:top_n]

    # Build Comment objects (no replies for now - would need to expand reply threads)
    comments: List[Comment] = []
    for r in top:
        comments.append(Comment(
            comment_id=r["comment_id"],
            video_id=video_id,
            username=r["username"] or None,
            text=r["text"],
            likes=r["likes"],
            reply_count=r["reply_count"],
            parent_comment_id=None,
            scraped_at=scraped_at,
        ))

    return comments

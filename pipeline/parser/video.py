"""
Video parser: extract video metadata and map to 3NF schema.

Uses rehydration JSON when available; falls back to DOM.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional, Tuple

from selenium.webdriver.remote.webdriver import WebDriver

from pipeline.storage import Author, AuthorMetricSnapshot, Video
from .utils import (
    extract_rehydration_json,
    get_video_item_payload,
    parse_count,
    safe_get,
)

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    from datetime import timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_video_from_json(
    item: dict,
    video_url: str,
    position: Optional[int] = None,
) -> Tuple[Author, Video, AuthorMetricSnapshot]:
    """
    Parse TikTok itemStruct into Author, Video, AuthorMetricSnapshot.
    """
    author_data = item.get("author") or {}
    music = item.get("music") or {}
    video_data = item.get("video") or {}
    stats = item.get("statsV2") or item.get("stats") or {}
    author_stats = item.get("authorStatsV2") or item.get("authorStats") or {}

    author_id = author_data.get("id") or ""
    username = author_data.get("uniqueId") or author_data.get("unique_id") or ""
    display_name = author_data.get("nickname") or ""

    author = Author(
        author_id=author_id,
        username=username,
        display_name=display_name or None,
        bio=(author_data.get("signature") or "").strip() or None,
        avatar_url=author_data.get("avatarThumb"),
        verified=bool(author_data.get("verified")),
    )

    video_id = item.get("id") or ""
    caption = (item.get("desc") or "").strip()
    challenges = item.get("challenges") or []
    hashtags = [c.get("title", "") for c in challenges if isinstance(c, dict) and c.get("title")]

    audio_name = music.get("title") or ""
    if music.get("authorName"):
        audio_name = f"{audio_name} - {music.get('authorName')}" if audio_name else music.get("authorName")
    if music.get("original"):
        audio_name = f"Original Sound - {audio_name}" if audio_name else "Original Sound"

    scraped_at = _iso_now()

    video = Video(
        video_id=video_id,
        author_id=author_id,
        scraped_at=scraped_at,
        url=video_url,
        caption=caption,
        hashtags=hashtags,
        audio_name=audio_name,
        audio_id=str(music.get("id")) if music.get("id") else None,
        duration_sec=video_data.get("duration"),
        thumbnail_url=video_data.get("cover") or video_data.get("originCover"),
        created_at=str(item.get("createTime")) if item.get("createTime") else None,
        likes=parse_count(str(stats.get("diggCount") or stats.get("likeCount", 0))),
        comments_count=parse_count(str(stats.get("commentCount", 0))),
        shares=parse_count(str(stats.get("shareCount", 0))),
        plays=parse_count(str(stats.get("playCount", 0))),
        source="for_you",
        position=position,
    )

    snapshot = AuthorMetricSnapshot(
        author_id=author_id,
        video_id=video_id,
        scraped_at=scraped_at,
        follower_count=parse_count(str(author_stats.get("followerCount", 0))),
        following_count=parse_count(str(author_stats.get("followingCount", 0))) or None,
        author_likes_count=parse_count(str(author_stats.get("heartCount", 0))) or None,
    )

    return author, video, snapshot


def parse_video_page(
    driver: WebDriver,
    video_url: str,
    position: Optional[int] = None,
    wait_sec: int = 8,
) -> Optional[Tuple[Author, Video, AuthorMetricSnapshot]]:
    """
    Load video page, extract JSON, parse into Author, Video, AuthorMetricSnapshot.
    Returns None if parsing fails.
    """
    driver.get(video_url)
    time.sleep(wait_sec)

    html = driver.page_source
    data = extract_rehydration_json(html)
    if not data:
        logger.warning("No rehydration JSON for %s", video_url[:60])
        return None

    payload = get_video_item_payload(data)
    if not payload:
        logger.warning("No itemInfo in rehydration for %s", video_url[:60])
        return None

    item = payload.get("itemInfo", {}).get("itemStruct") or payload.get("itemStruct")
    if not item:
        logger.warning("No itemStruct for %s", video_url[:60])
        return None

    return parse_video_from_json(item, video_url, position)

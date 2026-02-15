"""
3NF schema definitions for the pipeline.

Relations:
- authors: creator identity and static attributes
- videos: video metadata, linked to author
- author_metric_snapshots: follower count etc. at scrape time (for virality/controversy)
- comments: top-level and replies, linked to video
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Author:
    """Author relation (3NF)."""

    author_id: str
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    verified: bool = False


@dataclass
class Video:
    """Video relation (3NF). References author_id."""

    video_id: str
    author_id: str
    scraped_at: str  # ISO 8601
    url: Optional[str] = None
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    audio_name: str = ""
    audio_id: Optional[str] = None
    duration_sec: Optional[int] = None
    thumbnail_url: Optional[str] = None
    created_at: Optional[str] = None
    likes: int = 0
    comments_count: int = 0
    shares: int = 0
    plays: int = 0
    source: str = "for_you"
    position: Optional[int] = None


@dataclass
class AuthorMetricSnapshot:
    """Author metrics at scrape time (for virality/controversy analysis)."""

    author_id: str
    video_id: str
    scraped_at: str  # ISO 8601
    follower_count: int
    following_count: Optional[int] = None
    author_likes_count: Optional[int] = None


@dataclass
class Comment:
    """Comment relation (3NF). Top-level or reply (parent_comment_id set)."""

    comment_id: str
    video_id: str
    author_id: Optional[str] = None
    username: Optional[str] = None
    text: str = ""
    likes: int = 0
    reply_count: int = 0
    parent_comment_id: Optional[str] = None
    scraped_at: str = ""  # ISO 8601


def author_to_payload(a: Author) -> dict[str, Any]:
    """Convert Author to JSON-serializable payload."""
    return {
        "author_id": a.author_id,
        "username": a.username,
        "display_name": a.display_name,
        "bio": a.bio,
        "avatar_url": a.avatar_url,
        "verified": a.verified,
    }


def video_to_payload(v: Video) -> dict[str, Any]:
    """Convert Video to JSON-serializable payload."""
    return {
        "video_id": v.video_id,
        "author_id": v.author_id,
        "url": v.url,
        "scraped_at": v.scraped_at,
        "caption": v.caption,
        "hashtags": v.hashtags,
        "audio_name": v.audio_name,
        "audio_id": v.audio_id,
        "duration_sec": v.duration_sec,
        "thumbnail_url": v.thumbnail_url,
        "created_at": v.created_at,
        "likes": v.likes,
        "comments_count": v.comments_count,
        "shares": v.shares,
        "plays": v.plays,
        "source": v.source,
        "position": v.position,
    }


def author_snapshot_to_payload(s: AuthorMetricSnapshot) -> dict[str, Any]:
    """Convert AuthorMetricSnapshot to payload."""
    return {
        "author_id": s.author_id,
        "video_id": s.video_id,
        "scraped_at": s.scraped_at,
        "follower_count": s.follower_count,
        "following_count": s.following_count,
        "author_likes_count": s.author_likes_count,
    }


def comment_to_payload(c: Comment) -> dict[str, Any]:
    """Convert Comment to payload."""
    return {
        "comment_id": c.comment_id,
        "video_id": c.video_id,
        "author_id": c.author_id,
        "username": c.username,
        "text": c.text,
        "likes": c.likes,
        "reply_count": c.reply_count,
        "parent_comment_id": c.parent_comment_id,
        "scraped_at": c.scraped_at,
    }


def _iso_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

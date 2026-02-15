"""
Storage interface: payload-based, REST-ready.

Implementations: LocalStorage (SQLite), future RestStorage (HTTP POST).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .schema import (
    Author,
    AuthorMetricSnapshot,
    Comment,
    Video,
)


class StorageBackend(ABC):
    """Abstract storage backend. Implement for local (SQLite) or REST."""

    @abstractmethod
    def save_author(self, author: Author) -> None:
        """Upsert author by author_id."""
        ...

    @abstractmethod
    def save_video(self, video: Video) -> None:
        """Insert or replace video."""
        ...

    @abstractmethod
    def save_author_snapshot(self, snapshot: AuthorMetricSnapshot) -> None:
        """Insert author metric snapshot (author_id, video_id, scraped_at, follower_count, ...)."""
        ...

    @abstractmethod
    def save_comments(self, video_id: str, comments: List[Comment]) -> None:
        """Insert comments for a video. Replaces existing for that video if desired."""
        ...

    @abstractmethod
    def save_checkpoint(self, run_id: str, last_video_id: str, last_position: Optional[int] = None) -> None:
        """Save run checkpoint for resumability."""
        ...

    @abstractmethod
    def get_checkpoint(self, run_id: str) -> Optional[dict[str, Any]]:
        """Load checkpoint for run_id."""
        ...

    @abstractmethod
    def video_exists(self, video_id: str) -> bool:
        """Check if video already stored (for deduplication)."""
        ...

"""
Local SQLite storage backend (3NF).

Creates tables: authors, videos, author_metric_snapshots, comments, checkpoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from .interface import StorageBackend
from .schema import (
    Author,
    AuthorMetricSnapshot,
    Comment,
    Video,
)

logger = logging.getLogger(__name__)


def _ensure_db(db_path: Path) -> "sqlite3.Connection":
    import sqlite3
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: "sqlite3.Connection") -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS authors (
            author_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            display_name TEXT,
            bio TEXT,
            avatar_url TEXT,
            verified INTEGER DEFAULT 0,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            author_id TEXT NOT NULL,
            url TEXT,
            scraped_at TEXT NOT NULL,
            caption TEXT,
            hashtags TEXT,
            audio_name TEXT,
            audio_id TEXT,
            duration_sec INTEGER,
            thumbnail_url TEXT,
            created_at TEXT,
            likes INTEGER DEFAULT 0,
            comments_count INTEGER DEFAULT 0,
            shares INTEGER DEFAULT 0,
            plays INTEGER DEFAULT 0,
            source TEXT DEFAULT 'for_you',
            position INTEGER,
            FOREIGN KEY (author_id) REFERENCES authors(author_id)
        );

        CREATE TABLE IF NOT EXISTS author_metric_snapshots (
            author_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            scraped_at TEXT NOT NULL,
            follower_count INTEGER NOT NULL,
            following_count INTEGER,
            author_likes_count INTEGER,
            PRIMARY KEY (author_id, video_id, scraped_at),
            FOREIGN KEY (author_id) REFERENCES authors(author_id),
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        );

        CREATE TABLE IF NOT EXISTS comments (
            comment_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            author_id TEXT,
            username TEXT,
            text TEXT,
            likes INTEGER DEFAULT 0,
            reply_count INTEGER DEFAULT 0,
            parent_comment_id TEXT,
            scraped_at TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        );

        CREATE TABLE IF NOT EXISTS checkpoints (
            run_id TEXT PRIMARY KEY,
            last_video_id TEXT,
            last_position INTEGER,
            updated_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_videos_author ON videos(author_id);
        CREATE INDEX IF NOT EXISTS idx_comments_video ON comments(video_id);
        CREATE INDEX IF NOT EXISTS idx_snapshots_author ON author_metric_snapshots(author_id);
    """)


class LocalStorage(StorageBackend):
    """SQLite-backed storage. 3NF schema."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._connection: Optional[Any] = None

    def _get_conn(self) -> Any:
        import sqlite3
        if self._connection is None:
            self._connection = _ensure_db(self._db_path)
        return self._connection

    def save_author(self, author: Author) -> None:
        from datetime import datetime, timezone
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        conn.execute(
            """
            INSERT INTO authors (author_id, username, display_name, bio, avatar_url, verified, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(author_id) DO UPDATE SET
                username = excluded.username,
                display_name = excluded.display_name,
                bio = excluded.bio,
                avatar_url = excluded.avatar_url,
                verified = excluded.verified,
                updated_at = excluded.updated_at
            """,
            (
                author.author_id,
                author.username,
                author.display_name,
                author.bio,
                author.avatar_url,
                1 if author.verified else 0,
                now,
            ),
        )
        conn.commit()

    def save_video(self, video: Video) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO videos (
                video_id, author_id, url, scraped_at, caption, hashtags,
                audio_name, audio_id, duration_sec, thumbnail_url, created_at,
                likes, comments_count, shares, plays, source, position
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                author_id = excluded.author_id,
                url = excluded.url,
                scraped_at = excluded.scraped_at,
                caption = excluded.caption,
                hashtags = excluded.hashtags,
                audio_name = excluded.audio_name,
                audio_id = excluded.audio_id,
                duration_sec = excluded.duration_sec,
                thumbnail_url = excluded.thumbnail_url,
                created_at = excluded.created_at,
                likes = excluded.likes,
                comments_count = excluded.comments_count,
                shares = excluded.shares,
                plays = excluded.plays,
                source = excluded.source,
                position = excluded.position
            """,
            (
                video.video_id,
                video.author_id,
                video.url,
                video.scraped_at,
                video.caption,
                json.dumps(video.hashtags),
                video.audio_name,
                video.audio_id,
                video.duration_sec,
                video.thumbnail_url,
                video.created_at,
                video.likes,
                video.comments_count,
                video.shares,
                video.plays,
                video.source,
                video.position,
            ),
        )
        conn.commit()

    def save_author_snapshot(self, snapshot: AuthorMetricSnapshot) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO author_metric_snapshots
            (author_id, video_id, scraped_at, follower_count, following_count, author_likes_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.author_id,
                snapshot.video_id,
                snapshot.scraped_at,
                snapshot.follower_count,
                snapshot.following_count,
                snapshot.author_likes_count,
            ),
        )
        conn.commit()

    def save_comments(self, video_id: str, comments: List[Comment]) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM comments WHERE video_id = ?", (video_id,))
        for c in comments:
            conn.execute(
                """
                INSERT INTO comments (comment_id, video_id, author_id, username, text, likes, reply_count, parent_comment_id, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c.comment_id,
                    c.video_id,
                    c.author_id,
                    c.username,
                    c.text,
                    c.likes,
                    c.reply_count,
                    c.parent_comment_id,
                    c.scraped_at,
                ),
            )
        conn.commit()

    def save_checkpoint(self, run_id: str, last_video_id: str, last_position: Optional[int] = None) -> None:
        from datetime import datetime, timezone
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        conn.execute(
            """
            INSERT INTO checkpoints (run_id, last_video_id, last_position, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                last_video_id = excluded.last_video_id,
                last_position = excluded.last_position,
                updated_at = excluded.updated_at
            """,
            (run_id, last_video_id, last_position, now),
        )
        conn.commit()

    def get_checkpoint(self, run_id: str) -> Optional[dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT last_video_id, last_position, updated_at FROM checkpoints WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "last_video_id": row[0],
            "last_position": row[1],
            "updated_at": row[2],
        }

    def video_exists(self, video_id: str) -> bool:
        conn = self._get_conn()
        row = conn.execute("SELECT 1 FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return row is not None

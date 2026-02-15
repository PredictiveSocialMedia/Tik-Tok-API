"""
Storage layer: payload-based, 3NF schema, REST-ready interface.

- LocalStorage: SQLite backend
- Schema: Author, Video, AuthorMetricSnapshot, Comment
"""

from .interface import StorageBackend
from .local import LocalStorage
from .schema import (
    Author,
    AuthorMetricSnapshot,
    Comment,
    Video,
    author_to_payload,
    author_snapshot_to_payload,
    comment_to_payload,
    video_to_payload,
)

__all__ = [
    "StorageBackend",
    "LocalStorage",
    "Author",
    "Video",
    "AuthorMetricSnapshot",
    "Comment",
    "author_to_payload",
    "video_to_payload",
    "author_snapshot_to_payload",
    "comment_to_payload",
]

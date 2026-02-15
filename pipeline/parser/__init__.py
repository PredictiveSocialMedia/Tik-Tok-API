"""
Parser: video and comment extraction, mapped to 3NF schema.
"""

from .video import parse_video_from_json, parse_video_page
from .comments import parse_comments_from_dom

__all__ = [
    "parse_video_from_json",
    "parse_video_page",
    "parse_comments_from_dom",
]

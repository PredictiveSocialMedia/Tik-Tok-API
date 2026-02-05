"""
Shared pytest fixtures for TikTok scraper tests.

Uses minimal HTML/JSON snippets so we don't depend on live TikTok or real page structure.
"""

import pytest


@pytest.fixture
def sample_rehydration_html():
    """Minimal HTML containing the __UNIVERSAL_DATA_FOR_REHYDRATION__ script."""
    data = '{"__DEFAULT_SCOPE__":{"webapp.video-detail":{"itemInfo":{"itemStruct":{"id":"123","desc":"Hello #fyp","createTime":1700000000,"author":{"id":"a1","uniqueId":"user","nickname":"User"},"stats":{"playCount":1000,"diggCount":50,"commentCount":5},"video":{},"music":{}}}}}}'
    return f'<!DOCTYPE html><html><body><script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">{data}</script></body></html>'


@pytest.fixture
def sample_video_item_struct():
    """Minimal itemStruct-like dict for parse_video_item tests."""
    return {
        "id": "7123456789",
        "desc": "Check this out #viral #fyp",
        "createTime": 1700000000,
        "author": {
            "id": "author1",
            "uniqueId": "cooluser",
            "nickname": "Cool User",
            "verified": True,
            "signature": "Creator",
            "avatarThumb": "https://example.com/thumb.jpg",
        },
        "stats": {"playCount": 10000, "diggCount": 500, "commentCount": 20, "shareCount": 10},
        "statsV2": {"playCount": "12.5K", "diggCount": "1.2K", "commentCount": "50"},
        "video": {
            "playAddr": "https://example.com/video.mp4",
            "downloadAddr": "https://example.com/dl.mp4",
            "duration": 15,
            "width": 1080,
            "height": 1920,
        },
        "music": {
            "id": "music1",
            "title": "Song",
            "authorName": "Artist",
            "playUrl": "https://example.com/audio.mp3",
            "duration": 30,
        },
        "challenges": [{"id": "c1", "title": "fyp"}, {"id": "c2", "title": "viral"}],
        "authorStats": {"followerCount": 100000, "videoCount": 42},
        "locationCreated": "US",
    }


@pytest.fixture
def sample_comment_dict():
    """Raw comment dict as might appear in rehydration JSON."""
    return {
        "cid": "comment_abc",
        "text": "Great video!",
        "user": {"uniqueId": "viewer1", "nickname": "Viewer One"},
        "diggCount": 10,
        "replyCommentTotal": 2,
        "createTime": 1700000100,
    }

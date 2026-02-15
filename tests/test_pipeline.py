"""
Tests for the TikTok scraping pipeline.

Covers: storage (LocalStorage), parser utils, config, schema, parse_video_from_json.
No browser or network required.
"""

import tempfile
from pathlib import Path

import pytest

from pipeline.config import PipelineConfig
from pipeline.parser.utils import (
    extract_rehydration_json,
    find_in_dict,
    get_video_item_payload,
    parse_count,
    safe_get,
)
from pipeline.parser.video import parse_video_from_json
from pipeline.storage import Author, AuthorMetricSnapshot, Comment, LocalStorage, Video


# ---------------------------------------------------------------------------
# parse_count (from pipeline.parser.utils)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", 0),
        ("0", 0),
        ("9339", 9339),
        ("28.2K", 28200),
        ("1.9M", 1900000),
        ("2B", 2_000_000_000),
        ("1.5k", 1500),
        ("1,234", 1234),
        ("  42  ", 42),
        ("276.8K likes", 276800),
    ],
)
def test_parse_count(text, expected):
    assert parse_count(text) == expected


def test_parse_count_invalid_returns_zero():
    assert parse_count("nope") == 0
    assert parse_count("abc123") == 0


# ---------------------------------------------------------------------------
# safe_get
# ---------------------------------------------------------------------------


def test_safe_get():
    d = {"a": {"b": {"c": 1}}}
    assert safe_get(d, "a", "b", "c") == 1
    assert safe_get(d, "a", "b") == {"c": 1}
    assert safe_get(d, "a", "x", default=99) == 99
    assert safe_get(d, "x") is None
    assert safe_get({}, "a", default="default") == "default"


# ---------------------------------------------------------------------------
# find_in_dict
# ---------------------------------------------------------------------------


def test_find_in_dict():
    d = {"itemInfo": {"itemStruct": {"id": "1"}}, "other": {"itemInfo": {"itemStruct": {"id": "2"}}}}
    results = find_in_dict(d, "itemStruct")
    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[1]["id"] == "2"


def test_find_in_dict_with_list():
    d = {"items": [{"id": "a"}, {"id": "b"}]}
    assert find_in_dict(d, "id") == ["a", "b"]


# ---------------------------------------------------------------------------
# extract_rehydration_json
# ---------------------------------------------------------------------------


def test_extract_rehydration_json():
    data = '{"__DEFAULT_SCOPE__":{"webapp.video-detail":{"itemInfo":{"itemStruct":{"id":"123"}}}}}'
    html = f'<html><script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">{data}</script></html>'
    result = extract_rehydration_json(html)
    assert result is not None
    assert "__DEFAULT_SCOPE__" in result
    assert "itemInfo" in result["__DEFAULT_SCOPE__"]["webapp.video-detail"]


def test_extract_rehydration_json_no_script():
    assert extract_rehydration_json("<html><body></body></html>") is None


def test_extract_rehydration_json_invalid_json():
    html = '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">not valid json {</script>'
    assert extract_rehydration_json(html) is None


# ---------------------------------------------------------------------------
# get_video_item_payload
# ---------------------------------------------------------------------------


def test_get_video_item_payload_from_default_scope():
    data = {
        "__DEFAULT_SCOPE__": {
            "webapp.video-detail": {"itemInfo": {"itemStruct": {"id": "123"}}},
        }
    }
    payload = get_video_item_payload(data)
    assert payload is not None
    assert payload.get("itemInfo", {}).get("itemStruct", {}).get("id") == "123"


def test_get_video_item_payload_fallback_recursive():
    data = {"nested": {"itemInfo": {"itemStruct": {"id": "456"}}}}
    payload = get_video_item_payload(data)
    assert payload is not None
    assert payload.get("itemInfo", {}).get("itemStruct", {}).get("id") == "456"


def test_get_video_item_payload_missing():
    assert get_video_item_payload({}) is None
    assert get_video_item_payload({"__DEFAULT_SCOPE__": {}}) is None


# ---------------------------------------------------------------------------
# parse_video_from_json
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_item_struct():
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
            "duration": 15,
            "cover": "https://example.com/cover.jpg",
        },
        "music": {
            "id": "music1",
            "title": "Song",
            "authorName": "Artist",
            "playUrl": "https://example.com/audio.mp3",
            "original": False,
        },
        "challenges": [{"id": "c1", "title": "fyp"}, {"id": "c2", "title": "viral"}],
        "authorStats": {"followerCount": 100000, "followingCount": 500, "heartCount": 50000},
    }


def test_parse_video_from_json(sample_item_struct):
    author, video, snapshot = parse_video_from_json(
        sample_item_struct,
        "https://www.tiktok.com/@cooluser/video/7123456789",
        position=1,
    )
    assert author.author_id == "author1"
    assert author.username == "cooluser"
    assert author.display_name == "Cool User"
    assert author.verified is True

    assert video.video_id == "7123456789"
    assert video.author_id == "author1"
    assert video.caption == "Check this out #viral #fyp"
    assert video.hashtags == ["fyp", "viral"]
    assert "Song" in video.audio_name
    assert video.likes == 1200  # statsV2 "1.2K"
    assert video.comments_count == 50
    assert video.plays == 12500  # statsV2 "12.5K"

    assert snapshot.author_id == "author1"
    assert snapshot.video_id == "7123456789"
    assert snapshot.follower_count == 100000


# ---------------------------------------------------------------------------
# Storage (LocalStorage)
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp) / "test.db"


def test_storage_save_and_exists(temp_db):
    storage = LocalStorage(temp_db)
    author = Author(author_id="a1", username="user1")
    video = Video(video_id="v1", author_id="a1", scraped_at="2025-01-01T00:00:00Z", caption="Hi")

    storage.save_author(author)
    storage.save_video(video)

    assert storage.video_exists("v1")
    assert not storage.video_exists("v2")


def test_storage_save_comments(temp_db):
    storage = LocalStorage(temp_db)
    video = Video(video_id="v1", author_id="a1", scraped_at="2025-01-01T00:00:00Z")
    storage.save_video(video)

    comments = [
        Comment(comment_id="c1", video_id="v1", text="Nice", likes=10, scraped_at="2025-01-01T00:00:00Z"),
        Comment(comment_id="c2", video_id="v1", text="Cool", likes=5, scraped_at="2025-01-01T00:00:00Z"),
    ]
    storage.save_comments("v1", comments)
    # No assertion on read-back (no get_comments in interface); just ensure no error
    assert storage.video_exists("v1")


def test_storage_checkpoint(temp_db):
    storage = LocalStorage(temp_db)
    assert storage.get_checkpoint("run1") is None

    storage.save_checkpoint("run1", "v99", 42)
    cp = storage.get_checkpoint("run1")
    assert cp is not None
    assert cp["last_video_id"] == "v99"
    assert cp["last_position"] == 42


def test_storage_author_snapshot(temp_db):
    storage = LocalStorage(temp_db)
    storage.save_author(Author(author_id="a1", username="u1"))
    storage.save_video(Video(video_id="v1", author_id="a1", scraped_at="2025-01-01T00:00:00Z"))

    snapshot = AuthorMetricSnapshot(
        author_id="a1",
        video_id="v1",
        scraped_at="2025-01-01T00:00:00Z",
        follower_count=1000,
    )
    storage.save_author_snapshot(snapshot)
    # No error = success
    assert storage.video_exists("v1")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = PipelineConfig()
    assert cfg.headless is True
    assert cfg.top_comments_n == 20
    assert cfg.data_dir == Path("data/tiktok")
    assert cfg.db_path == Path("data/tiktok/scrape.db")


def test_config_custom_data_dir():
    cfg = PipelineConfig(data_dir=Path("/tmp/tiktok"))
    assert cfg.data_dir == Path("/tmp/tiktok")
    assert "scrape.db" in str(cfg.db_path)

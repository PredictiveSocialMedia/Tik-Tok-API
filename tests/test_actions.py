"""
Unit tests for actions.py.

Focuses on pure parsing and extraction logic so tests are fast and don't
depend on the network or a real browser. Scraper-specific tests that need
a driver use mocks (see test_scraper.py).
"""

import json
from unittest.mock import MagicMock

import pytest

from actions import (
    extract_rehydration_json,
    find_in_dict,
    get_comments_from_data,
    get_video_item_payload,
    parse_comment,
    parse_count,
    parse_video_item,
    safe_get,
)


# ---------------------------------------------------------------------------
# parse_count
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
        ("276.8K likes", 276800),  # parse_count only sees the number part via regex
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


def test_extract_rehydration_json(sample_rehydration_html):
    data = extract_rehydration_json(sample_rehydration_html)
    assert data is not None
    assert "__DEFAULT_SCOPE__" in data
    scope = data["__DEFAULT_SCOPE__"]
    assert "webapp.video-detail" in scope
    assert "itemInfo" in scope["webapp.video-detail"]


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
# get_comments_from_data
# ---------------------------------------------------------------------------


def test_get_comments_from_data():
    data = {"some": {"comments": [{"text": "Hi", "cid": "1"}]}}
    comments = get_comments_from_data(data)
    assert len(comments) == 1
    assert comments[0]["text"] == "Hi"


def test_get_comments_from_data_empty():
    assert get_comments_from_data({}) == []
    assert get_comments_from_data({"other": []}) == []


# ---------------------------------------------------------------------------
# parse_video_item
# ---------------------------------------------------------------------------


def test_parse_video_item(sample_video_item_struct):
    parsed = parse_video_item(sample_video_item_struct)
    assert parsed["id"] == "7123456789"
    assert parsed["caption"] == "Check this out #viral #fyp"
    assert parsed["author"]["uniqueId"] == "cooluser"
    assert parsed["stats"]["playCount"] == 12500  # statsV2 "12.5K"
    assert parsed["stats"]["likeCount"] == 1200   # statsV2 "1.2K"
    assert parsed["video"]["playUrl"] == "https://example.com/video.mp4"
    assert parsed["music"]["title"] == "Song"
    assert len(parsed["hashtags"]) == 2
    assert parsed["locationCreated"] == "US"


def test_parse_video_item_minimal():
    minimal = {"id": "1", "desc": "Hi", "author": {}, "video": {}, "music": {}}
    parsed = parse_video_item(minimal)
    assert parsed["id"] == "1"
    assert parsed["caption"] == "Hi"
    assert parsed["stats"]["playCount"] == 0
    assert parsed["authorStats"]["followerCount"] == 0


# ---------------------------------------------------------------------------
# parse_comment
# ---------------------------------------------------------------------------


def test_parse_comment(sample_comment_dict):
    parsed = parse_comment(sample_comment_dict)
    assert parsed["id"] == "comment_abc"
    assert parsed["text"] == "Great video!"
    assert parsed["username"] == "viewer1"
    assert parsed["likeCount"] == 10
    assert parsed["replyCount"] == 2
    assert parsed["createTime"] == 1700000100


def test_parse_comment_nested_text():
    raw = {"cid": "1", "comment": {"text": "Nested"}, "user": {}}
    assert parse_comment(raw)["text"] == "Nested"

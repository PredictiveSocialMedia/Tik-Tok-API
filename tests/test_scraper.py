"""
Tests for scraper.py using a mocked Chrome driver.

Avoids real browser and network: we inject fixture HTML as page source and
mock DOM comment scraping so the full scrape_tiktok_post flow can be tested.
"""

from unittest.mock import MagicMock, patch

import pytest

from scraper import scrape_tiktok_post


@pytest.fixture
def fixture_html():
    """HTML that contains valid __DEFAULT_SCOPE__ / video-detail / itemStruct."""
    data = (
        '{"__DEFAULT_SCOPE__":{"webapp.video-detail":{'
        '"itemInfo":{"itemStruct":{'
        '"id":"999","desc":"Test caption","createTime":1700000000,'
        '"author":{"id":"a1","uniqueId":"testuser","nickname":"Test User","verified":false},'
        '"stats":{"playCount":5000,"diggCount":200,"commentCount":8},'
        '"video":{"playAddr":"https://example.com/v.mp4","duration":10},'
        '"music":{"title":"Test Song","authorName":"Artist","playUrl":"https://example.com/a.mp3"}'
        "}}}}}"
    )
    return f'<html><script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">{data}</script></html>'


def test_scraper_success_with_mocked_driver(fixture_html):
    """Run full scrape_tiktok_post with driver that returns fixture HTML and no DOM comments."""
    mock_driver = MagicMock()
    mock_driver.page_source = fixture_html
    mock_driver.find_elements.return_value = []  # no comment wrappers

    with patch("scraper.load_tiktok_video_page", return_value=fixture_html):
        result = scrape_tiktok_post(
            "https://www.tiktok.com/@testuser/video/999",
            driver=mock_driver,
            max_comments=5,
        )

    assert result["success"] is True
    assert result["url"] == "https://www.tiktok.com/@testuser/video/999"
    assert result["caption"] == "Test caption"
    assert result["metadata"]["id"] == "999"
    assert result["metadata"]["author"]["uniqueId"] == "testuser"
    assert result["metadata"]["stats"]["playCount"] == 5000
    assert result["video"]["playUrl"] == "https://example.com/v.mp4"
    assert result["music"]["title"] == "Test Song"
    assert result["top_comments"] == []  # we mocked no DOM comments


def test_scraper_fails_gracefully_when_no_rehydration():
    """When page has no rehydration JSON, scraper returns success=False with error."""
    mock_driver = MagicMock()
    mock_driver.page_source = "<html><body>No script here</body></html>"

    with patch("scraper.load_tiktok_video_page", return_value=mock_driver.page_source):
        result = scrape_tiktok_post(
            "https://www.tiktok.com/@x/video/1",
            driver=mock_driver,
        )

    assert result["success"] is False
    assert "error" in result
    assert "rehydration" in result["error"].lower() or "video" in result["error"].lower()

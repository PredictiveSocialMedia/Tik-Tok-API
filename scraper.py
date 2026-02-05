"""
TikTok Post Scraper

Scrapes a single TikTok video page for:
- Caption / description and hashtags
- Video URL, cover image, and optional download
- Audio / music metadata and optional download
- Full metadata (likes, comments count, shares, saves, views, author stats)
- Top comments (from embedded JSON + live DOM fallback)

Uses Selenium to render the page and extracts data from TikTok's embedded
``__UNIVERSAL_DATA_FOR_REHYDRATION__`` JSON (no official API required).
"""

import json
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from selenium import webdriver

from actions import (
    create_driver,
    extract_rehydration_json,
    get_comments_from_data,
    get_comments_from_dom,
    get_video_item_payload,
    load_tiktok_video_page,
    parse_comment,
    parse_video_item,
)


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

_EMPTY_RESULT: dict = {
    "success": False,
    "error": None,
    "url": None,
    "caption": None,
    "metadata": None,
    "video": None,
    "music": None,
    "hashtags": [],
    "top_comments": [],
}


def _fail(url: str, reason: str) -> dict:
    return {**_EMPTY_RESULT, "url": url, "error": reason}


def scrape_tiktok_post(
    video_url: str,
    *,
    driver: Optional[webdriver.Chrome] = None,
    headless: bool = True,
    max_comments: int = 10,
    download_video_path: Optional[str] = None,
    download_audio_path: Optional[str] = None,
) -> dict:
    """
    Scrape a single TikTok post by URL.

    Parameters
    ----------
    video_url : str
        Full URL, e.g. ``https://www.tiktok.com/@user/video/123``
    driver : webdriver.Chrome, optional
        Re-use an existing browser session.
    headless : bool
        Launch headless if creating a new driver.
    max_comments : int
        Maximum top-level comments to return.
    download_video_path / download_audio_path : str, optional
        Save video / audio to these file paths.

    Returns
    -------
    dict
        Structured result with keys: ``success``, ``url``, ``caption``,
        ``metadata``, ``video``, ``music``, ``hashtags``, ``top_comments``,
        and optional download status fields.
    """
    own_driver = driver is None
    if own_driver:
        driver = create_driver(headless=headless)

    try:
        # ── Load page & extract JSON ────────────────────────────────
        html = load_tiktok_video_page(driver, video_url)
        data = extract_rehydration_json(html)
        if not data:
            return _fail(video_url, "Could not find __UNIVERSAL_DATA_FOR_REHYDRATION__ on page")

        payload = get_video_item_payload(data)
        if not payload:
            return _fail(video_url, "Could not locate video item in page data")

        item_info = payload.get("itemInfo") or {}
        item = item_info.get("itemStruct")
        if not item and payload.get("itemList"):
            item = payload["itemList"][0]
        if not item:
            return _fail(video_url, "Video itemStruct not found in payload")

        # ── Parse structured data ───────────────────────────────────
        parsed = parse_video_item(item)

        # ── Gather comments ─────────────────────────────────────────
        top_comments: list[dict] = []

        # 1) From embedded JSON (usually empty on desktop, but cheap to try)
        json_comments = get_comments_from_data(data)[:max_comments]
        top_comments = [
            parse_comment(c) for c in json_comments if isinstance(c, dict)
        ]

        # 2) From the live DOM if we still need more
        if len(top_comments) < max_comments:
            dom_comments = get_comments_from_dom(
                driver, max_comments - len(top_comments)
            )
            seen_texts = {c.get("text", "") for c in top_comments}
            for c in dom_comments:
                if len(top_comments) >= max_comments:
                    break
                if c.get("text", "") in seen_texts:
                    continue
                seen_texts.add(c["text"])
                top_comments.append(c)

        # ── Build result ────────────────────────────────────────────
        result: dict = {
            "success": True,
            "url": video_url,
            "caption": parsed["caption"],
            "metadata": {
                "id": parsed["id"],
                "createTime": parsed["createTime"],
                "author": parsed["author"],
                "stats": parsed["stats"],
                "authorStats": parsed["authorStats"],
                "locationCreated": parsed["locationCreated"],
                "diversificationLabels": parsed["diversificationLabels"],
            },
            "video": parsed["video"],
            "music": parsed["music"],
            "hashtags": parsed["hashtags"],
            "top_comments": top_comments,
        }

        # ── Optional downloads ──────────────────────────────────────
        if download_video_path and parsed["video"].get("playUrl"):
            try:
                urlretrieve(parsed["video"]["playUrl"], download_video_path)
                result["video_downloaded"] = download_video_path
            except Exception as exc:
                result["video_download_error"] = str(exc)

        if download_audio_path and parsed["music"].get("playUrl"):
            try:
                urlretrieve(parsed["music"]["playUrl"], download_audio_path)
                result["audio_downloaded"] = download_audio_path
            except Exception as exc:
                result["audio_download_error"] = str(exc)

        return result

    finally:
        if own_driver and driver:
            driver.quit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape a TikTok post (caption, metadata, audio, top comments)"
    )
    parser.add_argument(
        "url",
        help="TikTok video URL (e.g. https://www.tiktok.com/@user/video/123)",
    )
    parser.add_argument("--no-headless", action="store_true", help="Show the browser window")
    parser.add_argument(
        "--comments", type=int, default=5, help="Max top comments to fetch (default 5)"
    )
    parser.add_argument("--download-video", metavar="PATH", help="Download video to PATH")
    parser.add_argument("--download-audio", metavar="PATH", help="Download audio/music to PATH")
    parser.add_argument("--output", "-o", metavar="FILE", help="Write JSON result to FILE")
    args = parser.parse_args()

    result = scrape_tiktok_post(
        args.url,
        headless=not args.no_headless,
        max_comments=args.comments,
        download_video_path=args.download_video,
        download_audio_path=args.download_audio,
    )

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        print(f"Wrote result to {args.output}")
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

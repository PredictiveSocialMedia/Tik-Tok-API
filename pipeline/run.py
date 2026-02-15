#!/usr/bin/env python3
"""
TikTok scraping pipeline — main entry point.

Usage:
    python -m pipeline.run
    python -m pipeline.run --max-videos 50 --no-headless

Flow:
    1. Create browser, load cookies or wait for login
    2. Solve captcha if present
    3. Iterate For You feed
    4. For each video: parse metadata + comments, store (3NF)
    5. Checkpoint for resumability
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.browser import create_driver, load_cookies, save_cookies
from pipeline.captcha import solve_captcha
from pipeline.config import PipelineConfig
from pipeline.feed import iterate_for_you_feed
from pipeline.parser import parse_comments_from_dom, parse_video_page
from pipeline.storage import LocalStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run(config: PipelineConfig) -> int:
    """
    Run the pipeline. Returns number of videos processed.
    """
    driver = create_driver(
        headless=config.headless,
        window_width=config.window_width,
        window_height=config.window_height,
    )

    storage = LocalStorage(config.db_path)
    run_id = "for_you"  # Fixed ID for resumability

    try:
        # ── Auth ─────────────────────────────────────────────────
        driver.get(config.for_you_url)
        time.sleep(3)

        if config.cookies_path and config.cookies_path.exists():
            load_cookies(driver, config.cookies_path)
            driver.refresh()
            time.sleep(2)

        if "login" in driver.current_url.lower():
            logger.info("On login page; solve captcha or log in manually (60s)...")
            solve_captcha(driver, max_attempts=config.captcha_max_attempts)
            time.sleep(2)
            if "login" in driver.current_url.lower():
                time.sleep(55)  # Wait for manual login
            driver.get(config.for_you_url)
            time.sleep(3)
            if config.cookies_path:
                save_cookies(driver, config.cookies_path)

        solve_captcha(driver, max_attempts=2)
        time.sleep(2)

        # ── Feed iteration ───────────────────────────────────────
        videos_processed = 0
        position = 0

        for video_url in iterate_for_you_feed(
            driver,
            for_you_url=config.for_you_url,
            scroll_delay_sec=config.scroll_delay_sec,
            max_videos=config.max_videos_per_run,
            delay_between_scrolls_sec=config.delay_between_scrolls_sec,
        ):
            if config.max_videos_per_run and videos_processed >= config.max_videos_per_run:
                break

            position += 1
            video_id = ""
            if "/video/" in video_url:
                parts = video_url.rstrip("/").split("/video/")
                if len(parts) >= 2:
                    video_id = parts[-1].split("?")[0]

            if not video_id:
                continue

            if storage.video_exists(video_id):
                logger.debug("Skip existing video %s", video_id)
                continue

            # ── Parse video ──────────────────────────────────────
            result = parse_video_page(driver, video_url, position=position, wait_sec=6)
            if not result:
                logger.warning("Failed to parse %s", video_url[:60])
                time.sleep(config.delay_between_videos_sec)
                continue

            author, video, snapshot = result
            video_id = video.video_id  # Use canonical ID from JSON

            # ── Parse comments ────────────────────────────────────
            comments = parse_comments_from_dom(
                driver,
                video_id=video_id,
                top_n=config.top_comments_n,
                highest_liked_k=config.highest_liked_subset_k,
                replies_per_comment_m=config.replies_per_comment_m,
            )

            # ── Store ─────────────────────────────────────────────
            storage.save_author(author)
            storage.save_video(video)
            storage.save_author_snapshot(snapshot)
            storage.save_comments(video_id, comments)
            storage.save_checkpoint(run_id, video_id, position)

            videos_processed += 1
            logger.info("Stored video %s (%d comments) [%d total]", video_id, len(comments), videos_processed)

            # ── Captcha check ─────────────────────────────────────
            solve_captcha(driver, max_attempts=1)

            delay = config.delay_between_videos_sec + random.uniform(0, 1.0)
            time.sleep(delay)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        driver.quit()

    return videos_processed


def main() -> None:
    parser = argparse.ArgumentParser(description="TikTok For You feed scraper")
    parser.add_argument("--max-videos", type=int, default=None, help="Max videos per run")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tiktok"), help="Data directory")
    parser.add_argument("--top-comments", type=int, default=20, help="Top N comments per video")
    parser.add_argument("--replies-per-comment", type=int, default=5, help="Top M replies per comment")
    args = parser.parse_args()

    config = PipelineConfig(
        headless=not args.no_headless,
        max_videos_per_run=args.max_videos,
        data_dir=args.data_dir,
        top_comments_n=args.top_comments,
        replies_per_comment_m=args.replies_per_comment,
    )

    n = run(config)
    logger.info("Pipeline finished. Processed %d videos.", n)


if __name__ == "__main__":
    main()

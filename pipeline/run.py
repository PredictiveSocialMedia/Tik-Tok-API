#!/usr/bin/env python3
"""
TikTok scraping pipeline — main entry point.

Usage::

    python -m pipeline.run
    python -m pipeline.run --max-videos 50 --no-headless

Flow:
    1. Check rate-limit cooldown (skip if TikTok blocked us recently).
    2. Create browser (undetected-chromedriver + stealth scripts).
    3. Load cookies or auto-login with human-like interactions.
    4. Solve captcha if present.
    5. Iterate For You feed.
    6. For each video: parse metadata + comments, store (3NF).
    7. Checkpoint for resumability.

Anti-detection measures are applied automatically via
``pipeline.browser.driver`` (undetected-chromedriver, stealth JS) and
``pipeline.browser.stealth`` (human typing, jittered sleeps, cooldown file).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selenium.common.exceptions import NoSuchWindowException  # type: ignore[import-untyped]

from pipeline.auth import dismiss_overlays, is_login_button_visible, login_with_credentials
from pipeline.browser import create_driver, load_cookies, save_cookies
from pipeline.browser.stealth import clear_rate_limit, human_sleep, is_rate_limited
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
    Run the TikTok scraping pipeline.

    Checks for an active rate-limit cooldown before starting. If a
    cooldown is active (from a previous "Maximum attempts" block),
    the pipeline exits immediately with ``0`` videos processed.

    Returns:
        Number of videos successfully processed.
    """
    # --- Rate-limit guard ---
    if is_rate_limited():
        logger.warning("Exiting: rate-limit cooldown is active.")
        return 0

    driver = create_driver(
        headless=config.headless,
        window_width=config.window_width,
        window_height=config.window_height,
    )

    storage = LocalStorage(config.db_path)
    run_id = "for_you"
    videos_processed = 0

    try:
        # ── Auth ─────────────────────────────────────────────────
        driver.get(config.for_you_url)
        human_sleep(3.0, jitter=1.0)
        dismiss_overlays(driver)
        human_sleep(1.5, jitter=0.5)

        if config.cookies_path and config.cookies_path.exists():
            load_cookies(driver, config.cookies_path)
            driver.refresh()
            human_sleep(2.0, jitter=0.5)
            dismiss_overlays(driver)
            human_sleep(1.0, jitter=0.3)

        on_login_page = "login" in driver.current_url.lower()
        login_button_visible = is_login_button_visible(driver, timeout=8.0)
        on_foryou = (
            "foryou" in driver.current_url.lower()
            or "tiktok.com" in driver.current_url
        )
        need_login = (
            on_login_page
            or login_button_visible
            or (config.has_credentials() and on_foryou)
        )
        logger.info(
            "Auth check: on_login_page=%s, login_button_visible=%s, on_foryou=%s",
            on_login_page,
            login_button_visible,
            on_foryou,
        )

        if need_login:
            if config.has_credentials() and config.tiktok_email and config.tiktok_password:
                logger.info("Login required; logging in with TIKTOK_EMAIL / TIKTOK_PASSWORD...")
                ok = login_with_credentials(
                    driver,
                    email=config.tiktok_email,
                    password=config.tiktok_password,
                    login_url=config.login_url,
                )
                if ok:
                    clear_rate_limit()
                else:
                    logger.warning("Auto-login failed; waiting for manual login (60s)...")
                    human_sleep(60.0, jitter=5.0)
            else:
                logger.info("On login page; solve captcha or log in manually (60s)...")
                solve_captcha(driver, max_attempts=config.captcha_max_attempts)
                human_sleep(2.0, jitter=0.5)
                if "login" in driver.current_url.lower():
                    human_sleep(55.0, jitter=5.0)

            driver.get(config.for_you_url)
            human_sleep(3.0, jitter=1.0)
            if config.cookies_path:
                save_cookies(driver, config.cookies_path)

        solve_captcha(driver, max_attempts=2)
        human_sleep(2.0, jitter=0.5)

        # ── Feed iteration ───────────────────────────────────────
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
                human_sleep(config.delay_between_videos_sec, jitter=1.0)
                continue

            author, video, snapshot = result
            video_id = video.video_id

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
            logger.info(
                "Stored video %s (%d comments) [%d total]",
                video_id,
                len(comments),
                videos_processed,
            )

            # ── Captcha check ─────────────────────────────────────
            solve_captcha(driver, max_attempts=1)

            human_sleep(config.delay_between_videos_sec, jitter=1.0)

    except NoSuchWindowException:
        logger.warning("Browser window was closed; stopping pipeline.")
        return videos_processed
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        try:
            driver.quit()
        except NoSuchWindowException:
            pass
        except Exception:
            pass

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

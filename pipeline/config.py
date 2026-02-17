"""
Pipeline configuration and constants.

Override via environment variables or by passing a config dict to the orchestrator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the TikTok scraping pipeline."""

    # Browser
    headless: bool = True
    window_width: int = 1920
    window_height: int = 1080
    page_load_timeout_sec: float = 30.0

    # Auth / session
    cookies_path: Optional[Path] = None
    login_url: str = "https://www.tiktok.com/login"
    for_you_url: str = "https://www.tiktok.com/foryou"
    # Credentials from env (TIKTOK_EMAIL, TIKTOK_PASSWORD) — optional; when set, auto-login
    tiktok_email: Optional[str] = None
    tiktok_password: Optional[str] = None

    # Feed
    scroll_delay_sec: float = 1.5
    videos_per_batch: int = 10
    max_videos_per_run: Optional[int] = None  # None = no limit

    # Parsing
    top_comments_n: int = 20
    highest_liked_subset_k: int = 5
    replies_per_comment_m: int = 5

    # Captcha
    captcha_max_attempts: int = 3
    captcha_cooldown_sec: float = 60.0  # Pause after N failed attempts

    # Storage
    storage_backend: str = "local"  # "local" | "rest" (future)
    data_dir: Path = field(default_factory=lambda: Path("data/tiktok"))
    db_path: Optional[Path] = None  # SQLite; default = data_dir / "scrape.db"

    # Rate limiting
    delay_between_videos_sec: float = 2.0
    delay_between_scrolls_sec: float = 1.0

    def __post_init__(self) -> None:
        if self.cookies_path is None:
            self.cookies_path = self.data_dir / "cookies.json"
        if self.db_path is None:
            self.db_path = self.data_dir / "scrape.db"
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Load credentials from env if not explicitly set
        if self.tiktok_email is None:
            self.tiktok_email = os.environ.get("TIKTOK_EMAIL") or None
        if self.tiktok_password is None:
            self.tiktok_password = os.environ.get("TIKTOK_PASSWORD") or None

    def has_credentials(self) -> bool:
        """True if both email and password are set (for auto-login)."""
        return bool(self.tiktok_email and self.tiktok_password)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build config from environment variables."""
        data_dir = os.environ.get("TIKTOK_DATA_DIR", "data/tiktok")
        return cls(
            headless=os.environ.get("TIKTOK_HEADLESS", "true").lower() == "true",
            data_dir=Path(data_dir),
            top_comments_n=int(os.environ.get("TIKTOK_TOP_COMMENTS_N", "20")),
            highest_liked_subset_k=int(os.environ.get("TIKTOK_HIGHEST_LIKED_K", "5")),
            replies_per_comment_m=int(os.environ.get("TIKTOK_REPLIES_PER_COMMENT_M", "5")),
            max_videos_per_run=int(os.environ["TIKTOK_MAX_VIDEOS"]) if os.environ.get("TIKTOK_MAX_VIDEOS") else None,
        )

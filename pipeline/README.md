# TikTok Scraping Pipeline

Browser-based pipeline for scraping the **For You** feed: videos, metadata, comments, and author metrics. Designed for local storage first, REST API later.

## Architecture

```
pipeline/
├── run.py              # Main entry point (orchestrator)
├── config.py           # Configuration
├── auth/               # Login, session, cookies
├── browser/            # WebDriver setup
├── captcha/            # Detection + routing to solvers
├── feed/               # For You feed iteration
├── parser/             # Video + comment extraction (3NF)
└── storage/            # Payload-based persistence (SQLite)
```

## Quick Start

```bash
# From project root
python -m pipeline.run

# With options
python -m pipeline.run --max-videos 20 --no-headless --top-comments 30
```

## Prerequisites

- Chrome + ChromeDriver (or compatible)
- **Login**: either
  - Set `TIKTOK_EMAIL` and `TIKTOK_PASSWORD` env vars (auto-login), or
  - Place cookies at `data/tiktok/cookies.json` (from a previous run), or
  - Run with `--no-headless` and log in manually when prompted

## Data Schema (3NF)

- **authors**: author_id, username, display_name, bio, avatar_url, verified
- **videos**: video_id, author_id, caption, hashtags, audio_name, likes, comments_count, etc.
- **author_metric_snapshots**: author_id, video_id, scraped_at, follower_count (for virality/controversy)
- **comments**: comment_id, video_id, text, likes, reply_count, parent_comment_id

## Captcha Handling

The pipeline detects and routes to:

- **Slider puzzle** → `slider_puzzle.solve_slider_puzzle`
- **Rotation** → `rotation_captcha.solve_rotation_captcha`
- **Object selection** → `object_selection_captcha.handle_captcha`

## Storage

- **Local**: SQLite at `data/tiktok/scrape.db`
- **REST**: Implement `StorageBackend` and POST payloads to your API (future)

## Configuration

| Env / Arg | Default | Description |
|-----------|---------|-------------|
| `--max-videos` | None | Max videos per run |
| `--no-headless` | False | Show browser |
| `--data-dir` | data/tiktok | Data directory |
| `--top-comments` | 20 | Top N comments per video |
| `--replies-per-comment` | 5 | Top M replies per comment |
| `TIKTOK_EMAIL` | (env) | Email or username for auto-login |
| `TIKTOK_PASSWORD` | (env) | Password for auto-login |

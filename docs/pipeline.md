# TikTok Scraping Pipeline

## Overview

The pipeline scrapes the **For You** feed: videos, metadata, top comments, and author metrics. It uses a **browser** for login and captcha handling, stores data in **3NF** (authors, videos, author_metric_snapshots, comments), and is designed for **payload-based storage** (local SQLite now, REST API later).

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
│ Auth/Login  │──►│ Feed (For You)│──►│ Video Parse │──►│ Storage      │
│ + cookies   │   │ scroll       │   │ + comments  │   │ (SQLite)     │
└─────────────┘   └──────────────┘   └─────────────┘   └──────────────┘
       │                   │
       ▼                   ▼
  [Captcha detector + router → rotation_captcha, slider_puzzle, object_selection_captcha]
```

## Components

| Component | Purpose |
|-----------|---------|
| **browser** | WebDriver setup, cookie load/save |
| **auth** | Login flow, session reuse |
| **captcha** | Detect type, route to solver (slider, rotation, object selection) |
| **feed** | Iterate For You feed, yield video URLs |
| **parser** | Extract video + comments from rehydration JSON and DOM |
| **storage** | 3NF schema, LocalStorage (SQLite), REST-ready interface |

## Schema (3NF)

- **authors**: author_id, username, display_name, bio, avatar_url, verified
- **videos**: video_id, author_id, scraped_at, caption, hashtags, audio_name, likes, comments_count, etc.
- **author_metric_snapshots**: author_id, video_id, scraped_at, follower_count (for virality/controversy analysis)
- **comments**: comment_id, video_id, text, likes, reply_count, parent_comment_id

## Usage

```bash
python -m pipeline.run
python -m pipeline.run --max-videos 50 --no-headless --top-comments 30
```

## Configuration

See `pipeline/config.py` and `pipeline/README.md`. Key options: `--max-videos`, `--no-headless`, `--data-dir`, `--top-comments`, `--replies-per-comment`.

## See also

- [Rotation Captcha](rotation-captcha.md)
- [Slider Puzzle](slider-puzzle.md)
- [Object Selection Captcha](object-selection-captcha.md)

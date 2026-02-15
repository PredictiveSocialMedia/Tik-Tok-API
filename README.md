# TikTok Scraping Pipeline

A Python pipeline for scraping the **For You** feed: videos, metadata, top comments, and author metrics. Uses a browser for login and captcha handling, stores data in 3NF (SQLite), and is designed for REST API later.

## Features

- 📺 **For You feed** — Iterate and scrape videos at scale
- 📝 **Video metadata** — Caption, hashtags, audio name, likes, plays, shares
- 💬 **Top comments** — Configurable top N + highest-liked subset
- 📊 **Author metrics** — Follower count snapshots for virality/controversy analysis
- 🤖 **Captcha handling** — Auto-solves slider, rotation, and object-selection captchas
- 💾 **3NF storage** — authors, videos, author_metric_snapshots, comments

## Project layout

- **`pipeline/`** — Main scraping pipeline. Run with `python -m pipeline.run`. See [pipeline/README.md](pipeline/README.md) and [docs/pipeline.md](docs/pipeline.md).
- **`object_selection_captcha/`** — Shape-matching CAPTCHA solver (TikTok grid/click).
- **`rotation_captcha/`** — Rotation captcha solver (align inner circle). See [docs/rotation-captcha.md](docs/rotation-captcha.md).
- **`slider_puzzle/`** — Slider puzzle solver (drag piece into gap). See [docs/slider-puzzle.md](docs/slider-puzzle.md).
- **`funcaptcha/`** — FunCaptcha solvers (cycle match, quantity, dice, rotation).
- **`tests/`** — Pytest tests.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install ChromeDriver:
   - **macOS**: `brew install chromedriver`
   - **Linux**: Download from [ChromeDriver downloads](https://chromedriver.chromium.org/downloads)
   - **Windows**: Download from [ChromeDriver downloads](https://chromedriver.chromium.org/downloads)

   Make sure ChromeDriver is in your PATH.

## Usage

### Pipeline (For You feed)

```bash
python -m pipeline.run
```

With options:
```bash
# Limit videos and show browser
python -m pipeline.run --max-videos 50 --no-headless

# More comments per video
python -m pipeline.run --top-comments 30 --replies-per-comment 5
```

### Session / cookies

Place saved cookies at `data/tiktok/cookies.json` to reuse a session, or run with `--no-headless` and log in manually when prompted.

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests cover: pipeline (storage, parser, config), rotation_captcha, slider_puzzle, funcaptcha, object_selection_captcha. No browser or network required for unit tests.

## CAPTCHA solvers

The pipeline auto-solves:

- **Slider puzzle** — Drag the piece into the gap (`slider_puzzle`)
- **Rotation** — Align the inner circle (`rotation_captcha`)
- **Object selection** — Shape-matching grid (`object_selection_captcha`)

For object selection, train a model with:
```bash
python -m object_selection_captcha.train_tiktok_model --epochs 100
```

Model location: `object_selection_captcha/models/tiktok_captcha_best.pt`

## Documentation

- [Pipeline](docs/pipeline.md)
- [Rotation captcha](docs/rotation-captcha.md)
- [Slider puzzle](docs/slider-puzzle.md)
- [Object selection captcha](docs/object-selection-captcha.md)
- [Funcaptcha](docs/funcaptcha-rotation.md) (and related)

## Requirements

- Python 3.8+
- Chrome browser
- ChromeDriver (must match your Chrome version)

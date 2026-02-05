# TikTok Scraper

A Python scraper for extracting data from TikTok video pages, including captions, metadata, video/audio URLs, and top comments.

## Features

- 📝 Extract video captions and descriptions
- 📊 Get metadata (likes, comments, shares, views, author info)
- 🎥 Retrieve video URLs and optionally download videos
- 🎵 Extract audio/music metadata and optionally download audio
- 💬 Scrape top comments with like/reply counts
- 🤖 Uses Selenium to handle dynamic content loading

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

### Command Line

Basic usage:
```bash
python scraper.py "https://www.tiktok.com/@username/video/1234567890"
```

With options:
```bash
# Show browser window (useful for debugging)
python scraper.py "https://www.tiktok.com/@username/video/1234567890" --no-headless

# Get more comments
python scraper.py "https://www.tiktok.com/@username/video/1234567890" --comments 20

# Download video and audio
python scraper.py "https://www.tiktok.com/@username/video/1234567890" \
    --download-video video.mp4 \
    --download-audio audio.mp3

# Save output to JSON file
python scraper.py "https://www.tiktok.com/@username/video/1234567890" \
    --output result.json
```

### Python API

```python
from scraper import scrape_tiktok_post

# Basic usage
result = scrape_tiktok_post("https://www.tiktok.com/@username/video/1234567890")

# With options
result = scrape_tiktok_post(
    "https://www.tiktok.com/@username/video/1234567890",
    headless=False,
    max_comments=20,
    download_video_path="video.mp4",
    download_audio_path="audio.mp3"
)

print(result["caption"])
print(result["metadata"]["stats"])
print(result["top_comments"])
```

## Output Format

The scraper returns a dictionary with the following structure:

```json
{
  "success": true,
  "url": "https://www.tiktok.com/@username/video/1234567890",
  "caption": "Video caption text...",
  "metadata": {
    "id": "1234567890",
    "createTime": 1234567890,
    "author": {
      "uniqueId": "username",
      "nickname": "Display Name",
      "verified": false
    },
    "stats": {
      "playCount": 1000000,
      "likeCount": 50000,
      "commentCount": 1000,
      "shareCount": 500
    }
  },
  "video": {
    "playUrl": "https://...",
    "downloadUrl": "https://...",
    "duration": 30,
    "width": 1080,
    "height": 1920
  },
  "music": {
    "id": "1234567890",
    "title": "Song Title",
    "authorName": "Artist Name",
    "playUrl": "https://..."
  },
  "top_comments": [
    {
      "id": "comment_id",
      "text": "Comment text...",
      "likeCount": 100,
      "replyCount": 5,
      "createTime": 1234567890
    }
  ]
}
```

## Testing

Tests use **pytest** and avoid the network and a real browser where possible:

- **Unit tests** (`tests/test_actions.py`): `parse_count`, `safe_get`, `find_in_dict`, `extract_rehydration_json`, `get_video_item_payload`, `get_comments_from_data`, `parse_video_item`, `parse_comment` — all run against fixture data.
- **Scraper tests** (`tests/test_scraper.py`): `scrape_tiktok_post` is exercised with a **mocked** Chrome driver and fixture HTML, so no live TikTok or Selenium is required.

Run tests:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Scraper tests are inherently brittle (DOM/HTML can change); the suite is designed so parsing and business logic are well covered by unit tests, and only the orchestration is tested with mocks.

## Notes

- TikTok may show a slider puzzle captcha. If this happens, run with `--no-headless` and solve it manually once.
- The scraper handles cookie banners automatically.
- Comments are loaded dynamically, so the scraper scrolls and waits for them to appear.
- Video and audio URLs may expire after some time.

## Requirements

- Python 3.8+
- Chrome browser
- ChromeDriver (must match your Chrome version)
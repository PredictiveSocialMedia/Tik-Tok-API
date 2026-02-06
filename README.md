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

## CAPTCHA solver (object selection)

The scraper can try to **auto-solve** image-based “select all X” CAPTCHAs using a local YOLOv8 model (no external API). It is **optional** and only runs if the `captcha` package and its dependencies are installed.

### How powerful is it?

| Factor | Reality |
|--------|--------|
| **Object coverage** | Only the **80 COCO classes** (traffic lights, buses, bicycles, cars, fire hydrants, stop signs, etc.). Prompts like “crosswalk”, “chimney”, “palm tree” have **no** mapping and will fail unless we add heuristics or CLIP. |
| **Success rate (when it applies)** | On **grid** CAPTCHAs with clear images and a known prompt, expect roughly **50–80%** per attempt: YOLO can miss small or occluded objects, and tile boundaries / IoU thresholds can misassign. **Click** CAPTCHAs (single image, “click the X”) are often **60–90%** when the object is obvious. |
| **TikTok specifically** | TikTok uses **slider**, **rotation**, and **object-selection** CAPTCHAs. We only solve **object-selection** (grid or click). Slider and “rotate the image” are **not** implemented; those still need manual solve or `--no-headless`. |
| **Detection vs site layout** | Even when our model picks the right tiles, **site DOM** must match what we expect (e.g. reCAPTCHA-style containers and tile selectors). TikTok’s own CAPTCHA markup may differ; if our selectors don’t find the grid, the solver won’t run. |

**Summary:** Useful for generic “select all traffic lights / buses / bicycles” challenges on sites that use standard patterns. Not a silver bullet; expect failures and fall back to manual solve when needed.

### How to test it

1. **Detection only (no browser)** – see what the model would click on any image:
   ```bash
   python scratch/run_captcha_detection_demo.py path/to/image.jpg
   # or with a URL:
   python scratch/run_captcha_detection_demo.py "https://example.com/grid.jpg" --output result.png
   ```
   Prints detected objects and (optionally) saves an image with bounding boxes.

2. **Full flow in browser** – local fake CAPTCHA page, solver runs end-to-end:
   ```bash
   python scratch/run_captcha_e2e_demo.py
   ```
   Opens a 3×3 grid page in Chrome, runs the solver, and reports whether it “passed”.

3. **Live scraper** – run the scraper; when a CAPTCHA appears it will try to solve it (disable with `--no-captcha-solve` if you prefer to solve manually):
   ```bash
   python scraper.py "https://www.tiktok.com/@user/video/123" --no-headless
   ```

## Notes

- TikTok may show a **slider** or **rotation** captcha; those are not auto-solved. Run with `--no-headless` and solve manually if needed.
- The scraper handles cookie banners automatically.
- Comments are loaded dynamically, so the scraper scrolls and waits for them to appear.
- Video and audio URLs may expire after some time.

## Requirements

- Python 3.8+
- Chrome browser
- ChromeDriver (must match your Chrome version)
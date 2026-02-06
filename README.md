# TikTok Scraper

A Python scraper for extracting data from TikTok video pages, including captions, metadata, video/audio URLs, and top comments.

## Features

- 📝 Extract video captions and descriptions
- 📊 Get metadata (likes, comments, shares, views, author info)
- 🎥 Retrieve video URLs and optionally download videos
- 🎵 Extract audio/music metadata and optionally download audio
- 💬 Scrape top comments with like/reply counts
- 🤖 Uses Selenium to handle dynamic content loading

## Project layout

- **`object_selection_captcha/`** — CAPTCHA solver package (shape-matching, grid/click, training, models). Import as `object_selection_captcha` or `from object_selection_captcha import handle_captcha`.
- **`scratch/`** — One-off scripts and demos (e.g. `test_model_on_captcha.py`).
- **`tests/`** — Pytest tests for scraper and captcha.

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

The scraper can try to **auto-solve** image-based CAPTCHAs. Support is **optional** and only runs if the `captcha` package and its dependencies are installed.

### "Select 2 objects that are the same shape" (TikTok)

For TikTok's **shape-matching** CAPTCHA (3D letters, numbers, geometric shapes), the solver uses a **hybrid** approach:

1. **YOLO** (fine-tuned YOLOv11n) — locates every object and returns bounding boxes.
2. **Deep features** (MobileNetV2) — crops each detection, extracts 1280-d embeddings, and picks the pair with highest cosine similarity (plus a foreground-mask check).
3. **Fallback** — if the custom model is missing, it falls back to classical contour-based shape matching.

**Model location:** `object_selection_captcha/models/tiktok_captcha_best.pt`

**Train your own model:**

```bash
# From project root, with dataset in object_selection_captcha/tikdata.v1i.yolo8 (or pass --data)
python -m object_selection_captcha.train_tiktok_model --epochs 100
```

**Test the solver on images:**

```bash
# Uses dataset test images by default; pass image paths to test specific files
python scratch/test_model_on_captcha.py --save
# Output: scratch/test_results/*.jpg (annotated with detections and the chosen match)
```

**Run unit tests (captcha):**

```bash
# Run from repo root
pytest tests/test_tiktok_detector.py tests/test_shape_solver.py -v
```

### Generic "select all X" (COCO)

For grid/click CAPTCHAs that ask for COCO classes (e.g. traffic lights, buses), a local YOLO model is used. Object coverage is limited to **80 COCO classes**; prompts like "crosswalk" or "palm tree" have no mapping. Slider and rotation CAPTCHAs are **not** implemented — use `--no-headless` and solve those manually.

**Other ways to test CAPTCHA:**

1. **Detection only (no browser)** — see what the model would click on any image:
   ```bash
   python scratch/run_captcha_detection_demo.py path/to/image.jpg
   python scratch/run_captcha_detection_demo.py "https://example.com/grid.jpg" --output result.png
   ```

2. **Full flow in browser** — local fake CAPTCHA page:
   ```bash
   python scratch/run_captcha_e2e_demo.py
   ```

3. **Live scraper** — when a CAPTCHA appears it will try to solve it (disable with `--no-captcha-solve` if you prefer to solve manually):
   ```bash
   python scraper.py "https://www.tiktok.com/@user/video/123" --no-headless
   ```

---

## Contributing

We welcome contributions. To propose a change via pull request:

1. **Fork the repo**  
   Click "Fork" on GitHub so you have your own copy (e.g. `https://github.com/your-username/Tik-Tok-API`).

2. **Clone your fork and add the upstream remote**  
   ```bash
   git clone https://github.com/your-username/Tik-Tok-API.git
   cd Tik-Tok-API
   git remote add upstream https://github.com/ORIGINAL_OWNER/Tik-Tok-API.git
   ```
   Replace `your-username` and `ORIGINAL_OWNER` with your GitHub username and the original repo owner.

3. **Create a branch**  
   Work on a feature or fix in a dedicated branch (e.g. `git checkout -b fix-captcha-threshold`).

4. **Make your changes**  
   Edit code, add tests if needed, and run the test suite:
   ```bash
   pip install -r requirements.txt
   pytest tests/ -v
   ```

5. **Commit and push**  
   ```bash
   git add .
   git commit -m "Short description of the change"
   git push origin fix-captcha-threshold
   ```

6. **Open a pull request**  
   - Go to your fork on GitHub.
   - You should see a prompt to "Compare & pull request" for the branch you just pushed. Click it.
   - Or: **Branches** → select your branch → **New pull request**.
   - Choose the **base** repo and branch (usually `main` or `master` on the original repo).
   - Write a clear title and description (what changed, why, and how to test).
   - Submit the pull request. Maintainers will review and may ask for edits.

7. **Sync with upstream (optional)**  
   To update your fork with the latest changes from the original repo:
   ```bash
   git fetch upstream
   git checkout main   # or your default branch
   git merge upstream/main
   git push origin main
   ```

---

## Notes

- TikTok may show a **slider** or **rotation** captcha; those are not auto-solved. Run with `--no-headless` and solve manually if needed.
- The scraper handles cookie banners automatically.
- Comments are loaded dynamically, so the scraper scrolls and waits for them to appear.
- Video and audio URLs may expire after some time.

## Requirements

- Python 3.8+
- Chrome browser
- ChromeDriver (must match your Chrome version)
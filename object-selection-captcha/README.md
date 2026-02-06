# Object-selection CAPTCHA solver

TikTok shape CAPTCHA solver (YOLO + contour fallback). Used by the main scraper when `solve_captcha=True`.

## Tests

Tests for this implementation live in the repo root under **`tests/`**:

| Test file | What it covers |
|-----------|----------------|
| `tests/test_tiktok_detector.py` | TikTokDetection, `detect_tiktok_objects`, `find_matching_pair_yolo`, model availability, shape-solver integration |
| `tests/test_shape_solver.py` | Segment/contour logic, feature extraction, `compare_shapes`, `find_matching_pair` (classical path) |
| `tests/test_captcha.py` | COCO detector, `resolve_prompt_labels`, grid/click helpers, `handle_captcha` (mocked) |

They import the package as **`captcha`**. From the **repo root** (Tik-Tok-API), run:

```bash
# Use this folder as the captcha package (one-time): symlink captcha -> object-selection-captcha
ln -sf object-selection-captcha captcha

# Then run the object-selection-captcha tests
pytest tests/test_tiktok_detector.py tests/test_shape_solver.py tests/test_captcha.py -v
```

## Models

Weights go in **`models/`**. See `models/README.md`.

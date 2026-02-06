# Object-selection CAPTCHA solver

TikTok shape CAPTCHA solver (YOLO + MobileNetV2 deep features + contour fallback). Used by the main scraper when `solve_captcha=True`.

**Import:** `from object_selection_captcha import handle_captcha`

## Tests

From the repo root:

```bash
pytest tests/test_tiktok_detector.py tests/test_shape_solver.py tests/test_captcha.py -v
```

## Models

Weights go in `models/`. See `models/README.md`.

Train: `python -m object_selection_captcha.train_tiktok_model`

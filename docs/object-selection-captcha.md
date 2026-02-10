# Object-selection CAPTCHA (TikTok shape / grid / click)

## Overview

The **object_selection_captcha** package detects and solves **image-based selection CAPTCHAs** in the browser, including TikTok’s “Select 2 objects that are the same shape” challenge. It is used by the main scraper when `solve_captcha=True`.

**Main entry point:** `handle_captcha(driver, max_attempts=3)` — returns `True` if the CAPTCHA was solved (or none was present).

---

## Supported CAPTCHA types

| Type | Description | Solver |
|------|-------------|--------|
| **Shape** (TikTok) | “Select 2 objects that are the same shape” — 3D-rendered objects on a light background. | YOLO (TikTok-trained) + contour/Hu-moment fallback |
| **Grid** | “Select all images with X” on a 3×3 or 4×4 grid. | COCO YOLO tile classification |
| **Click** | “Click on the X” — single image, click target object(s). | COCO YOLO + click centres |
| **Slider / Rotate** | Drag puzzle piece or rotate to align. | Slider: [slider-puzzle.md](slider-puzzle.md). Rotate: manual or funcaptcha rotation. |

Unknown types are tried as shape → grid → click.

---

## Architecture

- **browser.py** — Detect CAPTCHA (iframes, containers), screenshot, click tiles/points, click verify.
- **solver.py** — Orchestration: `handle_captcha` and dispatch to shape/grid/click solvers.
- **tiktok_detector.py** — Fine-tuned YOLO for TikTok 3D objects; visual comparison for same-shape pair.
- **shape_solver.py** — Shape matching: Tier 1 = YOLO class match; Tier 2 = contour + Hu moments + similarity.
- **detector.py** — COCO YOLO for grid/click: tile classification, click-point resolution, prompt → labels.

---

## Shape solver (TikTok “same shape”)

**Tier 1 — Fine-tuned YOLO**  
If `object_selection_captcha/models/tiktok_captcha_best.pt` exists, it detects each 3D object and assigns a label. Two objects with the same label form the matching pair. Train with:

```bash
python -m object_selection_captcha.train_tiktok_model
```

**Tier 2 — Classical**  
Segment objects (colour + Otsu + connected components), filter noise, compute shape features (Hu moments, aspect ratio, extent, solidity, edge histogram). Compare every pair; return the two most similar by centre coordinates for clicking.

---

## API (public)

### `handle_captcha(driver, max_attempts=3, post_solve_wait=2.0) -> bool`

Detect and attempt to solve any visible CAPTCHA.

| Parameter | Description |
|-----------|-------------|
| `driver` | Selenium WebDriver (Chrome). |
| `max_attempts` | Max solve attempts before giving up. |
| `post_solve_wait` | Seconds to wait after clicking verify before checking success. |

**Returns:** `True` if CAPTCHA was solved or none was present; `False` after exhausting attempts.

---

## Data structures

**CaptchaInfo** (from `browser.py`) — `captcha_type`, `prompt`, `grid_rows`, `grid_cols`, `image_element`, `container_element`, `iframe`, `tile_elements`. Used internally by the solvers.

---

## Tests

From repo root:

```bash
pytest tests/test_tiktok_detector.py tests/test_shape_solver.py tests/test_captcha.py -v
```

---

## Models

Weights go in `object_selection_captcha/models/`. See `object_selection_captcha/models/README.md`.  
TikTok shape model: `tiktok_captcha_best.pt` (created by `train_tiktok_model`).

---

## See also

- [slider-puzzle.md](slider-puzzle.md) — Puzzle slider (drag the piece into the slot), e.g. TikTok.
- FunCaptcha puzzle docs: [funcaptcha-cycle-match.md](funcaptcha-cycle-match.md), [funcaptcha-rotation.md](funcaptcha-rotation.md), [funcaptcha-quantity.md](funcaptcha-quantity.md), [funcaptcha-dice-sum.md](funcaptcha-dice-sum.md).

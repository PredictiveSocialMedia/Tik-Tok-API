# FunCaptcha method: Quantity (match the number of objects)

## Overview

**Quantity** solves the FunCaptcha challenge where the user must **change the number of objects** in a panel until it **matches a reference**. For example, the left side shows “1 × heart” and the right side shows a controllable panel with zero or more hearts; the user clicks **+** or **−** until the count matches (e.g. one heart), then submits.

This module provides **object counting** on a single image (reference or current panel) and **quantity delta** (target count − current count). The caller (or `solve_quantity`) uses the delta to decide how many **+** or **−** clicks to perform.

---

## Challenge type

| Property | Description |
|----------|-------------|
| **Name** | Quantity + object / “Change the number until it matches” |
| **Difficulty** | High (variable difficulty spectrum) |
| **Input** | Two images: reference (e.g. “1 x heart”) and current panel |
| **Output** | Integer delta: positive = click + N times, negative = click − N times |
| **UI** | + and − buttons (or equivalent) to add/remove one object |

---

## Algorithm

### 1. Counting (three possible backends)

**Backend A — ML count regressor (default when model exists)**  
- If a trained model exists at `funcaptcha/models/count_quantity.pt`, the image is resized to **128×128**, converted to a tensor, and passed through the CNN.  
- The model outputs a single scalar (float); the count is `max(0, round(pred))`.  
- **Use:** `count_objects(..., prefer_ml=True)` (default). No model → try YOLO, then contour.

**Backend B — YOLO (object_selection_captcha)**  
- If `prefer_yolo=True` and the `object_selection_captcha` package is available, `detect_tiktok_objects` is run on the image.  
- The count is the **number of detections** (all classes).  
- Useful when the objects are the same as in the TikTok same-shape CAPTCHA (letters, numbers, shapes).  
- **Use:** `count_objects(..., prefer_yolo=True, prefer_ml=False)` to force YOLO when ML regressor is not desired.

**Backend C — Contour-based**  
- The image is converted to grayscale, blurred, and thresholded (Otsu, inverted).  
- Morphology (open, close) cleans the mask; contours are found.  
- Blobs with area between **1%** and **40%** of the image area are counted (filters noise and full-background).  
- **Use:** Automatic fallback when ML and YOLO are unavailable or disabled (`prefer_ml=False`, `prefer_yolo=False`).

### 2. Quantity delta

- **Target count** = `count_objects(reference_image)`.  
- **Current count** = `count_objects(current_image)`.  
- **Delta** = `target - current`.  
  - **Delta > 0** → click **+** `delta` times.  
  - **Delta < 0** → click **−** `|delta|` times.  
  - **Delta = 0** → matched; no clicks needed.

### 3. High-level solve loop

`solve_quantity` repeatedly:  
1. Gets reference and current images.  
2. Computes `delta = quantity_delta(ref, cur)`.  
3. If `delta == 0`, returns `True`.  
4. Otherwise clicks **+** or **−** (up to 5 times per step to avoid overshoot) and repeats.  
5. Stops after `max_steps` iterations or when images cannot be obtained.

---

## API reference

### `count_objects_contour(image) -> int`

Counts foreground blobs using thresholding and contours. No ML or YOLO.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `PIL.Image.Image` | Single panel image (reference or current). |

**Returns:** `int` (count ≥ 0).

---

### `count_objects_yolo(image, confidence_threshold=0.25) -> int`

Counts detections using `object_selection_captcha.tiktok_detector.detect_tiktok_objects`. Returns `-1` if YOLO is not available.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `PIL.Image.Image` | — | Panel image. |
| `confidence_threshold` | `float` | `0.25` | Minimum detection confidence. |

**Returns:** `int` (number of detections, or `-1` on error/unavailable).

---

### `count_objects_ml(image, image_size=128) -> Optional[int]`

Uses the trained count regressor if available. Returns `None` if the model is not loaded.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `PIL.Image.Image` | — | Panel image. |
| `image_size` | `int` | `128` | Resize size for the model. |

**Returns:** `int` (count ≥ 0) or `None`.

---

### `count_objects(image, prefer_yolo=True, prefer_ml=True, confidence_threshold=0.25) -> int`

Main counting entry point. Tries, in order: ML regressor (if `prefer_ml` and model exists), then YOLO (if `prefer_yolo` and available), then contour.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `PIL.Image.Image` | — | Panel image. |
| `prefer_yolo` | `bool` | `True` | Try YOLO after ML. |
| `prefer_ml` | `bool` | `True` | Try ML regressor first. |
| `confidence_threshold` | `float` | `0.25` | For YOLO. |

**Returns:** `int` (count ≥ 0).

---

### `quantity_delta(reference_image, current_image, prefer_yolo=True, prefer_ml=True) -> int`

Computes target count − current count.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_image` | `PIL.Image.Image` | — | Reference panel (e.g. “1 x heart”). |
| `current_image` | `PIL.Image.Image` | — | Current controllable panel. |
| `prefer_yolo` | `bool` | `True` | Passed to `count_objects`. |
| `prefer_ml` | `bool` | `True` | Passed to `count_objects`. |

**Returns:** `int`. Positive = need more (+ clicks); negative = need fewer (− clicks); zero = match.

---

### `solve_quantity(get_reference_image, get_current_image, click_plus, click_minus, max_steps=20, prefer_yolo=True) -> Optional[bool]`

High-level: loop until counts match or `max_steps` exceeded. Uses `quantity_delta` (with default `prefer_ml=True`) and caps at 5 + or − clicks per step.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `get_reference_image` | `callable() -> Image \| None` | — | Returns reference panel. |
| `get_current_image` | `callable() -> Image \| None` | — | Returns current panel. |
| `click_plus` | `callable()` | — | One “+” click. |
| `click_minus` | `callable()` | — | One “−” click. |
| `max_steps` | `int` | `20` | Max loop iterations. |
| `prefer_yolo` | `bool` | `True` | Passed to `quantity_delta`. |

**Returns:** `True` when matched, `False` if `max_steps` exceeded, `None` on image get error.

---

## Training the count regressor (optional)

The count model is trained on **synthetic** data: images with N circles on a light background, label N (no external images required).

```bash
python -m funcaptcha.train_quantity --epochs 20 --batch-size 32 --batches-per-epoch 50 --max-count 20
```

- **Output:** `funcaptcha/models/count_quantity.pt`.  
- Once this file exists, `count_objects(..., prefer_ml=True)` uses it when available.  
- The regressor may not generalize perfectly to real CAPTCHA art (hearts, animals); for production, consider training on real reference/current panels with known counts.

---

## Usage examples

### Logic only

```python
from PIL import Image
from funcaptcha.quantity import quantity_delta, count_objects

ref = Image.open("reference_1x_heart.png")
cur = Image.open("current_panel.png")
delta = quantity_delta(ref, cur)
if delta > 0:
    print("Click +", delta, "times")
elif delta < 0:
    print("Click -", -delta, "times")
else:
    print("Match")
```

### With browser

```python
from funcaptcha.browser_actions import click_plus_minus
from funcaptcha.quantity import solve_quantity

def get_ref(): ...
def get_cur(): ...

def click_plus():
    click_plus_minus(driver, plus=True, times=1, container=container)

def click_minus():
    click_plus_minus(driver, plus=False, times=1, container=container)

ok = solve_quantity(get_ref, get_cur, click_plus, click_minus)
```

---

## Constants and tuning

Contour-based counting uses:

- **Area range:** 1% to 40% of image area (filters tiny noise and full-background).  
- **Morphology:** Ellipse kernel (3×3), open then close.

For different CAPTCHA styles (e.g. very small or very large objects), you would need to adjust these inside `count_objects_contour` or rely on the ML/YOLO backends.

---

## Dependencies

- **Required:** `opencv-python-headless`, `numpy`, `Pillow`.  
- **Optional (ML):** `torch`; weights at `funcaptcha/models/count_quantity.pt`.  
- **Optional (YOLO):** `object_selection_captcha` and its TikTok YOLO model.

---

## See also

Other funcaptcha methods: [funcaptcha-cycle-match.md](funcaptcha-cycle-match.md), [funcaptcha-rotation.md](funcaptcha-rotation.md), [funcaptcha-dice-sum.md](funcaptcha-dice-sum.md). Object-selection CAPTCHA: [object-selection-captcha.md](object-selection-captcha.md). UI actions (slider, arrows, +/−, tiles): `funcaptcha.browser_actions`.

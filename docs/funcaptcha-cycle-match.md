# FunCaptcha method: Cycle match (2D object matching)

## Overview

**Cycle match** solves the FunCaptcha challenge where the user must **match the object in the left panel with the object in the right panel** by using **left/right arrow buttons** to cycle through options. The left panel shows a reference (e.g. “Match This!” with a crab); the right panel shows one of several options (e.g. teddy bear, crab, dog). The task is to click the arrows until the right panel shows the **same** object as the left, then submit.

This module provides **similarity scoring** between two panel images (reference vs current). The caller (or the high-level `solve_cycle_match`) repeatedly takes screenshots, computes the score, and clicks the right (or left) arrow until the score exceeds a threshold, indicating a match.

---

## Challenge type

| Property | Description |
|----------|-------------|
| **Name** | 2D object matching / “Match the animal” |
| **Difficulty** | Medium (variable difficulty spectrum) |
| **Input** | Two images: reference (left) and current (right) panel |
| **Output** | A similarity score in `[0, 1]`, or a boolean “matched” |
| **UI** | Arrow buttons (left/right) to cycle the right panel; Submit button |

---

## Algorithm

### 1. Image normalization

Both panels are resized to a fixed size (**128×128** by default) so that comparison is size-invariant. The implementation uses OpenCV BGR and resizes with `INTER_AREA`.

### 2. Similarity computation (three possible paths)

**Path A — Siamese network (default when model exists)**  
- If a trained Siamese model is present at `funcaptcha/models/siamese_cycle_match.pt`, it is loaded and used.  
- Both panels are normalized to 128×128, converted to tensors (float [0,1]), and passed through the shared backbone.  
- The model outputs a **cosine similarity** (dot product of L2-normalized embeddings) in the range `[-1, 1]`.  
- This is mapped to `[0, 1]` as `(sim + 1) / 2`.  
- **Use:** Set `use_siamese=True` (default). No model file → fallback to classical.

**Path B — Classical (histogram + structural)**  
- **Histogram similarity:** 3D colour histograms (32×32×32 bins) are computed for both panels, normalized, and compared with `cv2.compareHist(..., cv2.HISTCMP_CORREL)`. The correlation in `[-1, 1]` is mapped to `[0, 1]`.  
- **Structural similarity:** Per-channel correlation of the flattened images (mean, std, covariance) is computed and averaged; result is clipped to `[0, 1]`.  
- **Combined score:** `0.6 * hist_sim + 0.4 * struct_sim`.

**Path C — Deep features (optional, from object_selection_captcha)**  
- If `use_deep_features=True` and the `object_selection_captcha` package is available, the code can use YOLO to detect one object per panel and MobileNetV2 to compare the cropped regions.  
- The deep similarity is then blended: `combined = 0.5 * combined + 0.5 * deep_sim`.  
- This path is optional and slower; it relies on the TikTok-trained YOLO and MobileNetV2 feature extractor.

### 3. Match decision

- A **match** is declared when the score is **≥ threshold**. The default threshold is **0.75** (`_MATCH_THRESHOLD`).  
- So: `is_matched(ref, cur) ⟺ match_score(ref, cur) >= 0.75`.

---

## API reference

### `match_score(reference_image, current_image, use_deep_features=False, use_siamese=True) -> float`

Compares the two panel images and returns a similarity score in `[0, 1]`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_image` | `PIL.Image.Image` | — | Left/reference panel (RGB). |
| `current_image` | `PIL.Image.Image` | — | Right/current panel (RGB). |
| `use_deep_features` | `bool` | `False` | If True, and `object_selection_captcha` is available, blend in MobileNetV2 similarity on cropped objects. |
| `use_siamese` | `bool` | `True` | If True, use trained Siamese model when `funcaptcha/models/siamese_cycle_match.pt` exists; else classical. |

**Returns:** `float` in `[0, 1]`. Higher means “more similar”; typically **≥ 0.75** is treated as a match.

---

### `is_matched(reference_image, current_image, threshold=0.75, use_deep_features=False) -> bool`

Convenience function: returns `True` if `match_score(...) >= threshold`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_image` | `PIL.Image.Image` | — | Left panel. |
| `current_image` | `PIL.Image.Image` | — | Right panel. |
| `threshold` | `float` | `0.75` | Minimum score to consider a match. |
| `use_deep_features` | `bool` | `False` | Passed through to `match_score`. |

**Returns:** `True` if the panels are considered a match, else `False`.

---

### `solve_cycle_match(get_reference_image, get_current_image, click_right, click_left, max_steps=20, match_threshold=0.75, use_deep_features=False) -> Optional[bool]`

High-level solver: repeatedly gets reference and current images, computes the match score, and clicks the **right** arrow until the score exceeds `match_threshold` or `max_steps` is reached.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `get_reference_image` | `callable() -> Image \| None` | — | No arguments; returns PIL Image of the left panel or `None` on error. |
| `get_current_image` | `callable() -> Image \| None` | — | No arguments; returns PIL Image of the right panel or `None`. |
| `click_right` | `callable()` | — | Performs one “right arrow” click. |
| `click_left` | `callable()` | — | Performs one “left arrow” click (reserved for future use; current logic only clicks right). |
| `max_steps` | `int` | `20` | Maximum number of arrow clicks before giving up. |
| `match_threshold` | `float` | `0.75` | Minimum `match_score` to consider matched. |
| `use_deep_features` | `bool` | `False` | Passed to `match_score`. |

**Returns:**  
- `True` if a match was achieved (score ≥ threshold).  
- `False` if `max_steps` was exceeded without matching.  
- `None` if `get_reference_image()` or `get_current_image()` returned `None`.

**Note:** The current implementation always clicks **right** each step. A more advanced version could use a binary-search or direction heuristic (e.g. compare with previous score to decide left vs right).

---

## Training the Siamese model (optional)

To improve accuracy, you can train a Siamese network on same/different pairs:

```bash
python -m funcaptcha.train_cycle_match --images /path/to/image/dir --epochs 10 --batch-size 32
```

- **Data:** A directory of images (e.g. `object_selection_captcha/tikdata.v1i.yolov8/train/images`).  
- **Synthetic pairs:** “Same” = one image + augmentation (flip, colour jitter); “different” = two different random images.  
- **Output:** `funcaptcha/models/siamese_cycle_match.pt`.  
- Once this file exists, `match_score(..., use_siamese=True)` uses it automatically.

---

## Usage examples

### Logic only (no browser)

```python
from PIL import Image
from funcaptcha.cycle_match import match_score, is_matched

ref = Image.open("left_panel.png")
cur = Image.open("right_panel.png")

score = match_score(ref, cur)
if is_matched(ref, cur):
    print("Match — stop clicking")
else:
    print("Click right arrow and re-check")
```

### With browser (Selenium)

```python
from selenium import webdriver
from funcaptcha.browser_actions import click_arrow
from funcaptcha.cycle_match import solve_cycle_match

driver = webdriver.Chrome()
# ... load page, locate CAPTCHA container ...

container = driver.find_element(...)  # CAPTCHA container

def get_ref():
    left = container.find_element(...)
    return Image.open(io.BytesIO(left.screenshot_as_png)).convert("RGB")

def get_cur():
    right = container.find_element(...)
    return Image.open(io.BytesIO(right.screenshot_as_png)).convert("RGB")

def click_r():
    click_arrow(driver, right=True, container=container, times=1)

def click_l():
    click_arrow(driver, right=False, container=container, times=1)

ok = solve_cycle_match(get_ref, get_cur, click_r, click_l, max_steps=20)
if ok:
    # Click Submit
    ...
```

---

## Constants and tuning

| Constant | Value | Description |
|----------|--------|-------------|
| `_PANEL_SIZE` | `(128, 128)` | Size to which panels are resized before comparison. |
| `_MATCH_THRESHOLD` | `0.75` | Default threshold for `is_matched` and `solve_cycle_match`. |

You can pass a lower threshold (e.g. `0.7`) to accept slightly less similar panels, or higher (e.g. `0.8`) to reduce false positives.

---

## Dependencies

- **Required:** `opencv-python-headless`, `numpy`, `Pillow`.  
- **Optional (Siamese):** `torch`; trained weights at `funcaptcha/models/siamese_cycle_match.pt`.  
- **Optional (deep features):** `object_selection_captcha` (and its YOLO/MobileNetV2 stack).

---

## See also

Other funcaptcha methods: [funcaptcha-rotation.md](funcaptcha-rotation.md), [funcaptcha-quantity.md](funcaptcha-quantity.md), [funcaptcha-dice-sum.md](funcaptcha-dice-sum.md). Object-selection CAPTCHA: [object-selection-captcha.md](object-selection-captcha.md). UI actions (slider, arrows, +/−, tiles): `funcaptcha.browser_actions`.

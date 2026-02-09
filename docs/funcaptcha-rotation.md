# FunCaptcha method: Rotation (3D rotate to match angle)

## Overview

**Rotation** solves the FunCaptcha challenge where the user must **rotate a 3D object** (e.g. an animal) **to match a target orientation**, typically indicated by a reference (e.g. a hand pointing in the desired direction). The UI is usually either a **slider** (drag to rotate) or **left/right arrow buttons** (discrete rotation steps).

This module provides **angle estimation** (how much to rotate the current view to reach “upright” or “correct” orientation) and **UI helpers** to convert that angle into slider pixel delta or number of arrow clicks.

---

## Challenge type

| Property | Description |
|----------|-------------|
| **Name** | 3D object rotation / “Rotate to match this angle” |
| **Difficulty** | Medium–high (variable difficulty spectrum) |
| **Input** | One image (current rotated view) or, for fragment-to-background, fragment + background images |
| **Output** | Angle in degrees `[0, 360)` to apply (or pixel delta / arrow clicks) |
| **UI** | Slider (drag) or left/right arrow buttons |

---

## Algorithm

### 1. Angle estimation (two paths)

**Path A — RotNet (default when model exists)**  
- If a trained RotNet model exists at `funcaptcha/models/rotnet_rotation.pt`, it is loaded.  
- The image is resized to **128×128**, converted to a tensor (float [0,1]), and passed through the CNN.  
- The model has **128 output classes**, each corresponding to an angle bin (~2.8° per bin).  
- `predict_angle_deg(logits)` converts the softmax distribution over bins into a single angle in `[0, 360)`.  
- **Use:** `estimate_angle(..., use_rotnet=True)` (default). No model file → fallback to classical.

**Path B — Classical (Laplacian variance)**  
- The image is rotated at **multiple angles** (default step **5°**: 0°, 5°, 10°, …).  
- For each angle, the image is rotated by that amount using `cv2.getRotationMatrix2D` and `cv2.warpAffine`.  
- For the rotated image, the **Laplacian** (edge strength) is computed and its **variance** is taken as a score.  
- The **angle that maximizes this score** is returned. The heuristic is that “upright” orientations often have stronger or more coherent edges.  
- Center of rotation is the image center (or optional `center`); radius is optional (default half of min(width, height)).

**Path C — Fragment-to-background (special case)**  
- When the CAPTCHA provides a **separate fragment image** (the rotated piece) and a **background image** (the full scene with a hole), use `estimate_angle_fragment_to_background(fragment, background, angle_step)`.  
- For each angle, the fragment is rotated and overlaid (conceptually); the **mean absolute pixel difference** from the background is computed.  
- The **angle that minimizes this difference** is returned (best alignment).

### 2. From angle to UI action

**Slider:**  
- `angle_to_slider_delta(angle_degrees, track_length_px, full_rotation_degrees=360)`  
- Returns the number of **pixels to drag** the slider (positive = right):  
  `delta_x = angle_degrees / 360 * track_length_px`.

**Arrows:**  
- `angle_to_arrow_clicks(angle_degrees, degrees_per_click=15)`  
- Returns `(right: bool, n_clicks: int)`.  
- Example: 90° and 15° per click → `(True, 6)` (click right 6 times).

---

## API reference

### `estimate_angle_rotnet(image, image_size=128) -> Optional[float]`

Uses the trained RotNet model if available. Returns angle in `[0, 360)` or `None` if the model is not loaded.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `PIL.Image.Image` | — | Current rotated view (RGB). |
| `image_size` | `int` | `128` | Size to which the image is resized before feeding to the model. |

---

### `estimate_angle(image, use_rotnet=True, center=None, radius=None, angle_step=5) -> float`

Main entry point: estimates the rotation angle. Tries RotNet first when `use_rotnet=True` and the model exists; otherwise uses classical Laplacian search.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `PIL.Image.Image` | — | Current view. |
| `use_rotnet` | `bool` | `True` | Use RotNet when available. |
| `center` | `tuple[float, float] \| None` | `None` | Center of rotation (x, y); default image center. |
| `radius` | `float \| None` | `None` | Not used in current classical logic; reserved. |
| `angle_step` | `int` | `5` | Step in degrees for classical search (0, 5, 10, …). |

**Returns:** `float` in `[0, 360)`.

---

### `estimate_angle_classical(image, center=None, radius=None, angle_step=5) -> float`

Classical only: samples angles, rotates the image, scores by Laplacian variance, returns the best angle. No ML.

---

### `estimate_angle_fragment_to_background(fragment, background, angle_step=5) -> float`

For “fragment + background” CAPTCHAs: finds the angle that best aligns the rotated fragment with the background (min pixel difference). Returns angle in `[0, 360)`.

---

### `angle_to_slider_delta(angle_degrees, track_length_px, full_rotation_degrees=360) -> int`

Converts angle to slider pixel delta (positive = drag right).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `angle_degrees` | `float` | — | Target or correction angle in degrees. |
| `track_length_px` | `int` | — | Length of the slider track in pixels. |
| `full_rotation_degrees` | `float` | `360` | Full rotation corresponding to the whole track. |

**Returns:** `int` (pixel delta).

---

### `angle_to_arrow_clicks(angle_degrees, degrees_per_click=15) -> tuple[bool, int]`

Converts angle to number of arrow clicks and direction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `angle_degrees` | `float` | — | Angle to “apply” (e.g. correction angle). |
| `degrees_per_click` | `float` | `15` | Rotation per one arrow click (depends on CAPTCHA). |

**Returns:** `(right: bool, n_clicks: int)`. `right=True` means “click right arrow” `n_clicks` times; `right=False` means “click left arrow” `n_clicks` times.

---

### `solve_rotation_slider(get_image, drag_slider, track_length_px=300, angle_step=5, use_rotnet=True) -> Optional[bool]`

High-level: get current image, estimate angle, convert to slider delta, call `drag_slider(delta_x)`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `get_image` | `callable() -> Image \| None` | — | Returns current rotated view. |
| `drag_slider` | `callable(delta_x: int)` | — | Performs slider drag by `delta_x` pixels. |
| `track_length_px` | `int` | `300` | Slider track length for delta computation. |
| `angle_step` | `int` | `5` | Used for classical fallback. |
| `use_rotnet` | `bool` | `True` | Use RotNet when available. |

**Returns:** `True` if `drag_slider` was called successfully, `False`/`None` on error.

---

### `solve_rotation_arrows(get_image, click_left, click_right, degrees_per_click=15, angle_step=5, use_rotnet=True) -> Optional[bool]`

High-level: get image, estimate angle, convert to arrow clicks, then call `click_right()` or `click_left()` the required number of times.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `get_image` | `callable() -> Image \| None` | — | Returns current rotated view. |
| `click_left` | `callable()` | — | One left-arrow click. |
| `click_right` | `callable()` | — | One right-arrow click. |
| `degrees_per_click` | `float` | `15` | Rotation per click. |
| `angle_step` | `int` | `5` | For classical fallback. |
| `use_rotnet` | `bool` | `True` | Use RotNet when available. |

**Returns:** `True` after performing the clicks, or `None` if `get_image()` failed.

---

## Training the RotNet model (optional)

RotNet is trained **self-supervised**: no manual labels. For each image, a random rotation angle is chosen, the image is rotated by that angle, and the network is trained to predict the angle (128 classes).

```bash
python -m funcaptcha.train_rotation --images /path/to/image/dir --epochs 15 --batch-size 32
```

- **Data:** Any set of images (e.g. CAPTCHA screenshots or natural images).  
- **Output:** `funcaptcha/models/rotnet_rotation.pt`.  
- Once this file exists, `estimate_angle(..., use_rotnet=True)` uses it automatically.

---

## Usage examples

### Logic only

```python
from PIL import Image
from funcaptcha.rotation import estimate_angle, angle_to_slider_delta, angle_to_arrow_clicks

image = Image.open("rotated_captcha.png")
angle = estimate_angle(image)
delta_px = angle_to_slider_delta(angle, track_length_px=300)
# Or for arrows:
right, n = angle_to_arrow_clicks(angle, degrees_per_click=15)
# Then click right (or left) n times
```

### With browser (slider)

```python
from funcaptcha.rotation import estimate_angle, angle_to_slider_delta, solve_rotation_slider
from funcaptcha.browser_actions import slider_drag

def get_image():
    elem = driver.find_element(...)
    return Image.open(io.BytesIO(elem.screenshot_as_png)).convert("RGB")

def drag_slider(delta_x):
    slider_drag(driver, delta_x=delta_x, container=container)

solve_rotation_slider(get_image, drag_slider, track_length_px=300)
```

---

## Constants and tuning

| Constant | Value | Description |
|----------|--------|-------------|
| `_ANGLE_STEP` | `5` | Step (degrees) for classical angle search. Finer step (e.g. 2) improves precision but increases cost. |

---

## Dependencies

- **Required:** `opencv-python-headless`, `numpy`, `Pillow`.  
- **Optional (RotNet):** `torch`; weights at `funcaptcha/models/rotnet_rotation.pt`.

---

## See also

Other funcaptcha methods: [funcaptcha-cycle-match.md](funcaptcha-cycle-match.md), [funcaptcha-quantity.md](funcaptcha-quantity.md), [funcaptcha-dice-sum.md](funcaptcha-dice-sum.md). Object-selection CAPTCHA: [object-selection-captcha.md](object-selection-captcha.md). UI actions (slider, arrows, +/−, tiles): `funcaptcha.browser_actions`.

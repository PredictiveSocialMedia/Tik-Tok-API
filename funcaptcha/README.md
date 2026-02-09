# funcaptcha — Four FunCaptcha Solvers

In-house implementations for the four main FunCaptcha/Arkose-style challenge types (from the [variable difficulty](../docs/3d-spatial-captcha-research.md) spectrum and [solution frameworks](../docs/funcaptcha-solution-frameworks.md) research).

## Modules

| Module | Challenge type | What it does |
|--------|----------------|---------------|
| **cycle_match** | 2D object matching | Compare reference (left) vs current (right) panel; returns match score. Use with arrow clicks until matched. |
| **rotation** | 3D rotation | Estimate correction angle (classical CV: sample angles, pick best). Map to slider drag or arrow clicks. |
| **quantity** | Quantity + object | Count objects in reference and current panels; returns delta (+/- clicks needed). |
| **dice_sum** | Conditional selection | Parse "sum to X" from prompt; count pips per tile (1–6); return pair of tile indices that sum to X. |
| **browser_actions** | UI helpers | Slider drag, arrow left/right, plus/minus, tile clicks (Selenium ActionChains). |

## Do they use ML? Do we need training data?

| Module       | Default (no trained model)   | With trained model (optional) |
|-------------|------------------------------|--------------------------------|
| **cycle_match** | Histogram + structural similarity | **Siamese** (contrastive); train with `train_cycle_match.py` |
| **rotation**   | Classical: sample angles, Laplacian | **RotNet** (128-class); self-supervised `train_rotation.py` |
| **quantity**   | Contour blob count           | **Count regressor**; synthetic `train_quantity.py` |
| **dice_sum**   | Pip counting (contours)      | **6-way dice CNN**; synthetic `train_dice.py` |

Out of the box, all four work with **classical CV only** (no training). If you train the optional models and place weights in `funcaptcha/models/`, the solvers use them automatically for better accuracy.

## Training the ML upgrades (larger upgrades)

Requires **PyTorch** (`pip install torch`). Training uses **synthetic or existing image data**; no manual labeling for rotation (self-supervised) or quantity/dice (synthetic).

```bash
# From repo root

# 1. Cycle match — Siamese (same/different pairs). Needs image directory (e.g. TikTok CAPTCHA train images).
python -m funcaptcha.train_cycle_match --images object_selection_captcha/tikdata.v1i.yolov8/train/images --epochs 10

# 2. Rotation — RotNet (self-supervised: rotate image, predict angle). Same image dir.
python -m funcaptcha.train_rotation --images object_selection_captcha/tikdata.v1i.yolov8/train/images --epochs 15

# 3. Quantity — count regressor. Synthetic circles; no images needed.
python -m funcaptcha.train_quantity --epochs 20

# 4. Dice — 6-way classifier. Synthetic dice faces; no images needed.
python -m funcaptcha.train_dice --epochs 15
```

Trained weights are saved under `funcaptcha/models/` (e.g. `siamese_cycle_match.pt`, `rotnet_rotation.pt`, `count_quantity.pt`, `dice_classifier.pt`). The solvers load them automatically when present.

## Dependencies

- **Required:** `opencv-python-headless`, `numpy`, `Pillow`
- **Optional (for ML upgrades):** `torch` (for training and for using trained models)
- **Optional:** `object_selection_captcha` (same repo) for YOLO/MobileNetV2 in cycle_match and quantity.

## Usage (logic only)

Each solver takes images (and optionally prompt) and returns the answer; no browser.

### 1. Cycle match (2D object match)

```python
from PIL import Image
from funcaptcha.cycle_match import match_score, is_matched

ref = Image.open("reference_left.png")
cur = Image.open("current_right.png")
score = match_score(ref, cur)
if is_matched(ref, cur):
    print("Match — stop clicking arrows")
else:
    print("Click right (or left) arrow and re-check")
```

### 2. Rotation

```python
from funcaptcha.rotation import estimate_angle_classical, angle_to_slider_delta, angle_to_arrow_clicks

image = Image.open("rotated_captcha.png")
angle = estimate_angle_classical(image)
delta_px = angle_to_slider_delta(angle, track_length_px=300)
# Or: right, n = angle_to_arrow_clicks(angle); then click right/left n times
```

### 3. Quantity

```python
from funcaptcha.quantity import quantity_delta, count_objects

ref = Image.open("reference_1x_heart.png")
cur = Image.open("current_panel.png")
delta = quantity_delta(ref, cur)
# delta > 0 => click + delta times; delta < 0 => click - abs(delta) times
```

### 4. Dice sum

```python
from funcaptcha.dice_sum import solve_dice_sum, parse_target_sum

tiles = [Image.open(f"tile_{i}.png") for i in range(6)]
prompt = "Select the pair of dice whose top sides add up to 14"
pair = solve_dice_sum(tiles, prompt)
if pair:
    i, j = pair
    print(f"Click tiles {i} and {j}")
```

## Wiring with a browser

Use `funcaptcha.browser_actions` with Selenium (or similar):

```python
from selenium import webdriver
from funcaptcha.browser_actions import slider_drag, click_arrow, click_plus_minus, click_tiles
from funcaptcha.rotation import estimate_angle_classical, angle_to_slider_delta

driver = webdriver.Chrome()
# ... load CAPTCHA page ...

# Example: rotation with slider
img = ...  # screenshot of rotate challenge
angle = estimate_angle_classical(img)
slider_drag(driver, delta_x=angle_to_slider_delta(angle, 300))

# Example: dice sum — get tile elements from your CAPTCHA DOM, then:
# pair = solve_dice_sum(tile_images, prompt)
# click_tiles(driver, tile_elements, list(pair))
```

Selectors in `browser_actions` are defaults (e.g. `div[class*='slider']`); override via keyword args or by passing pre-found `WebElement`s.

## How to test

1. **Unit tests (no CAPTCHA, no training data)**  
   From repo root:
   ```bash
   python3 -m pytest tests/test_funcaptcha.py -v
   ```
   Tests use **synthetic images** (e.g. solid panels, simple shapes, fake dice tiles) to check that:
   - cycle_match gives high score for identical panels, lower for different ones
   - rotation returns an angle and slider/arrow helpers behave
   - quantity returns a delta (e.g. 1 vs 2 blobs → delta −1)
   - dice_sum parses the prompt and finds a pair that sums to the target (when tiles allow)

2. **With real screenshots**  
   Save reference/current panels, rotated CAPTCHA, or dice tiles as PNGs in a folder (e.g. `tests/fixtures/funcaptcha/`) and call the solvers in a small script or extra test cases. No extra fixtures are required by the code.

3. **End-to-end in a browser**  
   Use a site that shows FunCaptcha (or a local mock page with two panels, a slider, +/−, and a grid), wire `browser_actions` to your driver, and run the solvers on screenshots taken from the page. Success = CAPTCHA passes after clicks/drag.

## Notes

- **BDA / token:** These modules solve only the *visual* puzzle. Submitting a valid Arkose token (BDA/blob, tguess) requires a separate stack (e.g. [kiookp/funcaptcha--solver](https://github.com/kiookp/funcaptcha--solver) or real browser execution). See [funcaptcha-solution-frameworks.md](../docs/funcaptcha-solution-frameworks.md).
- **Rotation:** Classical angle search works for “rotate fragment to fit” or “most upright” heuristics; for higher accuracy, add a RotNet-style CNN (see [in-house-build-options-knowledge.md](../docs/in-house-build-options-knowledge.md)).
- **Dice:** Pip counting is sensitive to lighting and style; for odd dice styles, consider a small 6-class CNN per tile.

**Upgrades:** See [funcaptcha-upgrades.md](../docs/funcaptcha-upgrades.md) for research-backed improvements (CLIP for cycle_match, RotNet for rotation, density/count CNNs for quantity, 6-way CNN for dice, and quick wins that need no training).

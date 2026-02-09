# FunCaptcha method: Dice sum (conditional selection)

## Overview

**Dice sum** solves the FunCaptcha challenge where the user sees a **grid of dice faces** (typically 6 tiles in a 3×2 layout) and must **select the pair of dice whose top faces add up to a given number**. The instruction is usually in the form “Select the pair of dice whose top sides add up to 14” (or 7, 10, etc.). The user clicks **two tiles** (the two dice that sum to the target), then submits.

This module provides **prompt parsing** (extract target sum from text), **dice face value recognition** (1–6 per tile via pip counting or a 6-way CNN), **pair search** (which two indices sum to the target), and the combined **solve** that returns the two tile indices to click.

---

## Challenge type

| Property | Description |
|----------|-------------|
| **Name** | Conditional selection / “Dice sum to X” |
| **Difficulty** | Medium–high (logic + per-tile recognition) |
| **Input** | List of tile images (dice faces) + prompt string (or target int) |
| **Output** | `(i, j)` — 0-based indices of the two tiles to click, or `None` |
| **UI** | Grid of clickable tiles; user clicks the two that sum to the target |

---

## Algorithm

### 1. Parse target sum from prompt

- The prompt is searched for a number using several regex patterns, e.g.:  
  - `add\s+up\s+to\s+(\d+)`  
  - `sum\s+to\s+(\d+)`  
  - `total\s+(\d+)`  
  - etc.  
- The first captured number in the range **2–18** is taken as the target sum (two standard D6 dice sum to 2–12; 18 allows other variants).  
- **Function:** `parse_target_sum(prompt) -> Optional[int]`.

### 2. Per-tile value (1–6)

**Path A — 6-way CNN (default when model exists)**  
- If a trained classifier exists at `funcaptcha/models/dice_classifier.pt`, each tile is resized to **64×64**, converted to a tensor, and passed through the model.  
- The model has 6 classes (0–5); **value** = `argmax + 1` (1–6).  
- **Use:** `dice_values_from_tiles(..., use_ml=True)` (default). No model → pip counting.

**Path B — Pip counting (contours)**  
- Each tile is converted to grayscale and thresholded (Otsu, inverted) so pips (dark dots) are white.  
- Morphology (close) merges pip regions; contours are found.  
- Contours with area between **1%** and **35%** of the tile area are counted as pips.  
- The count is **clamped to 1–6** (valid dice face).  
- **Function:** `count_pips(dice_face_image) -> int`.

### 3. Find pair that sums to target

- **Values** = list of 1–6 per tile (length = number of tiles).  
- **Search:** over all pairs `(i, j)` with `i < j`, find one such that `values[i] + values[j] == target`.  
- **Function:** `find_pair_with_sum(values, target) -> Optional[tuple[int, int]]`.  
- Returns `(i, j)` (0-based indices) or `None` if no pair sums to the target (e.g. prompt said “14” but two D6 cannot sum to 14).

### 4. End-to-end solve

- Parse target from prompt (or use provided target).  
- Get values for all tiles (`dice_values_from_tiles`).  
- Call `find_pair_with_sum(values, target)`.  
- Return the pair or `None`.  
- **Function:** `solve_dice_sum(tile_images, prompt)` or `solve_dice_sum_with_prompt_parse(tile_images, target=..., prompt=...)`.

---

## API reference

### `parse_target_sum(prompt: str) -> Optional[int]`

Extracts the target sum from the instruction text.

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | Full instruction (e.g. “Select the pair whose top sides add up to 14”). |

**Returns:** `int` in 2–18, or `None` if no valid number found.

**Supported patterns:** “add up to X”, “sum to X”, “total X”, “equals X”, and a number at end of string (case-insensitive).

---

### `count_pips(dice_face_image: Image.Image) -> int`

Counts pips on a single dice face image using thresholding and contours. No ML.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dice_face_image` | `PIL.Image.Image` | Single tile (one dice face). |

**Returns:** `int` in 1–6.

---

### `dice_values_from_tiles_ml(tile_images, tile_size=64) -> Optional[list[int]]`

Uses the trained 6-way dice classifier if available. Returns one value (1–6) per tile, or `None` if the model is not loaded.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tile_images` | `list[PIL.Image.Image]` | — | One image per tile. |
| `tile_size` | `int` | `64` | Resize size for the model. |

**Returns:** `list[int]` (length = len(tile_images)), each in 1–6, or `None`.

---

### `dice_values_from_tiles(tile_images, use_ml=True) -> list[int]`

Main per-tile value routine. Tries ML first when `use_ml=True` and model exists; else returns `[count_pips(t) for t in tile_images]`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tile_images` | `list[PIL.Image.Image]` | — | One image per tile. |
| `use_ml` | `bool` | `True` | Use 6-way classifier when available. |

**Returns:** `list[int]` (1–6 per tile).

---

### `find_pair_with_sum(values: list[int], target: int) -> Optional[tuple[int, int]]`

Finds distinct indices `(i, j)` with `i < j` such that `values[i] + values[j] == target`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | `list[int]` | Per-tile values (e.g. 1–6). |
| `target` | `int` | Desired sum. |

**Returns:** `(i, j)` or `None`.

---

### `solve_dice_sum(tile_images, prompt) -> Optional[tuple[int, int]]`

Full solver: parse target from prompt, get values for all tiles, find pair that sums to target.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile_images` | `list[PIL.Image.Image]` | One image per tile (e.g. 6 for 3×2 grid). |
| `prompt` | `str` | Instruction text containing the target sum. |

**Returns:** `(i, j)` (0-based indices to click) or `None` if target could not be parsed or no pair sums to target.

---

### `solve_dice_sum_with_prompt_parse(tile_images, target=None, prompt=None) -> Optional[tuple[int, int]]`

Same as above but accepts either a numeric **target** or a **prompt** string (or both; target takes precedence over parsing prompt).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tile_images` | `list[PIL.Image.Image]` | — | One image per tile. |
| `target` | `int \| None` | `None` | If set, used directly (no prompt parsing). |
| `prompt` | `str \| None` | `None` | If target is None, parsed for target sum. |

**Returns:** `(i, j)` or `None`.

---

## Training the dice classifier (optional)

The 6-way classifier is trained on **synthetic** dice faces: generated images with 1–6 pips and augmentation (flip, colour jitter). No external images required.

```bash
python -m funcaptcha.train_dice --epochs 15 --batch-size 32 --batches-per-epoch 50
```

- **Output:** `funcaptcha/models/dice_classifier.pt`.  
- Once this file exists, `dice_values_from_tiles(..., use_ml=True)` uses it automatically.  
- For real CAPTCHA dice with different art or lighting, consider fine-tuning or training on real tile screenshots labeled 1–6.

---

## Usage examples

### Logic only

```python
from PIL import Image
from funcaptcha.dice_sum import solve_dice_sum, parse_target_sum

tiles = [Image.open(f"tile_{i}.png") for i in range(6)]
prompt = "Select the pair of dice whose top sides add up to 14"
pair = solve_dice_sum(tiles, prompt)
if pair:
    i, j = pair
    print("Click tiles", i, "and", j)
else:
    print("No pair found or target not parsed")
```

### With browser

```python
from funcaptcha.dice_sum import solve_dice_sum
from funcaptcha.browser_actions import click_tiles

# Assume tile_elements = list of WebElement for each tile
tile_images = [Image.open(io.BytesIO(el.screenshot_as_png)).convert("RGB") for el in tile_elements]
prompt = "..."  # From CAPTCHA DOM
pair = solve_dice_sum(tile_images, prompt)
if pair:
    click_tiles(driver, tile_elements, list(pair))
# Then click Submit
```

---

## Constants and tuning

| Constant | Value | Description |
|----------|--------|-------------|
| `_MIN_PIPS` | 1 | Minimum dice value (clamp). |
| `_MAX_PIPS` | 6 | Maximum dice value (clamp). |

Pip-counting area bounds: 1%–35% of tile area. For non-standard dice or very small/large pips, you may need to adjust these in `count_pips` or rely on the ML classifier.

---

## Dependencies

- **Required:** `opencv-python-headless`, `numpy`, `Pillow`.  
- **Optional (ML):** `torch`; weights at `funcaptcha/models/dice_classifier.pt`.

---

## See also

Other funcaptcha methods: [funcaptcha-cycle-match.md](funcaptcha-cycle-match.md), [funcaptcha-rotation.md](funcaptcha-rotation.md), [funcaptcha-quantity.md](funcaptcha-quantity.md). Object-selection CAPTCHA: [object-selection-captcha.md](object-selection-captcha.md). UI actions (slider, arrows, +/−, tiles): `funcaptcha.browser_actions`.

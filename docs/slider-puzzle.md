# Slider puzzle captcha (drag the piece into the slot)

## Overview

The **slider_puzzle** package solves “drag the puzzle piece into the slot” captchas, such as TikTok’s web verification. It uses **OpenCV template matching** to compute the slide distance and **human-like Selenium drag** (variable steps, overshoot, settle) so the motion passes bot checks.

**Main entry point:** `solve_slider_puzzle(driver, container=None, fudge_px=-6)` — extracts images from the page, computes offset, performs humanized drag. Returns `True` if the drag was executed.

---

## Challenge type

| Property | Description |
|----------|-------------|
| **Name** | Puzzle slider / “Drag the slider to fit the puzzle piece” |
| **Where** | TikTok web, some Geetest/DataDome-style flows |
| **Input** | Background image (with slot) + piece image; or live page with captcha visible |
| **Output** | Horizontal slide distance (px or proportion), then a drag action |
| **UI** | Slider handle; user drags it so the piece aligns with the slot |

---

## Algorithm

### 1. Position (where to slide)

- **Template matching:** The piece is used as a template and slid over the background. OpenCV `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)` gives a score at each x; the x with the best score is the slot position.
- **Optional Canny edges:** If `use_edges=True`, both images are converted to grayscale, Gaussian-blurred, and Canny-edged before matching. This often improves robustness to lighting and color.
- **Scaling:** If the piece is larger than the background, it is resized so the template fits. Output is in pixels (relative to background width) or as a proportion in [0, 1].
- **Functions:** `get_slide_offset(background_image, piece_image, use_edges=True)` → `(offset_x, confidence)`; `get_slide_offset_as_proportion(...)` → `(proportion, confidence)`.

### 2. From position to slider movement

- **Proportion × track width:** For TikTok, the puzzle area width (from the wrapper element) is used: `delta_x = proportion * puzzle_width + fudge_px`.
- **Fudge factor:** A small correction (e.g. `fudge_px=-6`) is often needed so the piece aligns; tune per environment.
- **Clamp:** Negative `delta_x` is clamped to 0 before dragging.

### 3. Humanized drag

- The slider handle is moved by `delta_x` pixels in small steps (default 6 px), with variable delay, a slight overshoot (e.g. 5 px) then correction, then release. This mimics human movement and helps pass detection.
- **Function:** `slider_drag_humanized(driver, delta_x, container=None, ...)`.

### 4. End-to-end solve

- **Image extraction:** Background and piece are taken from the page (by default from TikTok selectors). Images are loaded from `img` `src` URLs when available, otherwise from element screenshots. Puzzle width is read from the wrapper or background element.
- **Flow:** `get_captcha_images` → `get_slide_offset_as_proportion` → `delta_x = proportion * puzzle_width + fudge_px` → `slider_drag_humanized`.
- **Function:** `solve_slider_puzzle(driver, container=None, fudge_px=-6, ...)`.

---

## API reference

### `get_slide_offset(background_image, piece_image, use_edges=True, method=cv2.TM_CCOEFF_NORMED) -> tuple[float, float]`

Finds the x-offset (pixels) where the piece best matches the background.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `background_image` | `PIL.Image.Image` | — | Full background (with slot). |
| `piece_image` | `PIL.Image.Image` | — | Movable piece (same scale as background). |
| `use_edges` | `bool` | `True` | Use Canny edge detection before matching. |
| `method` | `int` | `cv2.TM_CCOEFF_NORMED` | OpenCV matchTemplate method. |

**Returns:** `(offset_x, confidence)`. `offset_x` is the horizontal distance to slide; `confidence` in [0, 1].

---

### `get_slide_offset_as_proportion(background_image, piece_image, use_edges=True) -> tuple[float, float]`

Same as above but returns `(proportion, confidence)` with proportion in [0, 1] (offset_x / background width). Use for scaling: `slide_px = proportion * track_width`.

---

### `slider_drag_humanized(driver, delta_x, container=None, handle_element=None, handle_selector=DEFAULT_SLIDER_HANDLE, step_size=6, step_delay_ms=10, overshoot_px=5, overshoot_delay_ms=250, randomize=True) -> bool`

Drags the slider handle by `delta_x` pixels with human-like motion.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `driver` | `WebDriver` | — | Selenium WebDriver. |
| `delta_x` | `int` | — | Pixels to move (positive = right). |
| `container` | `WebElement \| None` | `None` | Scope for finding the handle. |
| `handle_element` | `WebElement \| None` | `None` | Slider handle; if None, looked up via selector. |
| `handle_selector` | `str` | `.secsdk-captcha-drag-icon` etc. | CSS selector for the handle. |
| `step_size` | `int` | `6` | Pixels per step. |
| `step_delay_ms` | `int` | `10` | Delay between steps (ms). |
| `overshoot_px` | `int` | `5` | Overshoot then correct. |
| `randomize` | `bool` | `True` | Slight randomness in steps/delays. |

**Returns:** `True` if the drag was performed, `False` if the handle was not found or an error occurred.

---

### `get_captcha_images(driver, container=None, bg_selector=..., piece_selector=..., wrapper_selector=..., timeout=5.0) -> tuple[Image | None, Image | None, float | None]`

Extracts background image, piece image, and puzzle width from the page.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `driver` | `WebDriver` | — | Selenium WebDriver. |
| `container` | `WebElement \| None` | `None` | Scope for element search. |
| `bg_selector` | `str` | `#captcha-verify-image` | CSS selector for background image. |
| `piece_selector` | `str` | `.captcha_verify_img_slide` | CSS selector for piece image. |
| `wrapper_selector` | `str` | `.captcha_verify_img--wrapper` | CSS selector for wrapper (puzzle width). |

**Returns:** `(background_pil, piece_pil, puzzle_width)`. Any of the three may be `None` if not found.

---

### `solve_slider_puzzle(driver, container=None, fudge_px=-6, use_edges=True, bg_selector=..., piece_selector=..., wrapper_selector=..., get_images=None) -> bool`

Full solve: get images → compute offset → scale + fudge → humanized drag.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `driver` | `WebDriver` | — | Selenium WebDriver. |
| `container` | `WebElement \| None` | `None` | CAPTCHA container. |
| `fudge_px` | `int` | `-6` | Pixels to add to computed distance (tune for TikTok). |
| `use_edges` | `bool` | `True` | Use Canny in template matching. |
| `get_images` | `callable \| None` | `None` | If set, `get_images(driver, container)` must return `(bg, piece, width)`; overrides selectors. |

**Returns:** `True` if the drag was performed, `False` otherwise. The caller must click **Verify** / **Submit** after a successful drag if the page requires it.

---

## Default selectors (TikTok)

| Role | Selector |
|------|----------|
| Background image | `#captcha-verify-image` |
| Piece image | `.captcha_verify_img_slide` |
| Wrapper (width) | `.captcha_verify_img--wrapper` |
| Slider handle | `div[class*='slider'] div[class*='handle'], .secsdk-captcha-drag-icon` |

Override via `bg_selector`, `piece_selector`, `wrapper_selector` or a custom `get_images` in `solve_slider_puzzle`.

---

## Fudge factor

TikTok’s layout often needs a small correction. Use `fudge_px=-6` as a starting point; if the piece is consistently short or long, try -8, -4, or 0.

---

## Testing

1. **Unit tests:** `pytest tests/test_slider_puzzle.py -v` — position logic on synthetic images.
2. **Position only with saved images:** Save background and piece from a real captcha, then run `python scratch/test_slider_puzzle_images.py bg.png piece.png` to see offset and confidence.
3. **Full solve in browser:** Trigger a slider captcha (e.g. on TikTok), call `solve_slider_puzzle(driver, container=..., fudge_px=-6)`, then click Verify. Success = captcha closes.
4. **With scraper:** When `object_selection_captcha` detects `captcha_type == "slider"`, call `solve_slider_puzzle(driver, info.container_element)` and then click the verify button.

---

## Dependencies

- **Required:** `opencv-python-headless`, `numpy`, `Pillow` (position); `selenium` (drag + solve).
- No PyTorch or external solver APIs.

---

## See also

- [object-selection-captcha.md](object-selection-captcha.md) — Shape/grid/click CAPTCHAs; when slider is detected, you can call the slider_puzzle solver from there.
- Package README: `slider_puzzle/README.md`.

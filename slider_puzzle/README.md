# Slider puzzle captcha solver

Solves "drag the puzzle piece into the slot" captchas (e.g. TikTok web). Uses **OpenCV template matching** for the slide distance and **human-like Selenium drag** so the motion passes bot checks.

## Quick start

```python
from selenium import webdriver
from slider_puzzle import solve_slider_puzzle

driver = webdriver.Chrome()
# ... load page until captcha appears ...

container = driver.find_element("css selector", ".captcha container")
ok = solve_slider_puzzle(driver, container=container, fudge_px=-6)
if ok:
    # Click "Verify" / "Submit" if needed
    ...
```

## API

- **`get_slide_offset(background_image, piece_image, use_edges=True)`**  
  Returns `(offset_x_px, confidence)`. Use when you already have PIL Images.

- **`get_slide_offset_as_proportion(background_image, piece_image)`**  
  Returns `(proportion, confidence)` in [0, 1] for scaling to any track width.

- **`slider_drag_humanized(driver, delta_x, container=None, ...)`**  
  Drags the slider handle by `delta_x` pixels with variable steps, overshoot and settle.

- **`get_captcha_images(driver, container=None, ...)`**  
  Extracts background image, piece image, and puzzle width from the page (TikTok selectors by default).

- **`solve_slider_puzzle(driver, container=None, fudge_px=-6, ...)`**  
  Full flow: get images → compute offset → scale + fudge → humanized drag.

## TikTok selectors (default)

- Background: `#captcha-verify-image`
- Piece: `.captcha_verify_img_slide`
- Wrapper (for width): `.captcha_verify_img--wrapper`
- Handle: `.secsdk-captcha-drag-icon` (in drag module)

Override with `bg_selector`, `piece_selector`, `wrapper_selector` or pass a custom `get_images(driver, container)` to `solve_slider_puzzle`.

## Fudge factor

TikTok often needs a small correction (e.g. `fudge_px=-6`). Tune per environment.

## Dependencies

- `opencv-python-headless`, `numpy`, `Pillow` (position)
- `selenium` (drag + solve)

No PyTorch or external solver services.

## Testing

### 1. Unit tests (no browser, no captcha)

From the repo root:

```bash
pytest tests/test_slider_puzzle.py -v
```

This runs the position logic on synthetic images.

### 2. Position only with saved images

If you have a real captcha screenshot: save the **background** and **piece** as two images (e.g. `bg.png`, `piece.png`). Then:

```bash
python scratch/test_slider_puzzle_images.py bg.png piece.png
```

The script prints the computed offset (pixels and proportion) and confidence. Check that the offset is a plausible slide distance (e.g. 50–300 px for a ~340 px wide puzzle).

### 3. Full solve in the browser

1. Trigger a slider captcha (e.g. use your scraper with TikTok until a captcha appears, or open TikTok in a logged-out session and refresh/scroll until the verification pops up).
2. In your code, once the captcha is visible, run:

   ```python
   from slider_puzzle import solve_slider_puzzle

   container = driver.find_element("css selector", "div[class*='captcha']")  # or your container
   ok = solve_slider_puzzle(driver, container=container, fudge_px=-6)
   ```
3. If the slider moves to roughly the right place and the captcha passes after you click Verify, it’s working. If it’s consistently off, adjust `fudge_px` (e.g. try -8, -4, 0).

### 4. Quick live check with your scraper

Run your scraper with `solve_captcha=True` and wire slider handling: when `object_selection_captcha` detects `captcha_type == "slider"`, call `solve_slider_puzzle(driver, info.container_element)` then click the verify button. Success = captcha closes and the page continues.

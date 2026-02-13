# rotation_captcha

Solver for **circular rotation captchas** — the kind where an outer ring shows a
reference image and an inner circle is rotated; the user drags a slider to align
the inner piece with the outer.  Commonly seen on TikTok.

## Package structure

```
rotation_captcha/
├── __init__.py   # public API re-exports
├── angle.py      # angle estimation (overlay + ring methods)
├── drag.py       # human-like slider drag (Selenium)
└── solve.py      # end-to-end solver + image extraction
```

## Quick start

### End-to-end (with Selenium)

```python
from selenium import webdriver
from rotation_captcha import solve_rotation_captcha

driver = webdriver.Chrome()
driver.get("https://www.tiktok.com/login")
# … navigate to the point where the rotation captcha appears …

success = solve_rotation_captcha(driver)
print("Solved!" if success else "Failed.")
```

### Angle estimation only (no browser)

```python
from PIL import Image
from rotation_captcha import estimate_angle

outer = Image.open("outer.png")
inner = Image.open("inner.png")

angle = estimate_angle(inner, outer, method="overlay", angle_step=2)
print(f"Rotate inner by {angle:.1f}° to match outer")
```

## Angle estimation methods

| Method    | How it works | When to use |
|-----------|-------------|-------------|
| **overlay** (default) | Rotates the inner image at every candidate angle and computes the mean absolute pixel difference inside the inner circle region against the outer image. Picks the angle with the minimum difference. | General purpose; works well when both images have similar lighting. |
| **ring** | Samples a ring of pixels just inside and just outside the boundary between inner and outer images. Uses perceptual (YUV) color distance. Picks the angle with the smallest total deviation. | More robust when inner/outer have different brightness or slight vignetting. |

Both methods default to 3° steps (120 candidates). Set `angle_step=1` for
higher precision at the cost of speed.

## Slider drag

`rotation_drag_humanized()` mimics a real user:

- Small steps (4 px default) with ±1 px randomization.
- Tiny vertical jitter during drag.
- Brief overshoot past the target, then correction back.
- Random pauses for "visual checking".

## CSS selectors

Default selectors target TikTok's captcha DOM:

| Element | Default selector |
|---------|-----------------|
| Outer image | `#captcha-verify-image` |
| Inner image | `img.captcha_verify_img_slide, img[class*='whirl-inner'], img[class*='rotate-inner']` |
| Slider track | `div.captcha_verify_slide--slidebar, div[class*='slider-track'], div[class*='secsdk-captcha-drag']` |
| Handle | `div.secsdk-captcha-drag-icon, div[class*='slider'] div[class*='handle']` |

Override any selector via keyword arguments to `solve_rotation_captcha()`.

## Dependencies

- **opencv-python** (`cv2`) — image processing and rotation
- **Pillow** (`PIL`) — image I/O
- **NumPy** — array operations
- **Selenium** — browser automation (only needed for drag / end-to-end solve)

## Testing without a browser

Save outer and inner images locally, then:

```python
from PIL import Image
from rotation_captcha.angle import estimate_angle_overlay, estimate_angle_ring

outer = Image.open("outer.png")
inner = Image.open("inner.png")

angle_o, diff = estimate_angle_overlay(inner, outer, angle_step=2)
angle_r, dev  = estimate_angle_ring(inner, outer, angle_step=2)
print(f"overlay: {angle_o:.1f}°  (diff={diff:.2f})")
print(f"ring:    {angle_r:.1f}°  (dev={dev:.2f})")
```

# Rotation Captcha Solver

> Package: `rotation_captcha/`

## Overview

Solves circular rotation captchas — the variant where:

- An **outer ring** shows the reference (background) image.
- An **inner circle** is rotated out of alignment.
- A **horizontal slider** lets the user rotate the inner circle until it
  matches the outer ring.

This is the captcha TikTok commonly presents during login and sign-up flows.

---

## How it works

```
┌──────────────┐   ┌──────────────┐       angle        ┌────────────┐   delta_x   ┌──────────────┐
│ Outer (full)  │ + │ Inner (frag) │  ──► estimate  ──► │ angle→px   │ ──────────► │ humanised    │
│ image         │   │ image        │      angle          │ converter  │             │ slider drag  │
└──────────────┘   └──────────────┘                     └────────────┘             └──────────────┘
```

1. **Image extraction** – Grab the outer and inner images from the page (via
   `src` URL or element screenshot).
2. **Angle estimation** – Try every candidate angle (default 3° steps = 120
   candidates) and score each one.
3. **Angle → pixels** – Convert the winning angle to a pixel delta using the
   measured slider track length.
4. **Human-like drag** – Execute the slider movement with variable speed,
   vertical jitter, overshoot, and correction.

---

## Angle estimation methods

### 1. Overlay (default)

Rotates the inner image by each candidate angle and computes the **mean
absolute pixel difference** against the outer image, masked to the inner
circle region only.  The angle that produces the minimum difference wins.

```python
from rotation_captcha.angle import estimate_angle_overlay

angle, diff = estimate_angle_overlay(inner_pil, outer_pil, angle_step=2)
```

**Strengths:** Simple, fast, no training data.
**Weakness:** Sensitive to brightness or contrast differences between inner
and outer images.

### 2. Ring (boundary matching)

Samples a ring of pixels just *inside* the inner boundary and another ring
just *outside* (on the outer image).  Uses **perceptual YUV color distance**
at each point.  The angle that minimises total deviation wins.

```python
from rotation_captcha.angle import estimate_angle_ring

angle, dev = estimate_angle_ring(inner_pil, outer_pil, angle_step=2)
```

**Strengths:** More robust when lighting differs between inner and outer
regions (vignetting, shadow).
**Weakness:** Slightly slower; sensitive to `ring_offset` parameter.

### Choosing a method

| Scenario | Recommended |
|----------|------------|
| Same lighting across both regions | `overlay` |
| Noticeable brightness/vignette difference | `ring` |
| Unsure | Try both and pick the one with the better score |

---

## API reference

### `rotation_captcha.angle`

| Function | Description |
|----------|-------------|
| `estimate_angle_overlay(fragment, background, angle_step, inner_radius_ratio)` | Returns `(angle, diff)` |
| `estimate_angle_ring(inner_image, outer_image, angle_step, inner_radius, ring_offset, n_points)` | Returns `(angle, deviation)` |
| `estimate_angle(inner_image, outer_image, method, angle_step, ...)` | Unified wrapper; returns `float` angle |
| `angle_to_slider_delta(angle_degrees, track_length_px, full_rotation_degrees)` | Returns `int` pixel delta |

### `rotation_captcha.drag`

| Function | Description |
|----------|-------------|
| `rotation_drag_humanized(driver, delta_x, container, handle_element, handle_selector, *, step_size, step_delay_ms, overshoot_px, overshoot_delay_ms, randomize)` | Human-like slider drag; returns `bool` |

### `rotation_captcha.solve`

| Function | Description |
|----------|-------------|
| `get_captcha_images(driver, container, outer_selector, inner_selector, timeout)` | Returns `(outer_pil, inner_pil)` |
| `get_track_length(driver, container, track_selector, handle_selector)` | Returns `int` track length in px |
| `solve_rotation_captcha(driver, container, *, method, angle_step, fudge_px, ...)` | End-to-end solver; returns `bool` |

---

## Default CSS selectors (TikTok)

| Element | Selector |
|---------|----------|
| Outer image | `#captcha-verify-image` |
| Inner image | `img.captcha_verify_img_slide, img[class*='whirl-inner'], img[class*='rotate-inner']` |
| Slider track | `div.captcha_verify_slide--slidebar, div[class*='slider-track'], div[class*='secsdk-captcha-drag']` |
| Handle | `div.secsdk-captcha-drag-icon, div[class*='slider'] div[class*='handle']` |

Override via keyword arguments to `solve_rotation_captcha()` or
`get_captcha_images()`.

---

## Human-like drag details

`rotation_drag_humanized()` differs from the slider-puzzle drag:

- **Smaller steps** (4 px vs 6 px) — finer angular control.
- **Vertical jitter** on every step — real fingers don't move in a perfect line.
- **Smaller overshoot** (3 px) — rotation sliders are more sensitive.
- **Random "checking" pause** before the overshoot correction.

All timing and distance parameters are configurable.

---

## Tuning tips

| Parameter | Effect | Default |
|-----------|--------|---------|
| `angle_step` | Precision vs speed trade-off. 1° = best accuracy, 120× slower than 3°. | 3 |
| `fudge_px` | Constant correction added to the pixel delta. Adjust if consistently over/under. | 0 |
| `inner_radius_ratio` | Overlay method only — fraction of half-image that is the inner circle. | 0.4 |
| `ring_offset` | Ring method only — how many pixels inside/outside the boundary to sample. | 5 |
| `overshoot_px` | Pixels to overshoot before correcting. Larger = more human-like, riskier. | 3 |

---

## Testing without a browser

```python
from PIL import Image
from rotation_captcha.angle import estimate_angle_overlay, estimate_angle_ring

outer = Image.open("outer.png")
inner = Image.open("inner.png")

angle_o, diff = estimate_angle_overlay(inner, outer, angle_step=2)
angle_r, dev  = estimate_angle_ring(inner, outer, angle_step=2)
print(f"overlay: {angle_o:.1f}° (diff={diff:.2f})")
print(f"ring:    {angle_r:.1f}° (dev={dev:.2f})")
```

---

## See also

- [Slider Puzzle](slider-puzzle.md) — sliding puzzle piece captcha solver
- [Object Selection Captcha](object-selection-captcha.md) — shape/grid/click captchas
- [FunCaptcha – Rotation](funcaptcha-rotation.md) — the FunCaptcha 3D rotation variant (arrow-button based)
- [FunCaptcha – Cycle Match](funcaptcha-cycle-match.md)
- [FunCaptcha – Quantity](funcaptcha-quantity.md)
- [FunCaptcha – Dice Sum](funcaptcha-dice-sum.md)

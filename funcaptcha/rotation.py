"""
3D rotation CAPTCHA solver (FunCaptcha "rotate to match this angle" style).

Two strategies:
- Classical: sample rotation angles, rotate fragment and compare to background (min diff = correct angle).
- Slider/arrow UI: map angle to pixel drag or N arrow clicks.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Sample every N degrees for classical search
_ANGLE_STEP = 5
# For "rotate fragment to fit" type: we need fragment mask or assume full image is the rotated asset
_DEFAULT_SIZE = 256


def _to_bgr(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def estimate_angle_rotnet(image: Image.Image, image_size: int = 128) -> Optional[float]:
    """Use trained RotNet if available. Returns angle in [0, 360) or None."""
    try:
        from funcaptcha.ml_models import load_rotnet, _device
        model = load_rotnet()
        if model is None:
            return None
        torch = __import__("torch")
        device = _device()
        bgr = _to_bgr(image)
        bgr = cv2.resize(bgr, (image_size, image_size))
        x = torch.from_numpy(bgr.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            logits = model(x)
        angle = model.predict_angle_deg(logits).item()
        logger.debug("estimate_angle_rotnet: %.1f deg", angle)
        return float(angle % 360.0)
    except Exception as e:
        logger.debug("RotNet not used: %s", e)
        return None


def estimate_angle(
    image: Image.Image,
    use_rotnet: bool = True,
    center: Optional[tuple[float, float]] = None,
    radius: Optional[float] = None,
    angle_step: int = _ANGLE_STEP,
) -> float:
    """
    Estimate rotation angle. If use_rotnet and trained model exists, use it;
    else use classical (Laplacian variance) search.
    """
    if use_rotnet:
        a = estimate_angle_rotnet(image)
        if a is not None:
            return a
    return estimate_angle_classical(image, center=center, radius=radius, angle_step=angle_step)


def estimate_angle_classical(
    image: Image.Image,
    center: Optional[tuple[float, float]] = None,
    radius: Optional[float] = None,
    angle_step: int = _ANGLE_STEP,
) -> float:
    """
    Classical approach: treat image as a rotated fragment. Sample angles;
    rotate image and compute edge/color difference from a reference (upright).
    The angle with minimum difference is the correction needed to "upright" the image.

    Assumes the CAPTCHA shows a single rotated object; we rotate the whole image
    and use self-consistency: at correct angle, edges align best (e.g. minimal
    high-frequency residual). Simplified: we search for angle that minimizes
    a simple "rotation artifact" metric (e.g. variance of Laplacian after rotate).

    Actually for "rotate to fit" type: reference is the background, fragment is
    the rotated piece. Here we don't have separate background/fragment; we have
    one image. So we use: "angle that makes the image look most upright" by
    sampling rotations and picking the one that maximizes edge alignment or
    minimizes symmetric difference. A common approach: at 0° the content is
    "correct"; we search for angle θ such that rotating by -θ gives "best" result.
    Best = e.g. maximum vertical symmetry or minimum entropy.

    Simpler: return angle where the sum of squared gradients (edge strength) is
    maximized after rotation (many rotate CAPTCHAs have clearer edges when upright).
    Or: minimize difference between rotated image and 180° rotated version (symmetry).
    """
    bgr = _to_bgr(image)
    h, w = bgr.shape[:2]
    cx = center[0] if center else w / 2
    cy = center[1] if center else h / 2
    r = radius if radius else min(w, h) / 2

    best_angle = 0.0
    best_score = -1.0

    for deg in range(0, 360, angle_step):
        M = cv2.getRotationMatrix2D((cx, cy), float(deg), 1.0)
        rotated = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        # Score: Laplacian variance (edge strength) — often higher when "upright"
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        score = float(np.var(lap))
        if score > best_score:
            best_score = score
            best_angle = float(deg)

    logger.debug("estimate_angle_classical: best angle=%.1f (score=%.2f)", best_angle, best_score)
    return best_angle


def estimate_angle_fragment_to_background(
    fragment: Image.Image,
    background: Image.Image,
    angle_step: int = _ANGLE_STEP,
) -> float:
    """
    Given a fragment (rotated piece) and background (full image with hole),
    find the angle that best aligns the fragment with the background.
    Returns rotation in degrees (0–360) to apply to fragment to fit.
    """
    frag_bgr = _to_bgr(fragment)
    bg_bgr = _to_bgr(background)
    fh, fw = frag_bgr.shape[:2]
    bh, bw = bg_bgr.shape[:2]
    # Resize fragment to match background if needed
    if (fw, fh) != (bw, bh):
        frag_bgr = cv2.resize(frag_bgr, (bw, bh), interpolation=cv2.INTER_LINEAR)

    best_angle = 0.0
    best_diff = float("inf")
    cx, cy = bw / 2, bh / 2

    for deg in range(0, 360, angle_step):
        M = cv2.getRotationMatrix2D((cx, cy), float(deg), 1.0)
        rotated = cv2.warpAffine(frag_bgr, M, (bw, bh), borderMode=cv2.BORDER_REPLICATE)
        diff = np.mean(np.abs(rotated.astype(float) - bg_bgr.astype(float)))
        if diff < best_diff:
            best_diff = diff
            best_angle = float(deg)

    return best_angle


def angle_to_slider_delta(angle_degrees: float, track_length_px: int, full_rotation_degrees: float = 360.0) -> int:
    """Convert target rotation angle to slider pixel delta (positive = drag right)."""
    return int(angle_degrees / full_rotation_degrees * track_length_px)


def angle_to_arrow_clicks(angle_degrees: float, degrees_per_click: float = 15.0) -> tuple[bool, int]:
    """
    Convert angle to number of arrow clicks and direction.
    Returns (right: bool, n_clicks: int). Right=True means "click right arrow".
    """
    n = int(round(angle_degrees / degrees_per_click))
    if n >= 0:
        return True, n
    return False, -n


def solve_rotation_slider(
    get_image: callable,
    drag_slider: callable,
    track_length_px: int = 300,
    angle_step: int = _ANGLE_STEP,
    use_rotnet: bool = True,
) -> Optional[bool]:
    """
    High-level: estimate angle from current image, then drag slider by corresponding delta.
    *get_image* returns PIL Image of the current rotated state.
    *drag_slider* is callable(delta_x: int). Returns True if applied, False/None on error.
    """
    image = get_image()
    if image is None:
        return None
    angle = estimate_angle(image, use_rotnet=use_rotnet, angle_step=angle_step)
    delta = angle_to_slider_delta(angle, track_length_px)
    try:
        drag_slider(delta)
        return True
    except Exception as e:
        logger.warning("solve_rotation_slider: %s", e)
        return False


def solve_rotation_arrows(
    get_image: callable,
    click_left: callable,
    click_right: callable,
    degrees_per_click: float = 15.0,
    angle_step: int = _ANGLE_STEP,
    use_rotnet: bool = True,
) -> Optional[bool]:
    """
    High-level: estimate angle, then click right or left arrow N times.
    *click_left* / *click_right* are callables that perform one click.
    """
    image = get_image()
    if image is None:
        return None
    angle = estimate_angle(image, use_rotnet=use_rotnet, angle_step=angle_step)
    right, n = angle_to_arrow_clicks(angle, degrees_per_click)
    for _ in range(n):
        if right:
            click_right()
        else:
            click_left()
    return True

"""
Rotation angle estimation for circular rotation captchas.

Two methods:
- **Overlay** (primary): rotate the inner fragment at many angles, compare
  pixel difference against the full/background image; the angle with the
  minimum difference is the correct rotation.
- **Ring** (ycq0125-style alternative): compare only a ring of boundary
  pixels between inner and outer images; the angle with the minimum color
  deviation is chosen. Often more stable when lighting differs inside vs
  outside.

Both methods return an angle in [0, 360).
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_ANGLE_STEP = 3  # degrees


def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    arr = np.array(pil_image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _pil_to_hsv(pil_image: Image.Image) -> np.ndarray:
    bgr = _pil_to_bgr(pil_image)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image around its center by angle_deg (positive = CCW)."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Circular mask helpers
# ---------------------------------------------------------------------------

def _circle_mask(h: int, w: int, radius: float) -> np.ndarray:
    """Boolean mask: True inside a circle of given radius centered in (h, w)."""
    cy, cx = h / 2, w / 2
    Y, X = np.ogrid[:h, :w]
    return ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2


def _ring_pixels(img: np.ndarray, radius: float, n_points: int = 360) -> np.ndarray:
    """Sample n_points pixels along a circle of given radius (centered)."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    angles = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    xs = np.clip((cx + radius * np.cos(angles)).astype(int), 0, w - 1)
    ys = np.clip((cy + radius * np.sin(angles)).astype(int), 0, h - 1)
    return img[ys, xs]  # shape (n_points, channels)


# ---------------------------------------------------------------------------
# Method 1: Overlay (pixel difference in the inner region)
# ---------------------------------------------------------------------------

def estimate_angle_overlay(
    fragment: Image.Image,
    background: Image.Image,
    angle_step: int = _ANGLE_STEP,
    inner_radius_ratio: float = 0.4,
) -> Tuple[float, float]:
    """
    Find the rotation angle that best aligns the fragment (inner circle)
    with the background (full image) by minimizing mean absolute pixel
    difference inside the inner circle region.

    Parameters
    ----------
    fragment : PIL.Image
        The rotated inner circle (or full image; only the inner region is used).
    background : PIL.Image
        The full circular image (reference).
    angle_step : int
        Degrees between samples (smaller = more precise, slower).
    inner_radius_ratio : float
        Radius of the inner circle as a fraction of half the image size.

    Returns
    -------
    (angle_deg, min_diff) : tuple
        angle_deg in [0, 360); min_diff = mean abs diff at that angle (lower is better).
    """
    frag = _pil_to_bgr(fragment)
    bg = _pil_to_bgr(background)
    bh, bw = bg.shape[:2]

    # Resize fragment to match background
    if frag.shape[:2] != bg.shape[:2]:
        frag = cv2.resize(frag, (bw, bh), interpolation=cv2.INTER_LINEAR)

    # Mask: only compare inside the inner circle
    inner_r = inner_radius_ratio * min(bw, bh) / 2
    mask = _circle_mask(bh, bw, inner_r)

    best_angle = 0.0
    best_diff = float("inf")

    # --- Stage 1: coarse search over [0, 360) with given angle_step ---
    for deg in range(0, 360, angle_step):
        rotated = _rotate(frag, float(deg))
        diff = np.mean(np.abs(rotated[mask].astype(float) - bg[mask].astype(float)))
        if diff < best_diff:
            best_diff = diff
            best_angle = float(deg)

    # --- Stage 2: local refinement around the best coarse angle ---
    # Search a small window around best_angle at 1 degree resolution.
    # This improves precision without a full 360° x 1° sweep.
    fine_step = 1
    window = max(6, 2 * angle_step)  # at least ±6°, or wider than coarse step
    start = int(best_angle - window)
    end = int(best_angle + window) + 1
    for deg in range(start, end, fine_step):
        d = deg % 360
        rotated = _rotate(frag, float(d))
        diff = np.mean(np.abs(rotated[mask].astype(float) - bg[mask].astype(float)))
        if diff < best_diff:
            best_diff = diff
            best_angle = float(d)

    logger.debug("estimate_angle_overlay: angle=%.1f diff=%.2f", best_angle, best_diff)
    return (best_angle, best_diff)


# ---------------------------------------------------------------------------
# Method 2: Ring (boundary pixel comparison in HSV)
# ---------------------------------------------------------------------------

def _hsv_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """YUV-based perceptual color distance (ycq0125 style)."""
    y1 = 0.299 * c1[0] + 0.587 * c1[1] + 0.114 * c1[2]
    u1 = -0.14713 * c1[0] - 0.28886 * c1[1] + 0.436 * c1[2]
    v1 = 0.615 * c1[0] - 0.51498 * c1[1] - 0.10001 * c1[2]
    y2 = 0.299 * c2[0] + 0.587 * c2[1] + 0.114 * c2[2]
    u2 = -0.14713 * c2[0] - 0.28886 * c2[1] + 0.436 * c2[2]
    v2 = 0.615 * c2[0] - 0.51498 * c2[1] - 0.10001 * c2[2]
    return math.sqrt((y1 - y2) ** 2 + (u1 - u2) ** 2 + (v1 - v2) ** 2)


def estimate_angle_ring(
    inner_image: Image.Image,
    outer_image: Image.Image,
    angle_step: int = _ANGLE_STEP,
    inner_radius: Optional[int] = None,
    ring_offset: int = 5,
    n_points: int = 360,
) -> Tuple[float, float]:
    """
    Find the rotation angle by comparing a ring of pixels at the boundary
    between the inner (fragment) and outer (background) images.

    For each candidate angle, rotate the inner image and sample pixels on a
    circle just inside the boundary; sample the outer image on a circle just
    outside. The angle that minimizes the total color deviation is chosen.

    Parameters
    ----------
    inner_image : PIL.Image
        The rotated inner circle.
    outer_image : PIL.Image
        The full/outer image.
    angle_step : int
        Degrees per sample.
    inner_radius : int, optional
        Radius of the inner circle in pixels. If None, inferred as 40% of half
        the image size.
    ring_offset : int
        Pixels inside / outside the boundary to sample (default 5).
    n_points : int
        Number of points around the ring (default 360).

    Returns
    -------
    (angle_deg, min_deviation) : tuple
        angle_deg in [0, 360); min_deviation = total color distance at that angle.
    """
    inner_bgr = _pil_to_bgr(inner_image)
    outer_bgr = _pil_to_bgr(outer_image)

    # Match sizes
    oh, ow = outer_bgr.shape[:2]
    if inner_bgr.shape[:2] != (oh, ow):
        inner_bgr = cv2.resize(inner_bgr, (ow, oh), interpolation=cv2.INTER_LINEAR)

    if inner_radius is None:
        inner_radius = int(0.4 * min(ow, oh) / 2)

    # Outer ring (static): pixels just outside the inner circle on the full image
    outer_ring = _ring_pixels(outer_bgr, inner_radius + ring_offset, n_points)

    best_angle = 0.0
    best_dev = float("inf")

    # --- Stage 1: coarse search over [0, 360) with given angle_step ---
    for deg in range(0, 360, angle_step):
        rotated = _rotate(inner_bgr, float(deg))
        inner_ring = _ring_pixels(rotated, inner_radius - ring_offset, n_points)
        total_dev = sum(
            _hsv_distance(inner_ring[i].astype(float), outer_ring[i].astype(float))
            for i in range(n_points)
        )
        if total_dev < best_dev:
            best_dev = total_dev
            best_angle = float(deg)

    # --- Stage 2: local refinement around the best coarse angle ---
    fine_step = 1
    window = max(6, 2 * angle_step)
    start = int(best_angle - window)
    end = int(best_angle + window) + 1
    for deg in range(start, end, fine_step):
        d = deg % 360
        rotated = _rotate(inner_bgr, float(d))
        inner_ring = _ring_pixels(rotated, inner_radius - ring_offset, n_points)
        total_dev = sum(
            _hsv_distance(inner_ring[i].astype(float), outer_ring[i].astype(float))
            for i in range(n_points)
        )
        if total_dev < best_dev:
            best_dev = total_dev
            best_angle = float(d)

    logger.debug("estimate_angle_ring: angle=%.1f deviation=%.2f", best_angle, best_dev)
    return (best_angle, best_dev)


# ---------------------------------------------------------------------------
# Combined estimator
# ---------------------------------------------------------------------------

def estimate_angle(
    inner_image: Image.Image,
    outer_image: Image.Image,
    method: str = "overlay",
    angle_step: int = _ANGLE_STEP,
    inner_radius: Optional[int] = None,
    inner_radius_ratio: float = 0.4,
) -> float:
    """
    Estimate the rotation angle of the inner circle relative to the outer.

    Parameters
    ----------
    inner_image : PIL.Image
        The rotated inner piece.
    outer_image : PIL.Image
        The full/outer image.
    method : str
        "overlay" (default) or "ring".
    angle_step : int
        Degrees per sample.
    inner_radius : int, optional
        For ring method: radius in pixels (None = auto).
    inner_radius_ratio : float
        For overlay method: inner radius as fraction of half image size.

    Returns
    -------
    float
        Angle in [0, 360).
    """
    if method == "ring":
        angle, _ = estimate_angle_ring(
            inner_image, outer_image,
            angle_step=angle_step,
            inner_radius=inner_radius,
        )
    else:
        angle, _ = estimate_angle_overlay(
            inner_image, outer_image,
            angle_step=angle_step,
            inner_radius_ratio=inner_radius_ratio,
        )
    return angle


# ---------------------------------------------------------------------------
# Angle → slider
# ---------------------------------------------------------------------------

def angle_to_slider_delta(
    angle_degrees: float,
    track_length_px: int,
    full_rotation_degrees: float = 360.0,
) -> int:
    """Convert rotation angle to slider pixel delta (positive = drag right)."""
    return int(angle_degrees / full_rotation_degrees * track_length_px)

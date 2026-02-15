"""
Unit tests for rotation_captcha.angle — angle estimation on synthetic images.

These tests create known-rotation pairs and verify that both the overlay
and ring methods recover the angle within a reasonable tolerance.
"""

import math
import numpy as np
import cv2
import pytest
from PIL import Image

from rotation_captcha.angle import (
    estimate_angle_overlay,
    estimate_angle_ring,
    estimate_angle,
    angle_to_slider_delta,
)


# ---------------------------------------------------------------------------
# Helpers: generate synthetic outer / inner image pairs
# ---------------------------------------------------------------------------

def _make_test_pair(
    size: int = 200,
    rotation_deg: float = 45.0,
    inner_radius_ratio: float = 0.4,
) -> tuple:
    """
    Create a synthetic (outer, inner) image pair where the inner is rotated
    by *rotation_deg* from the outer.

    The image contains a recognisable asymmetric pattern (coloured wedges)
    so rotation can be detected.

    Returns (outer_pil, inner_pil).
    """
    # Draw a colourful asymmetric circle image
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = size // 2 - 2

    # Paint 8 coloured wedges (so there's a clear orientation)
    colours = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    ]
    for i, colour in enumerate(colours):
        start_angle = i * 45
        end_angle = (i + 1) * 45
        cv2.ellipse(img, (cx, cy), (r, r), 0, start_angle, end_angle, colour, -1)

    # The outer image is the full wedge circle
    outer = img.copy()

    # The inner image is the same circle rotated by rotation_deg
    M = cv2.getRotationMatrix2D((cx, cy), rotation_deg, 1.0)
    inner = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE)

    outer_pil = Image.fromarray(cv2.cvtColor(outer, cv2.COLOR_BGR2RGB))
    inner_pil = Image.fromarray(cv2.cvtColor(inner, cv2.COLOR_BGR2RGB))
    return outer_pil, inner_pil


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEstimateAngleOverlay:
    @pytest.mark.parametrize("true_angle", [0, 30, 90, 135, 270])
    def test_recovers_known_angle(self, true_angle: int):
        outer, inner = _make_test_pair(rotation_deg=float(true_angle))
        angle, diff = estimate_angle_overlay(inner, outer, angle_step=3)
        # The solver should find the *negative* of the rotation applied
        # (rotating inner by `angle` should undo the initial rotation).
        # Because estimate_angle_overlay tries all angles and picks the one
        # that minimises difference, it should find ≈ (360 - true_angle) % 360.
        expected = (360 - true_angle) % 360
        error = min(abs(angle - expected), 360 - abs(angle - expected))
        assert error <= 6, (
            f"true_angle={true_angle}, expected≈{expected}, got={angle}, error={error}"
        )

    def test_zero_rotation(self):
        outer, inner = _make_test_pair(rotation_deg=0)
        angle, diff = estimate_angle_overlay(inner, outer, angle_step=3)
        assert angle == 0.0 or angle >= 357.0, f"Expected ~0, got {angle}"


class TestEstimateAngleRing:
    @pytest.mark.parametrize("true_angle", [0, 45, 120, 200])
    def test_recovers_known_angle(self, true_angle: int):
        outer, inner = _make_test_pair(rotation_deg=float(true_angle))
        angle, dev = estimate_angle_ring(inner, outer, angle_step=3, ring_offset=3)
        expected = (360 - true_angle) % 360
        error = min(abs(angle - expected), 360 - abs(angle - expected))
        assert error <= 12, (
            f"true_angle={true_angle}, expected≈{expected}, got={angle}, error={error}"
        )


class TestEstimateAngle:
    def test_overlay_method(self):
        outer, inner = _make_test_pair(rotation_deg=60)
        angle = estimate_angle(inner, outer, method="overlay", angle_step=3)
        expected = (360 - 60) % 360  # 300
        error = min(abs(angle - expected), 360 - abs(angle - expected))
        assert error <= 6

    def test_ring_method(self):
        outer, inner = _make_test_pair(rotation_deg=60)
        angle = estimate_angle(inner, outer, method="ring", angle_step=3)
        expected = 300
        error = min(abs(angle - expected), 360 - abs(angle - expected))
        assert error <= 12


class TestAngleToSliderDelta:
    def test_basic(self):
        assert angle_to_slider_delta(180, 360) == 180
        assert angle_to_slider_delta(90, 360) == 90
        assert angle_to_slider_delta(45, 260) == 32  # int(45/360 * 260) = 32

    def test_zero(self):
        assert angle_to_slider_delta(0, 300) == 0

    def test_full_rotation(self):
        assert angle_to_slider_delta(360, 300) == 300

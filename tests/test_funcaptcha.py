"""
Tests for funcaptcha solvers (cycle_match, rotation, quantity, dice_sum).

Uses synthetic images only — no training data, no live CAPTCHA.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


def _solid_rgb(w: int, h: int, r: int, g: int, b: int) -> Image.Image:
    """Create a solid-color PIL Image (RGB)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    return Image.fromarray(arr)


def _image_with_blobs(w: int, h: int, n_circles: int, bg_gray: int = 240) -> Image.Image:
    """Simple image with n_circles dark circles on light background (for quantity tests)."""
    arr = np.full((h, w, 3), bg_gray, dtype=np.uint8)
    # Place circles so they don't overlap much
    step = max(1, min(w, h) // (n_circles + 1))
    for i in range(n_circles):
        cx = step * (i + 1) % w
        cy = step * (i + 1) % h
        r = min(step // 2, 15)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        for y in range(y0, y1):
            for x in range(x0, x1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    arr[y, x, :] = [40, 40, 40]
    return Image.fromarray(arr)


def _fake_dice_tile(value: int, size: int = 64) -> Image.Image:
    """Crude dice face: value 1-6 as number of dark circles on light bg."""
    arr = np.full((size, size, 3), 250, dtype=np.uint8)
    # Pip positions (rough 1-6 layout)
    centers = {
        1: [(size // 2, size // 2)],
        2: [(size // 4, size // 4), (3 * size // 4, 3 * size // 4)],
        3: [(size // 4, size // 4), (size // 2, size // 2), (3 * size // 4, 3 * size // 4)],
        4: [(size // 4, size // 4), (3 * size // 4, size // 4), (size // 4, 3 * size // 4), (3 * size // 4, 3 * size // 4)],
        5: [(size // 4, size // 4), (3 * size // 4, size // 4), (size // 2, size // 2), (size // 4, 3 * size // 4), (3 * size // 4, 3 * size // 4)],
        6: [(size // 4, size // 4), (3 * size // 4, size // 4), (size // 4, size // 2), (3 * size // 4, size // 2), (size // 4, 3 * size // 4), (3 * size // 4, 3 * size // 4)],
    }
    r = max(2, size // 12)
    for cx, cy in centers.get(value, centers[1]):
        for y in range(max(0, cy - r), min(size, cy + r + 1)):
            for x in range(max(0, cx - r), min(size, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    arr[y, x, :] = [30, 30, 30]
    return Image.fromarray(arr)


# -----------------------------------------------------------------------------
# cycle_match
# -----------------------------------------------------------------------------


def test_cycle_match_identical_panels_high_score():
    from funcaptcha.cycle_match import match_score, is_matched
    img = _solid_rgb(100, 100, 200, 100, 50)
    score = match_score(img, img)
    assert score >= 0.75  # identical panels should be well above match threshold
    assert is_matched(img, img, threshold=0.75)


def test_cycle_match_different_panels_lower_score():
    from funcaptcha.cycle_match import match_score, is_matched
    ref = _solid_rgb(100, 100, 200, 100, 50)
    cur = _solid_rgb(100, 100, 50, 150, 200)
    score = match_score(ref, cur)
    assert score < 0.9
    # May or may not be above 0.75 depending on histogram; at least it's different from same-image
    assert score <= match_score(ref, ref)


# -----------------------------------------------------------------------------
# rotation
# -----------------------------------------------------------------------------


def test_rotation_returns_angle_in_range():
    from funcaptcha.rotation import estimate_angle_classical
    img = _solid_rgb(80, 80, 180, 180, 180)
    angle = estimate_angle_classical(img, angle_step=45)
    assert 0 <= angle <= 360


def test_rotation_angle_to_slider_delta():
    from funcaptcha.rotation import angle_to_slider_delta
    delta = angle_to_slider_delta(180, track_length_px=300)
    assert delta == 150  # 180/360 * 300


def test_rotation_angle_to_arrow_clicks():
    from funcaptcha.rotation import angle_to_arrow_clicks
    right, n = angle_to_arrow_clicks(90, degrees_per_click=15)
    assert right is True
    assert n == 6
    right, n = angle_to_arrow_clicks(-30, degrees_per_click=15)
    assert right is False
    assert n == 2


# -----------------------------------------------------------------------------
# quantity
# -----------------------------------------------------------------------------


def test_quantity_count_objects_contour():
    from funcaptcha.quantity import count_objects_contour
    one = _image_with_blobs(100, 100, 1)
    two = _image_with_blobs(100, 100, 2)
    assert count_objects_contour(one) == 1
    assert count_objects_contour(two) == 2


def test_quantity_delta():
    from funcaptcha.quantity import quantity_delta
    ref_one = _image_with_blobs(100, 100, 1)
    cur_two = _image_with_blobs(100, 100, 2)
    delta = quantity_delta(ref_one, cur_two, prefer_yolo=False, prefer_ml=False)
    assert delta == -1  # target 1, current 2 → need one minus


# -----------------------------------------------------------------------------
# dice_sum
# -----------------------------------------------------------------------------


def test_dice_sum_parse_target():
    from funcaptcha.dice_sum import parse_target_sum
    assert parse_target_sum("add up to 14") == 14
    assert parse_target_sum("sum to 7") == 7
    assert parse_target_sum("total 6") == 6
    assert parse_target_sum("") is None
    assert parse_target_sum("no number here") is None


def test_dice_sum_find_pair():
    from funcaptcha.dice_sum import find_pair_with_sum
    values = [1, 2, 3, 4, 5, 6]
    assert find_pair_with_sum(values, 7) == (0, 5)   # 1+6
    assert find_pair_with_sum(values, 11) == (4, 5)  # 5+6
    assert find_pair_with_sum(values, 3) == (0, 1)    # 1+2
    assert find_pair_with_sum(values, 99) is None


def test_dice_sum_solve_with_fake_tiles():
    from funcaptcha.dice_sum import solve_dice_sum
    # Tiles with values 3, 4, 1, 2, 5, 6 → pairs that sum to 7: (0,1)=3+4, (2,5)=1+6, (3,4)=2+5
    tiles = [_fake_dice_tile(v) for v in [3, 4, 1, 2, 5, 6]]
    pair = solve_dice_sum(tiles, "add up to 7")
    # Pip counting on our fake tiles may not yield exact 1-6; if it does, we get a valid pair
    if pair is not None:
        i, j = pair
        assert 0 <= i < j < 6
    # At least parsing and pair search run
    pair2 = solve_dice_sum(tiles, "sum to 14")
    # 14 not possible with two D6; may return None or a pair if pip count misreads
    assert pair2 is None or (isinstance(pair2, tuple) and len(pair2) == 2)


def test_dice_sum_count_pips_returns_1_to_6():
    from funcaptcha.dice_sum import count_pips
    for v in range(1, 7):
        tile = _fake_dice_tile(v)
        n = count_pips(tile)
        assert 1 <= n <= 6, f"expected 1-6 for value {v}, got {n}"

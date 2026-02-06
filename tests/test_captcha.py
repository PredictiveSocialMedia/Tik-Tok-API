"""
Tests for the captcha package.

Covers pure logic (Detection, tile geometry, prompt resolution, edge density)
without requiring a real browser or YOLO model download.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from captcha.detector import (
    Detection,
    classify_tiles,
    find_click_points,
    resolve_prompt_labels,
    tile_edge_density,
)
from captcha.browser import CaptchaInfo
from captcha.solver import _extract_bold_word


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------


class TestDetection:
    def test_center(self):
        d = Detection("car", 0.9, 10, 20, 50, 60)
        assert d.center == (30.0, 40.0)

    def test_area(self):
        d = Detection("car", 0.9, 0, 0, 10, 10)
        assert d.area == 100.0

    def test_iou_with_itself(self):
        d = Detection("car", 0.9, 0, 0, 10, 10)
        assert abs(d.iou_with_box(0, 0, 10, 10) - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        d = Detection("car", 0.9, 0, 0, 10, 10)
        assert d.iou_with_box(20, 20, 30, 30) == 0.0

    def test_iou_partial(self):
        d = Detection("car", 0.9, 0, 0, 10, 10)
        iou = d.iou_with_box(5, 5, 15, 15)
        # Intersection = 5*5 = 25, union = 100 + 100 - 25 = 175
        assert abs(iou - 25 / 175) < 1e-6

    def test_repr(self):
        d = Detection("bus", 0.85, 1, 2, 3, 4)
        r = repr(d)
        assert "bus" in r
        assert "0.85" in r


# ---------------------------------------------------------------------------
# resolve_prompt_labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt,expected_contains",
    [
        ("Select all images with traffic lights", "traffic light"),
        ("Select all squares with a bus", "bus"),
        ("Click on the bicycle", "bicycle"),
        ("Find the fire hydrant", "fire hydrant"),
        ("Select all images with a stop sign", "stop sign"),
    ],
)
def test_resolve_prompt_labels(prompt, expected_contains):
    labels = resolve_prompt_labels(prompt)
    assert expected_contains in labels


def test_resolve_prompt_labels_unknown():
    labels = resolve_prompt_labels("Select all images with waffles")
    # "waffles" is not in COCO
    assert labels == []


# ---------------------------------------------------------------------------
# _extract_bold_word (solver helper)
# ---------------------------------------------------------------------------


def test_extract_bold_word_html():
    prompt = "Select all images with <strong>traffic lights</strong>"
    labels = _extract_bold_word(prompt)
    assert "traffic light" in labels


def test_extract_bold_word_markdown():
    prompt = "Select all images with **buses**"
    labels = _extract_bold_word(prompt)
    assert "bus" in labels


def test_extract_bold_word_fallback_tail():
    prompt = "Please click on the fire hydrant"
    labels = _extract_bold_word(prompt)
    assert "fire hydrant" in labels


# ---------------------------------------------------------------------------
# tile_edge_density
# ---------------------------------------------------------------------------


def test_edge_density_flat_image():
    """A solid colour tile should have near-zero edge density."""
    tile = Image.fromarray(np.full((50, 50, 3), 128, dtype=np.uint8))
    assert tile_edge_density(tile) < 1.0


def test_edge_density_noisy_image():
    """A random/noisy tile should have higher edge density than a flat one."""
    rng = np.random.RandomState(42)
    noisy = Image.fromarray(rng.randint(0, 255, (50, 50, 3), dtype=np.uint8))
    flat = Image.fromarray(np.full((50, 50, 3), 128, dtype=np.uint8))
    assert tile_edge_density(noisy) > tile_edge_density(flat)


# ---------------------------------------------------------------------------
# classify_tiles (mocked YOLO)
# ---------------------------------------------------------------------------


def _make_test_image(w=300, h=300):
    """Create a simple 300×300 test image."""
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def test_classify_tiles_with_mock_detections():
    """
    Mock detect_objects to return a detection in the top-left corner of a 3×3 grid.
    That detection should match tile 0.
    """
    fake_det = Detection("traffic light", 0.9, 10, 10, 90, 90)

    with patch("captcha.detector.detect_objects", return_value=[fake_det]):
        img = _make_test_image()
        indices = classify_tiles(img, 3, 3, ["traffic light"])

    assert 0 in indices  # top-left tile (0,0) → index 0


def test_classify_tiles_centre_detection():
    """Detection in the dead centre of a 3×3 grid → tile 4."""
    fake_det = Detection("bus", 0.8, 110, 110, 190, 190)  # centre of 300×300

    with patch("captcha.detector.detect_objects", return_value=[fake_det]):
        img = _make_test_image()
        indices = classify_tiles(img, 3, 3, ["bus"])

    assert 4 in indices


def test_classify_tiles_no_match():
    """No detections → empty result."""
    with patch("captcha.detector.detect_objects", return_value=[]):
        img = _make_test_image()
        indices = classify_tiles(img, 3, 3, ["car"])

    assert indices == []


# ---------------------------------------------------------------------------
# find_click_points (mocked YOLO)
# ---------------------------------------------------------------------------


def test_find_click_points():
    """find_click_points returns centres of detections."""
    fake_det = Detection("bicycle", 0.7, 100, 100, 200, 200)

    with patch("captcha.detector.detect_objects", return_value=[fake_det]):
        img = _make_test_image()
        points = find_click_points(img, ["bicycle"])

    assert len(points) == 1
    assert points[0] == (150.0, 150.0)


def test_find_click_points_empty():
    with patch("captcha.detector.detect_objects", return_value=[]):
        img = _make_test_image()
        points = find_click_points(img, ["airplane"])

    assert points == []


# ---------------------------------------------------------------------------
# CaptchaInfo dataclass
# ---------------------------------------------------------------------------


def test_captcha_info_defaults():
    info = CaptchaInfo(captcha_type="grid", prompt="Select traffic lights")
    assert info.grid_rows == 0
    assert info.grid_cols == 0
    assert info.tile_elements == []
    assert info.iframe is None


# ---------------------------------------------------------------------------
# handle_captcha integration (fully mocked)
# ---------------------------------------------------------------------------


def test_handle_captcha_no_captcha():
    """When no CAPTCHA is detected, handle_captcha returns True immediately."""
    from captcha.solver import handle_captcha

    mock_driver = MagicMock()
    with patch("captcha.solver.detect_captcha", return_value=None):
        assert handle_captcha(mock_driver) is True

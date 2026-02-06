"""
Tests for captcha/tiktok_detector.py.

Tests the hybrid approach: YOLO for object localization + visual crop
comparison for matching.  YOLO is mocked; the crop comparison functions
are tested with real synthetic images.
"""

from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from PIL import Image

from captcha.tiktok_detector import (
    TikTokDetection,
    find_matching_pair_yolo,
    detect_tiktok_objects,
    is_model_available,
    _crop_detection,
    _compare_crops,
    _make_foreground_mask,
)


# ---------------------------------------------------------------------------
# TikTokDetection dataclass
# ---------------------------------------------------------------------------


class TestTikTokDetection:
    def test_center(self):
        d = TikTokDetection("A", 0.9, 10, 20, 50, 60)
        assert d.center == (30, 40)

    def test_area(self):
        d = TikTokDetection("B", 0.8, 0, 0, 10, 20)
        assert d.area == 200.0

    def test_area_zero_when_degenerate(self):
        d = TikTokDetection("X", 0.5, 10, 10, 10, 10)
        assert d.area == 0.0

    def test_repr(self):
        d = TikTokDetection("star", 0.95, 100, 100, 200, 200)
        r = repr(d)
        assert "star" in r
        assert "0.95" in r
        assert "TikTokDetection" in r


# ---------------------------------------------------------------------------
# Helpers: create mock YOLO results
# ---------------------------------------------------------------------------


def _make_mock_box(cls_id: int, conf: float, xyxy: list[float]):
    """Create a mock YOLO box object."""
    box = MagicMock()
    box.cls = np.array([cls_id])
    box.conf = np.array([conf])
    box.xyxy = np.array([xyxy])
    return box


def _make_mock_results(boxes_data: list[tuple[int, float, list[float]]]):
    """Create mock YOLO results."""
    boxes = [_make_mock_box(cls_id, conf, xyxy) for cls_id, conf, xyxy in boxes_data]
    result = MagicMock()
    result.boxes = boxes
    return result


# ---------------------------------------------------------------------------
# Helpers: create synthetic images with distinct objects
# ---------------------------------------------------------------------------


def _make_image_with_shapes(w=400, h=400) -> Image.Image:
    """
    Create a test image with 4 distinct objects:
      - Two RED circles (at (60,60) and (340,340)) — the matching pair
      - One BLUE square (at (300,60))
      - One GREEN triangle (at (60,300))
    """
    img = np.full((h, w, 3), 220, dtype=np.uint8)  # light grey background

    # Red circle 1 (top-left)
    cv2.circle(img, (60, 60), 30, (0, 0, 200), -1)
    # Red circle 2 (bottom-right) — same shape & colour
    cv2.circle(img, (340, 340), 30, (0, 0, 200), -1)
    # Blue square (top-right)
    cv2.rectangle(img, (270, 30), (330, 90), (200, 0, 0), -1)
    # Green triangle (bottom-left)
    pts = np.array([[60, 270], [30, 330], [90, 330]])
    cv2.fillPoly(img, [pts], (0, 200, 0))

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _make_detections_for_shapes() -> list[TikTokDetection]:
    """Detections matching the shapes in _make_image_with_shapes."""
    return [
        TikTokDetection("chu_khoi", 0.95, 30, 30, 90, 90),     # red circle 1
        TikTokDetection("chu_khoi", 0.90, 270, 30, 330, 90),    # blue square
        TikTokDetection("chu_khoi", 0.88, 30, 270, 90, 330),    # green triangle
        TikTokDetection("chu_khoi", 0.92, 310, 310, 370, 370),  # red circle 2
    ]


# ---------------------------------------------------------------------------
# is_model_available
# ---------------------------------------------------------------------------


class TestModelAvailable:
    def test_returns_false_when_no_model(self, tmp_path):
        assert is_model_available(tmp_path / "nonexistent.pt") is False

    def test_returns_true_when_model_exists(self, tmp_path):
        model_file = tmp_path / "test_model.pt"
        model_file.write_bytes(b"fake model data")
        assert is_model_available(model_file) is True


# ---------------------------------------------------------------------------
# detect_tiktok_objects (mocked YOLO)
# ---------------------------------------------------------------------------


class TestDetectTikTokObjects:
    def test_returns_detections(self, tmp_path):
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.names = {0: "A", 1: "7", 2: "star"}
        results = _make_mock_results([
            (0, 0.90, [10, 10, 80, 80]),
            (1, 0.85, [100, 100, 170, 170]),
            (0, 0.88, [200, 50, 270, 120]),
        ])
        mock_model.return_value = [results]

        with patch("captcha.tiktok_detector._get_model", return_value=mock_model):
            dets = detect_tiktok_objects(
                _make_image_with_shapes(), confidence_threshold=0.25, model_path=model_path
            )

        assert len(dets) == 3
        assert dets[0].label == "A"
        assert dets[1].label == "7"
        assert dets[2].label == "A"

    def test_filters_low_confidence(self, tmp_path):
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.names = {0: "A", 1: "B"}
        results = _make_mock_results([
            (0, 0.90, [10, 10, 80, 80]),
            (1, 0.10, [100, 100, 170, 170]),
        ])
        mock_model.return_value = [results]

        with patch("captcha.tiktok_detector._get_model", return_value=mock_model):
            dets = detect_tiktok_objects(
                _make_image_with_shapes(), confidence_threshold=0.25, model_path=model_path
            )

        assert len(dets) == 1
        assert dets[0].label == "A"


# ---------------------------------------------------------------------------
# Crop comparison unit tests
# ---------------------------------------------------------------------------


class TestCropComparison:
    def test_identical_crops_score_high(self):
        """Two identical crops should have similarity close to 1."""
        # Create a crop with a coloured object on grey background
        crop = np.full((128, 128, 3), 210, dtype=np.uint8)
        crop[30:100, 30:100] = [0, 0, 200]  # red square
        score = _compare_crops(crop, crop)
        assert score > 0.9, f"Identical crops should score >0.9, got {score}"

    def test_different_shapes_score_lower(self):
        """A circle vs a square (both on grey bg) should score lower."""
        # Red circle on grey
        crop_a = np.full((128, 128, 3), 210, dtype=np.uint8)
        cv2.circle(crop_a, (64, 64), 40, (0, 0, 200), -1)
        # Blue rectangle on grey
        crop_b = np.full((128, 128, 3), 210, dtype=np.uint8)
        cv2.rectangle(crop_b, (20, 40), (108, 88), (200, 0, 0), -1)
        score = _compare_crops(crop_a, crop_b)
        assert score < 0.6, f"Different shapes should score <0.6, got {score}"

    def test_same_shape_different_colour_scores_higher(self):
        """Two circles of different colours should score higher than circle vs square."""
        # Red circle
        crop_a = np.full((128, 128, 3), 210, dtype=np.uint8)
        cv2.circle(crop_a, (64, 64), 40, (0, 0, 200), -1)
        # Green circle (same shape, different colour)
        crop_b = np.full((128, 128, 3), 210, dtype=np.uint8)
        cv2.circle(crop_b, (64, 64), 40, (0, 200, 0), -1)
        # Blue square (different shape)
        crop_c = np.full((128, 128, 3), 210, dtype=np.uint8)
        cv2.rectangle(crop_c, (24, 24), (104, 104), (200, 0, 0), -1)

        sim_same_shape = _compare_crops(crop_a, crop_b)
        sim_diff_shape = _compare_crops(crop_a, crop_c)
        assert sim_same_shape > sim_diff_shape, (
            f"Same-shape sim ({sim_same_shape:.3f}) should exceed "
            f"different-shape sim ({sim_diff_shape:.3f})"
        )

    def test_foreground_mask_detects_coloured_region(self):
        """A bright object on light grey should produce a non-empty mask."""
        # Light grey background with a red square in the middle
        crop = np.full((128, 128, 3), 210, dtype=np.uint8)
        crop[40:90, 40:90] = [0, 0, 200]  # red square (BGR)
        mask = _make_foreground_mask(crop)
        fg_pixels = np.count_nonzero(mask)
        assert fg_pixels > 100, f"Expected foreground pixels, got {fg_pixels}"

    def test_foreground_mask_empty_for_background(self):
        """A purely light-grey image should have ~no foreground."""
        crop = np.full((128, 128, 3), 220, dtype=np.uint8)
        mask = _make_foreground_mask(crop)
        fg_pixels = np.count_nonzero(mask)
        assert fg_pixels < 50, f"Background-only image should have few fg pixels, got {fg_pixels}"


class TestCropDetection:
    def test_crop_returns_correct_size(self):
        image = _make_image_with_shapes()
        det = TikTokDetection("A", 0.9, 30, 30, 90, 90)
        crop = _crop_detection(image, det)
        assert crop.shape == (128, 128, 3)

    def test_crop_clamps_to_image_bounds(self):
        """Crop at image edge should not crash."""
        image = _make_image_with_shapes()
        det = TikTokDetection("A", 0.9, 0, 0, 50, 50)
        crop = _crop_detection(image, det)
        assert crop.shape == (128, 128, 3)


# ---------------------------------------------------------------------------
# find_matching_pair_yolo — hybrid: YOLO detect + visual compare
# ---------------------------------------------------------------------------


class TestFindMatchingPairYolo:
    def _mock_detect(self, detections):
        return patch(
            "captcha.tiktok_detector.detect_tiktok_objects",
            return_value=detections,
        )

    def test_picks_visually_similar_pair(self):
        """
        Given 4 objects (2 red circles, 1 blue square, 1 green triangle)
        all labeled 'chu_khoi', the visual comparison should pick the
        two red circles as the matching pair.
        """
        image = _make_image_with_shapes()
        dets = _make_detections_for_shapes()

        with self._mock_detect(dets):
            result = find_matching_pair_yolo(image)

        assert result is not None
        (ax, ay), (bx, by) = result
        # The two red circles are at ~(60,60) and ~(340,340)
        centres = sorted([(ax, ay), (bx, by)])
        assert centres[0][0] < 100 and centres[0][1] < 100, (
            f"First match should be near (60,60), got {centres[0]}"
        )
        assert centres[1][0] > 300 and centres[1][1] > 300, (
            f"Second match should be near (340,340), got {centres[1]}"
        )

    def test_returns_none_with_too_few_detections(self):
        dets = [TikTokDetection("A", 0.90, 10, 10, 80, 80)]
        with self._mock_detect(dets):
            result = find_matching_pair_yolo(_make_image_with_shapes())
        assert result is None

    def test_returns_none_with_no_detections(self):
        with self._mock_detect([]):
            result = find_matching_pair_yolo(_make_image_with_shapes())
        assert result is None

    def test_falls_back_to_all_pairs_when_no_same_class(self):
        """When all detections are different classes, compare all pairs."""
        image = _make_image_with_shapes()
        dets = [
            TikTokDetection("A", 0.95, 30, 30, 90, 90),     # red circle 1
            TikTokDetection("B", 0.90, 270, 30, 330, 90),    # blue square
            TikTokDetection("C", 0.92, 310, 310, 370, 370),  # red circle 2
        ]
        with self._mock_detect(dets):
            result = find_matching_pair_yolo(image)

        # Should still find the two red circles via cross-class fallback
        assert result is not None

    def test_only_compares_within_same_class_first(self):
        """
        When there are same-class pairs, should not compare across classes.
        Two red circles (class A) should match even if a blue square is
        also class A but less similar.
        """
        image = _make_image_with_shapes()
        # Mark the two circles as class A, square as class B, triangle as class C
        dets = [
            TikTokDetection("A", 0.95, 30, 30, 90, 90),      # red circle 1
            TikTokDetection("B", 0.90, 270, 30, 330, 90),     # blue square
            TikTokDetection("C", 0.88, 30, 270, 90, 330),     # green triangle
            TikTokDetection("A", 0.92, 310, 310, 370, 370),   # red circle 2
        ]
        with self._mock_detect(dets):
            result = find_matching_pair_yolo(image)

        assert result is not None
        (ax, ay), (bx, by) = result
        centres = sorted([(ax, ay), (bx, by)])
        # Should match the two "A" class red circles
        assert centres[0][0] < 100
        assert centres[1][0] > 300


# ---------------------------------------------------------------------------
# Shape solver YOLO integration
# ---------------------------------------------------------------------------


class TestShapeSolverYoloIntegration:
    """Verify that shape_solver.find_matching_pair tries YOLO first."""

    def test_uses_yolo_when_available(self):
        expected = ((45, 45), (235, 85))
        with patch("captcha.shape_solver._try_yolo", return_value=expected):
            from captcha.shape_solver import find_matching_pair
            result = find_matching_pair(_make_image_with_shapes())
        assert result == expected

    def test_falls_back_to_classical_when_yolo_unavailable(self):
        expected_classical = ((80, 200), (320, 200))
        with patch("captcha.shape_solver._try_yolo", return_value=None):
            with patch(
                "captcha.shape_solver._find_matching_pair_classical",
                return_value=expected_classical,
            ):
                from captcha.shape_solver import find_matching_pair
                result = find_matching_pair(_make_image_with_shapes())
        assert result == expected_classical

    def test_skips_yolo_when_disabled(self):
        expected = ((80, 200), (320, 200))
        with patch(
            "captcha.shape_solver._find_matching_pair_classical",
            return_value=expected,
        ) as mock_classical:
            with patch("captcha.shape_solver._try_yolo") as mock_yolo:
                from captcha.shape_solver import find_matching_pair
                result = find_matching_pair(_make_image_with_shapes(), use_yolo=False)
        mock_yolo.assert_not_called()
        mock_classical.assert_called_once()
        assert result == expected

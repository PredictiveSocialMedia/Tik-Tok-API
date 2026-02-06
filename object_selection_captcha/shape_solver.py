"""
Shape-matching CAPTCHA solver.

Handles TikTok's "Select 2 objects that are the same shape" CAPTCHA.

The CAPTCHA shows several 3D-rendered objects (letters, numbers, geometric
shapes) scattered on a light background.  The user must click the **two**
objects that share the same shape.  Numbered blue circles only appear
*after* the user clicks; they are selection confirmations, NOT part of the
puzzle.

Strategy (two tiers)
--------------------
**Tier 1 — Fine-tuned YOLO (preferred)**:
If a custom-trained YOLO model exists at ``object_selection_captcha/models/tiktok_captcha_best.pt``,
it is used to classify each object by label (e.g. "A", "7", "star").
Two objects with the same label are the matching pair.  Train the model with
``python -m object_selection_captcha.train_tiktok_model``.

**Tier 2 — Classical contour analysis (fallback)**:
1. Segment individual objects from the light background using colour
   distance + Otsu thresholding + connected-component analysis.
2. Filter out noise and shadows; keep only significant blobs.
3. For each object, extract shape features: Hu moments, aspect ratio,
   extent, solidity, edge-orientation histogram.
4. Compare every pair; the two most similar are the answer.
5. Return their centre coordinates so the caller can click them.

Inspired by https://github.com/Lokno/click-captcha — uses visual feature
extraction and comparison rather than ML classification.

Requires: opencv-python-headless, numpy, Pillow
Optionally: ultralytics (for Tier 1)
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detected object
# ---------------------------------------------------------------------------


class DetectedObject:
    """A segmented object: bounding box, contour, and centre."""

    __slots__ = ("x", "y", "w", "h", "contour", "index", "area")

    def __init__(
        self,
        x: int, y: int, w: int, h: int,
        contour: np.ndarray,
        area: float,
        index: int,
    ):
        self.x = x          # bounding-box top-left
        self.y = y
        self.w = w
        self.h = h
        self.contour = contour
        self.area = area
        self.index = index

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def __repr__(self) -> str:
        cx, cy = self.center
        return f"DetectedObject(idx={self.index}, center=({cx},{cy}), size={self.w}x{self.h}, area={self.area:.0f})"


# ---------------------------------------------------------------------------
# Object segmentation from background
# ---------------------------------------------------------------------------

# The CAPTCHA background is near-white / light grey.
# We define "background" as pixels close to the median of the image
# (which is dominated by the light background).

_MIN_OBJECT_AREA = 400       # ignore blobs smaller than this (shadows, noise)
_MAX_OBJECT_AREA_RATIO = 0.4  # ignore blobs larger than 40% of the image (background)
_MIN_OBJECTS = 3              # expect at least 3 objects in the CAPTCHA
_MAX_OBJECTS = 12             # cap to avoid over-segmentation


def segment_objects(image: np.ndarray) -> list[DetectedObject]:
    """
    Find individual objects in a TikTok shape-CAPTCHA image.

    Parameters
    ----------
    image : np.ndarray
        BGR image of the CAPTCHA challenge area.

    Returns
    -------
    list[DetectedObject]
        Detected objects sorted left-to-right, each with bounding box,
        contour, and centre.
    """
    h, w = image.shape[:2]
    total_area = h * w

    # ── Background separation ────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Also use colour distance from the background.
    # Estimate background colour as the median (most of the image is bg).
    bg_color = np.median(image.reshape(-1, 3), axis=0).astype(np.float32)
    color_dist = np.linalg.norm(
        image.astype(np.float32) - bg_color[np.newaxis, np.newaxis, :],
        axis=2,
    ).astype(np.uint8)

    # Combine: Otsu on greyscale + colour distance threshold
    _, mask_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask_color = cv2.threshold(color_dist, 30, 255, cv2.THRESH_BINARY)

    # Union of both masks gives the best foreground coverage
    mask = cv2.bitwise_or(mask_otsu, mask_color)

    # ── Morphological cleanup ────────────────────────────────────
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    # Fill small holes inside objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_med, iterations=3)

    # ── Find contours ────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects: list[DetectedObject] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _MIN_OBJECT_AREA:
            continue
        if area > total_area * _MAX_OBJECT_AREA_RATIO:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        # Skip very thin or very flat blobs (likely shadows or edges)
        if bw < 15 or bh < 15:
            continue
        aspect = bw / max(bh, 1)
        if aspect > 6 or aspect < 0.16:
            continue
        objects.append(DetectedObject(bx, by, bw, bh, cnt, area, len(objects)))

    # ── Merge overlapping / touching blobs that are likely one object ──
    objects = _merge_close_objects(objects, image)

    # Sort left-to-right (reading order)
    objects.sort(key=lambda o: (o.x + o.w // 2))
    for i, obj in enumerate(objects):
        obj.index = i

    logger.info("segment_objects: found %d objects", len(objects))
    for obj in objects:
        logger.debug("  %s", obj)

    return objects[:_MAX_OBJECTS]


def _merge_close_objects(
    objects: list[DetectedObject], image: np.ndarray
) -> list[DetectedObject]:
    """
    Merge objects whose bounding boxes significantly overlap, since one
    3D object can fragment into multiple contours (e.g. body + shadow).
    """
    if len(objects) <= 1:
        return objects

    merged = list(objects)
    changed = True
    while changed:
        changed = False
        new_merged: list[DetectedObject] = []
        used = set()
        for i in range(len(merged)):
            if i in used:
                continue
            best = merged[i]
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                other = merged[j]
                if _boxes_overlap_significantly(best, other):
                    best = _combine_objects(best, other)
                    used.add(j)
                    changed = True
            new_merged.append(best)
        merged = new_merged

    return merged


def _boxes_overlap_significantly(a: DetectedObject, b: DetectedObject) -> bool:
    """True if the bounding boxes of a and b overlap by >30% of the smaller."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    smaller = min(a.w * a.h, b.w * b.h)
    return inter > smaller * 0.3 if smaller > 0 else False


def _combine_objects(a: DetectedObject, b: DetectedObject) -> DetectedObject:
    """Merge two objects into one with a combined bounding box."""
    x1 = min(a.x, b.x)
    y1 = min(a.y, b.y)
    x2 = max(a.x + a.w, b.x + b.w)
    y2 = max(a.y + a.h, b.y + b.h)
    # Use the larger contour
    cnt = a.contour if a.area >= b.area else b.contour
    area = a.area + b.area
    return DetectedObject(x1, y1, x2 - x1, y2 - y1, cnt, area, a.index)


# ---------------------------------------------------------------------------
# Crop an object from the image
# ---------------------------------------------------------------------------

_CROP_PADDING = 10  # pixels of padding around the bounding box


def crop_object(image: np.ndarray, obj: DetectedObject) -> np.ndarray:
    """Crop the image around an object's bounding box with padding."""
    h, w = image.shape[:2]
    x1 = max(0, obj.x - _CROP_PADDING)
    y1 = max(0, obj.y - _CROP_PADDING)
    x2 = min(w, obj.x + obj.w + _CROP_PADDING)
    y2 = min(h, obj.y + obj.h + _CROP_PADDING)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return crop


# ---------------------------------------------------------------------------
# Object mask from a crop
# ---------------------------------------------------------------------------


def segment_crop(crop: np.ndarray) -> np.ndarray:
    """
    Create a binary mask of the foreground object in a crop.
    Returns: mask (255 = object, 0 = background).
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def get_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest contour in a binary mask, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# ---------------------------------------------------------------------------
# Shape feature extraction
# ---------------------------------------------------------------------------


def extract_shape_features(crop: np.ndarray) -> dict:
    """
    Extract shape descriptors from a cropped object region.

    Features
    --------
    - ``hu_moments``: 7 Hu moments (log-transformed, rotation-invariant)
    - ``contour``: the largest contour (for cv2.matchShapes)
    - ``aspect_ratio``: bounding rect width / height
    - ``extent``: contour area / bounding rect area
    - ``solidity``: contour area / convex hull area
    - ``edge_hist``: normalised edge-orientation histogram (8 bins)
    - ``edge_density``: mean edge magnitude
    """
    mask = segment_crop(crop)
    contour = get_largest_contour(mask)

    features: dict = {
        "hu_moments": np.zeros(7),
        "contour": None,
        "aspect_ratio": 1.0,
        "extent": 0.0,
        "solidity": 0.0,
        "edge_hist": np.zeros(8),
        "edge_density": 0.0,
    }

    if contour is None or cv2.contourArea(contour) < 50:
        return features

    features["contour"] = contour

    # ── Hu moments ───────────────────────────────────────────────
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    # Log-scale (avoid log(0))
    hu = np.array([-np.sign(h) * np.log10(max(abs(h), 1e-30)) for h in hu])
    features["hu_moments"] = hu

    # ── Bounding rect properties ─────────────────────────────────
    x, y, bw, bh = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    features["aspect_ratio"] = bw / max(bh, 1)
    features["extent"] = area / max(bw * bh, 1)

    # ── Solidity (area / convex hull area) ───────────────────────
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    features["solidity"] = area / max(hull_area, 1)

    # ── Edge orientation histogram ───────────────────────────────
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx)  # -pi to pi
    # Quantise into 8 bins (each 45 degrees)
    angle_bins = ((angle + np.pi) / (2 * np.pi) * 8).astype(int).clip(0, 7)
    hist = np.zeros(8)
    for b in range(8):
        hist[b] = mag[angle_bins == b].sum()
    total = hist.sum()
    if total > 0:
        hist /= total
    features["edge_hist"] = hist
    features["edge_density"] = float(edges.mean())

    return features


# ---------------------------------------------------------------------------
# Shape comparison
# ---------------------------------------------------------------------------


def compare_shapes(feat_a: dict, feat_b: dict) -> float:
    """
    Compute a similarity score (higher = more similar) between two feature sets.

    Combines multiple signals:
    - Hu moment distance (cv2.matchShapes)
    - Aspect ratio similarity
    - Extent similarity
    - Solidity similarity
    - Edge histogram cosine similarity
    """
    score = 0.0
    weights_total = 0.0

    # ── Contour match (Hu moments via OpenCV) ────────────────────
    c_a = feat_a.get("contour")
    c_b = feat_b.get("contour")
    if c_a is not None and c_b is not None:
        match_i1 = cv2.matchShapes(c_a, c_b, cv2.CONTOURS_MATCH_I1, 0)
        match_i2 = cv2.matchShapes(c_a, c_b, cv2.CONTOURS_MATCH_I2, 0)
        hu_sim = 1.0 / (1.0 + match_i1 + match_i2 * 0.5)
        score += hu_sim * 5.0
        weights_total += 5.0

    # ── Aspect ratio similarity ──────────────────────────────────
    ar_a = feat_a.get("aspect_ratio", 1.0)
    ar_b = feat_b.get("aspect_ratio", 1.0)
    ar_sim = 1.0 - min(abs(ar_a - ar_b), 2.0) / 2.0
    score += ar_sim * 2.0
    weights_total += 2.0

    # ── Extent similarity ────────────────────────────────────────
    ext_a = feat_a.get("extent", 0.0)
    ext_b = feat_b.get("extent", 0.0)
    ext_sim = 1.0 - min(abs(ext_a - ext_b), 1.0)
    score += ext_sim * 1.5
    weights_total += 1.5

    # ── Solidity similarity ──────────────────────────────────────
    sol_a = feat_a.get("solidity", 0.0)
    sol_b = feat_b.get("solidity", 0.0)
    sol_sim = 1.0 - min(abs(sol_a - sol_b), 1.0)
    score += sol_sim * 1.5
    weights_total += 1.5

    # ── Edge histogram cosine similarity ─────────────────────────
    h_a = feat_a.get("edge_hist", np.zeros(8))
    h_b = feat_b.get("edge_hist", np.zeros(8))
    dot = np.dot(h_a, h_b)
    norm_a = np.linalg.norm(h_a)
    norm_b = np.linalg.norm(h_b)
    if norm_a > 0 and norm_b > 0:
        cos_sim = dot / (norm_a * norm_b)
    else:
        cos_sim = 0.0
    score += cos_sim * 2.0
    weights_total += 2.0

    return score / max(weights_total, 1.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def find_matching_pair(
    image: Image.Image,
    use_yolo: bool = True,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Analyse a TikTok shape CAPTCHA image and find the matching pair.

    Tries the fine-tuned YOLO model first (if available and *use_yolo* is
    True), then falls back to classical contour analysis.

    Parameters
    ----------
    image : PIL.Image
        Screenshot of the CAPTCHA challenge image.
    use_yolo : bool
        Whether to attempt YOLO-based detection first (default True).

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]] or None
        Pixel coordinates (x, y) of the centres of the two matching
        objects to click, or None if comparison failed.
    """
    # ── Tier 1: Fine-tuned YOLO ──────────────────────────────────
    if use_yolo:
        result = _try_yolo(image)
        if result is not None:
            return result

    # ── Tier 2: Classical contour analysis ────────────────────────
    logger.info("Using classical contour-based shape matching (Tier 2)")
    return _find_matching_pair_classical(image)


def _try_yolo(
    image: Image.Image,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Attempt YOLO-based matching. Returns None if unavailable or fails."""
    try:
        from .tiktok_detector import find_matching_pair_yolo, is_model_available
    except ImportError:
        logger.debug("tiktok_detector not importable; skipping YOLO tier")
        return None

    if not is_model_available():
        logger.info(
            "No fine-tuned model found at object_selection_captcha/models/tiktok_captcha_best.pt — "
            "falling back to classical solver. Train one with: "
            "python -m object_selection_captcha.train_tiktok_model"
        )
        return None

    try:
        result = find_matching_pair_yolo(image)
        if result is not None:
            logger.info("YOLO Tier 1 solver found a matching pair")
            return result
        logger.warning("YOLO Tier 1 ran but found no matching pair; falling back")
    except Exception as exc:
        logger.warning("YOLO Tier 1 failed: %s; falling back", exc)

    return None


def _find_matching_pair_classical(
    image: Image.Image,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Classical contour-based shape matching (Tier 2)."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    objects = segment_objects(img)

    if len(objects) < 2:
        logger.warning("Need at least 2 objects, found %d", len(objects))
        return None

    # Extract features for each object
    features: list[dict] = []
    for obj in objects:
        crop = crop_object(img, obj)
        feat = extract_shape_features(crop)
        features.append(feat)
        logger.debug(
            "Object %d at (%d,%d): aspect=%.2f extent=%.2f solidity=%.2f",
            obj.index, *obj.center, feat["aspect_ratio"],
            feat["extent"], feat["solidity"],
        )

    # Compare every pair; pick the most similar
    best_pair: Optional[tuple[int, int]] = None
    best_score = -1.0

    for i, j in combinations(range(len(objects)), 2):
        sim = compare_shapes(features[i], features[j])
        logger.debug(
            "Pair (%d, %d) similarity=%.4f",
            objects[i].index, objects[j].index, sim,
        )
        if sim > best_score:
            best_score = sim
            best_pair = (i, j)

    if best_pair is None:
        return None

    oi, oj = objects[best_pair[0]], objects[best_pair[1]]
    logger.info(
        "Shape solver (classical): best pair = objects %d & %d (similarity=%.4f) "
        "at (%d,%d) & (%d,%d)",
        oi.index, oj.index, best_score,
        *oi.center, *oj.center,
    )
    return (oi.center, oj.center)

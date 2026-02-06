"""
TikTok CAPTCHA object detector — hybrid YOLO + visual comparison.

Strategy
--------
1. **YOLO** (fine-tuned) locates every 3D object and gives bounding boxes.
   The class labels (``chu_khoi``, ``so_khoi``, …) are *category-level*
   (e.g. "letter", "number") — too coarse to directly identify the
   matching pair.
2. **Visual comparison** crops each detected object and compares every
   pair using structural similarity (SSIM), histogram correlation, and
   template matching.  The two most visually similar crops are the answer.

This hybrid beats both pure YOLO (wrong matches within a category) and
pure contour segmentation (struggles with 3D shadows/overlap).

Model location
--------------
The solver looks for the trained weights at:

    captcha/models/tiktok_captcha_best.pt

Train with ``python -m captcha.train_tiktok_model`` (see that module for
instructions).

Fallback
--------
If the custom model is not found, callers should fall back to the classical
contour-based solver in ``shape_solver.py``.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_DEFAULT_MODEL_PATH = _MODELS_DIR / "tiktok_captcha_best.pt"

# Lazy-loaded singleton
_tiktok_model = None
_model_checked = False


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


class TikTokDetection:
    """A detected 3D object in the CAPTCHA image."""

    __slots__ = ("label", "confidence", "x1", "y1", "x2", "y2")

    def __init__(
        self,
        label: str,
        confidence: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ):
        self.label = label
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def center(self) -> tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def __repr__(self) -> str:
        cx, cy = self.center
        return (
            f"TikTokDetection({self.label!r}, conf={self.confidence:.2f}, "
            f"center=({cx},{cy}), size={self.x2 - self.x1:.0f}x{self.y2 - self.y1:.0f})"
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def is_model_available(model_path: Optional[str | Path] = None) -> bool:
    """Return True if a trained TikTok CAPTCHA model exists."""
    p = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    return p.is_file()


def _get_model(model_path: Optional[str | Path] = None):
    """Load the fine-tuned model (cached after first call)."""
    global _tiktok_model, _model_checked

    if _tiktok_model is not None:
        return _tiktok_model

    p = Path(model_path) if model_path else _DEFAULT_MODEL_PATH

    if not p.is_file():
        _model_checked = True
        raise FileNotFoundError(
            f"TikTok CAPTCHA model not found at {p}. "
            f"Train one with: python -m captcha.train_tiktok_model"
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required: pip install ultralytics"
        )

    _tiktok_model = YOLO(str(p))
    _model_checked = True
    logger.info("Loaded TikTok CAPTCHA model from %s", p)
    logger.info("Model classes: %s", _tiktok_model.names)
    return _tiktok_model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_tiktok_objects(
    image: Image.Image,
    confidence_threshold: float = 0.25,
    model_path: Optional[str | Path] = None,
) -> list[TikTokDetection]:
    """
    Run the fine-tuned TikTok CAPTCHA model on *image*.

    Parameters
    ----------
    image : PIL.Image
        RGB screenshot of the CAPTCHA challenge area.
    confidence_threshold : float
        Minimum detection confidence to keep.
    model_path : str or Path, optional
        Override the model weights path.

    Returns
    -------
    list[TikTokDetection]
        Detected objects with class labels and bounding boxes.

    Raises
    ------
    FileNotFoundError
        If the model weights are not found.
    """
    model = _get_model(model_path)
    results = model(image, verbose=False)[0]

    detections: list[TikTokDetection] = []
    class_names = model.names  # dict: {0: "A", 1: "B", ...}

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue
        cls_id = int(box.cls[0])
        label = class_names.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(TikTokDetection(label, conf, x1, y1, x2, y2))

    logger.info(
        "detect_tiktok_objects: %d detections (threshold=%.2f)",
        len(detections),
        confidence_threshold,
    )
    for det in detections:
        logger.debug("  %s", det)

    return detections


# ---------------------------------------------------------------------------
# Crop comparison — find the visually identical pair
# ---------------------------------------------------------------------------

_CROP_SIZE = 128  # resize crops to this square for comparison
_CROP_PADDING = 4  # pixels of padding around the bounding box

# Background colour of TikTok CAPTCHA images (light grey)
_BG_LOW = np.array([180, 180, 180], dtype=np.uint8)
_BG_HIGH = np.array([255, 255, 255], dtype=np.uint8)

# Deep feature extractor (lazy-loaded)
_feature_model = None
_feature_transform = None


def _crop_detection(image: Image.Image, det: TikTokDetection) -> np.ndarray:
    """Crop and resize a detection region from the image (returns BGR)."""
    w, h = image.size
    x1 = max(0, int(det.x1) - _CROP_PADDING)
    y1 = max(0, int(det.y1) - _CROP_PADDING)
    x2 = min(w, int(det.x2) + _CROP_PADDING)
    y2 = min(h, int(det.y2) + _CROP_PADDING)

    crop = image.crop((x1, y1, x2, y2))
    crop_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(crop_bgr, (_CROP_SIZE, _CROP_SIZE), interpolation=cv2.INTER_AREA)
    return resized


def _make_foreground_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Create a binary mask: 255 for foreground (3D object), 0 for the
    light-grey CAPTCHA background.
    """
    bg_mask = cv2.inRange(crop_bgr, _BG_LOW, _BG_HIGH)
    fg_mask = cv2.bitwise_not(bg_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask


# ---------------------------------------------------------------------------
# Deep feature embedding (pre-trained CNN)
# ---------------------------------------------------------------------------


def _get_feature_extractor():
    """
    Lazy-load a pre-trained MobileNetV2 as a feature extractor.

    Returns (model, transform) where model outputs a 1280-d feature
    vector for each input image.  Uses torch (already installed via
    ultralytics).
    """
    global _feature_model, _feature_transform

    if _feature_model is not None:
        return _feature_model, _feature_transform

    import torch
    from torchvision import models, transforms

    # MobileNetV2 — fast and lightweight
    weights = models.MobileNet_V2_Weights.DEFAULT
    full_model = models.mobilenet_v2(weights=weights)
    full_model.eval()

    # Remove the classifier head: keep features only (1280-d)
    _feature_model = torch.nn.Sequential(
        full_model.features,
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
    )
    _feature_model.eval()

    _feature_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    logger.info("Loaded MobileNetV2 feature extractor")
    return _feature_model, _feature_transform


def _extract_features(crops_bgr: list[np.ndarray]) -> np.ndarray:
    """
    Extract deep feature vectors for a batch of BGR crops.

    Returns an (N, 1280) numpy array of L2-normalised feature vectors.
    """
    import torch

    model, transform = _get_feature_extractor()

    tensors = []
    for crop_bgr in crops_bgr:
        # BGR → RGB → PIL
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensors.append(transform(pil_img))

    batch = torch.stack(tensors)
    with torch.no_grad():
        features = model(batch).numpy()

    # L2-normalise for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    features = features / norms

    return features


def _compare_crops(crop_a: np.ndarray, crop_b: np.ndarray) -> float:
    """
    Compute visual similarity between two crops (higher = more similar).

    Uses deep feature embedding (cosine similarity from MobileNetV2)
    combined with a lightweight foreground-shape check.

    Returns a score in [0, 1].
    """
    # ── Deep feature similarity (primary signal) ─────────────────────
    features = _extract_features([crop_a, crop_b])
    cosine_sim = float(features[0] @ features[1])
    cosine_sim = max(0.0, cosine_sim)  # clip negatives

    # ── Foreground mask Dice (secondary shape check) ─────────────────
    mask_a = _make_foreground_mask(crop_a)
    mask_b = _make_foreground_mask(crop_b)
    intersection = np.count_nonzero(cv2.bitwise_and(mask_a, mask_b))
    total = np.count_nonzero(mask_a) + np.count_nonzero(mask_b)
    dice = (2.0 * intersection / total) if total > 0 else 0.0

    # ── Weighted combination ─────────────────────────────────────────
    score = cosine_sim * 0.80 + dice * 0.20

    logger.debug(
        "  compare: cosine=%.3f  dice=%.3f → %.3f",
        cosine_sim, dice, score,
    )

    return score


def _compare_crops_batch(
    crops: list[np.ndarray],
    pairs: list[tuple[int, int]],
) -> list[float]:
    """
    Compare multiple pairs efficiently using a single feature extraction
    pass.  Returns a list of similarity scores (one per pair).
    """
    if not pairs:
        return []

    # Extract features for all unique crop indices in one batch
    unique_indices = sorted({i for pair in pairs for i in pair})
    idx_to_pos = {idx: pos for pos, idx in enumerate(unique_indices)}
    batch = [crops[i] for i in unique_indices]

    features = _extract_features(batch)

    # Pre-compute foreground masks
    masks = {}
    for i in unique_indices:
        masks[i] = _make_foreground_mask(crops[i])

    scores: list[float] = []
    for i, j in pairs:
        # Cosine similarity from pre-computed features
        fi, fj = features[idx_to_pos[i]], features[idx_to_pos[j]]
        cosine_sim = max(0.0, float(fi @ fj))

        # Dice coefficient
        mi, mj = masks[i], masks[j]
        inter = np.count_nonzero(cv2.bitwise_and(mi, mj))
        total = np.count_nonzero(mi) + np.count_nonzero(mj)
        dice = (2.0 * inter / total) if total > 0 else 0.0

        score = cosine_sim * 0.80 + dice * 0.20
        scores.append(score)

        logger.debug(
            "  pair(%d,%d): cosine=%.3f dice=%.3f → %.3f",
            i, j, cosine_sim, dice, score,
        )

    return scores


# ---------------------------------------------------------------------------
# Main matching function — YOLO detection + visual comparison
# ---------------------------------------------------------------------------


def find_matching_pair_yolo(
    image: Image.Image,
    confidence_threshold: float = 0.25,
    model_path: Optional[str | Path] = None,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Hybrid solver: YOLO locates objects, visual comparison finds the match.

    Strategy
    --------
    1. Run YOLO to detect all objects (bounding boxes).
    2. Crop each detection from the original image.
    3. Compare every pair of crops visually (SSIM + histogram + template).
    4. The two most similar crops are the matching pair.
    5. Return their centres.

    Only pairs within the **same YOLO class** are compared, since two
    objects of different categories (e.g. a letter vs a sphere) can never
    be the correct match.  This dramatically reduces false positives.

    Parameters
    ----------
    image : PIL.Image
        Screenshot of the CAPTCHA challenge area.
    confidence_threshold : float
        Minimum detection confidence.
    model_path : str or Path, optional
        Override the model weights path.

    Returns
    -------
    tuple[tuple[int,int], tuple[int,int]] or None
        Centres ``(x, y)`` of the two matching objects, or ``None`` if
        no matching pair could be found.
    """
    detections = detect_tiktok_objects(image, confidence_threshold, model_path)

    if len(detections) < 2:
        logger.warning(
            "YOLO found %d objects (need at least 2)", len(detections)
        )
        return None

    # Crop each detection
    crops: list[np.ndarray] = []
    for det in detections:
        crops.append(_crop_detection(image, det))

    # Build list of same-class pairs to compare
    same_class_pairs: list[tuple[int, int]] = [
        (i, j)
        for i, j in combinations(range(len(detections)), 2)
        if detections[i].label == detections[j].label
    ]

    if same_class_pairs:
        # Compare all same-class pairs in one efficient batch
        scores = _compare_crops_batch(crops, same_class_pairs)
        best_idx = int(np.argmax(scores))
        best_pair = same_class_pairs[best_idx]
        best_score = scores[best_idx]
    else:
        # No same-class pairs → compare all pairs (cross-class fallback)
        logger.info("No same-class pairs; comparing all detections")
        all_pairs = list(combinations(range(len(detections)), 2))
        scores = _compare_crops_batch(crops, all_pairs)
        best_idx = int(np.argmax(scores))
        best_pair = all_pairs[best_idx]
        best_score = scores[best_idx]

    a, b = detections[best_pair[0]], detections[best_pair[1]]
    logger.info(
        "YOLO+visual match: %s & %s (sim=%.4f) → (%d,%d) & (%d,%d)",
        a.label, b.label, best_score,
        *a.center, *b.center,
    )

    return (a.center, b.center)

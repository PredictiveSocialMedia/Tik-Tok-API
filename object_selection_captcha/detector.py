"""
Object detection backend.

Uses YOLOv8-nano (ultralytics) for fast, offline object detection.
The model is downloaded automatically on first use (~6 MB).

For CAPTCHA prompts that reference objects outside the 80 COCO classes,
a simple colour / edge-density heuristic fallback is provided, plus an
optional CLIP zero-shot classifier if ``transformers`` is installed.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class names (YOLOv8 index → label)
# ---------------------------------------------------------------------------

COCO_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Common CAPTCHA prompt words  → COCO class(es) they map to
PROMPT_TO_COCO: dict[str, list[str]] = {
    "traffic light": ["traffic light"],
    "traffic lights": ["traffic light"],
    "bus": ["bus"],
    "buses": ["bus"],
    "bicycle": ["bicycle"],
    "bicycles": ["bicycle"],
    "bike": ["bicycle", "motorcycle"],
    "motorcycle": ["motorcycle"],
    "motorbike": ["motorcycle"],
    "car": ["car"],
    "cars": ["car"],
    "truck": ["truck"],
    "trucks": ["truck"],
    "fire hydrant": ["fire hydrant"],
    "hydrant": ["fire hydrant"],
    "crosswalk": [],  # not in COCO – needs heuristic / CLIP
    "stairs": [],
    "staircase": [],
    "chimney": [],
    "bridge": [],
    "palm tree": [],
    "boat": ["boat"],
    "airplane": ["airplane"],
    "stop sign": ["stop sign"],
}


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

class Detection:
    """One detected object: bounding box + label + confidence."""

    __slots__ = ("label", "confidence", "x1", "y1", "x2", "y2")

    def __init__(self, label: str, confidence: float, x1: float, y1: float, x2: float, y2: float):
        self.label = label
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou_with_box(self, bx1: float, by1: float, bx2: float, by2: float) -> float:
        """Intersection-over-union with an arbitrary rectangle."""
        ix1 = max(self.x1, bx1)
        iy1 = max(self.y1, by1)
        ix2 = min(self.x2, bx2)
        iy2 = min(self.y2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"Detection({self.label!r}, conf={self.confidence:.2f}, "
            f"box=[{self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}])"
        )


# ---------------------------------------------------------------------------
# YOLO detector (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_yolo_model = None


def _get_yolo():
    """Load the YOLOv8-nano model (cached after first call)."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n.pt")
            logger.info("YOLOv8-nano model loaded")
        except ImportError:
            raise ImportError(
                "ultralytics is required for CAPTCHA solving: pip install ultralytics"
            )
    return _yolo_model


def detect_objects(
    image: Image.Image,
    confidence_threshold: float = 0.25,
    target_labels: Optional[list[str]] = None,
) -> list[Detection]:
    """
    Run YOLOv8 on *image* and return detections.

    Parameters
    ----------
    image : PIL.Image
        RGB image to analyse.
    confidence_threshold : float
        Minimum detection confidence.
    target_labels : list[str], optional
        If provided, only return detections whose label is in this list.

    Returns
    -------
    list[Detection]
    """
    model = _get_yolo()
    results = model(image, verbose=False)[0]
    detections: list[Detection] = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue
        cls_id = int(box.cls[0])
        label = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class_{cls_id}"
        if target_labels and label not in target_labels:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(Detection(label, conf, x1, y1, x2, y2))

    return detections


# ---------------------------------------------------------------------------
# Prompt → COCO labels resolver
# ---------------------------------------------------------------------------

def resolve_prompt_labels(prompt: str) -> list[str]:
    """
    Convert a CAPTCHA prompt string (e.g. "Select all images with **traffic lights**")
    into a list of COCO class labels to look for.

    Returns an empty list if no known COCO mapping exists (caller should fall
    back to CLIP or heuristics).
    """
    prompt_lower = prompt.lower()

    # Direct lookup
    for key, labels in PROMPT_TO_COCO.items():
        if key in prompt_lower and labels:
            return labels

    # Fuzzy: check each COCO name
    matches = [name for name in COCO_NAMES if name in prompt_lower]
    return matches


# ---------------------------------------------------------------------------
# Tile classification helpers
# ---------------------------------------------------------------------------

def classify_tiles(
    full_image: Image.Image,
    grid_rows: int,
    grid_cols: int,
    target_labels: list[str],
    confidence_threshold: float = 0.20,
    iou_threshold: float = 0.10,
) -> list[int]:
    """
    Given a grid-based CAPTCHA image, return indices of tiles that contain
    one of *target_labels*.

    Strategy
    --------
    1. Run YOLO on the **full image** (better context than per-tile).
    2. For each detection matching *target_labels*, compute IoU with every tile.
    3. Mark tiles that have IoU > *iou_threshold* with any detection.

    Returns
    -------
    list[int]
        0-based tile indices (row-major: tile 0 = top-left, tile cols-1 = top-right).
    """
    w, h = full_image.size
    tile_w = w / grid_cols
    tile_h = h / grid_rows

    detections = detect_objects(full_image, confidence_threshold, target_labels)
    logger.info("classify_tiles: %d detections for labels %s", len(detections), target_labels)

    matched: set[int] = set()

    for det in detections:
        for row in range(grid_rows):
            for col in range(grid_cols):
                tx1 = col * tile_w
                ty1 = row * tile_h
                tx2 = tx1 + tile_w
                ty2 = ty1 + tile_h
                if det.iou_with_box(tx1, ty1, tx2, ty2) >= iou_threshold:
                    idx = row * grid_cols + col
                    matched.add(idx)

    # Fallback: also run per-tile detection for small objects
    if not matched:
        logger.info("classify_tiles: full-image found nothing, trying per-tile")
        for row in range(grid_rows):
            for col in range(grid_cols):
                tx1 = int(col * tile_w)
                ty1 = int(row * tile_h)
                tx2 = int(tx1 + tile_w)
                ty2 = int(ty1 + tile_h)
                tile_img = full_image.crop((tx1, ty1, tx2, ty2))
                tile_dets = detect_objects(tile_img, confidence_threshold, target_labels)
                if tile_dets:
                    matched.add(row * grid_cols + col)

    return sorted(matched)


def find_click_points(
    image: Image.Image,
    target_labels: list[str],
    confidence_threshold: float = 0.25,
) -> list[tuple[float, float]]:
    """
    For click-based CAPTCHAs: return (x, y) centres of detected target objects
    in *image* pixel coordinates.
    """
    detections = detect_objects(image, confidence_threshold, target_labels)
    return [d.center for d in detections]


# ---------------------------------------------------------------------------
# Edge-density heuristic (fallback for non-COCO objects)
# ---------------------------------------------------------------------------

def tile_edge_density(tile: Image.Image) -> float:
    """
    Simple Sobel-like edge-density score for a tile.
    Higher = more "stuff" in the tile (useful for distinguishing empty sky
    tiles from tiles with structures like chimneys, bridges, etc.).
    """
    grey = np.array(tile.convert("L"), dtype=np.float32)
    if grey.size == 0:
        return 0.0
    gx = np.abs(np.diff(grey, axis=1))
    gy = np.abs(np.diff(grey, axis=0))
    return float((gx.mean() + gy.mean()) / 2.0)

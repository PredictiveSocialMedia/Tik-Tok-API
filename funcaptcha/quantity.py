"""
Quantity + object CAPTCHA solver (FunCaptcha "change number until it matches" style).

Counts objects in reference (e.g. "1 x heart") and in current panel; returns how many
+ or - clicks are needed. Uses object_selection_captcha detector when available,
else simple contour-based count.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def count_objects_contour(image: Image.Image) -> int:
    """
    Count significant foreground blobs (contours) in a light-background image.
    Suitable for "N x object" panels where one object type is shown multiple times.
    """
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    area_min = (h * w) * 0.01
    area_max = (h * w) * 0.4
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area_min <= area <= area_max:
            count += 1
    return count


def count_objects_yolo(image: Image.Image, confidence_threshold: float = 0.25) -> int:
    """
    Count detections using object_selection_captcha YOLO (if available).
    Returns total number of detected objects (all classes).
    """
    try:
        from object_selection_captcha.tiktok_detector import detect_tiktok_objects
        detections = detect_tiktok_objects(image, confidence_threshold=confidence_threshold)
        return len(detections)
    except Exception as e:
        logger.debug("YOLO count not available: %s", e)
        return -1


def count_objects_ml(image: Image.Image, image_size: int = 128) -> Optional[int]:
    """Use trained count regressor if available. Returns count or None."""
    try:
        from funcaptcha.ml_models import load_count_regressor, _device
        model = load_count_regressor()
        if model is None:
            return None
        import numpy as np
        torch = __import__("torch")
        device = _device()
        arr = np.array(image.resize((image_size, image_size)).convert("RGB")).astype(np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).item()
        return max(0, int(round(pred)))
    except Exception as e:
        logger.debug("Count regressor not used: %s", e)
        return None


def count_objects(
    image: Image.Image,
    prefer_yolo: bool = True,
    prefer_ml: bool = True,
    confidence_threshold: float = 0.25,
) -> int:
    """
    Count objects in image. Tries ML regressor if prefer_ml and model exists,
    then YOLO if prefer_yolo, else contour count.
    """
    if prefer_ml:
        n = count_objects_ml(image)
        if n is not None:
            return n
    if prefer_yolo:
        n = count_objects_yolo(image, confidence_threshold)
        if n >= 0:
            return n
    return count_objects_contour(image)


def quantity_delta(
    reference_image: Image.Image,
    current_image: Image.Image,
    prefer_yolo: bool = True,
    prefer_ml: bool = True,
) -> int:
    """
    Return (target_count - current_count). Positive => need to click + N times;
    negative => click - N times.
    """
    target = count_objects(reference_image, prefer_yolo=prefer_yolo, prefer_ml=prefer_ml)
    current = count_objects(current_image, prefer_yolo=prefer_yolo, prefer_ml=prefer_ml)
    delta = target - current
    logger.info("quantity_delta: target=%d current=%d -> delta=%d", target, current, delta)
    return delta


def solve_quantity(
    get_reference_image: callable,
    get_current_image: callable,
    click_plus: callable,
    click_minus: callable,
    max_steps: int = 20,
    prefer_yolo: bool = True,
) -> Optional[bool]:
    """
    High-level: click + or - until current count matches reference.
    *click_plus* / *click_minus* are callables that perform one click.
    Returns True when matched, False if max_steps exceeded, None on error.
    """
    for step in range(max_steps):
        ref = get_reference_image()
        cur = get_current_image()
        if ref is None or cur is None:
            return None
        delta = quantity_delta(ref, cur, prefer_yolo=prefer_yolo)
        if delta == 0:
            logger.info("quantity: matched at step %d", step)
            return True
        if delta > 0:
            for _ in range(min(delta, 5)):  # cap 5 per step to avoid overshoot
                click_plus()
        else:
            for _ in range(min(-delta, 5)):
                click_minus()
    return False

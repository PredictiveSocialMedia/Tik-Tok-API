"""
Puzzle slider position: find horizontal offset (in pixels) to align piece with slot.

Uses OpenCV template matching (optionally with Canny edge) so the piece
is slid over the background and the best-match x is the slot position.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL RGB to OpenCV BGR."""
    arr = np.array(pil_image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def get_slide_offset(
    background_image: Image.Image,
    piece_image: Image.Image,
    use_edges: bool = True,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Tuple[float, float]:
    """
    Find the x-offset (in pixels) where the puzzle piece best matches the background.

    The piece is treated as a template; we slide it over the background and
    take the x of the best match as the slot position. For a piece starting at
    x=0, this is the distance to drag. If the piece is already at some offset
    in the image, the caller should use (match_x - piece_left) or similar.

    Parameters
    ----------
    background_image : PIL.Image
        Full background image (with the slot/gap in it).
    piece_image : PIL.Image
        The movable puzzle piece (same scale as background).
    use_edges : bool
        If True, run Canny edge detection on both images before matching
        (often improves robustness to lighting/color).
    method : int
        OpenCV matchTemplate method (default TM_CCOEFF_NORMED).

    Returns
    -------
    (offset_x, confidence) : tuple
        offset_x: horizontal pixel offset from left of background where the
        piece best aligns (i.e. distance to slide the piece).
        confidence: match score in [0, 1] for TM_CCOEFF_NORMED; higher is better.
    """
    bg = _pil_to_bgr(background_image)
    piece = _pil_to_bgr(piece_image)

    # Ensure same scale: piece may be smaller; template must not exceed image
    bh, bw = bg.shape[:2]
    ph, pw = piece.shape[:2]
    if pw > bw or ph > bh:
        # Resize piece down to fit
        scale = min((bw - 1) / pw, (bh - 1) / ph)
        nw, nh = int(pw * scale), int(ph * scale)
        if nw < 1 or nh < 1:
            nw, nh = max(1, nw), max(1, nh)
        piece = cv2.resize(piece, (nw, nh), interpolation=cv2.INTER_LINEAR)
        ph, pw = piece.shape[:2]

    if use_edges:
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.GaussianBlur(bg_gray, (3, 3), 0)
        piece_gray = cv2.GaussianBlur(piece_gray, (3, 3), 0)
        bg_gray = cv2.Canny(bg_gray, 50, 150)
        piece_gray = cv2.Canny(piece_gray, 50, 150)
        img = bg_gray
        templ = piece_gray
    else:
        img = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        templ = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(img, templ, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        x = min_loc[0]
        confidence = 1.0 - min_val if method == cv2.TM_SQDIFF_NORMED else 1.0 / (1.0 + min_val)
    else:
        x = max_loc[0]
        confidence = float(max_val)

    logger.debug("get_slide_offset: x=%s confidence=%.3f", x, confidence)
    return (float(x), max(0.0, min(1.0, confidence)))


def get_slide_offset_as_proportion(
    background_image: Image.Image,
    piece_image: Image.Image,
    use_edges: bool = True,
) -> Tuple[float, float]:
    """
    Same as get_slide_offset but returns (proportion, confidence) where
    proportion is in [0, 1] relative to background width. Useful when
    slider track width differs from image width: slide_px = proportion * track_width.
    """
    x, conf = get_slide_offset(background_image, piece_image, use_edges=use_edges)
    w = background_image.width
    prop = x / w if w > 0 else 0.0
    return (prop, conf)

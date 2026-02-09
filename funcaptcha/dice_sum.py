"""
Dice "sum to X" CAPTCHA solver (FunCaptcha conditional selection).

Given a grid of dice tile images and a prompt like "Select the pair whose top sides add up to 14",
returns the two tile indices to click. Uses pip counting (1-6) per tile; then finds the pair
that sums to the target.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Pip count for standard dice: 1-6. We clamp to this range.
_MIN_PIPS = 1
_MAX_PIPS = 6


def parse_target_sum(prompt: str) -> Optional[int]:
    """
    Extract target sum from prompt text.
    E.g. "add up to 14", "sum to 14", "top sides add up to 14" -> 14.
    """
    if not prompt:
        return None
    # Numbers 2-18 (possible dice pair sums)
    patterns = [
        r"add\s+up\s+to\s+(\d+)",
        r"sum\s+to\s+(\d+)",
        r"sum\s+(\d+)",
        r"(\d+)\s*$",
        r"total\s+(\d+)",
        r"equals?\s+(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, prompt, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 2 <= val <= 18:  # two D6: 2-12; allow higher for other variants
                return val
    return None


def count_pips(dice_face_image: Image.Image) -> int:
    """
    Count pips (dots) on a single dice face image. Returns 1-6.
    Uses dark-on-light detection: threshold and count connected components of appropriate size.
    """
    bgr = cv2.cvtColor(np.array(dice_face_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Pips are usually dark; background lighter
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Optional: morph to merge nearby pixels of same pip
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    area_min = (h * w) * 0.01
    area_max = (h * w) * 0.35
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area_min <= area <= area_max:
            count += 1
    # Clamp to valid dice face
    count = max(_MIN_PIPS, min(_MAX_PIPS, count))
    return count


def dice_values_from_tiles_ml(tile_images: list[Image.Image], tile_size: int = 64) -> Optional[list[int]]:
    """Use trained 6-way dice classifier if available. Returns list of 1-6 or None."""
    try:
        from funcaptcha.ml_models import load_dice_classifier, _device
        model = load_dice_classifier()
        if model is None:
            return None
        import numpy as np
        torch = __import__("torch")
        device = _device()
        values = []
        for t in tile_images:
            arr = np.array(t.resize((tile_size, tile_size)).convert("RGB")).astype(np.float32) / 255.0
            x = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
            val = model.predict_value(logits).item()
            values.append(int(val))
        return values
    except Exception as e:
        logger.debug("Dice classifier not used: %s", e)
        return None


def dice_values_from_tiles(tile_images: list[Image.Image], use_ml: bool = True) -> list[int]:
    """Return list of dice face values (1-6) for each tile. Uses ML classifier if available and use_ml."""
    if use_ml:
        ml_vals = dice_values_from_tiles_ml(tile_images)
        if ml_vals is not None:
            return ml_vals
    return [count_pips(t) for t in tile_images]


def find_pair_with_sum(values: list[int], target: int) -> Optional[tuple[int, int]]:
    """
    Find distinct indices (i, j) such that values[i] + values[j] == target.
    Returns (i, j) with i < j, or None.
    """
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] + values[j] == target:
                return (i, j)
    return None


def solve_dice_sum(
    tile_images: list[Image.Image],
    prompt: str,
) -> Optional[tuple[int, int]]:
    """
    Solve "dice pair sum to X" challenge.

    Parameters
    ----------
    tile_images : list of PIL.Image
        One image per tile (e.g. 6 tiles for 3x2 grid).
    prompt : str
        Instruction text containing target sum (e.g. "add up to 14").

    Returns
    -------
    (i, j) : tuple of int
        Indices of the two tiles to click (0-based), or None if not found.
    """
    target = parse_target_sum(prompt)
    if target is None:
        logger.warning("dice_sum: could not parse target from prompt %r", prompt)
        return None
    values = dice_values_from_tiles(tile_images)
    logger.info("dice_sum: target=%d values=%s", target, values)
    pair = find_pair_with_sum(values, target)
    if pair is None:
        logger.warning("dice_sum: no pair sums to %d", target)
        return None
    return pair


def solve_dice_sum_with_prompt_parse(
    tile_images: list[Image.Image],
    target: Optional[int] = None,
    prompt: Optional[str] = None,
) -> Optional[tuple[int, int]]:
    """
    Same as solve_dice_sum but accepts either *target* (int) or *prompt* (str).
    """
    if target is not None:
        t = target
    elif prompt is not None:
        t = parse_target_sum(prompt)
        if t is None:
            return None
    else:
        return None
    values = dice_values_from_tiles(tile_images)
    return find_pair_with_sum(values, t)

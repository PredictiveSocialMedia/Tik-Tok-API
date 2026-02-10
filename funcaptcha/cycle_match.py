"""
2D object matching via arrow cycling (FunCaptcha "match the animal" style).

Compares reference (left) and current (right) panel images; returns a match score.
Caller clicks left/right arrow until score exceeds threshold.
Uses histogram + structural similarity; optional integration with object_selection_captcha
for deep features when available.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Normalize panel size for comparison
_PANEL_SIZE = (128, 128)
_MATCH_THRESHOLD = 0.75  # above this, consider "matched"


def _to_bgr(pil_image: Image.Image) -> np.ndarray:
    """PIL RGB to OpenCV BGR."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _normalize_panel(im: np.ndarray, size: tuple[int, int] = _PANEL_SIZE) -> np.ndarray:
    """Resize and optionally crop center so comparison is size-invariant."""
    h, w = im.shape[:2]
    if w <= 0 or h <= 0:
        return np.zeros((*size, 3), dtype=np.uint8)
    resized = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    return resized


def _histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compare two BGR images using histogram correlation (0..1)."""
    hist_a = cv2.calcHist([a], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return float(max(0.0, (corr + 1) / 2.0))  # map [-1,1] -> [0,1]


def _structural_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Simple structural similarity: mean of per-channel correlation of flattened patches."""
    if a.shape != b.shape:
        return 0.0
    a_flat = a.astype(np.float32).reshape(-1, 3)
    b_flat = b.astype(np.float32).reshape(-1, 3)
    scores = []
    for c in range(3):
        ac, bc = a_flat[:, c], b_flat[:, c]
        ma, mb = np.mean(ac), np.mean(bc)
        sa, sb = np.std(ac) + 1e-8, np.std(bc) + 1e-8
        cov = np.mean((ac - ma) * (bc - mb))
        scores.append((cov / (sa * sb) + 1) / 2.0)
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _match_score_siamese(reference_image: Image.Image, current_image: Image.Image) -> Optional[float]:
    """Use trained Siamese model if available. Returns score in [0, 1] or None."""
    try:
        from funcaptcha.ml_models import load_siamese, _device
        model = load_siamese()
        if model is None:
            return None
        torch = __import__("torch")
        device = _device()
        ref_n = _normalize_panel(_to_bgr(reference_image))
        cur_n = _normalize_panel(_to_bgr(current_image))
        x1 = torch.from_numpy(ref_n.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        x2 = torch.from_numpy(cur_n.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            sim = model(x1, x2).item()
        return float(np.clip((sim + 1) / 2.0, 0.0, 1.0))
    except Exception as e:
        logger.debug("Siamese not used: %s", e)
        return None


def match_score(
    reference_image: Image.Image,
    current_image: Image.Image,
    use_deep_features: bool = False,
    use_siamese: bool = True,
) -> float:
    """
    Compare reference (e.g. left panel "Match This!") and current (right panel) images.

    Returns a similarity score in [0, 1]; higher means "same object".
    Above ~0.75 typically indicates a match for arrow-cycling challenges.

    Parameters
    ----------
    reference_image : PIL.Image
        Left/reference panel (RGB).
    current_image : PIL.Image
        Right/current panel (RGB).
    use_deep_features : bool
        If True and object_selection_captcha is available, use MobileNetV2
        feature comparison for better accuracy (slower).
    use_siamese : bool
        If True and a trained Siamese model exists (funcaptcha/models/siamese_cycle_match.pt),
        use it for similarity; fall back to classical otherwise.
    """
    if use_siamese:
        siamese_score = _match_score_siamese(reference_image, current_image)
        if siamese_score is not None:
            logger.debug("cycle_match score (Siamese): %.3f", siamese_score)
            return siamese_score

    ref_bgr = _to_bgr(reference_image)
    cur_bgr = _to_bgr(current_image)
    ref_n = _normalize_panel(ref_bgr)
    cur_n = _normalize_panel(cur_bgr)

    hist_sim = _histogram_similarity(ref_n, cur_n)
    struct_sim = _structural_similarity(ref_n, cur_n)
    combined = 0.6 * hist_sim + 0.4 * struct_sim

    if use_deep_features:
        try:
            from object_selection_captcha.tiktok_detector import _extract_features, _crop_detection
            from object_selection_captcha.tiktok_detector import detect_tiktok_objects
            det_ref = detect_tiktok_objects(reference_image, confidence_threshold=0.2)
            det_cur = detect_tiktok_objects(current_image, confidence_threshold=0.2)
            if det_ref and det_cur:
                crop_ref = _crop_detection(reference_image, det_ref[0])
                crop_cur = _crop_detection(current_image, det_cur[0])
                feats = _extract_features([crop_ref, crop_cur])
                deep_sim = float(np.clip(feats[0] @ feats[1], 0.0, 1.0))
                combined = 0.5 * combined + 0.5 * deep_sim
        except Exception as e:
            logger.debug("Deep features not used: %s", e)

    logger.debug("cycle_match score: hist=%.3f struct=%.3f -> %.3f", hist_sim, struct_sim, combined)
    return combined


def is_matched(
    reference_image: Image.Image,
    current_image: Image.Image,
    threshold: float = _MATCH_THRESHOLD,
    use_deep_features: bool = False,
) -> bool:
    """Return True if reference and current panels are considered a match."""
    return match_score(reference_image, current_image, use_deep_features=use_deep_features) >= threshold


def solve_cycle_match(
    get_reference_image: callable,
    get_current_image: callable,
    click_right: callable,
    click_left: callable,
    max_steps: int = 20,
    match_threshold: float = _MATCH_THRESHOLD,
    use_deep_features: bool = False,
) -> Optional[bool]:
    """
    High-level: cycle (click right or left) until reference and current match.

    *get_reference_image* and *get_current_image* are callables that return PIL Images.
    *click_right* / *click_left* are callables that perform one arrow click (e.g. from browser_actions).

    Returns True if match was achieved, False if max_steps exceeded, None on error.
    """
    for step in range(max_steps):
        ref = get_reference_image()
        cur = get_current_image()
        if ref is None or cur is None:
            return None
        score = match_score(ref, cur, use_deep_features=use_deep_features)
        if score >= match_threshold:
            logger.info("cycle_match: matched at step %d (score=%.3f)", step, score)
            return True
        # Always try "next" first (right); could be extended to binary search or direction heuristic
        click_right()
    logger.warning("cycle_match: max_steps=%d exceeded", max_steps)
    return False

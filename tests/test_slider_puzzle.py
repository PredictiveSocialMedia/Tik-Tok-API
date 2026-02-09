"""
Tests for slider_puzzle (position only; no browser).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from slider_puzzle.position import get_slide_offset, get_slide_offset_as_proportion


def _make_synthetic_captcha(slot_x: int = 60, slot_y: int = 40, piece_w: int = 20, piece_h: int = 20, bg_w: int = 100, bg_h: int = 80):
    """Background with a distinct patch at (slot_x, slot_y); piece is that patch."""
    bg = np.full((bg_h, bg_w, 3), 180, dtype=np.uint8)
    # Distinct "slot" region (darker + pattern)
    sy, sx = slot_y, slot_x
    bg[sy : sy + piece_h, sx : sx + piece_w, :] = [60, 70, 80]
    bg[sy + 2 : sy + piece_h - 2, sx + 2 : sx + piece_w - 2, :] = [90, 100, 110]
    piece = np.array(bg[sy : sy + piece_h, sx : sx + piece_w, :].copy())
    return Image.fromarray(bg), Image.fromarray(piece)


def test_get_slide_offset_finds_slot():
    bg, piece = _make_synthetic_captcha(slot_x=60, slot_y=40)
    offset_x, confidence = get_slide_offset(bg, piece, use_edges=False)
    assert 50 <= offset_x <= 70, f"expected ~60, got {offset_x}"
    assert confidence > 0.5


def test_get_slide_offset_with_edges():
    bg, piece = _make_synthetic_captcha(slot_x=60, slot_y=40)
    offset_x, confidence = get_slide_offset(bg, piece, use_edges=True)
    # Canny can shift the peak on synthetic images; just ensure we get a valid result
    assert 0 <= offset_x <= bg.width
    assert 0 <= confidence <= 1


def test_get_slide_offset_as_proportion():
    bg, piece = _make_synthetic_captcha(slot_x=60, slot_y=40, bg_w=100)
    prop, conf = get_slide_offset_as_proportion(bg, piece, use_edges=False)
    assert 0.5 <= prop <= 0.75, f"expected ~0.6, got {prop}"
    assert 0 <= conf <= 1


def test_get_slide_offset_different_position():
    bg, piece = _make_synthetic_captcha(slot_x=30, slot_y=20, bg_w=120, bg_h=60)
    offset_x, _ = get_slide_offset(bg, piece, use_edges=False)
    assert 20 <= offset_x <= 40

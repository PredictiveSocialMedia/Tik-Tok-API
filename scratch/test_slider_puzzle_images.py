#!/usr/bin/env python3
"""
Test slider_puzzle position with saved images (no browser).

Usage:
    python scratch/test_slider_puzzle_images.py background.png piece.png

Saves of a real captcha: use browser devtools or screenshot to get the
background image and the movable piece image, then run this to see the
computed offset and confidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from slider_puzzle.position import get_slide_offset, get_slide_offset_as_proportion


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scratch/test_slider_puzzle_images.py <background.png> <piece.png>")
        print("  Use saved images from a real slider captcha to test position detection.")
        sys.exit(1)
    bg_path = Path(sys.argv[1])
    piece_path = Path(sys.argv[2])
    if not bg_path.exists():
        print(f"Not found: {bg_path}")
        sys.exit(1)
    if not piece_path.exists():
        print(f"Not found: {piece_path}")
        sys.exit(1)

    bg = Image.open(bg_path).convert("RGB")
    piece = Image.open(piece_path).convert("RGB")
    print(f"Background: {bg.size[0]}x{bg.size[1]}")
    print(f"Piece:      {piece.size[0]}x{piece.size[1]}")

    offset_px, confidence = get_slide_offset(bg, piece, use_edges=True)
    print(f"\nWith edges:  offset_x = {offset_px:.1f} px, confidence = {confidence:.3f}")

    offset_px2, conf2 = get_slide_offset(bg, piece, use_edges=False)
    print(f"Without edges: offset_x = {offset_px2:.1f} px, confidence = {conf2:.3f}")

    prop, prop_conf = get_slide_offset_as_proportion(bg, piece, use_edges=True)
    print(f"\nProportion (for track width): {prop:.4f} (confidence {prop_conf:.3f})")
    print(f"  → For a 340px track: slide {int(prop * 340)} px (before fudge)")
    print(f"  → With fudge -6:      slide {int(prop * 340 - 6)} px")


if __name__ == "__main__":
    main()

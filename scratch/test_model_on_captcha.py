#!/usr/bin/env python3
"""
Visual test: run the trained TikTok CAPTCHA model on real images.

Shows each image with:
  - All detected objects (bounding boxes + class labels)
  - The matching pair highlighted (green circles on centres)
  - Console output of what the solver would click

Usage:
  # Test on all images in the test split:
  python scratch/test_model_on_captcha.py

  # Test on a specific image:
  python scratch/test_model_on_captcha.py path/to/captcha.jpg

  # Save annotated images instead of displaying:
  python scratch/test_model_on_captcha.py --save

  # Use the classical (contour) solver for comparison:
  python scratch/test_model_on_captcha.py --classical
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Implementation package
CAPTCHA_PKG = PROJECT_ROOT / "object_selection_captcha"

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Default test images
# ---------------------------------------------------------------------------

_TEST_DIR = CAPTCHA_PKG / "tikdata.v1i.yolov8" / "test" / "images"


def get_test_images() -> list[Path]:
    if _TEST_DIR.is_dir():
        return sorted(_TEST_DIR.glob("*.jpg")) + sorted(_TEST_DIR.glob("*.png"))
    return []


# ---------------------------------------------------------------------------
# Run detection + matching
# ---------------------------------------------------------------------------


def run_yolo_solver(image: Image.Image) -> dict:
    """Run the YOLO-based solver and return results."""
    from object_selection_captcha.tiktok_detector import detect_tiktok_objects, find_matching_pair_yolo

    detections = detect_tiktok_objects(image, confidence_threshold=0.25)
    pair = find_matching_pair_yolo(image, confidence_threshold=0.25)

    return {
        "method": "YOLO (fine-tuned)",
        "detections": detections,
        "pair": pair,
    }


def run_classical_solver(image: Image.Image) -> dict:
    """Run the classical contour-based solver and return results."""
    from object_selection_captcha.shape_solver import find_matching_pair

    pair = find_matching_pair(image, use_yolo=False)

    return {
        "method": "Classical (contour)",
        "detections": [],
        "pair": pair,
    }


# ---------------------------------------------------------------------------
# Draw results
# ---------------------------------------------------------------------------


def annotate_image(image: Image.Image, results: dict) -> Image.Image:
    """Draw detection boxes and matching pair on the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Try to get a font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Draw all detection boxes
    for det in results.get("detections", []):
        color = "cyan"
        draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=color, width=2)
        label = f"{det.label} {det.confidence:.0%}"
        draw.text((det.x1, det.y1 - 16), label, fill=color, font=font)

    # Draw the matching pair
    pair = results.get("pair")
    if pair:
        (ax, ay), (bx, by) = pair
        r = 15
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], outline="lime", width=3)
        draw.text((ax + r + 4, ay - 8), "MATCH", fill="lime", font=font)
        draw.ellipse([bx - r, by - r, bx + r, by + r], outline="lime", width=3)
        draw.text((bx + r + 4, by - 8), "MATCH", fill="lime", font=font)

        # Draw a connecting line
        draw.line([(ax, ay), (bx, by)], fill="lime", width=2)
    else:
        draw.text((10, 10), "NO MATCH FOUND", fill="red", font=font)

    # Method label
    method = results.get("method", "")
    draw.text((10, img.height - 20), method, fill="white", font=font)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_image(
    image_path: Path,
    use_classical: bool = False,
    save: bool = False,
    output_dir: Path | None = None,
) -> bool:
    """Process one image. Returns True if a matching pair was found."""
    print(f"\n{'='*60}")
    print(f"Image: {image_path.name}")
    print(f"{'='*60}")

    image = Image.open(image_path).convert("RGB")

    if use_classical:
        results = run_classical_solver(image)
    else:
        results = run_yolo_solver(image)

    # Print results
    print(f"Method: {results['method']}")

    for det in results.get("detections", []):
        cx, cy = det.center
        print(f"  Detected: {det.label:12s} conf={det.confidence:.2f}  center=({cx},{cy})")

    pair = results.get("pair")
    if pair:
        (ax, ay), (bx, by) = pair
        print(f"\n  >> MATCH: click ({ax},{ay}) and ({bx},{by})")
        found = True
    else:
        print(f"\n  >> NO MATCH FOUND")
        found = False

    # Annotate and show/save
    annotated = annotate_image(image, results)

    if save:
        out_dir = output_dir or Path("scratch/test_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"result_{image_path.stem}.jpg"
        annotated.save(out_path, quality=95)
        print(f"  Saved: {out_path}")
    else:
        # Display with matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(annotated)
            plt.title(f"{image_path.name} — {results['method']}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  (install matplotlib to display, or use --save)")

    return found


def main():
    parser = argparse.ArgumentParser(description="Test TikTok CAPTCHA model on images")
    parser.add_argument(
        "images", nargs="*",
        help="Image paths to test (default: all test split images)",
    )
    parser.add_argument(
        "--classical", action="store_true",
        help="Use classical contour solver instead of YOLO",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated images instead of displaying",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for --save (default: scratch/test_results)",
    )
    args = parser.parse_args()

    if args.images:
        image_paths = [Path(p) for p in args.images]
    else:
        image_paths = get_test_images()
        if not image_paths:
            print("No test images found. Provide image paths as arguments.")
            sys.exit(1)
        print(f"Found {len(image_paths)} test images in dataset")

    total = len(image_paths)
    found = 0
    for path in image_paths:
        if not path.is_file():
            print(f"WARNING: {path} not found, skipping")
            continue
        if process_image(path, args.classical, args.save, args.output_dir):
            found += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {found}/{total} images had a matching pair detected")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

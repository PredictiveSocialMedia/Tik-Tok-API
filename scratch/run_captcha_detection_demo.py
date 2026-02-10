#!/usr/bin/env python3
"""
Run the CAPTCHA object detector on a single image (no browser).

Shows what the solver would detect and where it would click. Useful to
test "how well does YOLO work on this image?" before relying on it in
the full scraper.

Usage
-----
  python run_captcha_detection_demo.py path/to/image.jpg
  python run_captcha_detection_demo.py path/to/image.jpg --output boxes.png
  # URL: may fail on some systems (SSL). Prefer a local file.
  python run_captcha_detection_demo.py "https://example.com/photo.jpg" --output boxes.png

Requires: ultralytics, Pillow (pip install ultralytics Pillow)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Add project root so we can import object_selection_captcha
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CAPTCHA object detection on an image and show/save results"
    )
    parser.add_argument(
        "image",
        help="Path to image file or URL",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save annotated image to this path (draws bounding boxes)",
    )
    parser.add_argument(
        "--prompt",
        default="traffic light",
        help="Filter detections to this label (default: traffic light). Use 'all' to show every detection.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence (default: 0.25)",
    )
    args = parser.parse_args()

    from PIL import Image
    from object_selection_captcha.detector import detect_objects, resolve_prompt_labels

    # Load image
    path = args.image
    if path.startswith("http://") or path.startswith("https://"):
        print(f"Downloading {path} ...")
        tmp = Path("/tmp/captcha_demo_image.jpg")
        urlretrieve(path, tmp)
        path = str(tmp)
    if not Path(path).exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    image = Image.open(path).convert("RGB")
    print(f"Image size: {image.size[0]}x{image.size[1]}")

    # Optional label filter
    target_labels = None if args.prompt.lower() == "all" else resolve_prompt_labels(args.prompt)
    if target_labels is None and args.prompt.lower() != "all":
        target_labels = [args.prompt.lower()]
    if args.prompt.lower() == "all":
        target_labels = None

    detections = detect_objects(image, args.confidence, target_labels)
    print(f"Detections (confidence >= {args.confidence}): {len(detections)}")

    for i, d in enumerate(detections):
        print(f"  {i+1}. {d.label} @ ({d.center[0]:.0f}, {d.center[1]:.0f}) conf={d.confidence:.2f}")

    if not detections:
        print("No matching objects. Try --prompt all to see every detection, or lower --confidence.")
    else:
        print("\nClick points (x, y) that would be sent to the CAPTCHA:")
        for d in detections:
            print(f"  ({d.center[0]:.1f}, {d.center[1]:.1f})")

    if args.output and detections:
        # Draw boxes and save
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)
        for d in detections:
            draw.rectangle(
                [(d.x1, d.y1), (d.x2, d.y2)],
                outline="lime",
                width=3,
            )
            draw.text((d.x1, max(0, d.y1 - 16)), f"{d.label} {d.confidence:.2f}", fill="lime")
        image.save(args.output)
        print(f"\nSaved annotated image to {args.output}")


if __name__ == "__main__":
    main()

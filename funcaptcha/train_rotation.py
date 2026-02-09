"""
Train RotNet for rotation (self-supervised: rotate image by θ, predict θ).

No labels: take any image, rotate by random angle, train to predict that angle.
Uses 128 classes (~2.8° per bin). Saves to funcaptcha/models/rotnet_rotation.pt.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        return torch, nn, F, DataLoader
    except ImportError as e:
        raise ImportError("Training requires torch: pip install torch") from e


def _default_image_dir():
    p = Path(__file__).resolve().parent.parent / "object_selection_captcha" / "tikdata.v1i.yolov8" / "train" / "images"
    return p if p.is_dir() else None


def _list_images(path: Path, limit: int = 2000):
    out = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        out.extend(path.glob(ext))
    random.shuffle(out)
    return [str(x) for x in out[:limit]]


def _load_rotate_and_label(image_path: str, num_classes: int = 128, image_size: int = 128):
    """Load image, pick random angle, rotate, return (rotated_image_tensor, class_index)."""
    torch, nn, F, _ = _require_torch()
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    h, w = img.shape[:2]
    angle_deg = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    rotated = cv2.resize(rotated, (image_size, image_size))
    rotated = rotated.astype(np.float32) / 255.0
    x = torch.from_numpy(rotated.transpose(2, 0, 1)).float()
    step = 360.0 / num_classes
    class_idx = min(int(angle_deg / step), num_classes - 1)
    return x, class_idx


def train(args):
    torch, nn, F, DataLoader = _require_torch()
    from funcaptcha.ml_models import _RotNet, get_rotnet_path, _device

    device = _device()
    image_paths = _list_images(Path(args.images), limit=args.max_images)
    if not image_paths:
        raise SystemExit("No images found. Use --images DIR")
    num_classes = 128
    model = _RotNet(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        random.shuffle(image_paths)
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i : i + args.batch_size]
            xs, ys = [], []
            for path in batch_paths:
                x, y = _load_rotate_and_label(path, num_classes=num_classes, image_size=128)
                xs.append(x)
                ys.append(y)
            x = torch.stack(xs).to(device)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/max(n_batches,1):.4f}")
    get_rotnet_path().parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), get_rotnet_path())
    print(f"Saved to {get_rotnet_path()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, default=None)
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    args.images = Path(args.images or _default_image_dir() or ".")
    train(args)


if __name__ == "__main__":
    main()

"""
Train Siamese network for cycle_match (contrastive / same-different).

Synthetic data: "same" = one image with augment (crop, color) to get pair;
"different" = two different random images. Uses a folder of images or
generates simple shapes. Saves to funcaptcha/models/siamese_cycle_match.pt.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

# Optional torch
def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        return torch, nn, F, Dataset, DataLoader
    except ImportError as e:
        raise ImportError("Training requires torch and torchvision: pip install torch torchvision") from e


def _default_image_dir():
    """Try object_selection_captcha TikTok images as source."""
    p = Path(__file__).resolve().parent.parent / "object_selection_captcha" / "tikdata.v1i.yolov8" / "train" / "images"
    return p if p.is_dir() else None


def _list_images(path: Path, limit: int = 2000):
    out = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        out.extend(path.glob(ext))
    random.shuffle(out)
    return [str(x) for x in out[:limit]]


class SyntheticPairDataset:
    """Generate same/different pairs. Same = augment one image; different = two images."""

    def __init__(self, image_paths, image_size=128, same_prob=0.5):
        self.image_paths = image_paths
        self.image_size = image_size
        self.same_prob = same_prob

    def __len__(self):
        return len(self.image_paths) * 2

    def _load_and_resize(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(img).astype(np.float32) / 255.0

    def _augment(self, arr):
        # Light augmentation
        if random.random() < 0.5:
            arr = arr[:, ::-1].copy()
        if random.random() < 0.3:
            jitter = np.random.uniform(0.9, 1.1, (1, 1, 3)).astype(np.float32)
            arr = np.clip(arr * jitter, 0, 1)
        return arr

    def __getitem__(self, idx):
        torch, nn, F, Dataset, DataLoader = _require_torch()
        i = idx % len(self.image_paths)
        path = self.image_paths[i]
        img = self._load_and_resize(path)
        is_same = random.random() < self.same_prob
        if is_same:
            img2 = self._augment(img.copy())
            label = 1.0
        else:
            j = random.randint(0, len(self.image_paths) - 1)
            if j == i:
                j = (j + 1) % len(self.image_paths)
            img2 = self._load_and_resize(self.image_paths[j])
            img2 = self._augment(img2)
            label = -1.0
        # NCHW, float32
        x1 = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        x2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0)
        return x1.squeeze(0), x2.squeeze(0), torch.tensor([label], dtype=torch.float32)


def train(args):
    torch, nn, F, Dataset, DataLoader = _require_torch()
    from funcaptcha.ml_models import _SiameseNetwork, get_siamese_path, _device

    device = _device()
    image_paths = _list_images(Path(args.images), limit=args.max_images)
    if not image_paths:
        raise SystemExit("No images found. Use --images DIR with JPG/PNG or add images to object_selection_captcha/tikdata.../train/images")
    dataset = SyntheticPairDataset(image_paths, image_size=128, same_prob=0.5)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    model = _SiameseNetwork().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for x1, x2, label in loader:
            x1, x2 = x1.to(device), x2.to(device)
            label = label.to(device).squeeze(1)
            sim = model(x1, x2)
            # same -> sim high (1), different -> sim low (-1). Treat sim as logit (scale for sigmoid).
            target = (label + 1) / 2  # -1,1 -> 0,1
            loss = F.binary_cross_entropy_with_logits(sim.clamp(-1, 1) * 5.0, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/max(n_batches,1):.4f}")
    get_siamese_path().parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), get_siamese_path())
    print(f"Saved to {get_siamese_path()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, default=None, help="Directory of images (default: object_selection_captcha/.../train/images)")
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    if args.images is None:
        args.images = _default_image_dir()
        if args.images is None:
            args.images = str(Path(__file__).resolve().parent.parent / "object_selection_captcha" / "tikdata.v1i.yolov8" / "train" / "images")
    args.images = Path(args.images)
    train(args)


if __name__ == "__main__":
    main()

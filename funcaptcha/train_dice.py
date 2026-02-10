"""
Train 6-way dice classifier (dice face -> 1-6).

Synthetic data: generate dice face images with 1-6 pips (like tests/test_funcaptcha.py).
Saves to funcaptcha/models/dice_classifier.pt.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError as e:
        raise ImportError("Training requires torch: pip install torch") from e


def _fake_dice_tile(value: int, size: int = 64) -> np.ndarray:
    """Synthetic dice face (value 1-6). Returns float [0,1] (size, size, 3)."""
    value = max(1, min(6, value))
    arr = np.full((size, size, 3), 0.95, dtype=np.float32)
    centers = {
        1: [(size // 2, size // 2)],
        2: [(size // 4, size // 4), (3 * size // 4, 3 * size // 4)],
        3: [(size // 4, size // 4), (size // 2, size // 2), (3 * size // 4, 3 * size // 4)],
        4: [(size // 4, size // 4), (3 * size // 4, size // 4), (size // 4, 3 * size // 4), (3 * size // 4, 3 * size // 4)],
        5: [(size // 4, size // 4), (3 * size // 4, size // 4), (size // 2, size // 2), (size // 4, 3 * size // 4), (3 * size // 4, 3 * size // 4)],
        6: [(size // 4, size // 4), (3 * size // 4, size // 4), (size // 4, size // 2), (3 * size // 4, size // 2), (size // 4, 3 * size // 4), (3 * size // 4, 3 * size // 4)],
    }
    r = max(2, size // 12)
    for cx, cy in centers[value]:
        for y in range(max(0, cy - r), min(size, cy + r + 1)):
            for x in range(max(0, cx - r), min(size, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    arr[y, x, :] = 0.1
    return arr


def _augment_dice(arr: np.ndarray) -> np.ndarray:
    """Light augmentation."""
    if random.random() < 0.5:
        arr = arr[:, ::-1].copy()
    if random.random() < 0.5:
        arr = arr[::-1, :].copy()
    if random.random() < 0.3:
        jitter = np.random.uniform(0.85, 1.15, (1, 1, 3)).astype(np.float32)
        arr = np.clip(arr * jitter, 0, 1)
    return arr


def train(args):
    torch, nn, F = _require_torch()
    from funcaptcha.ml_models import _DiceClassifier, get_dice_path, _device

    device = _device()
    model = _DiceClassifier(num_classes=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for _ in range(args.batches_per_epoch):
            batch_size = args.batch_size
            labels = [random.randint(1, 6) for _ in range(batch_size)]
            imgs = [_augment_dice(_fake_dice_tile(l, size=64)) for l in labels]
            x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs).transpose(0, 3, 1, 2))).float().to(device)
            y = torch.tensor([l - 1 for l in labels], dtype=torch.long, device=device)  # 0-5
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/max(n_batches,1):.4f}")
    get_dice_path().parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), get_dice_path())
    print(f"Saved to {get_dice_path()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches-per-epoch", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""
Train count regressor for quantity (image -> scalar count).

Synthetic data: place N circles/shapes on canvas, label N. Saves to funcaptcha/models/count_quantity.pt.
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


def _synthetic_count_image(n: int, size: int = 128, max_count: int = 20) -> np.ndarray:
    """Generate image with n circles on light background."""
    n = max(0, min(max_count, n))
    arr = np.full((size, size, 3), 240, dtype=np.float32)
    for _ in range(n):
        cx, cy = random.randint(20, size - 21), random.randint(20, size - 21)
        r = random.randint(5, 12)
        for y in range(max(0, cy - r), min(size, cy + r + 1)):
            for x in range(max(0, cx - r), min(size, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    arr[y, x, :] = [40, 40, 40]
    return arr / 255.0


def train(args):
    torch, nn, F = _require_torch()
    from funcaptcha.ml_models import _CountRegressor, get_count_path, _device

    device = _device()
    model = _CountRegressor(max_count=args.max_count).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    max_count = args.max_count
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for _ in range(args.batches_per_epoch):
            batch_size = args.batch_size
            counts = [random.randint(0, max_count) for _ in range(batch_size)]
            imgs = [_synthetic_count_image(c, size=128, max_count=max_count) for c in counts]
            x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs).transpose(0, 3, 1, 2))).float().to(device)
            y = torch.tensor(counts, dtype=torch.float32, device=device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/max(n_batches,1):.4f}")
    get_count_path().parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), get_count_path())
    print(f"Saved to {get_count_path()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches-per-epoch", type=int, default=50)
    ap.add_argument("--max-count", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()

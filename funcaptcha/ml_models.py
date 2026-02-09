"""
PyTorch model definitions for funcaptcha upgrades.

- SiameseNetwork: same/different or similarity for cycle_match (contrastive).
- RotNet: rotation angle regression/classification for rotation (CSL or angular loss).
- CountRegressor: scalar count prediction for quantity.
- DiceClassifier: 6-way (1-6) for dice_sum.
"""

from __future__ import annotations

from pathlib import Path

def _torch():
    import torch
    return torch

def _device():
    t = _torch()
    if t.cuda.is_available():
        return t.device("cuda")
    if hasattr(t.backends, "mps") and t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")


def _make_backbone(in_channels=3, out_dim=256):
    import torch.nn as nn
    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(128, out_dim),
                nn.ReLU(inplace=True),
            )
            self.out_dim = out_dim

        def forward(self, x):
            return self.net(x)
    return _Backbone()


# ---------------------------------------------------------------------------
# Siamese network (cycle_match)
# ---------------------------------------------------------------------------

def _SiameseNetwork():
    import torch.nn as nn
    class _Siamese(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _make_backbone(3, 256)
            self.embed_dim = 256
            self.image_size = 128

        def forward(self, x1, x2):
            e1 = self.backbone(x1)
            e2 = self.backbone(x2)
            e1 = e1 / (e1.norm(dim=1, keepdim=True) + 1e-8)
            e2 = e2 / (e2.norm(dim=1, keepdim=True) + 1e-8)
            return (e1 * e2).sum(dim=1)

        def embed(self, x):
            e = self.backbone(x)
            return e / (e.norm(dim=1, keepdim=True) + 1e-8)
    return _Siamese()


# ---------------------------------------------------------------------------
# RotNet (rotation)
# ---------------------------------------------------------------------------

def _RotNet(num_classes=128):
    import torch.nn as nn
    class _Rot(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _make_backbone(3, 256)
            self.num_classes = num_classes
            self.head = nn.Linear(256, num_classes)

        def forward(self, x):
            return self.head(self.backbone(x))

        def predict_angle_deg(self, logits):
            t = _torch()
            probs = t.softmax(logits, dim=1)
            step = 360.0 / num_classes
            indices = t.arange(num_classes, device=logits.device, dtype=logits.dtype)
            angles = (indices + 0.5) * step
            return (probs * angles).sum(dim=1)
    return _Rot()


# ---------------------------------------------------------------------------
# Count regressor (quantity)
# ---------------------------------------------------------------------------

def _CountRegressor(max_count=20):
    import torch.nn as nn
    class _Count(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _make_backbone(3, 256)
            self.max_count = max_count
            self.head = nn.Linear(256, 1)

        def forward(self, x):
            return self.head(self.backbone(x)).squeeze(-1)
    return _Count()


# ---------------------------------------------------------------------------
# Dice classifier (dice_sum)
# ---------------------------------------------------------------------------

def _DiceClassifier(num_classes=6):
    import torch.nn as nn
    class _Dice(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _make_backbone(3, 256)
            self.num_classes = num_classes
            self.head = nn.Linear(256, num_classes)

        def forward(self, x):
            return self.head(self.backbone(x))

        def predict_value(self, logits):
            return logits.argmax(dim=1) + 1
    return _Dice()


# ---------------------------------------------------------------------------
# Model paths and loaders
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).resolve().parent / "models"

def get_siamese_path() -> Path:
    return _MODELS_DIR / "siamese_cycle_match.pt"

def get_rotnet_path() -> Path:
    return _MODELS_DIR / "rotnet_rotation.pt"

def get_count_path() -> Path:
    return _MODELS_DIR / "count_quantity.pt"

def get_dice_path() -> Path:
    return _MODELS_DIR / "dice_classifier.pt"


def _load_state_dict(path, device):
    t = _torch()
    try:
        return t.load(path, map_location=device, weights_only=True)
    except TypeError:
        return t.load(path, map_location=device)


def load_siamese(device=None):
    """Load Siamese model if exists. Returns model or None."""
    p = get_siamese_path()
    if not p.is_file():
        return None
    t = _torch()
    device = device or _device()
    model = _SiameseNetwork()
    model.load_state_dict(_load_state_dict(p, device))
    model.to(device)
    model.eval()
    return model


def load_rotnet(device=None):
    p = get_rotnet_path()
    if not p.is_file():
        return None
    t = _torch()
    device = device or _device()
    model = _RotNet(128)
    model.load_state_dict(_load_state_dict(p, device))
    model.to(device)
    model.eval()
    return model


def load_count_regressor(device=None):
    p = get_count_path()
    if not p.is_file():
        return None
    t = _torch()
    device = device or _device()
    model = _CountRegressor(20)
    model.load_state_dict(_load_state_dict(p, device))
    model.to(device)
    model.eval()
    return model


def load_dice_classifier(device=None):
    p = get_dice_path()
    if not p.is_file():
        return None
    t = _torch()
    device = device or _device()
    model = _DiceClassifier(6)
    model.load_state_dict(_load_state_dict(p, device))
    model.to(device)
    model.eval()
    return model

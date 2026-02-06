#!/usr/bin/env python3
"""
Train a YOLOv8 model on TikTok CAPTCHA data from Roboflow.

This script downloads a labeled dataset of TikTok's "select 2 objects that
are the same shape" CAPTCHA images (3D letters, numbers, geometric shapes)
and fine-tunes YOLOv8 on them.

Usage
-----
Simplest (auto-detects dataset in ``object_selection_captcha/``)::

    python -m object_selection_captcha.train_tiktok_model

With explicit dataset path::

    python -m object_selection_captcha.train_tiktok_model --data object_selection_captcha/tikdata.v1i.yolov8/data.yaml

Download from Roboflow::

    export ROBOFLOW_API_KEY="your_key_here"
    python -m object_selection_captcha.train_tiktok_model --roboflow

After training, the best model is saved to ``object_selection_captcha/models/tiktok_captcha_best.pt``.
The solver picks it up automatically.

Requirements
------------
- ultralytics
- roboflow (only for ``--roboflow`` download; ``pip install roboflow``)
- A GPU / Apple MPS is recommended but not required (CPU works, just slower)
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_CAPTCHA_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _CAPTCHA_DIR / "models"
_DEFAULT_MODEL_NAME = "tiktok_captcha_best.pt"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _find_dataset_yaml() -> Path | None:
    """
    Auto-detect a ``data.yaml`` inside ``object_selection_captcha/`` by looking for
    Roboflow-style dataset directories.
    """
    for candidate in sorted(_CAPTCHA_DIR.glob("*/data.yaml")):
        return candidate
    for candidate in sorted(_CAPTCHA_DIR.glob("datasets/*/data.yaml")):
        return candidate
    return None


def _validate_and_fix_data_yaml(yaml_path: Path) -> Path:
    """
    Ensure the paths inside ``data.yaml`` actually point to existing
    directories.  Roboflow exports often use ``../train/images`` when the
    correct path is ``train/images`` (relative to data.yaml's parent).

    If any path is broken, write a corrected copy as ``data_fixed.yaml``
    in the same directory and return its path.  Otherwise return the
    original path unchanged.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    dataset_dir = yaml_path.parent.resolve()
    needs_fix = False

    for key in ("train", "val", "test"):
        raw = data.get(key)
        if raw is None:
            continue

        raw_path = Path(raw)

        # If already absolute and exists, leave it
        if raw_path.is_absolute() and raw_path.is_dir():
            continue

        # Resolve relative to data.yaml's directory
        resolved = (dataset_dir / raw_path).resolve()
        if resolved.is_dir():
            # Works as-is → make absolute so YOLO never struggles
            data[key] = str(resolved)
            needs_fix = True
            continue

        # Common Roboflow bug: "../train/images" should be "train/images"
        # Strip leading ".." components and retry
        parts = list(raw_path.parts)
        while parts and parts[0] == "..":
            parts.pop(0)
        fixed = dataset_dir / Path(*parts)
        if fixed.is_dir():
            data[key] = str(fixed.resolve())
            needs_fix = True
            print(f"  Fixed {key} path: {raw!r} → {data[key]}")
            continue

        # Last resort: scan common subfolder patterns
        for pattern in (
            f"{key}/images",
            f"images/{key}",
            key,
        ):
            guess = dataset_dir / pattern
            if guess.is_dir():
                data[key] = str(guess.resolve())
                needs_fix = True
                print(f"  Fixed {key} path: {raw!r} → {data[key]}")
                break
        else:
            if key != "test":  # test is optional
                print(f"  WARNING: could not resolve {key} path: {raw!r}")

    if not needs_fix:
        return yaml_path

    fixed_path = yaml_path.parent / "data_fixed.yaml"
    with open(fixed_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  Wrote corrected config to: {fixed_path}")
    return fixed_path


def _detect_device() -> str:
    """Pick the best available device: CUDA > MPS > CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Roboflow download
# ---------------------------------------------------------------------------


def download_dataset(api_key: str, workspace: str, project: str, version: int) -> str:
    """
    Download a dataset from Roboflow Universe.

    Returns the path to the ``data.yaml`` file.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: 'roboflow' package not installed. Run: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dest = str(_CAPTCHA_DIR / "datasets")
    dataset = proj.version(version).download("yolov8", location=dest)
    return os.path.join(dataset.location, "data.yaml")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 15,
    device: str = "",
) -> Path:
    """
    Fine-tune YOLOv8 on the TikTok CAPTCHA dataset.

    Parameters
    ----------
    data_yaml : str
        Absolute path to the YOLO-format ``data.yaml``.
    model_size : str
        YOLOv8 variant: "n" (nano), "s" (small), "m" (medium).
    epochs : int
        Max training epochs.
    imgsz : int
        Input image size.
    batch : int
        Batch size.
    patience : int
        Early-stopping patience (epochs without improvement).
    device : str
        Device string (e.g. "0" for GPU 0, "cpu", "mps").
        Empty string = auto-detect.

    Returns
    -------
    Path
        Path to the best model weights file.
    """
    from ultralytics import YOLO

    # Auto-detect device if not specified
    if not device:
        device = _detect_device()
    print(f"Device: {device}")

    base_model = f"yolo11{model_size}.pt"
    print(f"Loading base model: {base_model}")
    model = YOLO(base_model)

    print(f"Training on {data_yaml} for up to {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project=str(_CAPTCHA_DIR / "runs"),
        name="tiktok_captcha",
        exist_ok=True,
        verbose=True,
    )

    # Find best weights
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path(results.save_dir) / "weights" / "last.pt"

    # Copy to models/ for the solver to find
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _MODELS_DIR / _DEFAULT_MODEL_NAME
    shutil.copy2(best_weights, dest)
    print(f"\nBest model saved to: {dest}")
    print("The CAPTCHA solver will automatically use this model.")

    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on TikTok CAPTCHA data"
    )
    parser.add_argument(
        "--data",
        help="Path to data.yaml (relative or absolute)",
    )
    parser.add_argument(
        "--roboflow",
        action="store_true",
        help="Download dataset from Roboflow (requires ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--workspace",
        default="testch-ylnei",
        help="Roboflow workspace (default: testch-ylnei)",
    )
    parser.add_argument(
        "--project",
        default="tikdata",
        help="Roboflow project (default: tikdata)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Roboflow dataset version (default: 1)",
    )
    parser.add_argument(
        "--model-size",
        choices=["n", "s", "m"],
        default="n",
        help="YOLOv8 size: n=nano, s=small, m=medium (default: n)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--patience", type=int, default=15, help="Early-stop patience (default: 15)")
    parser.add_argument("--device", default="", help="Device: '0' for GPU, 'cpu', 'mps', or '' for auto")
    args = parser.parse_args()

    # ── Resolve data.yaml ────────────────────────────────────────
    if args.data:
        yaml_path = Path(args.data).resolve()
    elif args.roboflow:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            print("ERROR: Set ROBOFLOW_API_KEY environment variable.")
            print("  Get a free key at https://app.roboflow.com/settings/api")
            print("  Then: export ROBOFLOW_API_KEY='your_key_here'")
            sys.exit(1)
        print(f"Downloading dataset from Roboflow: {args.workspace}/{args.project} v{args.version}")
        yaml_path = Path(download_dataset(api_key, args.workspace, args.project, args.version)).resolve()
    else:
        # Auto-detect dataset in object_selection_captcha/
        found = _find_dataset_yaml()
        if found is None:
            print("ERROR: No dataset found. Provide one of:")
            print("  --data /path/to/data.yaml     (local dataset)")
            print("  --roboflow                     (download from Roboflow)")
            print(f"\nOr place a YOLO dataset folder inside: {_CAPTCHA_DIR}")
            sys.exit(1)
        yaml_path = found.resolve()
        print(f"Auto-detected dataset: {yaml_path}")

    if not yaml_path.is_file():
        print(f"ERROR: data.yaml not found at: {yaml_path}")
        sys.exit(1)

    # ── Validate & fix paths inside data.yaml ────────────────────
    print(f"Dataset config: {yaml_path}")
    print("Validating dataset paths...")
    yaml_path = _validate_and_fix_data_yaml(yaml_path)

    # ── Train ────────────────────────────────────────────────────
    train(
        data_yaml=str(yaml_path),
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()

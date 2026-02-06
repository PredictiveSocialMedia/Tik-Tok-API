# CAPTCHA Models

Place trained model weights here. The solver looks for:

- `tiktok_captcha_best.pt` — Fine-tuned YOLO (v11n) for TikTok shape CAPTCHA

Base models (downloaded by Ultralytics when training; can also be placed here):

- `yolo11n.pt` — YOLOv11 nano (default base for training)
- `yolov8n.pt` — YOLOv8 nano (alternative base)

## Training

```bash
# Set your Roboflow API key
export ROBOFLOW_API_KEY="your_key_here"

# Train the model
python -m captcha.train_tiktok_model

# Or use a local dataset
python -m captcha.train_tiktok_model --data /path/to/data.yaml
```

The training script automatically copies the best weights to this directory.

## Note

Model weights (`.pt` files) are excluded from git via `.gitignore`.

# Scratch

Experimental / one-off scripts.

## Object selection game (`object_select.py`)

A small “verification” game: click one of the colored boxes, then see a result screen.

**Run:**

```bash
pip install matplotlib
python object_select.py
```

- **Game:** Three boxes appear; click any one. Hover highlights the box under the cursor.
- **Result:** After your click, a second window shows **“You're verified!”** (green) or **“Try again”** (red) with a clear visual.

No browser or pynput required — matplotlib only.

---

## CAPTCHA solver demos

Test the project’s CAPTCHA solver (object detection + grid/click logic) without a real site.

### 1. Detection only (no browser)

Run YOLO on any image and see what would be detected / clicked:

```bash
# From repo root or scratch/
pip install ultralytics Pillow
python run_captcha_detection_demo.py path/to/image.jpg
python run_captcha_detection_demo.py path/to/image.jpg --output boxes.png   # draw boxes
python run_captcha_detection_demo.py "https://example.com/photo.jpg" --prompt car
```

Use `--prompt all` to show every COCO detection; default filter is “traffic light”.

### 2. Full E2E in browser

Open a local 3×3 grid page and run the solver (detect → click tiles → verify):

```bash
pip install selenium ultralytics Pillow
python run_captcha_e2e_demo.py
```

Requires Chrome and ChromeDriver. The demo page uses a grid of car images; the solver should select all 9 and click Verify.

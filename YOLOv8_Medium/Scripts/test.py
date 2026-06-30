"""
YOLOv8 Camera Classifier — Test a single image or folder
---------------------------------------------------------
Run this after training to test your model.

Usage:
    python predict.py --source path/to/image.jpg
    python predict.py --source path/to/folder/
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIGURATION  —  edit these
# ─────────────────────────────────────────────

MODEL_PATH = r".\runs\camera_classifier_v4\weights\best.pt"
CONFIDENCE  = 0.5   # minimum confidence to report a detection

# ─────────────────────────────────────────────

def predict(source: str) -> None:
    model  = YOLO(MODEL_PATH)
    source = Path(source)

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # Collect images
    if source.is_dir():
        images = sorted([
            p for p in source.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ])
    else:
        images = [source]

    print(f"\n  Model  : {MODEL_PATH}")
    print(f"  Images : {len(images)}")
    print(f"  Conf   : {CONFIDENCE}\n")
    print(f"  {'Image':<40} {'Result':<12} {'Confidence'}")
    print(f"  {'-'*65}")

    for img in images:
        results = model(str(img), verbose=False)
        probs   = results[0].probs
        top1    = probs.top1
        conf    = probs.top1conf.item()
        label   = results[0].names[top1]

        marker = "📷" if label == "camera" else "  "
        print(f"  {marker} {img.name:<38} {label:<12} {conf:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image file or folder path")
    args = parser.parse_args()
    predict(args.source)
"""
YOLOv8 Image Classification Trainer — Camera Detector
------------------------------------------------------
Trains a YOLOv8 classification model on your dataset.

Folder structure expected:
    og_images/
        train/
            camera/
            no_camera/
        valid/
            camera/
            no_camera/
        test/
            camera/
            no_camera/

Requirements:
    pip install ultralytics

Usage:
    python train.py
"""

from ultralytics import YOLO
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIGURATION  —  edit these
# ─────────────────────────────────────────────

DATA_DIR   = r".\og_images"   # your dataset root
PROJECT    = r".\runs"         # where results are saved
RUN_NAME   = "camera_classifier_v1"

MODEL      = "yolov8m-cls.pt"   # m = medium, good balance of speed vs accuracy
                                 # options: yolov8n-cls.pt (fast), yolovs8m-cls.pt, yolov8l-cls.pt (best)

EPOCHS     = 20
IMG_SIZE   = 640
BATCH      = 16    # lower to 8 if you get out-of-memory errors
WORKERS    = 8
DEVICE     = 0     # 0 = first GPU, 'cpu' = CPU only

# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────

def main():
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    # Quick sanity check
    for split in ["train", "valid", "test"]:
        split_path = data_path / split
        if not split_path.exists():
            print(f"  [WARN] Missing split folder: {split_path}")
        else:
            classes = [d.name for d in split_path.iterdir() if d.is_dir()]
            count   = sum(
                len(list(d.glob("*.*")))
                for d in split_path.iterdir() if d.is_dir()
            )
            print(f"  {split:<8} classes: {classes}  |  images: {count}")

    print(f"\n  Model      : {MODEL}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Image size : {IMG_SIZE}")
    print(f"  Batch size : {BATCH}")
    print(f"  Device     : {DEVICE}")
    print(f"  Output     : {PROJECT}/{RUN_NAME}")
    print("\n  Starting training...\n")

    model = YOLO(MODEL)

    results = model.train(
        data      = DATA_DIR,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        workers   = WORKERS,
        device    = DEVICE,
        project   = PROJECT,
        name      = RUN_NAME,
        patience  = 15,
        save      = True,
        plots     = True,
        verbose   = True,
    )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model saved to: {PROJECT}/{RUN_NAME}/weights/best.pt")
    print(f"  Results & plots:     {PROJECT}/{RUN_NAME}/")


if __name__ == "__main__":
    main()
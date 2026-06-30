"""
YOLOv8 Classification Model Evaluator
--------------------------------------
Evaluates your trained model on the test set and reports:
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - Per-class breakdown

Requirements:
    pip install ultralytics scikit-learn

Usage:
    python evaluate.py
"""

from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import numpy as np

# ─────────────────────────────────────────────
#  CONFIGURATION  —  edit these
# ─────────────────────────────────────────────

MODEL_PATH = r".\runs\camera_classifier_v4\weights\best.pt"
TEST_DIR   = r".\dataset\test"   # must have camera/ no_camera/ subfolders
CONF       = 0.0   # use 0.0 to evaluate all predictions regardless of confidence

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─────────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────────

def evaluate():
    model    = YOLO(MODEL_PATH)
    test_dir = Path(TEST_DIR)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    # Discover classes from subfolder names
    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    if not class_names:
        raise ValueError(f"No class subfolders found in {TEST_DIR}")

    print(f"\n{'='*60}")
    print(f"  YOLOv8 Classification Evaluator")
    print(f"{'='*60}")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Test dir: {TEST_DIR}")
    print(f"  Classes : {class_names}")

    # Map class name → index (matches model's internal ordering)
    # model.names is a dict like {0: 'camera', 1: 'no_camera'}
    model_class_to_idx = {v: k for k, v in model.names.items()}
    print(f"  Model class map: {model.names}")

    # Collect all images and their true labels
    all_images   = []
    true_labels  = []

    for class_dir in class_dirs:
        images = [p for p in class_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
        for img in images:
            all_images.append(img)
            true_labels.append(class_dir.name)

    print(f"\n  Total test images: {len(all_images)}")
    for cls in class_names:
        count = sum(1 for l in true_labels if l == cls)
        print(f"    {cls:<20} {count} images")

    # Run inference
    print(f"\n  Running inference...")
    pred_labels = []
    pred_confs  = []

    for img_path in tqdm(all_images, unit="img"):
        results = model(str(img_path), verbose=False)
        probs   = results[0].probs
        top1    = probs.top1
        conf    = probs.top1conf.item()
        label   = results[0].names[top1]
        pred_labels.append(label)
        pred_confs.append(conf)

    # ── Metrics ───────────────────────────────────────────────────────
    precision = precision_score(true_labels, pred_labels,
                                average="weighted", labels=class_names, zero_division=0)
    recall    = recall_score(true_labels, pred_labels,
                             average="weighted", labels=class_names, zero_division=0)
    f1        = f1_score(true_labels, pred_labels,
                         average="weighted", labels=class_names, zero_division=0)

    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / len(true_labels)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total images : {len(all_images)}")
    print(f"  Correct      : {correct}")
    print(f"  Accuracy     : {accuracy:.1%}")
    print(f"  Precision    : {precision:.1%}")
    print(f"  Recall       : {recall:.1%}")
    print(f"  F1 Score     : {f1:.1%}")

    # ── Per-class breakdown ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PER-CLASS BREAKDOWN")
    print(f"{'='*60}")
    report = classification_report(true_labels, pred_labels,
                                   labels=class_names, zero_division=0)
    print(report)

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    print(f"{'='*60}")
    print(f"  CONFUSION MATRIX")
    print(f"{'='*60}")
    header = f"{'':>15}" + "".join(f"{c:>15}" for c in class_names)
    print(f"  {header}")
    for i, row_name in enumerate(class_names):
        row = f"  {row_name:>15}" + "".join(f"{cm[i][j]:>15}" for j in range(len(class_names)))
        print(row)

    # ── Failures ──────────────────────────────────────────────────────
    failures = [
        (all_images[i], true_labels[i], pred_labels[i], pred_confs[i])
        for i in range(len(all_images))
        if true_labels[i] != pred_labels[i]
    ]

    print(f"\n{'='*60}")
    print(f"  FAILURES  ({len(failures)} wrong out of {len(all_images)})")
    print(f"{'='*60}")
    if failures:
        print(f"  {'Image':<45} {'True':<15} {'Predicted':<15} {'Conf'}")
        print(f"  {'-'*85}")
        for img, true, pred, conf in failures[:30]:   # show max 30
            print(f"  {img.name:<45} {true:<15} {pred:<15} {conf:.1%}")
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
    else:
        print("  No failures — perfect score on test set!")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    evaluate()
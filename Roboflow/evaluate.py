from inference import get_model
from pathlib import Path
import sys

# Initialize the model
model = get_model(
    model_id="camerav2-u58ps/2",  
    api_key="0h1TYROakqSQIO7OyqGC"
)

# Paths
test_dir = Path("D:/Projects/SaxionPA/CameraModel/ROBOFLOW/test")

# Counters for our Confusion Matrix
# Positive class = "camera", Negative class = "no_camera"
tp = 0  # True Positives: Actual camera, Predicted camera
fn = 0  # False Negatives: Actual camera, Predicted no_camera
fp = 0  # False Positives: Actual no_camera, Predicted camera
tn = 0  # True Negatives: Actual no_camera, Predicted no_camera

print("\n--- Starting Bulk Evaluation (Calculating Core Metrics) ---")

# 1. Evaluate the CAMERA folder (Actual Positives)
camera_path = test_dir / "camera"
if camera_path.exists():
    images = [p for p in camera_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    total = len(images)
    print(f"\nProcessing [CAMERA] folder...")
    for idx, img in enumerate(images, 1):
        result = model.infer(str(img))
        res_obj = result[0] if isinstance(result, list) else result
        
        if res_obj and hasattr(res_obj, "predictions") and res_obj.predictions:
            pred = getattr(res_obj.predictions[0], "class_name", getattr(res_obj.predictions[0], "class", "Unknown")).lower().strip()
            if pred == "camera":
                tp += 1
            else:
                fn += 1
        else:
            fn += 1 # Count unparsed as a miss
        sys.stdout.write(f"\r progress: {idx}/{total}")
        sys.stdout.flush()
    print()

# 2. Evaluate the NO_CAMERA folder (Actual Negatives)
no_camera_path = test_dir / "no_camera"
if no_camera_path.exists():
    images = [p for p in no_camera_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    total = len(images)
    print(f"\nProcessing [NO_CAMERA] folder...")
    for idx, img in enumerate(images, 1):
        result = model.infer(str(img))
        res_obj = result[0] if isinstance(result, list) else result
        
        if res_obj and hasattr(res_obj, "predictions") and res_obj.predictions:
            pred = getattr(res_obj.predictions[0], "class_name", getattr(res_obj.predictions[0], "class", "Unknown")).lower().strip()
            if pred == "camera":
                fp += 1
            else:
                tn += 1
        else:
            tn += 1
        sys.stdout.write(f"\r progress: {idx}/{total}")
        sys.stdout.flush()
    print()

# --- Metric Calculations ---
total_predictions = tp + tn + fp + fn
correct_predictions = tp + tn

accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# --- THE FINAL SCOREBOARD ---
print("\n================ EVALUATION METRICS ================")
print(f"✅ Correct Predictions : {correct_predictions} / {total_predictions}")
print(f"📊 Accuracy            : {accuracy:.2%}")
print(f"🎯 Precision           : {precision:.2%}")
print(f"📈 Recall              : {recall:.2%}")
print(f"⚖️  F1 Score            : {f1_score:.2%}")

print("\n============ CONFUSION MATRIX ============")
print("                   PREDICTED")
print("               Camera   No Camera")
print(f"ACTUAL Camera    [ {tp:<5} ] [ {fn:<5} ]  (True Pos / False Neg)")
print(f"       No Cam    [ {fp:<5} ] [ {tn:<5} ]  (False Pos / True Neg)")
print("====================================================")
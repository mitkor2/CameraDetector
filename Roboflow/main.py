from inference import get_model
from pathlib import Path

# Initialize the model (Make sure to change the '1' to your actual version number!)
model = get_model(
    model_id="camerav2-u58ps/2",  
    api_key="0h1TYROakqSQIO7OyqGC"
)

image_dir = Path("./images") # Change this to your images directory

print("\n--- Starting Inference Results ---")

for image_path in image_dir.glob("*"):
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        continue

    # Run the model
    result = model.infer(str(image_path))
    
    # If it's wrapped inside a list, unpack the first item
    if isinstance(result, list):
        response_obj = result[0] if len(result) > 0 else None
    else:
        response_obj = result

    if response_obj and hasattr(response_obj, "predictions") and response_obj.predictions:
        # Get the top prediction from the object's predictions list
        top_pred = response_obj.predictions[0]
        
        # Access attributes directly using dot notation (.class_name and .confidence)
        # Note: Roboflow uses 'class_name' or 'predicted_class' in their objects
        class_name = getattr(top_pred, "class_name", getattr(top_pred, "class", "Unknown"))
        confidence = getattr(top_pred, "confidence", 0.0)
        
        print(f"📄 {image_path.name} -> Prediction: [{class_name}] | Confidence: {confidence:.2%}")
    else:
        print(f"📄 {image_path.name} -> ⚠️ No predictions found on the response object.")
                
print("--- Processing Complete ---")
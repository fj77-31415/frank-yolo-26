import os
import ultralytics
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- 1. RUN TTA VALIDATION ---
# CRITICAL FIX: Load your specifically trained model, NOT the base yolo11n.pt
trained_model_path = "/workspace/my_volume/runs/run_5090_aerial/weights/best.pt"
model = YOLO(trained_model_path) 

# Run validation with Test Time Augmentation (TTA) enabled
print("Running TTA Validation...")
metrics = model.val(split='test', augment=True)

# --- 2. RUN SAHI INFERENCE ---
print("Loading model into SAHI...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8', # 'yolov8' handles YOLO11 weights in SAHI
    model_path=trained_model_path,
    confidence_threshold=0.25,
    device="cuda:0" 
)

# Run Sliced Inference on a single high-resolution test image
print("Running SAHI prediction...")
result = get_sliced_prediction(
    "/dataset/images/test/DS1_gss221_jpg.rf.4a4446da30cf676ef2a5ce5211e3de51.jpg",
    detection_model,
    slice_height=1024, # Matches training imgsz
    slice_width=1024,  
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# Save the visual results to your persistent volume
output_dir = "/workspace/my_volume/sahi_results/"
os.makedirs(output_dir, exist_ok=True)
result.export_visuals(export_dir=output_dir)
print(f"Done. Results saved to {output_dir}")
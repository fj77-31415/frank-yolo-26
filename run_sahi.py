import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- Configuration ---
MODEL_PATH = '/home/frankjunkar/CUAir/YOLO-testing/finetuned.pt'  # Path to your trained YOLO11 model weights
INPUT_DIR = '/home/frankjunkar/CUAir/YOLO-testing/rando-imgs'
OUTPUT_DIR = '/home/frankjunkar/CUAir/YOLO-testing/sahi_output_3'

SLICE_SIZE = 1536
OVERLAP_RATIO = 0.05
CONF_THRESH = 0.1 # change as needed
# ---------------------

# --- MPS Mac Specific ---
# run in terminal after moving files to Mac:
"""
yolo export model=finetuned.pt format=coreml half=True
"""
# then update your MODEL_PATH in the SAHI script to point to this new CoreML file instead of the .pt file


# ---------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov11',
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESH,
        device="cuda:0" 
    )


    # replace above with this when on Mac:
    """
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESH,
        device="mps" 
    )
    """

    detection_model.category_mapping = {'0': 'human', '1': 'tent', '2': 'tree'}

    for img_name in os.listdir(INPUT_DIR):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(INPUT_DIR, img_name)
        print(f"Processing: {img_name}")

        try:
            result = get_sliced_prediction(
                img_path,
                detection_model,
                slice_height=SLICE_SIZE,
                slice_width=SLICE_SIZE,
                overlap_height_ratio=OVERLAP_RATIO,
                overlap_width_ratio=OVERLAP_RATIO
            )
            
            result.export_visuals(export_dir=OUTPUT_DIR, file_name=os.path.splitext(img_name)[0])
            
        except Exception as e:
            print(f"FAILED on {img_name}. Skipping file. Error: {e}")
            continue

    print(f"Inference complete. Reconstituted images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
import os

# --- Configuration ---
# Ensure this is the EXACT path to your fine-tuning dataset
DATASET_DIR = "/home/frankjunkar/CUAir/YOLO-testing/new_annotations_sliced"
# ---------------------

def clean_empty_images(base_dir):
    splits = ['train', 'valid', 'test']
    total_deleted = 0
    
    print(f"Scanning dataset at: {base_dir}...\n")

    for split in splits:
        print(f"Checking [{split.upper()}] split...")
        
        # Updated to match your structure: base_dir/split/images and base_dir/split/labels
        images_dir = os.path.join(base_dir, split, 'images')
        labels_dir = os.path.join(base_dir, split, 'labels')

        if not os.path.exists(images_dir):
            print(f"  -> WARNING: Missing image folder: {images_dir}")
            continue
        if not os.path.exists(labels_dir):
            print(f"  -> WARNING: Missing label folder: {labels_dir}")
            continue

        deleted_count = 0
        kept_count = 0

        for img_filename in os.listdir(images_dir):
            if img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                
                base_name = os.path.splitext(img_filename)[0]
                txt_filename = base_name + '.txt'
                
                img_path = os.path.join(images_dir, img_filename)
                txt_path = os.path.join(labels_dir, txt_filename)

                # TRIGGER 1: Text file doesn't exist at all
                if not os.path.exists(txt_path):
                    os.remove(img_path)
                    deleted_count += 1
                    total_deleted += 1
                
                # TRIGGER 2: Text file exists, but is completely empty (0 bytes)
                elif os.path.getsize(txt_path) == 0:
                    os.remove(img_path)      # Delete the image
                    os.remove(txt_path)      # Delete the useless empty text file
                    deleted_count += 1
                    total_deleted += 1
                    
                else:
                    kept_count += 1

        print(f"  - Kept: {kept_count} populated images")
        print(f"  - Deleted: {deleted_count} empty images/labels\n")

    print(f"Cleanup complete. Removed {total_deleted} total empty backgrounds.")

if __name__ == "__main__":
    if os.path.exists(DATASET_DIR):
        clean_empty_images(DATASET_DIR)
    else:
        print(f"FATAL ERROR: The root directory '{DATASET_DIR}' does not exist. Check your path.")
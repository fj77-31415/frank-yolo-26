import os
import cv2
import random

# --- Configuration ---
INPUT_DIR = "/home/frankjunkar/CUAir/YOLO-testing/4-26" # Folder containing train/, valid/, test/
OUTPUT_DIR = "/home/frankjunkar/CUAir/YOLO-testing/4-26_sliced"
TILE_SIZE = 1536
BG_RETENTION_RATE = 0.10 # Retains 10% of empty background tiles
# ---------------------

def create_dirs(base_path, split):
    img_dir = os.path.join(base_path, split, 'images')
    lbl_dir = os.path.join(base_path, split, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    return img_dir, lbl_dir

def process_image(img_path, lbl_path, out_img_dir, out_lbl_dir):
    img = cv2.imread(img_path)
    if img is None:
        return

    img_h, img_w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Read original normalized labels
    boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_cen, y_cen, w, h = map(float, parts[1:])
                    # Denormalize to absolute pixels
                    xmin = (x_cen - w / 2) * img_w
                    ymin = (y_cen - h / 2) * img_h
                    xmax = (x_cen + w / 2) * img_w
                    ymax = (y_cen + h / 2) * img_h
                    boxes.append((class_id, xmin, ymin, xmax, ymax))

    # Slide 1536x1536 window across the image
    for y in range(0, img_h, TILE_SIZE):
        for x in range(0, img_w, TILE_SIZE):
            # Handle edges by snapping window to the border
            y1 = min(y, img_h - TILE_SIZE)
            x1 = min(x, img_w - TILE_SIZE)
            y2 = y1 + TILE_SIZE
            x2 = x1 + TILE_SIZE
            
            # Prevent out-of-bounds indexing if original image is smaller than 1536
            if y2 > img_h or x2 > img_w:
                continue

            tile_boxes = []
            for (cid, b_xmin, b_ymin, b_xmax, b_ymax) in boxes:
                # Check for intersection with current tile
                overlap_xmin = max(b_xmin, x1)
                overlap_ymin = max(b_ymin, y1)
                overlap_xmax = min(b_xmax, x2)
                overlap_ymax = min(b_ymax, y2)

                if overlap_xmin < overlap_xmax and overlap_ymin < overlap_ymax:
                    # Translate to tile coordinates
                    new_xmin = overlap_xmin - x1
                    new_ymin = overlap_ymin - y1
                    new_xmax = overlap_xmax - x1
                    new_ymax = overlap_ymax - y1

                    # Renormalize to 0-1 for the 1536x1536 tile
                    new_w = (new_xmax - new_xmin) / TILE_SIZE
                    new_h = (new_ymax - new_ymin) / TILE_SIZE
                    new_x_cen = (new_xmin + (new_xmax - new_xmin) / 2) / TILE_SIZE
                    new_y_cen = (new_ymin + (new_ymax - new_ymin) / 2) / TILE_SIZE
                    
                    # Filter out boxes clipped to less than 10 pixels wide/tall to prevent ghost artifacts
                    if new_w * TILE_SIZE > 10 and new_h * TILE_SIZE > 10:
                        tile_boxes.append(f"{cid} {new_x_cen:.6f} {new_y_cen:.6f} {new_w:.6f} {new_h:.6f}\n")

            # Determine whether to save the tile
            if len(tile_boxes) > 0 or random.random() < BG_RETENTION_RATE:
                tile_name = f"{base_name}_tile_{x1}_{y1}"
                tile_img = img[y1:y2, x1:x2]
                
                cv2.imwrite(os.path.join(out_img_dir, f"{tile_name}.jpg"), tile_img)
                
                if len(tile_boxes) > 0:
                    with open(os.path.join(out_lbl_dir, f"{tile_name}.txt"), 'w') as f:
                        f.writelines(tile_boxes)

def main():
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        in_img_dir = os.path.join(INPUT_DIR, split, 'images')
        in_lbl_dir = os.path.join(INPUT_DIR, split, 'labels')
        
        if not os.path.exists(in_img_dir):
            continue
            
        out_img_dir, out_lbl_dir = create_dirs(OUTPUT_DIR, split)
        
        for img_file in os.listdir(in_img_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(in_img_dir, img_file)
                lbl_path = os.path.join(in_lbl_dir, os.path.splitext(img_file)[0] + '.txt')
                process_image(img_path, lbl_path, out_img_dir, out_lbl_dir)

    print(f"Slicing complete. Data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
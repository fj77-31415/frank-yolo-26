import os
import zipfile
import tempfile
import shutil

# Define paths
downloads_dir = "/home/frankjunkar/CUAir/YOLO-testing/"
zip_files = ['2026-02-04T05-46-33.zip']
output_jpg_zip = os.path.join(downloads_dir, 'cleaned_4-26.zip')

def process_zips():
    with zipfile.ZipFile(output_jpg_zip, 'w', zipfile.ZIP_DEFLATED) as jpg_out_zip:
        for zip_name in zip_files:
            zip_path = os.path.join(downloads_dir, zip_name)
            
            if not os.path.exists(zip_path):
                print(f"File not found: {zip_path}")
                continue

            # Create a temporary file for the cleaned zip
            fd, temp_zip_path = tempfile.mkstemp(suffix='.zip')
            os.close(fd)

            with zipfile.ZipFile(zip_path, 'r') as original_zip:
                with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as temp_zip:
                    for item in original_zip.infolist():
                        if item.filename.lower().endswith('.jpg'):
                            # Skip 0-byte images
                            if item.file_size == 0:
                                continue
                                
                            # Add to combined jpg zip, prefixing to avoid name collisions
                            content = original_zip.read(item.filename)
                            new_name = f"{os.path.splitext(zip_name)[0]}_{os.path.basename(item.filename)}"
                            jpg_out_zip.writestr(new_name, content)
                        else:
                            # Keep non-jpg files in the original zip (via temp zip)
                            content = original_zip.read(item.filename)
                            temp_zip.writestr(item, content)
            
            # Replace original zip with the cleaned temp zip
            shutil.move(temp_zip_path, zip_path)
            print(f"Cleaned: {zip_name}")

    print(f"Success. Combined jpgs saved to: {output_jpg_zip}")

if __name__ == "__main__":
    process_zips()
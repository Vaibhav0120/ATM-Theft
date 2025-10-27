import os
import glob
import sys
import shutil
from tqdm import tqdm
from PIL import Image

# This is YOUR original remapping logic, which we use to SORT the crops.
CLASS_REMAPPING = {
    # --- Map to NEW ID 0 (Face_Covered) ---
    1: 0,  # Helmet
    7: 0,  # balaclava
    8: 0,  # concealing glasses
    9: 0,  # cover
    11: 0, # hand
    12: 0, # mask
    13: 0, # medicine mask
    17: 0, # person-with-mask
    19: 0, # scarf
    20: 0, # thief_mask
    # --- Map to NEW ID 1 (Face_Uncovered) ---
    0: 1,  # Cuong
    2: 1,  # Hung
    3: 1,  # Lau-Ka-Fai
    4: 1,  # Trung
    5: 1,  # Tuan
    6: 1,  # Vu
    10: 1, # face
    14: 1, # non-concealing glasses
    15: 1, # normal
    16: 1, # nothing
    18: 1, # person-without-mask
}

# Define our new class names
CLASS_NAMES = {
    0: "covered",
    1: "uncovered"
}

# Define the output directory structure
OUTPUT_DIR = os.path.join('dataset_configs', 'classifier_data')

def yolo_to_pixel(yolo_box, img_width, img_height):
    """Converts YOLO format (x_c, y_c, w, h) [0-1] to pixel [x_min, y_min, x_max, y_max]"""
    x_c, y_c, w, h = yolo_box
    x_c_px = x_c * img_width
    y_c_px = y_c * img_height
    w_px = w * img_width
    h_px = h * img_height
    
    x_min = int(x_c_px - (w_px / 2))
    y_min = int(y_c_px - (h_px / 2))
    x_max = int(x_c_px + (w_px / 2))
    y_max = int(y_c_px + (h_px / 2))
    
    # Clamp values to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)
    
    return [x_min, y_min, x_max, y_max]

def create_classifier_dataset(dataset_root):
    """
    Crops faces from the original dataset and saves them into
    'covered' and 'uncovered' folders for classifier training.
    """
    print(f"Creating classifier dataset in '{OUTPUT_DIR}'...")
    
    # Clean/Create output directories
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'valid', 'test']:
        for class_name in CLASS_NAMES.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)
            
    crop_count = 0
    
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(dataset_root, split, 'labels')
        image_dir = os.path.join(dataset_root, split, 'images')
        
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        
        if not label_files:
            print(f"Warning: No label files found in {label_dir}")
            continue

        print(f"Processing {split} split...")
        for label_file in tqdm(label_files, desc=f"Creating '{split}' crops"):
            base_name = os.path.basename(label_file)
            image_name = base_name.replace('.txt', '.jpg') # Assumes .jpg, adjust if needed
            image_path = os.path.join(image_dir, image_name)
            
            if not os.path.exists(image_path):
                # Check for .png, .jpeg
                image_name_png = base_name.replace('.txt', '.png')
                image_path_png = os.path.join(image_dir, image_name_png)
                if os.path.exists(image_path_png):
                    image_path = image_path_png
                else:
                    print(f"Warning: No image found for {label_file}")
                    continue

            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    old_class_id = int(parts[0])
                    
                    # Use YOUR mapping to decide which folder it goes in
                    if old_class_id in CLASS_REMAPPING:
                        new_class_id = CLASS_REMAPPING[old_class_id]
                        class_name = CLASS_NAMES[new_class_id]
                        
                        # Get YOLO box coordinates
                        yolo_box = [float(p) for p in parts[1:5]]
                        
                        # Convert to pixel coordinates
                        pixel_box = yolo_to_pixel(yolo_box, img_width, img_height)
                        
                        # Crop the image
                        with Image.open(image_path) as img_to_crop:
                            # Add a 10% padding to the crop, optional but often helps
                            pad_w = int((pixel_box[2] - pixel_box[0]) * 0.10)
                            pad_h = int((pixel_box[3] - pixel_box[1]) * 0.10)
                            
                            padded_box = [
                                max(0, pixel_box[0] - pad_w),
                                max(0, pixel_box[1] - pad_h),
                                min(img_width, pixel_box[2] + pad_w),
                                min(img_height, pixel_box[3] + pad_h)
                            ]
                            
                            crop = img_to_crop.crop(padded_box)
                        
                        # Save the cropped image
                        crop_filename = f"{base_name.replace('.txt', '')}_crop_{i}.jpg"
                        save_path = os.path.join(OUTPUT_DIR, split, class_name, crop_filename)
                        crop.save(save_path, "JPEG")
                        crop_count += 1

            except Exception as e:
                print(f"Error processing file {image_path}: {e}")
                
    print(f"\n✅ Classifier dataset creation complete. Saved {crop_count} cropped images.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide the dataset root path.")
        print("Usage: python prepare_classifier_data.py <path_to_dataset_root>")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found at '{dataset_path}'")
        sys.exit(1)
        
    create_classifier_dataset(dataset_path)
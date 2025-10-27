import os
import glob
import sys
from tqdm import tqdm

def remap_all_to_zero(dataset_root):
    """
    Finds all label files in train/valid/test and rewrites them,
    forcing every single bounding box to be class ID 0.
    """
    print(f"Remapping all classes in {dataset_root} to a single class '0'...")
    
    label_files = []
    for split in ['train', 'valid', 'test']:
        path_pattern = os.path.join(dataset_root, split, 'labels', '*.txt')
        label_files.extend(glob.glob(path_pattern))

    if not label_files:
        print(f"Warning: No label files found in '{dataset_root}'.")
        return

    print(f"Found {len(label_files)} label files to process for the detector.")
    remapped_count = 0

    for file_path in tqdm(label_files, desc="Remapping Detector Labels"):
        temp_lines = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                continue

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5: # Ensure it's a valid YOLO line
                    # Overwrite old class ID (parts[0]) with '0'
                    new_line = f"0 {' '.join(parts[1:])}\n"
                    temp_lines.append(new_line)
            
            if temp_lines:
                with open(file_path, 'w') as f:
                    f.writelines(temp_lines)
                remapped_count += 1
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(f"\n✅ Detector remapping complete. {remapped_count} files remapped to class '0'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide the dataset root path.")
        print("Usage: python remap_detector_labels.py <path_to_dataset_root>")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found at '{dataset_path}'")
        sys.exit(1)
        
    remap_all_to_zero(dataset_path)
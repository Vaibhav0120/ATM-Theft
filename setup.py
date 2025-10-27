import os
import subprocess
import sys
from roboflow import Roboflow

# --- 1. Download Dataset from Roboflow ---
print("Step 1/3: Downloading dataset from Roboflow...")

try:
    rf = Roboflow(api_key="5NOA4aA1WX8FH7q8CypK")
    project = rf.workspace("vaibhav-7tcrm").project("atm-theft-detection-f8ezg")
    version = project.version(4)
    # The dataset will be downloaded to 'ATM-Theft-Detection-4'
    dataset = version.download("yolov8") 
    dataset_path = dataset.location
    print(f"Dataset downloaded to: {dataset_path}")

except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Please check your API key and project/workspace names.")
    sys.exit(1)

# --- 2. Run Detector Label Remapping ---
print("\nStep 2/3: Preparing data for Face Detector (Model 1)...")
# This script will merge all 21 classes into 1 class: 'face'
detector_script_path = os.path.join('data_preparation', 'remap_detector_labels.py')
try:
    # Pass the downloaded dataset path to the script
    subprocess.run([sys.executable, detector_script_path, dataset_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running detector remapping: {e}")
    sys.exit(1)

# --- 3. Run Classifier Data Preparation ---
print("\nStep 3/3: Preparing data for Mask Classifier (Model 2)...")
# This script will crop faces and sort them into 'covered'/'uncovered' folders
classifier_script_path = os.path.join('data_preparation', 'prepare_classifier_data.py')
try:
    # Pass the downloaded dataset path to the script
    subprocess.run([sys.executable, classifier_script_path, dataset_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running classifier data preparation: {e}")
    sys.exit(1)

print("\n✅ All setup and data preparation steps completed successfully!")
print(f"Detector data is ready in '{dataset_path}' (labels are remapped).")
print("Classifier data is ready in 'dataset_configs/classifier_data/'.")
print("You are now ready to run the training scripts in the 'training/' folder.")
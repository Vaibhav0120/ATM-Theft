import os
import subprocess
import sys
import time
from roboflow import Roboflow

# --- 1. Download Dataset from Roboflow ---
print("➡️ Step 1/3: Starting dataset download from Roboflow...")
start_time = time.time()

try:
    rf = Roboflow(api_key="5NOA4aA1WX8FH7q8CypK")
    project = rf.workspace("vaibhav-7tcrm").project("atm-theft-detection-f8ezg")
    version = project.version(4)
    # The dataset will be downloaded to 'ATM-Theft-Detection-4'
    dataset = version.download("yolov8") #
    dataset_path = dataset.location
    end_time = time.time()
    print(f"✅ Dataset downloaded successfully to: {dataset_path} (took {end_time - start_time:.2f} seconds)")

except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("Please check your API key and project/workspace names.")
    sys.exit(1)

# --- 2. Run Detector Label Remapping ---
print("\n➡️ Step 2/3: Preparing data for Face Detector (Model 1)...")
# This script will merge all 21 classes into 1 class: 'face'
detector_script_path = os.path.join('data_preparation', 'remap_detector_labels.py') #
try:
    # Pass the downloaded dataset path to the script
    # The script itself contains a tqdm progress bar. Removed capture_output.
    subprocess.run([sys.executable, detector_script_path, dataset_path], check=True) #
    print(f"\n✅ Detector data remapping complete.") # Added newline for clarity
except subprocess.CalledProcessError as e:
    print(f"❌ Error running detector remapping: {e}")
    print("--- Error Output ---")
    print(e.stderr)
    print("--------------------")
    sys.exit(1)
except FileNotFoundError:
    print(f"❌ Error: Script not found at '{detector_script_path}'. Make sure it exists.")
    sys.exit(1)


# --- 3. Run Classifier Data Preparation ---
print("\n➡️ Step 3/3: Preparing data for Mask Classifier (Model 2)...")
# This script will crop faces and sort them into 'covered'/'uncovered' folders
classifier_script_path = os.path.join('data_preparation', 'prepare_classifier_data.py') #
try:
    # Pass the downloaded dataset path to the script
    # The script itself contains a tqdm progress bar. Removed capture_output.
    subprocess.run([sys.executable, classifier_script_path, dataset_path], check=True) #
    print(f"\n✅ Classifier data preparation complete.") # Added newline for clarity
except subprocess.CalledProcessError as e:
    print(f"❌ Error running classifier data preparation: {e}")
    print("--- Error Output ---")
    print(e.stderr)
    print("--------------------")
    sys.exit(1)
except FileNotFoundError:
    print(f"❌ Error: Script not found at '{classifier_script_path}'. Make sure it exists.")
    sys.exit(1)

print("\n🎉 All setup and data preparation steps completed successfully!")
print(f"Detector data is ready in '{dataset_path}' (labels remapped to class 0).") #
print("Classifier data (cropped images) is ready in 'dataset_configs/classifier_data/'.") #
print("You are now ready to run the training scripts in the 'training/' folder.") #
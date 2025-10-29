import os
import subprocess
import sys
import time
from roboflow import Roboflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Get API Key and Download Dataset ---
print("➡️ Step 1/3: Starting dataset download from Roboflow...")
start_time = time.time()

# Get API key from environment
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    print("❌ Error: ROBOFLOW_API_KEY not found.")
    print("Please create a file named '.env' in this directory and add the line:")
    print("ROBOFLOW_API_KEY=YOUR_KEY_HERE")
    sys.exit(1)

try:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("vaibhav-7tcrm").project("atm-theft-detection-f8ezg")
    version = project.version(4)
    # The dataset will be downloaded to 'ATM-Theft-Detection-4'
    dataset = version.download("yolov8")
    dataset_path = dataset.location
    end_time = time.time()
    print(f"✅ Dataset downloaded successfully to: {dataset_path} (took {end_time - start_time:.2f} seconds)")

except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("Please check your API key (in the .env file) and project/workspace names.")
    sys.exit(1)

# --- 2. Run Classifier Data Preparation (FIXED ORDER: This must run FIRST) ---
print("\n➡️ Step 2/3: Preparing data for Mask Classifier (Model 2)...")
# This script will crop faces and sort them into 'covered'/'uncovered' folders
# It needs the ORIGINAL labels before they are remapped.
classifier_script_path = os.path.join('data_preparation', 'prepare_classifier_data.py')
try:
    # Pass the downloaded dataset path to the script
    subprocess.run([sys.executable, classifier_script_path, dataset_path], check=True)
    print(f"\n✅ Classifier data preparation complete.")
except subprocess.CalledProcessError as e:
    print(f"❌ Error running classifier data preparation: {e}")
    print("--- Error Output ---")
    print(e.stderr)
    print("--------------------")
    sys.exit(1)
except FileNotFoundError:
    print(f"❌ Error: Script not found at '{classifier_script_path}'. Make sure it exists.")
    sys.exit(1)


# --- 3. Run Detector Label Remapping (FIXED ORDER: This must run SECOND) ---
print("\n➡️ Step 3/3: Preparing data for Face Detector (Model 1)...")
# This script will merge all 21 classes into 1 class: 'face'
# This modifies the labels in-place, which is why it must run AFTER classifier prep.
detector_script_path = os.path.join('data_preparation', 'remap_detector_labels.py')
try:
    # Pass the downloaded dataset path to the script
    subprocess.run([sys.executable, detector_script_path, dataset_path], check=True)
    print(f"\n✅ Detector data remapping complete.")
except subprocess.CalledProcessError as e:
    print(f"❌ Error running detector remapping: {e}")
    print("--- Error Output ---")
    print(e.stderr)
    print("--------------------")
    sys.exit(1)
except FileNotFoundError:
    print(f"❌ Error: Script not found at '{detector_script_path}'. Make sure it exists.")
    sys.exit(1)


print("\n🎉 All setup and data preparation steps completed successfully!")
print(f"Detector data is ready in '{dataset_path}' (labels remapped to class 0).")
print("Classifier data (cropped images) is ready in 'dataset_configs/classifier_data/'.")
print("You are now ready to run the training scripts in the 'training/' folder.")

from ultralytics import YOLO
import os
import yaml
import traceback # Added for better error reporting

# --- Config ---
DATASET_YAML = os.path.join('..', 'dataset_configs', 'detector.yaml')

# --- Point to the PREVIOUSLY trained weights ---
# Check your 'runs/detect/' folder to confirm 'detector_train_run' is the correct folder name
# This should be the folder created by the FIRST successful training run.
PREVIOUS_RUN_DIR = 'detector_train_run' # IMPORTANT: Verify this folder name exists!
BEST_WEIGHTS_PATH = os.path.join(
    'runs', 'detect', PREVIOUS_RUN_DIR, 'weights', 'best.pt'
)

IMG_SIZE = 320 # Keep consistent with training
MODEL_SAVE_PATH = os.path.join('..', 'models')
FINAL_MODEL_NAME = 'detector'
# --- End Config ---

# Create save directory if it doesn't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def export_only():
    """Loads a previously trained YOLO model and exports it to TFLite INT8."""

    # 1. Load the previously trained model weights
    print(f"Loading best weights from: {BEST_WEIGHTS_PATH}")
    if not os.path.exists(BEST_WEIGHTS_PATH):
        print(f"❌ Error: Weights file not found at '{BEST_WEIGHTS_PATH}'")
        print("Please double-check the 'PREVIOUS_RUN_DIR' variable in this script.")
        print("Ensure the specified training run completed and the 'best.pt' file exists.")
        return

    try:
        model = YOLO(BEST_WEIGHTS_PATH) # Load the .pt file directly
        print("Successfully loaded pre-trained weights.")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        traceback.print_exc()
        return

    # --- Training Section is REMOVED ---
    # The model.train() call is intentionally omitted here.

    # 2. Export to TFLite (Quantized INT8)
    print("\nExporting to TFLite INT8 format...")
    print(f"Using dataset config for calibration: {DATASET_YAML}")
    print(f"Using image size: {IMG_SIZE}")

    try:
        # Export the loaded model
        # The 'data' argument is crucial for providing the representative dataset for INT8 calibration
        tflite_path = model.export(
            format='tflite',
            int8=True,          # Enable INT8 quantization
            data=DATASET_YAML,  # Path to dataset YAML for calibration images
            imgsz=IMG_SIZE,     # Input image size used during training/calibration
            simplify=True       # Apply model simplification
        )
        print(f"Intermediate TFLite file saved to: {tflite_path}")

        # Define the final desired path for the quantized model
        final_tflite_path = os.path.join(MODEL_SAVE_PATH, f"{FINAL_MODEL_NAME}_int8.tflite")

        # Check if the exported file exists before attempting to rename/move
        if os.path.exists(tflite_path):
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(final_tflite_path), exist_ok=True)
            # Rename/Move the exported file to the final destination
            os.rename(tflite_path, final_tflite_path)
            print(f"✅ Successfully exported and saved quantized TFLite model to: {final_tflite_path}")
        else:
            print(f"❌ Error: Export process completed, but the expected TFLite file was not found at '{tflite_path}'")
            print("Please check the Ultralytics export logs above for any specific errors during conversion.")

    except Exception as e:
        print(f"❌ An error occurred during the TFLite export process: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")
        print("Common issues could be related to package versions (TensorFlow, ONNX, etc.) or the calibration dataset.")


if __name__ == '__main__':
    # Verify dataset YAML exists before starting
    if not os.path.exists(DATASET_YAML):
        print(f"Error: Dataset config file not found at '{DATASET_YAML}'")
        print("Please ensure the path exists and is correct.")
    else:
        # Run the export function
        export_only()
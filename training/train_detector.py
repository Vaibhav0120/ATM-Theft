from ultralytics import YOLO
import os
import yaml
import traceback
import shutil

# --- Config ---
DATASET_YAML = os.path.join('..', 'dataset_configs', 'detector.yaml')
MODEL_SAVE_PATH = os.path.join('..', 'models')
FINAL_MODEL_NAME = 'detector'

# --- Training Parameters ---
EPOCHS = 100       # Increased from 50 for better accuracy
IMG_SIZE = 320
BATCH_SIZE = 16   # Adjust based on your GPU VRAM (RTX 4050 6GB should handle 16)
DEVICE = 0        # 0 for GPU, 'cpu' for CPU
RUN_NAME = 'detector_train_run'
# --- End Config ---

# Create save directory if it doesn't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def train_and_export():
    """Trains a new YOLO model and exports it to TFLite INT8."""
    
    print(f"Starting detector training for {EPOCHS} epochs...")
    print(f"Dataset: {DATASET_YAML}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")

    try:
        # 1. Load the base model
        model = YOLO('yolov8n.pt') # Start from pre-trained nano model

        # 2. Train the model
        results = model.train(
            data=DATASET_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            name=RUN_NAME,
            device=DEVICE
        )
        
        print("✅ Training complete.")
        print(f"Trained model weights saved in 'runs/detect/{RUN_NAME}/'")

        # 3. Export to TFLite (Quantized INT8)
        print("\nExporting to TFLite INT8 format...")
        
        # The 'model' object is now the trained model
        tflite_path = model.export(
            format='tflite',
            int8=True,          # Enable INT8 quantization
            data=DATASET_YAML,  # Path to dataset YAML for calibration images
            imgsz=IMG_SIZE,     # Input image size used during training/calibration
            simplify=True       # Apply model simplification
        )
        print(f"Intermediate TFLite file saved to: {tflite_path}")

        # 4. Move the final model
        final_tflite_path = os.path.join(MODEL_SAVE_PATH, f"{FINAL_MODEL_NAME}_int8.tflite")
        
        if os.path.exists(tflite_path):
            shutil.move(tflite_path, final_tflite_path)
            print(f"✅ Successfully exported and saved quantized TFLite model to: {final_tflite_path}")
        else:
            print(f"❌ Error: Export process completed, but the expected TFLite file was not found at '{tflite_path}'")

    except Exception as e:
        print(f"❌ An error occurred during the training or export process: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")
        print("Common issues: CUDA/GPU memory (try reducing BATCH_SIZE), or dataset path.")

if __name__ == '__main__':
    # Verify dataset YAML exists before starting
    if not os.path.exists(DATASET_YAML):
        print(f"Error: Dataset config file not found at '{DATASET_YAML}'")
        print("Please ensure the path exists and is correct.")
    else:
        # Run the training and export function
        train_and_export()


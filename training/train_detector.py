from ultralytics import YOLO
import os
import yaml

# --- Config ---
DATASET_YAML = os.path.join('..', 'dataset_configs', 'detector.yaml')
MODEL_NAME = 'yolov8n.pt'  # Use v8n for a great balance.
EPOCHS = 50
IMG_SIZE = 320
MODEL_SAVE_PATH = os.path.join('..', 'models')
FINAL_MODEL_NAME = 'detector'
# --- End Config ---

# Create save directory
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def train():
    # 1. Load the model
    # Use yolov8n.pt for a good balance of speed and accuracy
    # yolov11n.pt is also a great choice if available in your ultralytics version
    model = YOLO(MODEL_NAME)

    # 2. Train the model
    print(f"Starting training for {EPOCHS} epochs...")
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,
        name=f"{FINAL_MODEL_NAME}_train_run"
    )
    
    print("Training complete.")

    # 3. Save the final PyTorch model
    final_pt_path = os.path.join(MODEL_SAVE_PATH, f"{FINAL_MODEL_NAME}.pt")
    model.save(final_pt_path)
    print(f"Final PyTorch model saved to {final_pt_path}")

    # 4. Export to TFLite (Quantized INT8)
    print("Exporting to TFLite INT8...")
    # We use data='coco8.yaml' for the representative_dataset for quantization
    # You can also use your own dataset's YAML file
    tflite_path = model.export(
        format='tflite',
        int8=True,
        data=DATASET_YAML, 
        imgsz=IMG_SIZE,
        simplify=True
    )
    
    # Rename and move the TFLite file
    final_tflite_path = os.path.join(MODEL_SAVE_PATH, f"{FINAL_MODEL_NAME}_int8.tflite")
    os.rename(tflite_path, final_tflite_path)
    print(f"✅ Quantized TFLite model saved to: {final_tflite_path}")

if __name__ == '__main__':
    # Verify dataset YAML exists
    if not os.path.exists(DATASET_YAML):
        print(f"Error: Dataset config file not found at {DATASET_YAML}")
        print("Please ensure the path in dataset_configs/detector.yaml is correct.")
    else:
        with open(DATASET_YAML, 'r') as f:
            print("--- Detector Config (detector.yaml) ---")
            print(f.read())
            print("---------------------------------------")
        train()
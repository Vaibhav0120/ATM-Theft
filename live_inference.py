import os
import cv2
import numpy as np
import time
import tensorflow as tf # Use full TensorFlow on laptop for easier TFLite loading

# --- Configuration ---
DETECTOR_MODEL_PATH = os.path.join('models', 'detector_int8.tflite') # Path relative to script

# Input size expected by the model
DETECTOR_IMG_SIZE = (320, 320)

# Threshold
DETECTION_THRESHOLD = 0.5

# Camera settings
CAM_SOURCE = 1 # <-- CHANGED TO 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# --- End Configuration ---

def load_tflite_model(model_path):
    """Loads a TFLite model and allocates tensors using TensorFlow."""
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at {model_path}")
        return None, None, None
    try:
        # Use TensorFlow's TFLite interpreter on the laptop
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ Model loaded successfully: {os.path.basename(model_path)}")
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"❌ ERROR: Failed to load model {model_path}: {e}")
        return None, None, None

def preprocess_image(image, target_size, is_quantized=True):
    """Resizes and preprocesses an image for TFLite inference."""
    h, w, _ = image.shape
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full((th, tw, 3), 114, dtype=np.uint8) # Pad with gray
    dw, dh = (tw - nw) // 2, (th - nh) // 2
    image_padded[dh:dh+nh, dw:dw+nw, :] = image_resized

    if is_quantized:
        return image_padded
    else:
        # Should not be needed for INT8 model
        image_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
        return np.expand_dims(image_rgb.astype(np.float32) / 255.0, axis=0)

def main():
    # --- Load Model ---
    detector, det_input_details, det_output_details = load_tflite_model(DETECTOR_MODEL_PATH)
    if detector is None:
        return

    det_input_index = det_input_details[0]['index']
    det_output_index = det_output_details[0]['index'] # Adjust if needed

    # --- Setup Camera ---
    print(f"Attempting to open camera source: {CAM_SOURCE}") # Added print
    cap = cv2.VideoCapture(CAM_SOURCE)
    time.sleep(2) # <-- ADDED DELAY

    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open camera source {CAM_SOURCE}")
        # Try the other index as a fallback
        fallback_source = 1 if CAM_SOURCE == 0 else 0
        print(f"Trying fallback camera source: {fallback_source}")
        cap = cv2.VideoCapture(fallback_source)
        time.sleep(2) # Wait again
        if not cap.isOpened():
             print(f"❌ ERROR: Cannot open fallback camera source {fallback_source} either.")
             return
        else:
             print(f"✅ Fallback camera {fallback_source} opened.")
    else:
         print(f"✅ Camera {CAM_SOURCE} opened successfully.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Camera opened. Starting inference test loop...")
    frame_times = []
    display_fps = "N/A"

    while True:
        start_time = time.time() # Start timing loop

        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            # Add a small delay and try again, or break
            time.sleep(0.1)
            ret, frame = cap.read() # Try one more time
            if not ret:
                print("ERROR: Still failed to grab frame. Exiting loop.")
                break # Exit if reading fails persistently

        # --- Check if frame is valid ---
        if frame is None or frame.size == 0:
            print("WARN: Got empty frame.")
            continue # Skip processing for empty frames

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_h, original_w, _ = frame.shape

        # --- Face Detection ---
        det_input_image = preprocess_image(frame_rgb, DETECTOR_IMG_SIZE, is_quantized=True)
        det_input_tensor = np.expand_dims(det_input_image, axis=0)

        detector.set_tensor(det_input_index, det_input_tensor)
        inference_start = time.time()
        detector.invoke()
        inference_time = (time.time() - inference_start) * 1000 # Milliseconds

        # --- Process Detector Output ---
        detector_output = detector.get_tensor(det_output_index)[0].T

        faces_detected = []
        for detection in detector_output:
            confidence = detection[4]
            if confidence >= DETECTION_THRESHOLD:
                cx, cy, w, h = detection[0:4]
                x_min = int((cx - w / 2) * original_w)
                y_min = int((cy - h / 2) * original_h)
                x_max = int((cx + w / 2) * original_w)
                y_max = int((cy + h / 2) * original_h)

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(original_w, x_max)
                y_max = min(original_h, y_max)

                # Add check for valid box dimensions
                if x_max > x_min and y_max > y_min:
                    faces_detected.append(((x_min, y_min, x_max, y_max), confidence))

        # --- Draw Results ---
        for box_pixels, det_conf in faces_detected:
            x_min, y_min, x_max, y_max = box_pixels
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {det_conf:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Calculate FPS ---
        loop_time = time.time() - start_time
        frame_times.append(loop_time)
        if len(frame_times) > 30: # Average over last 30 frames
            frame_times.pop(0)

        # Avoid division by zero if frame_times is empty
        if frame_times:
            avg_loop_time = sum(frame_times) / len(frame_times)
            if avg_loop_time > 0: # Avoid division by zero
                 display_fps = f"{1.0 / avg_loop_time:.1f}"
            else:
                 display_fps = "Inf" # Should not happen often
        else:
            display_fps = "Calculating..."


        cv2.putText(frame, f"Laptop FPS: {display_fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Inference: {inference_time:.1f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # --- Show Frame ---
        try:
            cv2.imshow('Live Detector Test (Laptop)', frame)
        except cv2.error as e:
            # Handle potential display errors in WSL without proper GUI setup
            if "display cannot be opened" in str(e) or "GTK" in str(e) or "cannot connect to X server" in str(e):
                print("---")
                print("ERROR: Cannot display window in this environment.")
                print("GUI applications in WSL might require an X server (like VcXsrv or Xming) running on Windows.")
                print("Or, if using Windows 11, ensure WSLg is enabled and updated ('wsl --update').")
                print("Stopping script as window cannot be shown.")
                print("---")
                break # Exit loop if display fails
            else:
                # Re-raise other OpenCV errors
                raise e


        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Stream stopped.")

if __name__ == '__main__':
    main()
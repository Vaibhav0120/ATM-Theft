import os
import cv2
import numpy as np
import time

# --- Configuration ---
DETECTOR_MODEL_PATH = os.path.join('..', 'models', 'detector_int8.tflite')
CLASSIFIER_MODEL_PATH = os.path.join('..', 'models', 'classifier_int8.tflite')

# Input sizes expected by the models (verify these!)
DETECTOR_IMG_SIZE = (320, 320)
CLASSIFIER_IMG_SIZE = (224, 224)

# Thresholds
DETECTION_THRESHOLD = 0.5 # Minimum confidence score for a face detection
CLASSIFIER_THRESHOLD = 0.7 # Minimum confidence score for 'uncovered' (adjust as needed)

# Classifier class mapping (MAKE SURE THIS MATCHES train_classifier.py output)
CLASSIFIER_CLASSES = {0: "COVERED", 1: "UNCOVERED"} # 0='covered', 1='uncovered'

# Camera settings
CAM_SOURCE = 0 # 0 for default camera (usually USB), adjust if using Pi Camera module
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# --- End Configuration ---

def load_tflite_model(model_path):
    """Loads a TFLite model and allocates tensors."""
    try:
        # Try importing tflite_runtime first (more lightweight)
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        # Fallback to full TensorFlow if tflite_runtime is not available
        print("WARN: tflite_runtime not found. Falling back to TensorFlow Lite interpreter.")
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at {model_path}")
        return None, None, None

    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ Model loaded successfully: {os.path.basename(model_path)}")
        # Print model input/output details (useful for debugging)
        # print("Input Details:", input_details)
        # print("Output Details:", output_details)
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"❌ ERROR: Failed to load model {model_path}: {e}")
        return None, None, None

def preprocess_image(image, target_size, is_quantized=True):
    """Resizes and preprocesses an image for TFLite inference."""
    # Resize keeping aspect ratio, padding if needed (letterbox)
    h, w, _ = image.shape
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    # Create a new image with padding
    image_padded = np.full((th, tw, 3), 114, dtype=np.uint8) # Pad with gray
    dw, dh = (tw - nw) // 2, (th - nh) // 2
    image_padded[dh:dh+nh, dw:dw+nw, :] = image_resized

    if is_quantized:
        # For uint8 quantized models, input is usually just uint8
        return image_padded
    else:
        # For float models (if you ever use them)
        image_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
        return np.expand_dims(image_rgb.astype(np.float32) / 255.0, axis=0)

def main():
    # --- Load Models ---
    detector, det_input_details, det_output_details = load_tflite_model(DETECTOR_MODEL_PATH)
    classifier, cls_input_details, cls_output_details = load_tflite_model(CLASSIFIER_MODEL_PATH)

    if detector is None or classifier is None:
        print("Exiting due to model loading failure.")
        return

    # Assuming UINT8 quantization for both based on training scripts
    det_input_index = det_input_details[0]['index'] # type: ignore
    det_output_index = det_output_details[0]['index'] # type: ignore # Adjust if model output differs
    
    cls_input_index = cls_input_details[0]['index'] # type: ignore
    cls_output_index = cls_output_details[0]['index'] # type: ignore

    # --- Setup Camera ---
    cap = cv2.VideoCapture(CAM_SOURCE)
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open camera source {CAM_SOURCE}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30) # Request FPS

    print("Camera opened. Starting inference loop...")
    fps_start_time = time.time()
    fps_frame_count = 0
    display_fps = "N/A"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Models usually expect RGB
        original_h, original_w, _ = frame.shape

        # --- 1. Face Detection ---
        det_input_image = preprocess_image(frame_rgb, DETECTOR_IMG_SIZE, is_quantized=True)
        # Add batch dimension
        det_input_tensor = np.expand_dims(det_input_image, axis=0)

        detector.set_tensor(det_input_index, det_input_tensor)
        detector.invoke()
        
        # --- Process Detector Output ---
        # NOTE: Output format depends heavily on how the YOLO model was exported to TFLite.
        # This is a common format assumption for YOLOv8 TFLite: [1, Boxes+Classes, NumDetections]
        # Boxes are often [x_center, y_center, width, height, confidence, class_prob_0, ...]
        # For our detector, class_prob_0 should always be the highest as we only have one class.
        # You MIGHT need to adjust indices based on model inspection (e.g., using Netron app)
        detector_output = detector.get_tensor(det_output_index)[0] # Shape (Boxes+Classes, NumDetections)
        detector_output = detector_output.T # Transpose to (NumDetections, Boxes+Classes)

        faces_detected = [] # Store tuples of (box_pixels, confidence)
        for detection in detector_output:
            # Example: Assuming [cx, cy, w, h, confidence] (indices 0-4)
            # Adjust indices if your model output is different!
            confidence = detection[4]
            if confidence >= DETECTION_THRESHOLD:
                # Get box in relative coords (0-1)
                cx, cy, w, h = detection[0:4]
                # Convert to absolute pixel coordinates for original frame
                x_min = int((cx - w / 2) * original_w)
                y_min = int((cy - h / 2) * original_h)
                x_max = int((cx + w / 2) * original_w)
                y_max = int((cy + h / 2) * original_h)

                # Clamp box to image boundaries
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(original_w, x_max)
                y_max = min(original_h, y_max)

                faces_detected.append(((x_min, y_min, x_max, y_max), confidence))

        # --- 2. Mask Classification (for each detected face) ---
        warning_triggered = False
        for box_pixels, det_conf in faces_detected:
            x_min, y_min, x_max, y_max = box_pixels

            # Ensure box dimensions are valid before cropping
            if x_min >= x_max or y_min >= y_max:
                continue

            # Crop face from the ORIGINAL frame (BGR)
            face_crop_bgr = frame[y_min:y_max, x_min:x_max]

            # Ensure crop is not empty
            if face_crop_bgr.size == 0:
                continue

            # Preprocess crop for classifier
            face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            cls_input_image = preprocess_image(face_crop_rgb, CLASSIFIER_IMG_SIZE, is_quantized=True)
            cls_input_tensor = np.expand_dims(cls_input_image, axis=0)

            # Run classifier inference
            classifier.set_tensor(cls_input_index, cls_input_tensor)
            classifier.invoke()
            
            # --- Process Classifier Output ---
            # Output depends on the model structure (sigmoid or softmax) and quantization
            # Assuming UINT8 output for binary sigmoid (0-255 range)
            classifier_output = classifier.get_tensor(cls_output_index)[0] # Get the single output value
            
            # Dequantize (if necessary, depends on how model was converted)
            # For UINT8 output, scale/zero_point are needed for precise probability
            # Simpler approach: Check if value is above a threshold in the UINT8 range
            # Find the midpoint corresponding to CLASSIFIER_THRESHOLD (e.g., 0.7)
            # This requires knowing the scale and zero_point from the TFLite model details.
            # Example Placeholder (Needs real scale/zero point):
            # scale, zero_point = cls_output_details[0]['quantization']
            # probability_uncovered = scale * (int(classifier_output[0]) - zero_point)
            
            # --- Simplified Logic (assuming higher UINT8 value means 'uncovered') ---
            # Assume 0 (covered) maps near 0, 1 (uncovered) maps near 255
            # We need a threshold in the 0-255 range. Let's map CLASSIFIER_THRESHOLD
            uint8_threshold = int(CLASSIFIER_THRESHOLD * 255) 
            
            pred_score_uint8 = classifier_output[0] # Get the raw uint8 output
            
            if pred_score_uint8 >= uint8_threshold:
                label = CLASSIFIER_CLASSES[1] # Uncovered
                color = (0, 255, 0) # Green
            else:
                label = CLASSIFIER_CLASSES[0] # Covered
                color = (0, 0, 255) # Red
                warning_triggered = True

            # --- Draw Results on Frame ---
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{label}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Optional: Display detection confidence
            # cv2.putText(frame, f"Det: {det_conf:.2f}", (x_min, y_max + 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # --- Trigger Warning ---
        if warning_triggered:
            print("ALERT: Face Covering Detected!")
            # Add your warning mechanism here (e.g., sound, light, message)
            cv2.putText(frame, "WARNING: FACE COVERED", (10, FRAME_HEIGHT - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


        # --- Calculate and Display FPS ---
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 1.0:
            display_fps = f"{fps_frame_count / elapsed_time:.1f}"
            fps_frame_count = 0
            fps_start_time = time.time()

        cv2.putText(frame, f"FPS: {display_fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Show Frame ---
        cv2.imshow('ATM Security Feed', frame)

        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Stream stopped.")

if __name__ == '__main__':
    main()
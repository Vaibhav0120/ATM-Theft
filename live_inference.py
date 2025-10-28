"""
Test INT8 TFLite models on Windows CPU
No GPU required - perfect for testing before Raspberry Pi deployment!

SETUP:
1. Using Python 3.11
2. pip install tensorflow opencv-python numpy
3. Run: python live_inference.py

"""

import os
import cv2
import numpy as np
import time

# ===== Configuration =====
DETECTOR_MODEL_PATH = 'models/detector_int8.tflite'
CLASSIFIER_MODEL_PATH = 'models/classifier_int8.tflite'

DETECTOR_IMG_SIZE = (320, 320)
CLASSIFIER_IMG_SIZE = (224, 224)

DETECTION_THRESHOLD = 0.5
CLASSIFIER_THRESHOLD = 0.7

CAM_SOURCE = 0  # Change to 1 if your camera doesn't work
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CLASSIFIER_CLASSES = {0: "COVERED", 1: "UNCOVERED"}
# ===== End Configuration =====

def load_tflite_model(model_path):
    """Load TFLite model on CPU"""
    import tensorflow as tf
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None, None, None
    
    try:
        # Load model (runs on CPU by default)
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        model_name = os.path.basename(model_path)
        print(f"✅ Loaded: {model_name}")
        print(f"   Input: {input_details[0]['shape']} ({input_details[0]['dtype'].__name__})")
        print(f"   Output: {output_details[0]['shape']} ({output_details[0]['dtype'].__name__})")
        
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return None, None, None

def preprocess_for_detector(image, target_size, input_dtype):
    """Preprocess image for YOLO detector"""
    h, w = image.shape[:2]
    th, tw = target_size
    
    # Letterbox resize (keeps aspect ratio)
    scale = min(tw / w, th / h)
    nw, nh = int(scale * w), int(scale * h)
    
    resized = cv2.resize(image, (nw, nh))
    padded = np.full((th, tw, 3), 114, dtype=np.uint8)
    
    dw, dh = (tw - nw) // 2, (th - nh) // 2
    padded[dh:dh+nh, dw:dw+nw, :] = resized
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    
    # For UINT8 quantized model, keep as uint8
    if input_dtype == np.uint8:
        return np.expand_dims(rgb, axis=0).astype(np.uint8)
    else:
        # For float32 models
        normalized = rgb.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

def preprocess_for_classifier(image, target_size, input_dtype):
    """Preprocess image for MobileNetV2 classifier"""
    # Resize
    resized = cv2.resize(image, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # For UINT8 quantized model, keep as uint8
    if input_dtype == np.uint8:
        return np.expand_dims(rgb, axis=0).astype(np.uint8)
    else:
        # MobileNetV2 preprocessing for float32
        normalized = (rgb.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(normalized, axis=0)

def dequantize_output(output, output_details):
    """Dequantize INT8/UINT8 output to float"""
    dtype = output_details[0]['dtype']
    
    if dtype in [np.int8, np.uint8]:
        scale, zero_point = output_details[0]['quantization']
        if scale != 0:  # Valid quantization params
            return scale * (output.astype(np.float32) - zero_point)
    
    return output

def detect_faces(interpreter, input_details, output_details, frame):
    """Run face detection"""
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    input_dtype = input_details[0]['dtype']
    
    # Preprocess
    input_tensor = preprocess_for_detector(frame, DETECTOR_IMG_SIZE, input_dtype)
    
    # Run inference
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    
    # Get output and dequantize if needed
    output = interpreter.get_tensor(output_index)
    output = dequantize_output(output, output_details)
    
    # Process detections
    faces = []
    h, w = frame.shape[:2]
    
    # YOLO output: [1, num_predictions, 5+num_classes]
    # For single class: [cx, cy, w, h, confidence]
    predictions = output[0]
    
    for pred in predictions:
        confidence = pred[4]
        
        if confidence >= DETECTION_THRESHOLD:
            cx, cy, bw, bh = pred[0:4]
            
            # Convert to pixel coordinates
            x_min = int((cx - bw / 2) * w)
            y_min = int((cy - bh / 2) * h)
            x_max = int((cx + bw / 2) * w)
            y_max = int((cy + bh / 2) * h)
            
            # Clamp to image boundaries
            x_min = max(0, min(x_min, w))
            y_min = max(0, min(y_min, h))
            x_max = max(0, min(x_max, w))
            y_max = max(0, min(y_max, h))
            
            if x_max > x_min and y_max > y_min:
                faces.append(((x_min, y_min, x_max, y_max), float(confidence)))
    
    return faces

def classify_face(interpreter, input_details, output_details, face_crop):
    """Classify if face is covered or uncovered"""
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    input_dtype = input_details[0]['dtype']
    
    # Preprocess
    input_tensor = preprocess_for_classifier(face_crop, CLASSIFIER_IMG_SIZE, input_dtype)
    
    # Run inference
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    
    # Get output and dequantize
    output = interpreter.get_tensor(output_index)
    output = dequantize_output(output, output_details)
    
    # Binary classification
    prob_uncovered = float(output[0][0])
    
    # Apply sigmoid if output is logit (not already probability)
    if prob_uncovered < 0 or prob_uncovered > 1:
        prob_uncovered = 1.0 / (1.0 + np.exp(-prob_uncovered))
    
    if prob_uncovered >= CLASSIFIER_THRESHOLD:
        return 1, prob_uncovered  # Uncovered
    else:
        return 0, 1.0 - prob_uncovered  # Covered

def main():
    print("=" * 70)
    print("Testing INT8 TFLite Models on CPU")
    print("(Same models for Raspberry Pi 5)")
    print("=" * 70)
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"\n✅ TensorFlow version: {tf.__version__}")
        print("✅ Running on CPU (no GPU needed)")
    except ImportError:
        print("\n❌ TensorFlow not installed!")
        print("\nInstall with:")
        print("  pip install tensorflow")
        print("  OR")
        print("  pip install tensorflow==2.10.0")
        return
    
    # Load models
    print("\n" + "=" * 70)
    print("Loading Models...")
    print("=" * 70 + "\n")
    
    detector, det_in, det_out = load_tflite_model(DETECTOR_MODEL_PATH)
    if detector is None:
        print("\n❌ Detector model not found!")
        print(f"Make sure '{DETECTOR_MODEL_PATH}' exists.")
        print("Copy it from WSL2 or train it first.")
        return
    
    print()
    classifier, cls_in, cls_out = load_tflite_model(CLASSIFIER_MODEL_PATH)
    if classifier is None:
        print("\n❌ Classifier model not found!")
        print(f"Make sure '{CLASSIFIER_MODEL_PATH}' exists.")
        return
    
    # Open camera
    print("\n" + "=" * 70)
    print("Opening Camera...")
    print("=" * 70)
    
    print(f"Trying camera index: {CAM_SOURCE}")
    cap = cv2.VideoCapture(CAM_SOURCE)
    time.sleep(2)  # Give camera time to initialize
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {CAM_SOURCE}")
        fallback = 1 if CAM_SOURCE == 0 else 0
        print(f"Trying camera index: {fallback}")
        cap = cv2.VideoCapture(fallback)
        time.sleep(2)
        
        if not cap.isOpened():
            print("❌ No camera available!")
            print("\nTroubleshooting:")
            print("- Check if another app is using the camera")
            print("- Try different CAM_SOURCE values (0, 1, 2)")
            return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera opened successfully!")
    print("\n" + "=" * 70)
    print("Starting Live Inference on CPU")
    print("Press 'q' to quit")
    print("=" * 70 + "\n")
    
    # Performance tracking
    frame_times = []
    detect_times = []
    classify_times = []
    
    frame_count = 0
    
    while True:
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        
        # Face detection
        detect_start = time.time()
        faces = detect_faces(detector, det_in, det_out, frame)
        detect_ms = (time.time() - detect_start) * 1000
        detect_times.append(detect_ms)
        
        # Classification
        warning = False
        total_classify_ms = 0
        
        for (x1, y1, x2, y2), det_conf in faces:
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Classify
            cls_start = time.time()
            class_id, prob = classify_face(classifier, cls_in, cls_out, crop)
            cls_ms = (time.time() - cls_start) * 1000
            total_classify_ms += cls_ms
            
            label = CLASSIFIER_CLASSES[class_id]
            color = (0, 0, 255) if class_id == 0 else (0, 255, 0)
            
            if class_id == 0:
                warning = True
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {prob:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if total_classify_ms > 0:
            classify_times.append(total_classify_ms)
        
        # Warning message
        if warning:
            cv2.putText(frame, "WARNING: FACE COVERED!", (10, FRAME_HEIGHT-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Calculate FPS
        loop_time = time.time() - loop_start
        frame_times.append(loop_time)
        
        # Keep last 30 frames for averaging
        if len(frame_times) > 30:
            frame_times.pop(0)
        if len(detect_times) > 30:
            detect_times.pop(0)
        if len(classify_times) > 30:
            classify_times.pop(0)
        
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
        avg_detect = sum(detect_times) / len(detect_times)
        avg_classify = sum(classify_times) / len(classify_times) if classify_times else 0
        
        # Display stats
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Detection: {avg_detect:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if avg_classify > 0:
            cv2.putText(frame, f"Classification: {avg_classify:.1f}ms", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Model info
        cv2.putText(frame, "CPU Mode | INT8 TFLite", (10, FRAME_HEIGHT-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show frame
        cv2.imshow('TFLite Models Test (CPU)', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
    print("\n" + "=" * 70)
    print("Session Summary")
    print("=" * 70)
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average Detection Time: {avg_detect:.1f}ms")
    if avg_classify > 0:
        print(f"Average Classification Time: {avg_classify:.1f}ms")
    print(f"Total inference time per frame: {avg_detect + avg_classify:.1f}ms")
    print("\n✅ Models tested successfully on CPU!")
    print("Performance on Raspberry Pi 5 will be similar (5-15 FPS expected)")
    print("=" * 70)

if __name__ == '__main__':
    main()
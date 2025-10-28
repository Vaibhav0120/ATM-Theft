"""
FIXED ATM Security System - INT8 TFLite Inference
Handles both FLOAT32 and UINT8 models automatically

CRITICAL FIXES:
- Fixed representative dataset in training (train_classifier.py)
- Improved dequantization logic
- Better probability interpretation
- Enhanced debugging output
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

DETECTION_THRESHOLD = 0.4  # Lowered slightly for better face detection
CLASSIFIER_THRESHOLD = 0.5  # Binary threshold for covered/uncovered

CAM_SOURCE = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# CRITICAL: This must match your training folder structure (alphabetically sorted)
# covered/ comes before uncovered/ alphabetically
# So: Class 0 = COVERED, Class 1 = UNCOVERED
CLASSIFIER_CLASSES = {0: "COVERED", 1: "UNCOVERED"}
# ===== End Configuration =====

def load_tflite_model(model_path):
    """Load TFLite model with detailed information"""
    import tensorflow as tf
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None, None, None
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        model_name = os.path.basename(model_path)
        print(f"✅ Loaded: {model_name}")
        
        # Print input/output details
        input_dtype = input_details[0]['dtype']
        output_dtype = output_details[0]['dtype']
        print(f"   Input:  {input_details[0]['shape']} - {input_dtype.__name__}")
        print(f"   Output: {output_details[0]['shape']} - {output_dtype.__name__}")
        
        # Print quantization parameters
        quant_params = output_details[0].get('quantization_parameters', {})
        if 'scales' in quant_params and len(quant_params['scales']) > 0:
            scale = quant_params['scales'][0]
            zero_point = quant_params['zero_points'][0] if len(quant_params['zero_points']) > 0 else 0
            print(f"   Output quantization: scale={scale:.6f}, zero_point={zero_point}")
        
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return None, None, None

def preprocess_image(image, target_size, input_dtype, letterbox=True):
    """
    Preprocess image for TFLite inference
    Automatically handles FLOAT32 or UINT8 input types
    """
    h, w = image.shape[:2]
    th, tw = target_size
    
    if letterbox:
        # Letterbox resize (maintains aspect ratio with padding)
        scale = min(tw / w, th / h)
        nw, nh = int(scale * w), int(scale * h)
        
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (gray padding)
        padded = np.full((th, tw, 3), 114, dtype=np.uint8)
        dw, dh = (tw - nw) // 2, (th - nh) // 2
        padded[dh:dh+nh, dw:dw+nw, :] = resized
        
        processed = padded
    else:
        # Simple resize (for classifier)
        processed = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Handle different input types
    if input_dtype == np.float32:
        # For FLOAT32 models: normalize to 0-1 range
        normalized = rgb.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)
    else:
        # For UINT8 models: keep as uint8 in 0-255 range
        return np.expand_dims(rgb, axis=0).astype(np.uint8)

def dequantize_value(quantized_value, scale, zero_point):
    """Dequantize INT8/UINT8 value to float"""
    if scale == 0:
        # No quantization or invalid params
        return float(quantized_value) / 255.0
    return scale * (float(quantized_value) - zero_point)

def dequantize_tensor(tensor, output_details):
    """Dequantize tensor if needed"""
    quant_params = output_details[0].get('quantization_parameters', {})
    
    if 'scales' in quant_params and len(quant_params['scales']) > 0:
        scale = quant_params['scales'][0]
        zero_point = quant_params['zero_points'][0] if len(quant_params['zero_points']) > 0 else 0
        if scale != 0:
            return scale * (tensor.astype(np.float32) - zero_point)
    
    return tensor.astype(np.float32)

def detect_faces(interpreter, input_details, output_details, frame):
    """Run face detection with automatic type handling"""
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    input_dtype = input_details[0]['dtype']
    
    # Preprocess with letterbox resize
    input_tensor = preprocess_image(frame, DETECTOR_IMG_SIZE, input_dtype, letterbox=True)
    
    # Run inference
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    
    # Get output and dequantize if needed
    output = interpreter.get_tensor(output_index)
    output = dequantize_tensor(output, output_details)
    
    # Process detections
    faces = []
    h, w = frame.shape[:2]
    
    # YOLO output format: [1, num_predictions, 5+num_classes]
    # For single class: [cx, cy, w, h, confidence]
    predictions = output[0].T
    
    for pred in predictions:
        confidence = pred[4]
        
        if confidence >= DETECTION_THRESHOLD:
            cx, cy, bw, bh = pred[0:4]
            
            # Convert normalized coords to pixels
            x_min = int((cx - bw / 2) * w)
            y_min = int((cy - bh / 2) * h)
            x_max = int((cx + bw / 2) * w)
            y_max = int((cy + bh / 2) * h)
            
            # Clamp to image boundaries
            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(0, min(x_max, w))
            y_max = max(0, min(y_max, h))
            
            # Validate box dimensions
            if x_max > x_min and y_max > y_min:
                # Add padding to bounding box (helps classifier)
                pad_w = int((x_max - x_min) * 0.1)
                pad_h = int((y_max - y_min) * 0.1)
                
                x_min = max(0, x_min - pad_w)
                y_min = max(0, y_min - pad_h)
                x_max = min(w, x_max + pad_w)
                y_max = min(h, y_max + pad_h)
                
                faces.append(((x_min, y_min, x_max, y_max), float(confidence)))
    
    return faces

def classify_face(interpreter, input_details, output_details, face_crop, debug=False):
    """
    Classify if face is covered or uncovered
    
    FIXED VERSION:
    - Proper dequantization of UINT8 output
    - Correct probability interpretation
    - Sigmoid output: low values → class 0 (covered), high values → class 1 (uncovered)
    """
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    input_dtype = input_details[0]['dtype']
    output_dtype = output_details[0]['dtype']
    
    # Preprocess (no letterbox for classifier)
    input_tensor = preprocess_image(face_crop, CLASSIFIER_IMG_SIZE, input_dtype, letterbox=False)
    
    # Run inference
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    
    # Get raw output
    output_raw = interpreter.get_tensor(output_index)
    
    # Handle different output types
    if output_dtype == np.uint8 or output_dtype == np.int8:
        # Quantized output - need to dequantize
        pred_score_quantized = int(output_raw[0][0])
        
        # Get quantization parameters
        quant_params = output_details[0].get('quantization_parameters', {})
        if 'scales' in quant_params and len(quant_params['scales']) > 0:
            scale = quant_params['scales'][0]
            zero_point = quant_params['zero_points'][0] if len(quant_params['zero_points']) > 0 else 0
            
            # CRITICAL FIX: Proper dequantization
            prob_class_1 = dequantize_value(pred_score_quantized, scale, zero_point)
            
            if debug:
                print(f"  [DEBUG] Raw UINT8: {pred_score_quantized}, Scale: {scale:.6f}, ZP: {zero_point}")
                print(f"  [DEBUG] Dequantized probability: {prob_class_1:.4f}")
        else:
            # Fallback: simple normalization
            prob_class_1 = pred_score_quantized / 255.0
            if debug:
                print(f"  [DEBUG] No quant params, using simple norm: {prob_class_1:.4f}")
    else:
        # FLOAT32 output - already dequantized
        prob_class_1 = float(output_raw[0][0])
        if debug:
            print(f"  [DEBUG] Float32 output: {prob_class_1:.4f}")
    
    # Clamp probability to valid range [0, 1]
    prob_class_1 = np.clip(prob_class_1, 0.0, 1.0)
    prob_class_0 = 1.0 - prob_class_1
    
    # Determine class based on threshold
    # Higher probability of class 1 (uncovered) → classify as uncovered
    if prob_class_1 >= CLASSIFIER_THRESHOLD:
        return 1, prob_class_1  # UNCOVERED
    else:
        return 0, prob_class_0  # COVERED

def main():
    print("=" * 70)
    print("ATM Security System - Face Coverage Detection")
    print("FIXED VERSION - Proper Quantization & Dequantization")
    print("=" * 70)
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"\n✅ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("\n❌ TensorFlow not installed!")
        print("Install with: pip install tensorflow")
        return
    
    # Load models
    print("\n" + "=" * 70)
    print("Loading Models...")
    print("=" * 70 + "\n")
    
    detector, det_in, det_out = load_tflite_model(DETECTOR_MODEL_PATH)
    if detector is None:
        return
    
    print()
    classifier, cls_in, cls_out = load_tflite_model(CLASSIFIER_MODEL_PATH)
    if classifier is None:
        return
    
    # Open camera
    print("\n" + "=" * 70)
    print("Opening Camera...")
    print("=" * 70)
    
    cap = cv2.VideoCapture(CAM_SOURCE)
    time.sleep(2)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {CAM_SOURCE}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera opened successfully!")
    print("\n" + "=" * 70)
    print("Starting Live Detection")
    print("Press 'q' to quit, 'd' for debug mode")
    print("=" * 70 + "\n")
    
    # Performance tracking
    covered_count = 0
    uncovered_count = 0
    total_detections = 0
    debug_mode = False
    
    fps_start = time.time()
    fps_count = 0
    display_fps = 0
    
    while True:
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue
        
        fps_count += 1
        
        # Detect faces
        faces = detect_faces(detector, det_in, det_out, frame)
        
        # Classify each face
        warning = False
        frame_covered = 0
        frame_uncovered = 0
        
        for (x1, y1, x2, y2), det_conf in faces:
            # Validate crop dimensions
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Classify with debug info
            if debug_mode:
                print(f"\n[Face Detection] Confidence: {det_conf:.3f}")
            
            class_id, prob = classify_face(classifier, cls_in, cls_out, crop, debug=debug_mode)
            label = CLASSIFIER_CLASSES[class_id]
            
            if debug_mode:
                print(f"[Classification] Label: {label}, Probability: {prob:.3f}")
            
            # Update statistics
            total_detections += 1
            if class_id == 0:
                covered_count += 1
                frame_covered += 1
                color = (0, 0, 255)  # Red for covered
                warning = True
            else:
                uncovered_count += 1
                frame_uncovered += 1
                color = (0, 255, 0)  # Green for uncovered
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with probability
            label_text = f"{label} {prob:.2f}"
            
            # Add background to text for readability
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show detection confidence
            det_text = f"Det: {det_conf:.2f}"
            cv2.putText(frame, det_text, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Warning banner
        if warning:
            h, w = frame.shape[:2]
            # Draw semi-transparent red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Warning text
            cv2.putText(frame, "WARNING: FACE COVERED DETECTED!", 
                       (w // 2 - 250, h - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if not debug_mode:  # Don't spam in debug mode
                print(f"⚠️  ALERT: Face covering detected! (Frame covered: {frame_covered})")
        
        # Calculate FPS
        if time.time() - fps_start >= 1.0:
            display_fps = fps_count / (time.time() - fps_start)
            fps_count = 0
            fps_start = time.time()
        
        # Display info panel
        info_y = 30
        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Faces: {len(faces)}", (10, info_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if total_detections > 0:
            cv2.putText(frame, f"Covered: {covered_count} ({100*covered_count/total_detections:.1f}%)", 
                       (10, info_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Uncovered: {uncovered_count} ({100*uncovered_count/total_detections:.1f}%)", 
                       (10, info_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Model info
        det_type = "FP32" if det_in[0]['dtype'] == np.float32 else "INT8"
        cls_type = "FP32" if cls_in[0]['dtype'] == np.float32 else "INT8"
        cv2.putText(frame, f"Detector: {det_type} | Classifier: {cls_type}", 
                   (10, FRAME_HEIGHT - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Debug mode indicator
        if debug_mode:
            cv2.putText(frame, "[DEBUG MODE]", (FRAME_WIDTH - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('ATM Security - Face Coverage Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"\n{'='*50}")
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            print(f"{'='*50}\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 70)
    print("Session Summary")
    print("=" * 70)
    print(f"Total detections: {total_detections}")
    if total_detections > 0:
        print(f"Covered faces: {covered_count} ({100*covered_count/total_detections:.1f}%)")
        print(f"Uncovered faces: {uncovered_count} ({100*uncovered_count/total_detections:.1f}%)")
    print(f"Average FPS: {display_fps:.1f}")
    print("=" * 70)

if __name__ == '__main__':
    main()
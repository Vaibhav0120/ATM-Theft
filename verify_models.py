#!/usr/bin/env python3
"""
Model Verification Script
Checks if your trained models are correctly quantized and configured
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print its size"""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ {description}: {filepath} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def check_tflite_model(model_path, model_name):
    """Verify TFLite model quantization parameters"""
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ TensorFlow not installed. Cannot verify model details.")
        return False
    
    if not os.path.exists(model_path):
        print(f"❌ {model_name} not found at {model_path}")
        return False
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n{'='*70}")
        print(f"{model_name} Details")
        print(f"{'='*70}")
        
        # Input details
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        print(f"Input Shape: {input_shape}")
        print(f"Input Type: {input_dtype.__name__}")
        
        # Output details
        output_shape = output_details[0]['shape']
        output_dtype = output_details[0]['dtype']
        print(f"Output Shape: {output_shape}")
        print(f"Output Type: {output_dtype.__name__}")
        
        # Quantization parameters
        quant_params = output_details[0].get('quantization_parameters', {})
        if 'scales' in quant_params and len(quant_params['scales']) > 0:
            scale = quant_params['scales'][0]
            zero_point = quant_params['zero_points'][0] if len(quant_params['zero_points']) > 0 else 0
            print(f"Output Quantization:")
            print(f"  - Scale: {scale:.6f}")
            print(f"  - Zero Point: {zero_point}")
            
            # Check if quantization is reasonable
            if scale > 0 and scale < 1:
                print(f"  ✅ Quantization parameters look good")
            else:
                print(f"  ⚠️  WARNING: Unusual quantization parameters")
        else:
            print("⚠️  No quantization parameters found (might be float32 model)")
        
        # Verify INT8 quantization
        import numpy as np
        if input_dtype == np.uint8 and output_dtype == np.uint8:
            print(f"\n✅ Model is properly quantized to INT8 (UINT8)")
        elif input_dtype == np.float32 or output_dtype == np.float32:
            print(f"\n⚠️  Model uses FLOAT32 (not quantized or partially quantized)")
        
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return False

def check_classifier_data():
    """Verify classifier dataset structure"""
    data_dir = os.path.join('dataset_configs', 'classifier_data')
    
    print(f"\n{'='*70}")
    print("Classifier Dataset Verification")
    print(f"{'='*70}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Classifier data directory not found: {data_dir}")
        print("   Run 'python setup.py' to prepare data")
        return False
    
    all_good = True
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"❌ Missing split: {split}")
            all_good = False
            continue
        
        for class_name in ['covered', 'uncovered']:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"❌ Missing class folder: {split}/{class_name}")
                all_good = False
                continue
            
            # Count images
            image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            count = len(image_files)
            
            if count > 0:
                print(f"✅ {split}/{class_name}: {count} images")
            else:
                print(f"⚠️  {split}/{class_name}: No images found")
                all_good = False
    
    print(f"{'='*70}\n")
    return all_good

def check_detector_dataset():
    """Verify detector dataset"""
    dataset_root = 'ATM-Theft-Detection-4'
    yaml_path = os.path.join('dataset_configs', 'detector.yaml')
    
    print(f"\n{'='*70}")
    print("Detector Dataset Verification")
    print(f"{'='*70}")
    
    if not os.path.exists(dataset_root):
        print(f"❌ Detector dataset not found: {dataset_root}")
        print("   Run 'python setup.py' to download dataset")
        return False
    
    if not os.path.exists(yaml_path):
        print(f"❌ Detector config not found: {yaml_path}")
        return False
    
    print(f"✅ Dataset root: {dataset_root}")
    print(f"✅ Config file: {yaml_path}")
    
    # Check splits
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(dataset_root, split, 'images')
        labels_dir = os.path.join(dataset_root, split, 'labels')
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            num_images = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            num_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
            print(f"✅ {split}: {num_images} images, {num_labels} labels")
        else:
            print(f"❌ {split}: Missing images or labels directory")
    
    print(f"{'='*70}\n")
    return True

def test_inference_imports():
    """Test if all required packages are installed"""
    print(f"\n{'='*70}")
    print("Package Verification")
    print(f"{'='*70}")
    
    packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'tqdm': 'TQDM',
        'ultralytics': 'Ultralytics',
        'roboflow': 'Roboflow'
    }
    
    all_installed = True
    for package, name in packages.items():
        try:
            if package == 'cv2':
                import cv2
                version = cv2.__version__
            elif package == 'PIL':
                from PIL import Image
                version = "OK"
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'OK')
            
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name} not installed")
            all_installed = False
    
    print(f"{'='*70}\n")
    return all_installed

def main():
    print("\n" + "="*70)
    print("ATM SECURITY SYSTEM - MODEL VERIFICATION")
    print("="*70 + "\n")
    
    # Check packages
    print("Step 1: Checking installed packages...")
    packages_ok = test_inference_imports()
    
    # Check datasets
    print("\nStep 2: Checking datasets...")
    detector_dataset_ok = check_detector_dataset()
    classifier_data_ok = check_classifier_data()
    
    # Check models
    print("\nStep 3: Checking trained models...")
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        print("   Models have not been trained yet.")
        detector_ok = False
        classifier_ok = False
    else:
        # Check detector
        detector_path = os.path.join(models_dir, 'detector_int8.tflite')
        detector_ok = check_tflite_model(detector_path, "Face Detector (YOLO)")
        
        # Check classifier
        classifier_path = os.path.join(models_dir, 'classifier_int8.tflite')
        classifier_ok = check_tflite_model(classifier_path, "Mask Classifier (MobileNetV2)")
        
        # Check Keras model
        keras_path = os.path.join(models_dir, 'classifier.h5')
        check_file_exists(keras_path, "Keras Classifier (for reference)")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    status = []
    status.append(("Packages Installed", packages_ok))
    status.append(("Detector Dataset", detector_dataset_ok))
    status.append(("Classifier Data", classifier_data_ok))
    status.append(("Detector Model", detector_ok))
    status.append(("Classifier Model", classifier_ok))
    
    for item, ok in status:
        symbol = "✅" if ok else "❌"
        print(f"{symbol} {item}")
    
    all_ok = all(ok for _, ok in status)
    
    if all_ok:
        print("\n" + "="*70)
        print("🎉 ALL CHECKS PASSED!")
        print("You can now run: python live_inference.py")
        print("="*70 + "\n")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  SOME CHECKS FAILED")
        print("\nNext steps:")
        if not detector_dataset_ok or not classifier_data_ok:
            print("  1. Run: python setup.py")
        if not detector_ok:
            print("  2. Train detector: cd training && python train_detector.py")
        if not classifier_ok:
            print("  3. Train classifier: cd training && python train_classifier.py")
        print("\nThen run this script again to verify.")
        print("="*70 + "\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())

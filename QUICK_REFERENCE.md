# ATM Security System - Quick Reference

## 🚨 Emergency Commands

### I need to start from scratch
```bash
cd /app
rm -rf models/ runs/ dataset_configs/classifier_data/ ATM-Theft-Detection-4/
mkdir -p models
python setup.py
```

### My classifier is broken (always predicts UNCOVERED)
```bash
cd /app
rm models/classifier_int8.tflite models/classifier.h5
cd training
python train_classifier.py
cd ..
python live_inference.py
```

### Check if everything is working
```bash
python verify_models.py
```

---

## 📋 Complete Workflow (From Zero to Working)

```bash
# Step 1: Clean slate
cd /app
rm -rf models/ runs/ dataset_configs/classifier_data/ ATM-Theft-Detection-4/
mkdir -p models

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Prepare data
python setup.py
# Wait ~5-10 minutes for download and processing

# Step 4: Train detector
cd training
python -c "
from ultralytics import YOLO
import os, shutil

model = YOLO('yolov8n.pt')
results = model.train(
    data='../dataset_configs/detector.yaml',
    epochs=50,
    imgsz=320,
    batch=16,
    name='detector_train_run',
    device=0
)

tflite_path = model.export(
    format='tflite',
    int8=True,
    data='../dataset_configs/detector.yaml',
    imgsz=320,
    simplify=True
)

if os.path.exists(tflite_path):
    shutil.move(tflite_path, '../models/detector_int8.tflite')
    print('✅ Done')
"
# Wait ~1-2 hours

# Step 5: Train classifier (FIXED VERSION)
python train_classifier.py
# Wait ~10-15 minutes

cd ..

# Step 6: Verify everything
python verify_models.py

# Step 7: Test live
python live_inference.py
# Press 'q' to quit, 'd' for debug mode
```

---

## 🔍 Debug Checklist

### Classifier always predicts UNCOVERED?

- [ ] Using FIXED version of `train_classifier.py`?
  ```bash
  grep "FIXED VERSION" training/train_classifier.py
  ```

- [ ] Verified quantization parameters?
  ```bash
  python verify_models.py
  # Check: scale ~0.0039, zero_point = 0
  ```

- [ ] Checked raw probabilities in debug mode?
  ```bash
  # In live_inference.py, press 'd'
  # Raw UINT8 should vary (not always >150)
  ```

- [ ] Verified class mapping?
  ```bash
  # Training output should show:
  # Found classes: ['covered', 'uncovered']
  # Class mapping: covered = 0, uncovered = 1
  ```

### No faces detected?

- [ ] Lower detection threshold?
  ```python
  # In live_inference.py line 25:
  DETECTION_THRESHOLD = 0.3  # Try 0.3 instead of 0.4
  ```

- [ ] Camera working?
  ```bash
  python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
  # Should print: True
  ```

- [ ] Good lighting?
  - Face should be well-lit
  - Avoid backlighting

### Training fails?

- [ ] GPU available?
  ```bash
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  ```

- [ ] Enough disk space?
  ```bash
  df -h .
  # Need at least 10GB free
  ```

- [ ] All dependencies installed?
  ```bash
  python verify_models.py
  ```

---

## 🎯 What Each File Does

### Setup & Data Preparation
| File | Purpose | When to Run |
|------|---------|-------------|
| `setup.py` | Downloads dataset, prepares data | Once at start |
| `data_preparation/remap_detector_labels.py` | Converts 21 classes → 1 | Called by setup.py |
| `data_preparation/prepare_classifier_data.py` | Crops faces → covered/uncovered | Called by setup.py |

### Training
| File | Purpose | Output |
|------|---------|--------|
| `training/train_detector.py` | Exports YOLO to TFLite | `models/detector_int8.tflite` |
| `training/train_classifier.py` | Trains MobileNetV2 (FIXED) | `models/classifier_int8.tflite` |

### Inference & Deployment
| File | Purpose | Platform |
|------|---------|----------|
| `live_inference.py` | Testing on laptop (FIXED) | Laptop/WSL |
| `deployment/run_atm_security.py` | Production script | Raspberry Pi 5 |

### Utilities
| File | Purpose |
|------|----------|
| `verify_models.py` | Check models & datasets |
| `COMPLETE_SOLUTION.md` | Full bug analysis & guide |
| `QUICK_REFERENCE.md` | This file |

---

## 📊 Expected File Sizes

```
models/
├── detector_int8.tflite       (~6 MB)   ← INT8 quantized YOLO
├── classifier_int8.tflite     (~3 MB)   ← INT8 quantized MobileNetV2
└── classifier.h5              (~10 MB)  ← Full Keras model (reference)

dataset_configs/
└── classifier_data/
    ├── train/
    │   ├── covered/           (1000-5000 images)
    │   └── uncovered/         (1000-5000 images)
    ├── valid/
    └── test/

ATM-Theft-Detection-4/
├── train/
│   ├── images/                (1000-5000 images)
│   └── labels/                (1000-5000 .txt files)
├── valid/
└── test/
```

---

## 🎮 Live Inference Controls

```bash
python live_inference.py
```

**Keyboard Controls:**
- `q` - Quit
- `d` - Toggle debug mode (shows raw probabilities)

**Debug Mode Output Example:**
```
[Face Detection] Confidence: 0.872
  [DEBUG] Raw UINT8: 45, Scale: 0.003922, ZP: 0
  [DEBUG] Dequantized probability: 0.1765
[Classification] Label: COVERED, Probability: 0.824
```

**Interpretation:**
- Raw UINT8: 45 (out of 255) from model
- Dequantized: 45 * 0.003922 = 0.1765 (probability of class 1)
- Since 0.1765 < 0.5, classified as class 0 (COVERED)
- Display probability: 1 - 0.1765 = 0.824 (for COVERED)

---

## 🔬 Testing Scenarios

### Test 1: Normal Face (No Mask)
**Expected:**
- ✅ Green bounding box
- ✅ Label: "UNCOVERED"
- ✅ Probability: >0.5
- ✅ No warning banner

### Test 2: Masked Face
**Expected:**
- ✅ Red bounding box
- ✅ Label: "COVERED"
- ✅ Probability: >0.5
- ✅ Red warning banner at bottom
- ✅ Console message: "ALERT: Face covering detected!"

### Test 3: Partial Coverage (Hand, Scarf)
**Expected:**
- ✅ Red or green box (depends on coverage amount)
- ✅ Probability between 0.4-0.6 (uncertain cases)

### Test 4: Multiple Faces
**Expected:**
- ✅ Each face gets its own box and classification
- ✅ Warning triggers if ANY face is covered

---

## 🚀 Deploy to Raspberry Pi

### Quick Deploy
```bash
# On laptop
cd /app
mkdir deploy_package
cp models/*.tflite deploy_package/
cp live_inference.py deploy_package/
cp -r deployment/ deploy_package/

# Transfer
scp -r deploy_package/ pi@192.168.1.XXX:~/atm_security/

# On Pi
ssh pi@192.168.1.XXX
cd ~/atm_security/deploy_package/
pip install tflite-runtime opencv-python-headless numpy
python deployment/run_atm_security.py
```

---

## ⚡ Performance Tips

### Laptop Testing
- **Use GPU:** Ensure CUDA is available
- **Close other apps:** Free up GPU memory
- **Good lighting:** Helps detection accuracy

### Raspberry Pi Deployment
- **Use lite OS:** Raspberry Pi OS Lite (no desktop)
- **Disable desktop:** `sudo systemctl set-default multi-user.target`
- **Overclock (optional):** Increase CPU/GPU speeds
- **Use USB 3.0 camera:** Faster than USB 2.0
- **Lower resolution:** 640x480 is enough for detection

---

## 📈 Model Training Tips

### Detector Training
- **Epochs:** 50 is good, 100 is better (diminishing returns)
- **Batch size:** 16 for RTX 4050, adjust based on GPU memory
- **Image size:** 320 is fast, 640 is more accurate (slower)
- **Device:** `device=0` for GPU 0, `device='cpu'` for CPU

### Classifier Training
- **Epochs:** 10 is usually enough (transfer learning)
- **Batch size:** 32 is good, lower if out of memory
- **Data balance:** Ensure ~equal covered/uncovered images
- **Image size:** 224 is standard for MobileNetV2

---

## 🆘 Common Error Messages

### "Model file not found"
**Cause:** Models not trained yet
**Fix:**
```bash
cd training
python train_classifier.py  # or train detector script
cd ..
```

### "Cannot open camera"
**Cause:** Camera not connected or in use
**Fix:**
```bash
# Check camera
ls /dev/video*
# Try different camera index
# In live_inference.py: CAM_SOURCE = 1  (instead of 0)
```

### "Out of memory" during training
**Cause:** GPU memory full
**Fix:**
```bash
# Reduce batch size in training script
# Detector: batch=8 (instead of 16)
# Classifier: BATCH_SIZE = 16 (instead of 32)
```

### "No images found" in dataset
**Cause:** setup.py didn't complete
**Fix:**
```bash
rm -rf ATM-Theft-Detection-4/ dataset_configs/classifier_data/
python setup.py
```

---

## 🎓 Understanding the Output

### Classifier Output Interpretation

**Binary Sigmoid Classifier:**
- Output range: 0.0 to 1.0
- Class 0 (COVERED): output near 0.0
- Class 1 (UNCOVERED): output near 1.0
- Threshold: 0.5 (adjustable)

**Example Outputs:**
| Raw Output | Interpretation | Label |
|-----------|----------------|--------|
| 0.05 | Very confident COVERED | COVERED (0.95) |
| 0.25 | Likely COVERED | COVERED (0.75) |
| 0.45 | Slightly COVERED | COVERED (0.55) |
| 0.55 | Slightly UNCOVERED | UNCOVERED (0.55) |
| 0.75 | Likely UNCOVERED | UNCOVERED (0.75) |
| 0.95 | Very confident UNCOVERED | UNCOVERED (0.95) |

---

## 🔧 Configuration Quick Reference

### In `live_inference.py`
```python
# Detection sensitivity
DETECTION_THRESHOLD = 0.4  # Lower = more faces detected (0.3-0.6)

# Classification threshold
CLASSIFIER_THRESHOLD = 0.5  # Binary threshold (0.4-0.6)

# Camera settings
CAM_SOURCE = 0  # Camera index (0, 1, 2, ...)
FRAME_WIDTH = 640  # Resolution
FRAME_HEIGHT = 480
```

### In `train_classifier.py`
```python
IMG_SIZE = (224, 224)  # MobileNetV2 input size (don't change)
BATCH_SIZE = 32  # Lower if out of memory
EPOCHS = 10  # More epochs = better accuracy (diminishing returns)
```

---

**Last Updated:** 2025  
**Status:** ✅ All bugs fixed, ready for deployment
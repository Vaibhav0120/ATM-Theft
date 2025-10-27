```bash
atm_security_project/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── setup.py                     # <-- Use this to download & prepare all data
│
├── data_preparation/
│   ├── remap_detector_labels.py   # Script 1: Merges 21 classes -> 1 class ('face')
│   └── prepare_classifier_data.py # Script 2: Crops faces -> 'covered'/'uncovered' folders
│
├── dataset_configs/
│   ├── detector.yaml              # Config file for training the detector (1 class)
│   └── classifier_data/           # This folder will be CREATED by setup.py
│       ├── train/
│       │   ├── covered/
│       │   └── uncovered/
│       ├── valid/
│       │   ├── covered/
│       │   └── uncovered/
│       └── test/
│           ├── covered/
│           └── uncovered/
│
├── training/
│   ├── train_detector.py          # Trains the YOLOv11/v8n Face Detector
│   └── train_classifier.py        # Trains the MobileNetV2 Mask Classifier
│
├── deployment/
│   └── run_atm_security.py        # Final RPi5 script (loads both .tflite models)
│
└── models/
    ├── detector.pt                # Trained detector (PyTorch)
    ├── detector_int8.tflite       # Quantized detector (for Pi)
    ├── classifier.h5              # Trained classifier (Keras)
    └── classifier_int8.tflite     # Quantized classifier (for Pi)
```
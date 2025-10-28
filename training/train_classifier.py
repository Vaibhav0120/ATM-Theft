import tensorflow as tf
import os
import numpy as np

# --- Config ---
DATA_DIR = os.path.join('..', 'dataset_configs', 'classifier_data')
MODEL_SAVE_PATH = os.path.join('..', 'models')
FINAL_MODEL_NAME = 'classifier'
IMG_SIZE = (224, 224) # MobileNetV2 preferred input size
BATCH_SIZE = 32
EPOCHS = 10
# --- End Config ---

# Create save directory
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def build_model(num_classes):
    """Builds a MobileNetV2 classifier model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False, # Don't include the final 1000-class layer
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add our custom classifier head
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    # We must preprocess the images the same way MobileNetV2 expects
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) 
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # Use 1 output neuron with sigmoid for binary classification
    if num_classes == 2:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else: # Fallback for multi-class
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'

    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    return model

def train():
    # 1. Load Datasets
    print(f"Loading classifier data from '{DATA_DIR}'...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary' # 'binary' for 2 classes
    )
    
    valid_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'valid'),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    
    class_names = train_dataset.class_names
    print(f"Found classes: {class_names}")
    print(f"Class mapping: {class_names[0]} = 0, {class_names[1]} = 1")
    
    if len(class_names) != 2:
        print(f"Error: Expected 2 classes, but found {len(class_names)}")
        return

    # 2. Build and Train the model
    model = build_model(num_classes=len(class_names))
    model.summary()
    
    print(f"Starting training for {EPOCHS} epochs...")
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS
    )
    
    # Save the Keras model
    keras_save_path = os.path.join(MODEL_SAVE_PATH, f"{FINAL_MODEL_NAME}.h5")
    model.save(keras_save_path)
    print(f"Keras model saved to {keras_save_path}")

    # 3. Convert and Quantize to TFLite INT8
    print("\n" + "="*70)
    print("Exporting to TFLite INT8 (FIXED VERSION)...")
    print("="*70)
    
    # CRITICAL FIX: Create a proper representative dataset
    # Load raw images without preprocessing
    quant_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),
        image_size=IMG_SIZE,
        batch_size=1, # Must be batch_size 1 for calibration
        label_mode='binary'
    )
    
    # FIXED: Proper representative dataset generator
    # The model expects uint8 input (0-255), and has preprocessing inside
    def representative_gen():
        """Generate calibration data for INT8 quantization."""
        print("Generating calibration data for quantization...")
        for i, (images, _) in enumerate(quant_dataset.take(150)):
            if i % 30 == 0:
                print(f"  Calibration step {i}/150")
            # CRITICAL FIX: Keep as float32 in 0-255 range for the preprocessing layer
            # The model's internal preprocessing will scale it correctly
            # TFLite converter will handle the quantization at the input
            calibration_data = tf.cast(images, tf.float32)
            yield [calibration_data]
        print("Calibration data generation complete.")

    # Convert the Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    
    # FIXED: Proper INT8 quantization settings
    # Force full integer quantization for all ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # Input is raw image (0-255)
    converter.inference_output_type = tf.uint8 # Output is quantized
    
    print("\nStarting TFLite conversion...")
    tflite_quant_model = converter.convert()
    print("TFLite conversion complete.")

    # 4. Save the final TFLite model
    final_tflite_path = os.path.join(MODEL_SAVE_PATH, f"{FINAL_MODEL_NAME}_int8.tflite")
    with open(final_tflite_path, 'wb') as f:
        f.write(tflite_quant_model)
        
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS: Quantized TFLite model saved to: {final_tflite_path}")
    print(f"{'='*70}")
    print(f"\nClass Index Mapping (alphabetical order):")
    print(f"  {class_names[0]} = 0 (sigmoid output ~0.0)")
    print(f"  {class_names[1]} = 1 (sigmoid output ~1.0)")
    print(f"\nModel ready for deployment on Raspberry Pi 5!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"Error: Classifier data not found at '{DATA_DIR}'")
        print("Please run 'python setup.py' first.")
    else:
        train()
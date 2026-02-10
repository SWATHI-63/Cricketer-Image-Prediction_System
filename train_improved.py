"""
CrickAI - Improved Model Training
=================================
Better training with more augmentation for small datasets.
"""

import os
import json
import numpy as np
from pathlib import Path

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data" / "images"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cricketer_model.keras"
LABELS_PATH = MODEL_DIR / "label_mapping.json"

# Training parameters - optimized for small datasets
IMG_SIZE = 224
BATCH_SIZE = 8  # Smaller batch size for small dataset
EPOCHS_PHASE1 = 30  # More epochs
EPOCHS_PHASE2 = 30

# Check for GPU
print("💻 Running on", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")


def create_data_generators():
    """Create data generators with HEAVY augmentation for small dataset."""
    
    # HEAVY augmentation to artificially increase dataset size
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,  # More rotation
        width_shift_range=0.3,  # More shift
        height_shift_range=0.3,
        shear_range=0.3,  # More shear
        zoom_range=0.3,  # More zoom
        horizontal_flip=True,
        brightness_range=[0.6, 1.4],  # More brightness variation
        fill_mode='nearest',
        validation_split=0.15  # Less validation, more training
    )
    
    train_generator = train_datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def create_model(num_classes):
    """Create model using EfficientNetB0 - better for small datasets."""
    
    # Use EfficientNetB0 - more efficient and better accuracy
    base_model = keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  # More dropout for small dataset
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model


def save_labels(train_generator):
    """Save the class labels mapping."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    class_indices = train_generator.class_indices
    classes = list(class_indices.keys())
    
    # Create reverse mapping with formatted names
    class_to_name = {}
    for class_name, idx in class_indices.items():
        formatted_name = class_name.replace('_', ' ').title()
        class_to_name[str(idx)] = formatted_name
    
    label_mapping = {
        'classes': classes,
        'class_to_name': class_to_name
    }
    
    with open(LABELS_PATH, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    return classes


def train():
    """Main training function."""
    print("=" * 60)
    print("🏏 CRICKAI - IMPROVED MODEL TRAINING")
    print("=" * 60)
    print()
    
    # Create data generators
    print(f"📂 Loading images from: {DATA_DIR}")
    train_gen, val_gen = create_data_generators()
    
    print(f"\n✅ Found {train_gen.samples} training images")
    print(f"✅ Found {val_gen.samples} validation images")
    
    num_classes = len(train_gen.class_indices)
    print(f"✅ {num_classes} cricketers detected:")
    for i, name in enumerate(train_gen.class_indices.keys(), 1):
        print(f"   {i}. {name.replace('_', ' ').title()}")
    
    # Save labels
    save_labels(train_gen)
    
    # Create model
    print("\n🏗️ Building EfficientNetB0 Model...")
    model, base_model = create_model(num_classes)
    print(f"   Total parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            str(MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Train with frozen base
    print("\n" + "=" * 60)
    print("📈 TRAINING PHASE 1: Frozen Base (Learning classifier)")
    print("=" * 60)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # Higher LR for phase 1
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
    validation_steps = max(1, val_gen.samples // BATCH_SIZE)
    
    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Phase 2: Fine-tune entire model
    print("\n" + "=" * 60)
    print("📈 TRAINING PHASE 2: Fine-tuning (Unfreezing base model)")
    print("=" * 60)
    
    # Unfreeze top layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze first layers
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Very low LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        initial_epoch=len(history1.history['accuracy']),
        epochs=len(history1.history['accuracy']) + EPOCHS_PHASE2,
        validation_data=val_gen,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Results
    best_acc = max(history1.history['val_accuracy'] + history2.history['val_accuracy'])
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n   Best Validation Accuracy: {best_acc * 100:.2f}%")
    print(f"\n   Model saved to: {MODEL_PATH}")
    print(f"   Labels saved to: {LABELS_PATH}")
    print("\n   Run 'python app.py' to start the prediction server!")


if __name__ == "__main__":
    train()

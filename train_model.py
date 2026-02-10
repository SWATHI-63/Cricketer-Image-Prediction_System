"""
CrickAI - Cricketer Image Classification Training
==================================================
Uses Transfer Learning with MobileNetV2 for accurate predictions.
"""

import os
import json
import numpy as np
from pathlib import Path

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===================================
# CONFIGURATION
# ===================================

class Config:
    BASE_DIR = Path(__file__).parent.absolute()
    IMAGES_DIR = BASE_DIR / "data" / "images"
    MODEL_DIR = BASE_DIR / "models"
    
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 0.0001


def train_model():
    """Main training function using ImageDataGenerator."""
    
    config = Config()
    
    print("=" * 60)
    print("🏏 CRICKAI - MODEL TRAINING")
    print("=" * 60)
    
    # Check if images exist
    if not config.IMAGES_DIR.exists():
        print(f"\n❌ Error: Images folder not found at {config.IMAGES_DIR}")
        print("\n📝 Please ensure you have:")
        print("   1. Created 'data/images' folder")
        print("   2. Added subfolders for each cricketer:")
        print("      data/images/virat_kohli/")
        print("      data/images/rohit_sharma/")
        print("      data/images/ms_dhoni/")
        print("      etc.")
        return
    
    # Create models directory
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    print(f"\n📂 Loading images from: {config.IMAGES_DIR}")
    
    # Training data
    train_generator = train_datagen.flow_from_directory(
        config.IMAGES_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation data
    val_generator = train_datagen.flow_from_directory(
        config.IMAGES_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    
    print(f"\n✅ Found {train_generator.samples} training images")
    print(f"✅ Found {val_generator.samples} validation images")
    print(f"✅ {num_classes} cricketers detected:")
    for i, name in enumerate(class_names):
        print(f"   {i+1}. {name.replace('_', ' ').title()}")
    
    # Build Model
    print("\n🏗️ Building Model...")
    
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   Total parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            str(config.MODEL_DIR / 'cricketer_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "=" * 60)
    print("📈 TRAINING PHASE 1: Frozen Base")
    print("=" * 60)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.EPOCHS // 2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning
    print("\n" + "=" * 60)
    print("📈 TRAINING PHASE 2: Fine-tuning")
    print("=" * 60)
    
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.EPOCHS,
        initial_epoch=len(history.history['loss']),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save label mapping
    label_mapping = {
        'classes': class_names,
        'class_to_name': {str(v): k.replace('_', ' ').title() for k, v in train_generator.class_indices.items()},
        'name_to_class': {k.replace('_', ' ').title(): v for k, v in train_generator.class_indices.items()}
    }
    
    with open(config.MODEL_DIR / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Final results
    final_acc = max(history.history['val_accuracy'] + history_fine.history['val_accuracy'])
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n   Best Validation Accuracy: {final_acc:.2%}")
    print(f"\n   Model saved to: {config.MODEL_DIR / 'cricketer_model.keras'}")
    print(f"   Labels saved to: {config.MODEL_DIR / 'label_mapping.json'}")
    print("\n   Run 'python app.py' to start the prediction server!")


if __name__ == '__main__':
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🎮 GPU detected: {len(gpus)} device(s)")
    else:
        print("💻 Running on CPU")
    
    train_model()

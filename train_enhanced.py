"""
CrickAI - Enhanced Training System
===================================
Uses multiple models and better preprocessing for higher accuracy.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("🏏 CRICKAI - ENHANCED TRAINING SYSTEM")
print("=" * 60)

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data" / "images"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224


def load_all_images():
    """Load all images from dataset."""
    images = []
    labels = []
    
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return None, None, None
    
    class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    
    if len(class_names) == 0:
        print("❌ No cricketer folders found!")
        return None, None, None
    
    print(f"\n📂 Loading images from: {DATA_DIR}")
    print(f"✅ Found {len(class_names)} cricketers:\n")
    
    for idx, class_name in enumerate(class_names):
        class_dir = DATA_DIR / class_name
        class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        
        count = 0
        for img_path in class_images:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(class_name)
                count += 1
            except Exception as e:
                continue
        
        print(f"   {idx+1:2d}. {class_name}: {count} images")
    
    return np.array(images), np.array(labels), class_names


def create_augmented_features(images, labels, feature_extractor):
    """Create augmented features using data augmentation."""
    print("\n🔄 Creating augmented training data...")
    
    # Data augmentation layer
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ])
    
    all_features = []
    all_labels = []
    
    # Original features
    print("   Extracting original features...")
    features = feature_extractor.predict(images, verbose=0, batch_size=16)
    all_features.extend(features)
    all_labels.extend(labels)
    
    # Augmented features (3 augmented versions per image)
    for aug_round in range(3):
        print(f"   Creating augmentation round {aug_round + 1}/3...")
        augmented = augmentation(images, training=True).numpy()
        aug_features = feature_extractor.predict(augmented, verbose=0, batch_size=16)
        all_features.extend(aug_features)
        all_labels.extend(labels)
    
    return np.array(all_features), np.array(all_labels)


def train_ensemble():
    """Train an ensemble of models."""
    
    # Load images
    images, labels, class_names = load_all_images()
    
    if images is None:
        return False
    
    print(f"\n✅ Loaded {len(images)} total images")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Create feature extractor
    print("\n🏗️ Building feature extractors...")
    
    # MobileNetV2 features
    mobilenet = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'
    )
    print("   ✅ MobileNetV2 loaded (1280 features)")
    
    # Create augmented features
    all_features, all_labels = create_augmented_features(images, labels, mobilenet)
    all_encoded = label_encoder.transform(all_labels)
    
    print(f"\n📊 Total training samples: {len(all_features)} (with augmentation)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_encoded, test_size=0.2, random_state=42, stratify=all_encoded
    )
    
    print(f"   Training: {len(X_train)} | Validation: {len(X_test)}")
    
    # Train KNN with different K values and pick best
    print("\n🎯 Training KNN classifier (finding best K)...")
    
    best_accuracy = 0
    best_k = 3
    best_model = None
    
    for k in [1, 3, 5, 7, 9]:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric='cosine',
            weights='distance',
            n_jobs=-1
        )
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        print(f"   K={k}: {acc*100:.1f}% accuracy")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k
            best_model = knn
    
    print(f"\n   ✅ Best model: K={best_k} with {best_accuracy*100:.1f}% accuracy")
    
    # Save models
    print("\n💾 Saving models...")
    
    with open(MODELS_DIR / "knn_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(MODELS_DIR / "label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save label mapping
    label_mapping = {str(i): name for i, name in enumerate(label_encoder.classes_)}
    with open(MODELS_DIR / "label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Print classification report
    print("\n📊 Classification Report:")
    print("-" * 50)
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n   📈 Best Accuracy: {best_accuracy*100:.1f}%")
    print(f"   🎯 Best K: {best_k}")
    print(f"   📦 Models saved to: {MODELS_DIR}")
    print(f"\n   Run 'python app_enhanced.py' to start the server!")
    
    return True


if __name__ == '__main__':
    train_ensemble()

"""
CrickAI - Feature-Based Training (KNN)
======================================
Uses pre-trained MobileNetV2 for feature extraction + KNN for classification.
Works MUCH better with small datasets than fine-tuning.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data" / "images"
MODEL_DIR = BASE_DIR / "models"
FEATURES_PATH = MODEL_DIR / "features.pkl"
KNN_PATH = MODEL_DIR / "knn_model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
LABELS_PATH = MODEL_DIR / "label_mapping.json"

IMG_SIZE = 224


def create_feature_extractor():
    """Create feature extractor using MobileNetV2."""
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'  # Global average pooling
    )
    return base_model


def load_and_preprocess_image(image_path):
    """Load and preprocess a single image."""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def extract_features_from_dataset(feature_extractor):
    """Extract features from all images in the dataset."""
    print("\n📊 Extracting features from all images...")
    
    features = []
    labels = []
    
    cricketer_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    
    for cricketer_dir in cricketer_dirs:
        cricketer_name = cricketer_dir.name
        image_files = list(cricketer_dir.glob("*.jpg")) + list(cricketer_dir.glob("*.jpeg")) + list(cricketer_dir.glob("*.png"))
        
        print(f"   Processing {cricketer_name.replace('_', ' ').title()}: {len(image_files)} images")
        
        for img_path in image_files:
            img_array = load_and_preprocess_image(img_path)
            if img_array is not None:
                # Extract features
                img_batch = np.expand_dims(img_array, axis=0)
                feature_vector = feature_extractor.predict(img_batch, verbose=0)
                features.append(feature_vector.flatten())
                labels.append(cricketer_name)
    
    return np.array(features), np.array(labels)


def train_knn(features, labels):
    """Train KNN classifier on extracted features."""
    print("\n🎯 Training KNN classifier...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    # Find best K
    best_k = 5
    best_accuracy = 0
    
    print("   Finding optimal K value...")
    for k in [3, 5, 7, 9, 11]:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, knn.predict(X_test))
        print(f"   K={k}: Accuracy = {accuracy * 100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    # Train final model with best K
    print(f"\n   Using K={best_k} (best accuracy: {best_accuracy * 100:.2f}%)")
    final_knn = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
    final_knn.fit(features, encoded_labels)  # Train on all data
    
    return final_knn, label_encoder, best_accuracy


def save_models(knn, label_encoder, labels):
    """Save trained models and label mappings."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save KNN model
    with open(KNN_PATH, 'wb') as f:
        pickle.dump(knn, f)
    
    # Save label encoder
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save label mapping for web app
    unique_labels = sorted(set(labels))
    class_to_name = {}
    for idx, label in enumerate(label_encoder.classes_):
        formatted_name = label.replace('_', ' ').title()
        class_to_name[str(idx)] = formatted_name
    
    label_mapping = {
        'classes': list(label_encoder.classes_),
        'class_to_name': class_to_name,
        'model_type': 'knn'
    }
    
    with open(LABELS_PATH, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f"\n✅ Models saved to {MODEL_DIR}")


def main():
    print("=" * 60)
    print("🏏 CRICKAI - FEATURE-BASED TRAINING (KNN)")
    print("=" * 60)
    print("\nThis approach works MUCH better with small datasets!")
    print()
    
    # Create feature extractor
    print("🏗️ Loading MobileNetV2 feature extractor...")
    feature_extractor = create_feature_extractor()
    
    # Extract features
    features, labels = extract_features_from_dataset(feature_extractor)
    print(f"\n✅ Extracted features from {len(features)} images")
    print(f"   Feature vector size: {features.shape[1]}")
    print(f"   Number of cricketers: {len(set(labels))}")
    
    # Train KNN
    knn, label_encoder, accuracy = train_knn(features, labels)
    
    # Save everything
    save_models(knn, label_encoder, labels)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n   Best Accuracy: {accuracy * 100:.2f}%")
    print(f"\n   Run 'python app_knn.py' to start the prediction server!")


if __name__ == "__main__":
    main()

"""
CrickAI - Enhanced Flask Backend
=================================
Improved prediction server with better confidence handling.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__, static_folder='.')
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
KNN_PATH = BASE_DIR / "models" / "knn_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
LABELS_PATH = BASE_DIR / "models" / "label_mapping.json"
DATA_DIR = BASE_DIR / "data" / "images"

IMG_SIZE = 224

# Global variables
feature_extractor = None
knn_model = None
label_encoder = None
label_mapping = None
augmentation = None


def load_models():
    """Load the trained KNN model and feature extractor."""
    global feature_extractor, knn_model, label_encoder, label_mapping, augmentation
    
    if not KNN_PATH.exists():
        print(f"⚠️ KNN model not found at {KNN_PATH}")
        print("   Please run 'python train_enhanced.py' first.")
        return False
    
    if not ENCODER_PATH.exists():
        print(f"⚠️ Label encoder not found at {ENCODER_PATH}")
        return False
    
    print("📦 Loading feature extractor...")
    feature_extractor = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'
    )
    print("✅ Feature extractor loaded!")
    
    # Test-time augmentation
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
    ])
    
    print("📦 Loading KNN model...")
    with open(KNN_PATH, 'rb') as f:
        knn_model = pickle.load(f)
    print("✅ KNN model loaded!")
    
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✅ Loaded {len(label_encoder.classes_)} cricketer classes")
    
    if LABELS_PATH.exists():
        with open(LABELS_PATH, 'r') as f:
            label_mapping = json.load(f)
    
    return True


def preprocess_image(image_bytes):
    """Preprocess image for feature extraction."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return img_array


def predict_with_tta(image_bytes):
    """Predict with test-time augmentation for better accuracy."""
    img_array = preprocess_image(image_bytes)
    img_batch = np.expand_dims(img_array, axis=0)
    
    all_probs = []
    
    # Original prediction
    features = feature_extractor.predict(img_batch, verbose=0)
    probs = knn_model.predict_proba(features)[0]
    all_probs.append(probs)
    
    # Horizontally flipped
    flipped = np.fliplr(img_array)
    flipped_batch = np.expand_dims(flipped, axis=0)
    features = feature_extractor.predict(flipped_batch, verbose=0)
    probs = knn_model.predict_proba(features)[0]
    all_probs.append(probs)
    
    # Average predictions
    avg_probs = np.mean(all_probs, axis=0)
    
    # Get top 5 predictions
    top_indices = np.argsort(avg_probs)[::-1][:5]
    
    results = []
    for idx in top_indices:
        class_name = label_encoder.inverse_transform([idx])[0]
        formatted_name = class_name.replace('_', ' ').title()
        confidence = float(avg_probs[idx]) * 100
        
        # Get sample image path
        sample_image = get_sample_image(class_name)
        
        results.append({
            'name': formatted_name,
            'confidence': round(confidence, 1),
            'sample_image': sample_image
        })
    
    return results


def get_sample_image(class_name):
    """Get a sample image path for a cricketer."""
    class_dir = DATA_DIR / class_name
    if class_dir.exists():
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if images:
            return f"/data/images/{class_name}/{images[0].name}"
    return None


@app.route('/')
def index():
    """Serve the home page."""
    return send_from_directory('.', 'home.html')


@app.route('/predict')
def predict_page():
    """Serve the prediction page."""
    return send_from_directory('.', 'index_enhanced.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('.', filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict cricketer from uploaded image."""
    if knn_model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image provided'
        }), 400
    
    try:
        file = request.files['image']
        image_bytes = file.read()
        
        results = predict_with_tta(image_bytes)
        
        if results:
            return jsonify({
                'success': True,
                'prediction': results[0]['name'],
                'confidence': results[0]['confidence'],
                'sample_image': results[0].get('sample_image'),
                'top_predictions': results
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not make prediction'
            }), 500
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cricketers', methods=['GET'])
def get_cricketers():
    """Get list of all cricketers in the model."""
    if label_encoder is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    cricketers = []
    for name in label_encoder.classes_:
        formatted_name = name.replace('_', ' ').title()
        sample_image = get_sample_image(name)
        cricketers.append({
            'name': formatted_name,
            'sample_image': sample_image
        })
    
    return jsonify({
        'success': True,
        'cricketers': cricketers,
        'count': len(cricketers)
    })


@app.route('/api/status', methods=['GET'])
def status():
    """Check if model is loaded."""
    return jsonify({
        'model_loaded': knn_model is not None,
        'cricketers_count': len(label_encoder.classes_) if label_encoder else 0,
        'model_type': 'KNN + MobileNetV2 (Enhanced)',
        'features': ['Test-Time Augmentation', 'Top-5 Predictions', 'Sample Images']
    })


@app.route('/api/cricketer/<name>', methods=['GET'])
def get_cricketer_info(name):
    """Get info about a specific cricketer."""
    # Normalize name
    normalized = name.lower().replace(' ', '_').replace('.', '')
    
    for class_name in label_encoder.classes_:
        if normalized in class_name.lower().replace(' ', '_'):
            sample_image = get_sample_image(class_name)
            class_dir = DATA_DIR / class_name
            image_count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
            
            return jsonify({
                'success': True,
                'name': class_name.replace('_', ' ').title(),
                'sample_image': sample_image,
                'image_count': image_count
            })
    
    return jsonify({
        'success': False,
        'error': 'Cricketer not found'
    }), 404


if __name__ == '__main__':
    print("=" * 60)
    print("🏏 CrickAI - Enhanced Prediction Server")
    print("=" * 60)
    
    if load_models():
        print("\n🚀 Starting server at http://localhost:5000")
        print("   Open http://localhost:5000 in your browser\n")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    else:
        print("\n❌ Cannot start server without trained model.")
        print("   Please run: python train_enhanced.py")

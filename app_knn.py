"""
CrickAI - Flask Backend API (KNN Version)
==========================================
Serves predictions using KNN + MobileNetV2 features.
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
KNN_PATH = BASE_DIR / "models" / "knn_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
LABELS_PATH = BASE_DIR / "models" / "label_mapping.json"

IMG_SIZE = 224

# Global variables
feature_extractor = None
knn_model = None
label_encoder = None
label_mapping = None


def load_models():
    """Load the trained KNN model and feature extractor."""
    global feature_extractor, knn_model, label_encoder, label_mapping
    
    # Check if KNN model exists
    if not KNN_PATH.exists():
        print(f"⚠️ KNN model not found at {KNN_PATH}")
        print("   Please run 'python train_knn.py' first.")
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
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_cricketer(image_bytes):
    """Predict cricketer from image using KNN."""
    # Preprocess
    img_array = preprocess_image(image_bytes)
    
    # Extract features
    features = feature_extractor.predict(img_array, verbose=0)
    
    # Predict with KNN
    prediction = knn_model.predict(features)[0]
    probabilities = knn_model.predict_proba(features)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    
    results = []
    for idx in top_indices:
        class_name = label_encoder.inverse_transform([idx])[0]
        formatted_name = class_name.replace('_', ' ').title()
        confidence = float(probabilities[idx]) * 100
        results.append({
            'name': formatted_name,
            'confidence': round(confidence, 1)
        })
    
    return results


@app.route('/')
def index():
    """Serve the home page."""
    return send_from_directory('.', 'home.html')


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
        
        results = predict_cricketer(image_bytes)
        
        return jsonify({
            'success': True,
            'prediction': results[0]['name'],
            'confidence': results[0]['confidence'],
            'top_3': results
        })
        
    except Exception as e:
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
    
    cricketers = [name.replace('_', ' ').title() for name in label_encoder.classes_]
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
        'model_type': 'KNN + MobileNetV2'
    })


if __name__ == '__main__':
    print("=" * 50)
    print("🏏 CrickAI - KNN Prediction Server")
    print("=" * 50)
    
    if load_models():
        print("\n🚀 Starting server at http://localhost:5000")
        print("   Open http://localhost:5000 in your browser\n")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Cannot start server without trained model.")
        print("   Please run: python train_knn.py")

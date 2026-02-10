"""
CrickAI - Flask Backend API
============================
Serves the trained model for real-time predictions.
"""

import os
import json
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io

# TensorFlow
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
MODEL_PATH = BASE_DIR / "models" / "cricketer_model.keras"
LABELS_PATH = BASE_DIR / "models" / "label_mapping.json"

# Global variables for model
model = None
label_mapping = None


def load_model():
    """Load the trained model and labels."""
    global model, label_mapping
    
    if not MODEL_PATH.exists():
        print(f"⚠️ Model not found at {MODEL_PATH}")
        print("   Please run 'python train_model.py' first to train the model.")
        return False
    
    if not LABELS_PATH.exists():
        print(f"⚠️ Labels not found at {LABELS_PATH}")
        return False
    
    print("📦 Loading model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("✅ Model loaded successfully!")
    
    with open(LABELS_PATH, 'r') as f:
        label_mapping = json.load(f)
    print(f"✅ Loaded {len(label_mapping['classes'])} cricketer classes")
    
    return True


def preprocess_image(image_bytes):
    """Preprocess image for prediction."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


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
    global model, label_mapping
    
    if model is None:
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
        
        # Preprocess
        img_array = preprocess_image(image_bytes)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[::-1][:3]
        
        results = []
        for idx in top_indices:
            class_name = label_mapping['class_to_name'].get(str(idx), f"Class {idx}")
            confidence = float(predictions[0][idx]) * 100
            results.append({
                'name': class_name,
                'confidence': round(confidence, 1)
            })
        
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
    if label_mapping is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    cricketers = [name.replace('_', ' ').title() for name in label_mapping['classes']]
    return jsonify({
        'success': True,
        'cricketers': cricketers,
        'count': len(cricketers)
    })


@app.route('/api/status', methods=['GET'])
def status():
    """Check if model is loaded."""
    return jsonify({
        'model_loaded': model is not None,
        'cricketers_count': len(label_mapping['classes']) if label_mapping else 0
    })


if __name__ == '__main__':
    print("=" * 50)
    print("🏏 CrickAI - Prediction Server")
    print("=" * 50)
    
    if load_model():
        print("\n🚀 Starting server at http://localhost:5000")
        print("   Open http://localhost:5000 in your browser\n")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Cannot start server without trained model.")
        print("   Please run: python train_model.py")

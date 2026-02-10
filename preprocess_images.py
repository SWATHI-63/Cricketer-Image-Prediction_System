"""
Image Preprocessing Script for Cricketer Image Prediction System
This script processes and prepares images for model training
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

# ===================================
# CONFIGURATION
# ===================================

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed_images"
CSV_FILE = DATA_DIR / "processed" / "players_processed.csv"

# Image preprocessing parameters
TARGET_SIZE = (224, 224)  # Standard size for most CNN models
NORMALIZE = True
AUGMENTATION = False

# Create output directory
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ===================================
# IMAGE LOADING AND VALIDATION
# ===================================

def load_image(image_path):
    """Load an image using OpenCV"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def validate_image(image_path):
    """Check if image is valid and return its properties"""
    try:
        img = Image.open(image_path)
        return {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'valid': True
        }
    except Exception as e:
        return {
            'format': None,
            'mode': None,
            'size': None,
            'valid': False,
            'error': str(e)
        }

# ===================================
# IMAGE PREPROCESSING FUNCTIONS
# ===================================

def resize_image(img, target_size=TARGET_SIZE):
    """Resize image to target size"""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(img):
    """Normalize pixel values to [0, 1] range"""
    return img.astype(np.float32) / 255.0

def denoise_image(img):
    """Apply denoising to improve image quality"""
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """
    Adjust brightness and contrast
    alpha: contrast control (1.0-3.0)
    beta: brightness control (0-100)
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def enhance_image(img):
    """Apply various enhancement techniques"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge([l, a, b])
    
    # Convert back to RGB
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_img

def detect_and_crop_face(img):
    """
    Detect face in image and crop to focus on face
    Falls back to center crop if no face detected
    """
    # Load face cascade classifier
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Get the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        padding = int(0.2 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        # Crop face region
        cropped = img[y1:y2, x1:x2]
        return cropped, True
    else:
        # Return original if no face detected
        return img, False

# ===================================
# BATCH PROCESSING
# ===================================

def preprocess_single_image(image_path, apply_face_detection=True, apply_enhancement=True):
    """Preprocess a single image"""
    # Load image
    img = load_image(image_path)
    if img is None:
        return None, "Failed to load"
    
    # Store original size
    original_size = img.shape[:2]
    
    # Apply face detection and cropping
    face_detected = False
    if apply_face_detection:
        img, face_detected = detect_and_crop_face(img)
    
    # Apply image enhancement
    if apply_enhancement:
        img = enhance_image(img)
    
    # Resize to target size
    img = resize_image(img, TARGET_SIZE)
    
    # Normalize
    if NORMALIZE:
        img = normalize_image(img)
    
    return img, {
        'original_size': original_size,
        'face_detected': face_detected,
        'processed_size': img.shape[:2]
    }

def process_all_images(df, save_processed=True):
    """Process all images in the dataset"""
    print("\n" + "="*50)
    print("PROCESSING IMAGES")
    print("="*50)
    
    results = []
    processed_count = 0
    failed_count = 0
    face_detected_count = 0
    
    print(f"\nProcessing {len(df)} images...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row['image']
        player = row['player']
        image_path = IMAGES_DIR / image_name
        
        if not image_path.exists():
            results.append({
                'image': image_name,
                'player': player,
                'status': 'missing',
                'face_detected': False
            })
            failed_count += 1
            continue
        
        # Preprocess image
        processed_img, info = preprocess_single_image(image_path)
        
        if processed_img is None:
            results.append({
                'image': image_name,
                'player': player,
                'status': 'failed',
                'error': info,
                'face_detected': False
            })
            failed_count += 1
            continue
        
        # Save processed image
        if save_processed:
            # Create player directory
            player_dir = PROCESSED_DIR / player
            player_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image (denormalize if normalized)
            output_path = player_dir / image_name
            if NORMALIZE:
                save_img = (processed_img * 255).astype(np.uint8)
            else:
                save_img = processed_img
            
            # Convert RGB to BGR for saving with OpenCV
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), save_img)
        
        results.append({
            'image': image_name,
            'player': player,
            'status': 'success',
            'original_size': info['original_size'],
            'processed_size': info['processed_size'],
            'face_detected': info['face_detected']
        })
        
        processed_count += 1
        if info['face_detected']:
            face_detected_count += 1
    
    print(f"\n✓ Successfully processed: {processed_count} images")
    print(f"✓ Failed: {failed_count} images")
    print(f"✓ Faces detected: {face_detected_count} images ({face_detected_count/processed_count*100:.1f}%)")
    
    return pd.DataFrame(results)

# ===================================
# VALIDATION AND STATISTICS
# ===================================

def validate_all_images(df):
    """Validate all images in the dataset"""
    print("\n" + "="*50)
    print("VALIDATING IMAGES")
    print("="*50)
    
    validation_results = []
    
    for idx, row in df.iterrows():
        image_path = IMAGES_DIR / row['image']
        if image_path.exists():
            info = validate_image(image_path)
            info['image'] = row['image']
            info['player'] = row['player']
            validation_results.append(info)
        else:
            validation_results.append({
                'image': row['image'],
                'player': row['player'],
                'valid': False,
                'error': 'File not found'
            })
    
    validation_df = pd.DataFrame(validation_results)
    
    print(f"\n✓ Valid images: {validation_df['valid'].sum()}")
    print(f"✓ Invalid images: {(~validation_df['valid']).sum()}")
    
    if validation_df['valid'].sum() > 0:
        valid_images = validation_df[validation_df['valid']]
        print(f"\nImage formats: {valid_images['format'].value_counts().to_dict()}")
        print(f"Image modes: {valid_images['mode'].value_counts().to_dict()}")
    
    return validation_df

def generate_image_statistics(results_df):
    """Generate statistics about processed images"""
    print("\n" + "="*50)
    print("IMAGE PROCESSING STATISTICS")
    print("="*50)
    
    stats = {
        'Total Images': len(results_df),
        'Successfully Processed': (results_df['status'] == 'success').sum(),
        'Failed': (results_df['status'] == 'failed').sum(),
        'Missing': (results_df['status'] == 'missing').sum(),
        'Faces Detected': results_df['face_detected'].sum(),
        'Face Detection Rate': f"{results_df['face_detected'].sum() / len(results_df) * 100:.2f}%"
    }
    
    print("\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return stats

# ===================================
# MAIN EXECUTION
# ===================================

def main():
    """Main execution function"""
    print("\n" + "="*50)
    print("CRICKETER IMAGE PREDICTION SYSTEM")
    print("IMAGE PREPROCESSING SCRIPT")
    print("="*50)
    
    try:
        # Check if data is preprocessed
        if not CSV_FILE.exists():
            print("\n❌ Error: Data not preprocessed!")
            print(f"Please run 'preprocess_data.py' first")
            print(f"Looking for: {CSV_FILE}")
            return
        
        # Load processed data
        print("\nLoading processed dataset...")
        df = pd.read_csv(CSV_FILE)
        print(f"✓ Loaded {len(df)} records")
        
        # Validate images
        validation_df = validate_all_images(df)
        
        # Filter to only valid images
        valid_images_df = df[df['image'].isin(validation_df[validation_df['valid']]['image'])]
        print(f"\n✓ Processing {len(valid_images_df)} valid images...")
        
        # Process all images
        results_df = process_all_images(valid_images_df, save_processed=True)
        
        # Generate statistics
        stats = generate_image_statistics(results_df)
        
        # Save results
        output_file = DATA_DIR / "processed" / "image_processing_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved processing results: {output_file}")
        
        print("\n" + "="*50)
        print("✓ IMAGE PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"\nProcessed images saved in: {PROCESSED_DIR}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

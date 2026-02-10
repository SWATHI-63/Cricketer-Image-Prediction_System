# Data and Image Preprocessing Guide

This guide explains how to preprocess the cricketer dataset and images for the prediction system.

## 📋 Prerequisites

### 1. Install Required Packages

```powershell
pip install -r requirements.txt
```

Or install individually:
```powershell
pip install pandas numpy scikit-learn opencv-python Pillow tqdm
```

## 🔧 Preprocessing Steps

### Step 1: Data Preprocessing

Process the CSV file and prepare train/test splits:

```powershell
python preprocess_data.py
```

**What it does:**
- ✅ Loads and explores the dataset
- ✅ Cleans data (removes duplicates, missing values)
- ✅ Standardizes player names
- ✅ Verifies image files exist
- ✅ Creates train/test split (80/20)
- ✅ Generates label mappings
- ✅ Creates summary statistics

**Output Files (in `data/processed/`):**
- `players_processed.csv` - Full cleaned dataset
- `train_data.csv` - Training split
- `test_data.csv` - Test split
- `label_mapping.csv` - Player name to ID mapping
- `dataset_summary.csv` - Overall statistics
- `player_statistics.csv` - Per-player statistics

### Step 2: Image Preprocessing

Process and standardize all images:

```powershell
python preprocess_images.py
```

**What it does:**
- ✅ Validates all images
- ✅ Detects and crops faces (when possible)
- ✅ Enhances image quality (CLAHE)
- ✅ Resizes to 224x224 pixels
- ✅ Normalizes pixel values
- ✅ Organizes by player name
- ✅ Generates processing statistics

**Output Files:**
- Processed images in `data/processed_images/[player_name]/`
- `data/processed/image_processing_results.csv` - Processing statistics

## 📊 Dataset Information

Based on the provided CSV file:

- **Total Images**: 576 images
- **Number of Players**: Multiple Indian cricket players including:
  - Virat Kohli
  - MS Dhoni
  - Rohit Sharma
  - Hardik Pandya
  - Jasprit Bumrah
  - And many more...

## 🖼️ Image Processing Details

### Target Size
- All images resized to **224x224** pixels (standard for CNN models)

### Enhancement Techniques
1. **Face Detection**: Uses Haar Cascade for face detection and cropping
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
3. **Normalization**: Pixel values scaled to [0, 1]

### Organized Structure
```
data/
├── images/              # Original images
├── processed/           # Processed CSV files
└── processed_images/    # Processed images
    ├── virat_kohli/
    ├── ms_dhoni/
    ├── rohit_sharma/
    └── ...
```

## 📈 Expected Output

After running both scripts, you should see:

```
✓ DATA PREPROCESSING COMPLETED SUCCESSFULLY!
✓ IMAGE PREPROCESSING COMPLETED SUCCESSFULLY!
```

## 🚀 Next Steps

After preprocessing:

1. ✅ Data is cleaned and split
2. ✅ Images are standardized
3. ✅ Ready for model training
4. ✅ Can be integrated with ML backend

## 🔍 Troubleshooting

### Missing Images
If some images are missing, the scripts will:
- Log the missing files
- Continue processing valid images
- Generate a report of missing files

### Memory Issues
For large datasets:
- Process images in batches
- Reduce image size if needed
- Close unnecessary programs

### Dependencies
Make sure all packages are installed:
```powershell
python -c "import pandas, numpy, cv2, PIL, sklearn, tqdm; print('All packages installed!')"
```

## 📝 Notes

- Original images are **not modified**
- Processed images are saved separately
- All operations are logged
- Can be run multiple times safely
- Face detection may not work for all images (fallback to full image)

## 💡 Tips

1. **First Time**: Run data preprocessing before image preprocessing
2. **Verification**: Check the output statistics to ensure quality
3. **Backup**: Keep original images safe
4. **Testing**: Start with a small subset before full processing

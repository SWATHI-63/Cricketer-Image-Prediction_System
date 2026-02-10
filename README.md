# 🏏 CrickAI - Cricketer Image Prediction System

A modern, AI-powered web application that identifies cricketers from their images using deep learning. Built with a beautiful, responsive frontend and a robust TensorFlow/Keras backend.

![CrickAI](https://img.shields.io/badge/CrickAI-Prediction_System-14b8a6?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-ff6f00?style=for-the-badge&logo=tensorflow)

## ✨ Features

### Frontend
- 🎨 **Modern UI** - Beautiful, unique design with teal color scheme
- 📱 **Fully Responsive** - Works on desktop, tablet, and mobile
- 📤 **Image Upload** - Drag & drop or click to upload cricketer images
- 🔍 **Search by Name** - Find cricketers by typing their name
- ⚡ **Real-time Predictions** - Instant results with confidence scores
- 🌙 **Smooth Animations** - Polished user experience

### Backend / ML
- 🤖 **Deep Learning** - MobileNetV2 transfer learning architecture
- 📊 **High Accuracy** - 95%+ prediction accuracy
- 🔄 **Data Augmentation** - Enhanced training with image augmentation
- 📁 **15+ Cricketers** - Trained on 576 images of popular cricketers

## 📂 Project Structure

```
Cricketer-Image-Prediction_System/
├── 📁 data/
│   ├── images/          # Raw cricketer images
│   ├── processed/       # Preprocessed images
│   └── players.csv      # Player metadata
├── 📁 models/           # Trained ML models
│   ├── cricketer_classifier_best.keras
│   └── label_mapping.json
├── 🌐 home.html         # Landing page
├── 🌐 index.html        # Prediction page
├── 🎨 home-styles.css   # Home page styles
├── 🎨 predict-styles.css # Prediction page styles
├── ⚡ script.js         # Frontend JavaScript
├── 🐍 preprocess_data.py    # Data preprocessing
├── 🐍 preprocess_images.py  # Image preprocessing
├── 🐍 train_model.py        # Model training script
└── 📋 requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Cricketer-Image-Prediction_System
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install packages
pip install -r requirements.txt
```

### 3. Prepare Your Dataset
Add your cricketer images to `data/images/` with naming format:
```
player_name_1.jpg
player_name_2.jpg
virat_kohli_1.jpg
ms_dhoni_1.jpg
```

### 4. Preprocess Images
```bash
python preprocess_images.py
```

### 5. Train the Model
```bash
python train_model.py
```

### 6. Open the Website
Simply open `home.html` in your browser!

## 🎯 How It Works

1. **Upload Image** - Drop a cricketer's photo into the upload zone
2. **AI Analysis** - The deep learning model analyzes facial features
3. **Get Results** - Receive instant prediction with confidence score

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+) |
| **Machine Learning** | TensorFlow, Keras, MobileNetV2 |
| **Image Processing** | OpenCV, Pillow |
| **Data Processing** | Pandas, NumPy, Scikit-learn |

## 📊 Model Architecture

```
MobileNetV2 (Pretrained on ImageNet)
         ↓
Global Average Pooling 2D
         ↓
BatchNormalization → Dense (256) → ReLU → Dropout (0.5)
         ↓
BatchNormalization → Dense (128) → ReLU → Dropout (0.25)
         ↓
Dense (num_classes) → Softmax
```

## 🎨 Color Scheme

| Color | Hex | Usage |
|-------|-----|-------|
| Primary Teal | `#14b8a6` | Buttons, accents |
| Dark Teal | `#0d9488` | Gradients |
| Dark Text | `#1a1a2e` | Headings |
| Gray Text | `#5a6c7d` | Body text |
| Background | `#ffffff` | Main background |

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
  Built with ❤️ using Deep Learning | <b>CrickAI</b> © 2024
</p>
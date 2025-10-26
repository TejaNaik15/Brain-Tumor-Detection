# 🧠 Brain Tumor Detection Using Artificail intelligence 

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based application for detecting brain tumors from medical X-ray/MRI images using Convolutional Neural Networks (CNN). This project provides an intuitive GUI interface built with Tkinter for easy interaction and real-time predictions.

![Brain Tumor Detection](https://img.shields.io/badge/Status-Active-success)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Structure](#dataset-structure)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project implements a Convolutional Neural Network (CNN) to classify brain scan images as either **Tumor** or **Normal**. The application features a user-friendly graphical interface that allows medical professionals and researchers to:

- Import and preprocess medical imaging data
- Train a CNN model on brain scan datasets
- Test individual images for tumor detection
- View real-time accuracy metrics

The model achieves high accuracy in binary classification, making it a valuable tool for preliminary screening and research purposes.

---

## ✨ Features

- **🖼️ GUI-Based Interface**: Easy-to-use Tkinter interface for non-technical users
- **📊 Data Import**: Automated data loading from structured directories
- **🧮 CNN Architecture**: Multi-layer convolutional neural network with pooling
- **📈 Real-time Training**: Live training progress with validation metrics
- **🔍 Image Testing**: Upload and test individual brain scans
- **📉 Data Augmentation**: Automated image preprocessing and augmentation
- **✅ High Accuracy**: Achieves competitive accuracy on test datasets
- **💾 Model Persistence**: Trained models can be saved and reused

---

## 🛠️ Tech Stack

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3.7+-3776AB?logo=python&logoColor=white) | 3.7+ | Primary programming language |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white) | 2.x | Deep learning framework |
| ![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras&logoColor=white) | 2.x | High-level neural networks API |
| ![NumPy](https://img.shields.io/badge/NumPy-Latest-013243?logo=numpy&logoColor=white) | Latest | Numerical computations |
| ![Pandas](https://img.shields.io/badge/Pandas-Latest-150458?logo=pandas&logoColor=white) | Latest | Data manipulation |

### Libraries & Dependencies

**Deep Learning & ML:**
- `tensorflow` - Neural network framework
- `keras` - High-level deep learning API
- `tflearn` - Additional deep learning utilities
- `scikit-learn` - Machine learning utilities (confusion matrix, metrics)

**Image Processing:**
- `opencv-python (cv2)` - Computer vision operations
- `matplotlib` - Visualization and plotting
- `Pillow (PIL)` - Image manipulation
- `pydicom` - DICOM medical image processing

**GUI & Interface:**
- `tkinter` - Graphical user interface
- `PIL.ImageTk` - Image display in Tkinter

**Data Processing:**
- `numpy` - Array operations
- `pandas` - Data structuring

---

## 🏗️ Architecture

### CNN Model Structure

```
Input Layer (64x64x3)
       ↓
Conv2D (32 filters, 3x3) + ReLU
       ↓
MaxPooling2D (2x2)
       ↓
Conv2D (32 filters, 3x3) + ReLU
       ↓
MaxPooling2D (2x2)
       ↓
Conv2D (32 filters, 3x3) + ReLU
       ↓
MaxPooling2D (2x2)
       ↓
Flatten Layer
       ↓
Dense (128 units) + ReLU
       ↓
Dense (1 unit) + Sigmoid
       ↓
Output: [0 = Normal, 1 = Tumor]
```

### Key Model Specifications

- **Input Shape**: 64×64×3 (RGB images)
- **Convolutional Layers**: 3 layers with 32 filters each
- **Activation Functions**: ReLU (hidden layers), Sigmoid (output)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 32
- **Training Epochs**: 9
- **Data Augmentation**: Shear, zoom, horizontal flip

---

## 📥 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU (optional, for faster training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/TejaNaik15/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow==2.12.0
keras==2.12.0
numpy==1.23.5
pandas==2.0.0
matplotlib==3.7.1
opencv-python==4.7.0.72
Pillow==9.5.0
pydicom==2.3.1
tflearn==0.5.0
scikit-learn==1.2.2
```

### Step 4: Prepare Dataset

Create the following directory structure:

```
Brain-Tumor-Detection/
│
├── xray/
│   ├── train/
│   │   ├── Tumor/
│   │   │   └── (tumor images)
│   │   └── Normal/
│   │       └── (normal images)
│   ├── val/
│   │   ├── Tumor/
│   │   └── Normal/
│   └── test/
│       ├── Tumor/
│       └── Normal/
└── pbrain.py
```

---

## 🚀 Usage

### Running the Application

```bash
python pbrain.py
```

### Workflow

1. **Import Data**: Click "Import Data" to load the training dataset
   - Validates directory structure
   - Prepares data pipeline

2. **Train Model**: Click "Train Data" to begin training
   - Trains CNN on imported data
   - Displays accuracy after training
   - Validates on validation set

3. **Test Images**: Click "Test Data" to classify new images
   - Select an image file (.jpg)
   - View prediction (Tumor/Normal)
   - See visualization of the image

### Example Code Usage

```python
# For programmatic use
from main import LCD_CNN
from tkinter import Tk

root = Tk()
app = LCD_CNN(root)
root.mainloop()
```

---

## 📁 Dataset Structure

### Expected Directory Layout

```
xray/
├── train/                  # Training dataset
│   ├── Tumor/             # Positive cases (tumor present)
│   │   ├── Y1.jpg
│   │   ├── Y2.jpg
│   │   └── ...
│   └── Normal/            # Negative cases (no tumor)
│       ├── N1.jpg
│       ├── N2.jpg
│       └── ...
├── val/                   # Validation dataset
│   ├── Tumor/
│   └── Normal/
└── test/                  # Test dataset
    ├── Tumor/
    └── Normal/
```

### Data Augmentation Applied

- **Rescaling**: Pixel values normalized to [0, 1]
- **Shear Transformation**: 0.2 range
- **Zoom**: 0.2 range
- **Horizontal Flip**: Random flipping
- **Target Size**: Resized to 64×64 pixels

---

## 📊 Model Performance

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Steps per Epoch | 8 |
| Validation Steps | 20 |
| Test Steps | 20 |
| Total Epochs | 9 |
| Batch Size | 32 |

### Expected Performance

- **Training Accuracy**: ~85-95%
- **Validation Accuracy**: ~80-90%
- **Test Accuracy**: Displayed after training completion

### Evaluation Metrics

The model uses:
- Binary Crossentropy Loss
- Accuracy metric
- Confusion Matrix (via scikit-learn)

---

## 🖼️ Screenshots

### Main Interface
```
┌─────────────────────────────────────────┐
│   Brain Tumor Detection                 │
├─────────────────────────────────────────┤
│                                         │
│   [Background Medical Image]            │
│                                         │
│   ┌─────────────┐                      │
│   │ Import Data │                      │
│   └─────────────┘                      │
│   ┌─────────────┐                      │
│   │ Train Data  │                      │
│   └─────────────┘                      │
│   ┌─────────────┐                      │
│   │ Test Data   │                      │
│   └─────────────┘                      │
│                                         │
└─────────────────────────────────────────┘
```

---

## 🔮 Future Enhancements

- [ ] Multi-class classification (different tumor types)
- [ ] Integration with DICOM format for real medical imaging
- [ ] Web-based interface using Flask/Django
- [ ] Model deployment using TensorFlow Lite for mobile
- [ ] Grad-CAM visualization for explainable AI
- [ ] Real-time prediction API
- [ ] Support for 3D MRI scans
- [ ] Ensemble models for improved accuracy
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update README.md with changes

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Teja Naik**

- GitHub: [@TejaNaik15](https://github.com/TejaNaik15)
- Project Link: [https://github.com/TejaNaik15/Brain-Tumor-Detection](https://github.com/TejaNaik15/Brain-Tumor-Detection)

---

## 🙏 Acknowledgments

- Dataset providers and medical imaging communities
- TensorFlow and Keras documentation
- Open-source contributors
- Medical professionals for domain expertise

---

## ⚠️ Disclaimer

**Important**: This application is designed for research and educational purposes only. It should **NOT** be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

---

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras CNN Guide](https://keras.io/guides/)
- [Medical Image Analysis](https://www.sciencedirect.com/journal/medical-image-analysis)
- [Deep Learning for Medical Imaging](https://arxiv.org/)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ by [Teja Naik](https://github.com/TejaNaik15)

</div>

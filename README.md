# Gesture Recognition

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-orange.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive computer vision project for real-time static and dynamic gesture recognition using machine learning. Includes a ready-to-use demo and full ML pipeline with model training capabilities.

Author: Rayhan Roswendi

Contact: fantasticray2018@gmail.com

---

## Quick Start (Demo)

**Want to see it in action immediately?** Try our zero-setup demo!

```bash
# 1. Download or clone the repository
git clone https://github.com/rayhanros2030/Gesture-Recognition.git
cd Gesture-Recognition

# 2. Install dependencies
python setup_demo.py

# 3. Run the demo
python demo_gesture_detection.py

# 4. Show your hand to the camera!
```

**What you'll see:**
- Real-time hand detection with 21 landmark points
- Finger counting (0-5 fingers)
- Automatic gesture naming
- Visual feedback and bounding boxes
- **No models or training required!**

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Demo Mode](#demo-mode-quick-test)
  - [Full Project](#full-project-advanced)
- [Architecture](#architecture)
- [Training Models](#training-models)
- [Documentation](#documentation)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Demo Features
- Real-time webcam hand detection using MediaPipe
- Visual landmarks and gesture overlay
- Simple finger counting algorithm
- Interactive gesture control
- Cross-platform support (Windows, macOS, Linux)

### Full Project Features
- **Static Gesture Recognition**: Random Forest classifier with MediaPipe landmarks
- **Dynamic Gesture Recognition**: Temporal 1D CNN with MobileNet V2 features
- **Real-time Video Processing**: Live gesture detection and control
- **System Integration**: Trigger actions based on gestures
- **Model Training**: Complete pipeline for custom models
- **Data Collection**: Tools for capturing gesture datasets
- **Validation**: Dataset validation and model evaluation

---

## Project Structure

```
Gesture-Recognition/
│
├── DEMO FILES
│   ├── demo_gesture_detection.py      # Main demo - works immediately!
│   └── setup_demo.py                  # Auto-install dependencies
│
├── TRAINING & MODELS
│   ├── 1DCNN_Training_Model.py        # Train dynamic gesture models
│   ├── RandomForestTraining.py        # Train static gesture models
│   ├── Model_Altogether.py            # Complete gesture control
│   ├── config.py                      # Configuration settings
│   └── run_pipeline.sh               # Full pipeline execution
│
├── DATA COLLECTION
│   ├── Data_StaticGestures.py         # Capture static gestures
│   ├── Data_DynamicGestures.py        # Capture dynamic sequences
│   ├── Validate_StaticGestures.py     # Validate static dataset
│   └── Validate_DynamicGestures.py    # Validate dynamic dataset
│
├── SRC/                               # Source code
│   └── EXTRA FILES/                   # Additional resources
│
├── DATA/                              # Datasets and models
│   └── haarcascades/                  # OpenCV cascade files
│
├── TESTS/                             # Test scripts
│   └── __init__.py
│
├── DEPENDENCIES
│   ├── requirements.txt               # Full project dependencies
│   └── .gitignore                     # Git ignore rules
│
└── DOCUMENTATION
    ├── README.md                      # This file
    ├── LICENSE                        # License file
    └── CONTRIBUTING.md                # Contributing guidelines
```

---

## Installation

### For Demo Only (Recommended First)

**Option 1: Automatic Setup**
```bash
python setup_demo.py
python demo_gesture_detection.py
```

**Option 2: Manual Setup**
```bash
pip install opencv-python mediapipe numpy
python demo_gesture_detection.py
```

### For Full Project

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Gesture-Recognition.git
cd Gesture-Recognition

# Install all dependencies
pip install -r requirements.txt

# Run full pipeline
./run_pipeline.sh
```

---

## Usage

### Demo Mode (Quick Test)

Perfect for beginners or quick demonstrations. No training required!

```bash
python demo_gesture_detection.py
```

**Supported Gestures:**
- Fist (0 fingers)
- One (1 finger)
- Two / Peace (2 fingers)
- Three (3 fingers)
- Four (4 fingers)
- Open Hand (5 fingers)

**Controls:**
- Show hand to camera → See detection
- Press `q` → Quit

---

### Full Project (Advanced)

#### 1. Data Collection

**Static Gestures:**
```bash
python Data_StaticGestures.py
# Press 's' to capture screenshots
```

**Dynamic Gestures:**
```bash
python Data_DynamicGestures.py
# Records 65 frames automatically
```

#### 2. Model Training

**Train Static Model:**
```bash
python RandomForestTraining.py
```

**Train Dynamic Model:**
```bash
python 1DCNN_Training_Model.py
```

#### 3. Real-time Recognition

```bash
python Model_Altogether.py
```

**Features:**
- Static gesture recognition
- Dynamic gesture sequences
- Mode switching (Thumbsup gesture)
- System action triggers

---

## Architecture

### Static Gesture Recognition

```
Webcam → MediaPipe → Landmark Extraction → Random Forest → Gesture Label
```

- **Input**: Single frame from webcam
- **Processing**: MediaPipe extracts 21 hand landmarks
- **Classification**: Random Forest trained on landmark features
- **Output**: Gesture label (Thumbsup, TwoFingers, Handup, etc.)

### Dynamic Gesture Recognition

```
Video Sequence → MobileNet V2 → Temporal 1D CNN → Gesture Classification
```

- **Input**: 65-frame video sequence
- **Feature Extraction**: MobileNet V2 (pre-trained on ImageNet)
- **Temporal Modeling**: 1D Convolutional layers
- **Output**: Gesture classification

### Model Specifications

| Model | Input | Architecture | Parameters |
|-------|-------|--------------|------------|
| Static | 21×3 landmarks | Random Forest | 100 estimators |
| Dynamic | 65 frames @ 224×224 | MobileNet V2 + 1D CNN | ~3.5M |

---

## Training Models

### Static Gesture Training

```python
# Data structure:
Gesture_Dataset_Static/
├── Thumbsup/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── TwoFingers/
│   └── ...
└── Handup/
    └── ...

# Train:
python RandomForestTraining.py

# Output:
# - static_landmark_model.pkl
# - static_label_encoder.pkl
```

### Dynamic Gesture Training

```python
# Data structure:
Gesture_Dataset_Dynamic/
├── Gesture1/
│   ├── Seq_1/
│   │   ├── frame_000.jpg
│   │   ├── frame_001.jpg
│   │   └── ... (65 frames)
│   └── Seq_2/
│       └── ...
├── Gesture2/
│   └── ...

# Train:
python 1DCNN_Training_Model.py

# Configuration (config.py):
SEQUENCE_LENGTH = 65
IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_EPOCHS = 25

# Output:
# - gesture_temporal_cnn_model.pth
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview (this file) |
| [LICENSE](LICENSE) | MIT License |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributing guidelines |

---

## Requirements

### Demo Requirements
- Python 3.7+
- opencv-python >= 4.5.0
- mediapipe >= 0.8.0
- numpy >= 1.21.0

### Full Project Requirements
See [requirements.txt](requirements.txt) for complete list:
- Core: numpy, opencv-python, pandas, matplotlib
- ML: scikit-learn, joblib
- Deep Learning: torch, torchvision, tensorflow, keras
- CV: mediapipe, Pillow
- Utilities: tqdm

---

## Learning Path

### Beginner
1. Start with `demo_gesture_detection.py`
2. Understand MediaPipe basics
3. Modify finger counting logic
4. Read code comments

### Intermediate
1. Collect static gesture data
2. Train Random Forest model
3. Validate dataset quality
4. Test recognition accuracy

### Advanced
1. Capture dynamic sequences
2. Train CNN temporal model
3. Implement gesture control
4. Add custom gestures

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **MediaPipe** by Google for hand detection
- **OpenCV** community for computer vision tools
- **PyTorch** team for deep learning framework
- Contributors and users of this project

---

## Contact

Have questions or suggestions? Feel free to open an issue or reach out!

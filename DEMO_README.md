# 🎯 Gesture Recognition Demo

A simple, ready-to-use demo that showcases hand detection and basic gesture recognition without requiring any pre-trained models or datasets.

## ✨ Features

- ✅ **Real-time hand detection** using MediaPipe
- ✅ **Visual hand landmarks** - see 21 key points on your hand
- ✅ **Simple finger counting** - detects 0-5 raised fingers
- ✅ **Gesture naming** - automatically names common gestures
- ✅ **Zero setup required** - no models, no datasets, no training
- ✅ **Works on Windows, Mac, and Linux**

## 🚀 Quick Start

### Option 1: Using Existing Requirements

If you already have the project dependencies installed:

```bash
python demo_gesture_detection.py
```

### Option 2: Minimal Setup (New Installation)

Create a new Python environment and install only the demo dependencies:

```bash
# Create virtual environment (optional but recommended)
python -m venv demo_env

# Activate virtual environment
# On Windows:
demo_env\Scripts\activate
# On Mac/Linux:
source demo_env/bin/activate

# Install dependencies
pip install opencv-python mediapipe numpy
```

## 🎮 How to Use

1. **Run the script:**
   ```bash
   python demo_gesture_detection.py
   ```

2. **Show your hand** to the camera

3. **Try different gestures:**
   - ✊ **Fist** (0 fingers)
   - ☝️ **One** (1 finger)
   - ✌️ **Two** (Peace sign)
   - 🤟 **Three** (3 fingers)
   - 🖖 **Four** (4 fingers)
   - 🖐️ **Open Hand** (5 fingers)

4. **Press 'q'** to quit

## 📋 What You'll See

- **Video feed** from your webcam (mirrored)
- **Hand landmarks** drawn in real-time
- **Bounding box** around detected hands
- **Gesture name** displayed on screen
- **Console output** showing detected gestures

## 🔍 How It Works

1. **MediaPipe Hands** detects hand landmarks (21 key points)
2. **Finger counting algorithm** analyzes landmark positions:
   - Compares finger tip positions with joint positions
   - Determines which fingers are raised
3. **Gesture mapping** converts finger counts to gesture names

## 🎓 For Beginners

This demo is perfect for learning about:
- Computer vision basics
- Real-time image processing
- Hand tracking with MediaPipe
- Simple gesture recognition concepts

## 🐛 Troubleshooting

**"Could not open camera"**
- Make sure your webcam is connected
- Check if another program is using the camera
- Try closing other video applications

**No hand detection**
- Ensure good lighting
- Hold hand clearly in front of camera
- Keep hand visible and not too close to edges

**Performance issues**
- Close other resource-intensive programs
- Try reducing the frame rate in the code

## 📚 Next Steps

Want to learn more? Try these advanced features:

1. **Train your own gesture classifier** - See `RandomForestTraining.py`
2. **Dynamic gesture recognition** - See `1DCNN_Training_Model.py`
3. **Full gesture control** - See `Model_Altogether.py`

## 💡 Code Structure

- `demo_gesture_detection.py` - Main demo script
- Uses MediaPipe for hand detection
- Simple algorithm for finger counting
- Clean, commented code for learning

## 🤝 Contributing

Found this helpful? Want to add features? Feel free to:
- Add more gesture recognition
- Improve the finger counting algorithm
- Add gesture recording capability
- Create visualizations

## 📝 Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe
- NumPy

## 🎉 Enjoy!

Have fun exploring gesture recognition! This is just the beginning. The full project includes machine learning models for complex gesture recognition.

---

**Note:** This demo is a simplified version that doesn't require any training data or pre-trained models. It's designed to showcase the core concepts and get you started quickly!


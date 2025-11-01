# 🚀 Quick Start: Gesture Recognition Demo

Get up and running in **under 5 minutes**!

## 📦 Step 1: Install Dependencies

### Automatic Setup (Recommended)

```bash
python setup_demo.py
```

This script will check and install all required packages automatically.

### Manual Setup

```bash
pip install opencv-python mediapipe numpy
```

## 🎮 Step 2: Run the Demo

```bash
python demo_gesture_detection.py
```

## ✋ Step 3: Try These Gestures

| Gesture | What to Do | Detected As |
|---------|------------|-------------|
| ✊ | Make a fist | "Fist" |
| ☝️ | One finger up | "One" |
| ✌️ | Two fingers (peace) | "Two (Peace)" |
| 🤟 | Three fingers up | "Three" |
| 🖖 | Four fingers up | "Four" |
| 🖐️ | Open hand | "Open Hand" |

## 🎯 What Should Happen

1. **Camera window opens** showing your video feed
2. **Hand landmarks appear** when you show your hand (21 white dots)
3. **Gesture name displays** above your hand
4. **Console prints** detected gestures every ~30 frames

## ❓ Troubleshooting

### Camera Issues

**Problem:** "Could not open camera"  
**Solution:** 
- Close other programs using the camera (Zoom, Teams, etc.)
- Try running as administrator
- Check camera permissions

### Import Errors

**Problem:** "No module named 'cv2'" or similar  
**Solution:** Run the setup script: `python setup_demo.py`

### No Detection

**Problem:** Hand not detected  
**Solution:**
- Ensure good lighting
- Keep hand in center of frame
- Don't get too close to camera
- Try making gestures slowly

## 🎓 What's Next?

After trying the basic demo, explore the full project:

### For Learners
- 📖 Read through `demo_gesture_detection.py` code
- 🧪 Try modifying the finger counting logic
- 🎨 Add custom visualizations

### For Developers
- 🤖 Train your own models with training scripts
- 📹 Record gesture datasets
- 🎯 Implement custom gesture actions

### Full Project Features
- **Static Gesture Recognition** - `RandomForestTraining.py`
- **Dynamic Gesture Recognition** - `1DCNN_Training_Model.py`
- **Complete Gesture Control** - `Model_Altogether.py`

## 📚 Files in This Demo

- `demo_gesture_detection.py` - Main demo script (run this!)
- `setup_demo.py` - Automatic dependency installer
- `DEMO_README.md` - Detailed documentation
- `QUICK_START_DEMO.md` - This file

## 🎉 Have Fun!

This is a simplified demo to get you started. The full project includes machine learning models for more complex gesture recognition and system control.

**Questions?** Check the full README.md for the complete project documentation.

---

**Pro Tip:** For best results, use good lighting and ensure you have enough room to move your hands freely!


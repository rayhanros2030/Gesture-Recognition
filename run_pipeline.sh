#!/bin/bash

# Gesture Recognition Pipeline Automation Script
# This script runs the complete gesture recognition pipeline in the correct order

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

echo "Starting Gesture Recognition Pipeline..."
echo "================================================"

# Phase 1: Data Collection
echo "PHASE 1: DATA COLLECTION"
echo "----------------------------------------"

echo "Step 1: Collecting static gesture images..."
echo "Manual step required: Run this interactively to capture gesture images"
echo "   Press 's' to capture screenshots, ESC to exit"
read -p "Press Enter when ready to collect static gesture data, or 's' to skip: " skip_static
if [[ "$skip_static" != "s" ]]; then
    python "Data_StaticGestures.py"
fi

echo ""
echo "Step 2: Collecting dynamic gesture sequences..."
echo "Manual step required: Perform gestures in front of camera"
echo "   Each gesture needs multiple sequences recorded"
read -p "Press Enter when ready to collect dynamic gesture data, or 's' to skip: " skip_dynamic
if [[ "$skip_dynamic" != "s" ]]; then
    python "Data_DynamicGestures.py"
fi

echo ""
echo "Data collection phase completed!"

# Phase 2: Model Training
echo ""
echo "PHASE 2: MODEL TRAINING"
echo "----------------------------------------"

echo "Step 3: Training static gesture classifier..."
echo "   This will create: static_landmark_model.pkl & static_label_encoder.pkl"
python "Validate_StaticGestures.py"

echo ""
echo "Step 4: Training dynamic gesture model..."
echo "   This will create: gesture_temporal_cnn_model.pth"
python "1DCNN_Training_Model.py"

echo ""
echo "Model training phase completed!"

# Phase 3: Validation & Testing
echo ""
echo "PHASE 3: VALIDATION & TESTING"
echo "----------------------------------------"

echo "Step 5: Testing dynamic gesture recognition..."
echo "Manual step: This will open webcam for real-time testing"
read -p "Press Enter to test dynamic gesture recognition, or 's' to skip: " skip_test
if [[ "$skip_test" != "s" ]]; then
    python "DynamicGesture.py"
fi

# Phase 4: Full Application
echo ""
echo "PHASE 4: FULL APPLICATION"
echo "----------------------------------------"

echo "Step 6: Running complete gesture control system..."
echo "This combines both static and dynamic gesture recognition"
echo "   Gesture combinations will trigger computer actions (e.g., open YouTube)"
echo "   - Thumbsup switches between static/dynamic modes"
echo "   - TwoFingers + Handup sequence opens YouTube"
read -p "Press Enter to start the full application, or 's' to skip: " skip_app
if [[ "$skip_app" != "s" ]]; then
    python "Model_Altogether.py"
fi

echo ""
echo "Pipeline completed successfully!"
echo "================================================"
echo ""
echo "Generated Files:"
echo "   - static_landmark_model.pkl"
echo "   - static_label_encoder.pkl" 
echo "   - gesture_temporal_cnn_model.pth"
echo ""
echo "To run just the application (after training):"
echo "   python \"Model_Altogether.py\""
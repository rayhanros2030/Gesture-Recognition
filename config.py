"""
Centralized configuration for Gesture Recognition Pipeline
"""
import os

# Model parameters
SEQUENCE_LENGTH = 65  # Standardized across all models
IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_EPOCHS = 25

# Model file paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    'static_model': os.path.join(MODEL_DIR, 'static_landmark_model.pkl'),
    'static_encoder': os.path.join(MODEL_DIR, 'static_label_encoder.pkl'),
    'dynamic_model': os.path.join(MODEL_DIR, 'gesture_temporal_cnn_model.pth')
}

# Dataset paths
DATASET_PATHS = {
    'static': 'Gesture_Dataset_Static',
    'dynamic': 'Gesture_Dataset_Dynamic'
}

# MediaPipe configuration
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Training configuration
TRAINING_CONFIG = {
    'learning_rate': 1e-4,
    'device': 'cuda',  # Will be auto-detected in code
    'save_interval': 2000
}

# Gesture control configuration
GESTURE_CONTROL = {
    'cooldown_seconds': 8,
    'mode_switch_gesture': 'Thumbsup',
    'action_sequence': ['TwoFingers', 'Handup']
}

# Platform-specific configurations
import platform
PLATFORM = platform.system().lower()
PLATFORM_ACTIONS = {
    'darwin': {  # macOS
        'open_browser': ['command', 'space']
    },
    'windows': {
        'open_browser': ['win', 'r']
    },
    'linux': {
        'open_browser': ['alt', 'f2']
    }
}
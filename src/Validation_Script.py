import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def is_frame_too_dark(frame, threshold=40):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = cv2.mean(gray)[0]
    return mean_intensity < threshold

def enhance_contrast(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def validate_sequences(root_dir, min_valid_frames=8, darkness_threshold=40):
    invalid_sequences = []
    valid_sequences = []

    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue
        for sequence in os.listdir(label_path):
            sequence_path = os.path.join(label_path, sequence)
            if not os.path.isdir(sequence_path):
                continue

            valid_frame_count = 0
            for frame_file in sorted(os.listdir(sequence_path)):
                frame_path = os.path.join(sequence_path, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                frame = enhance_contrast(frame)
                if is_frame_too_dark(frame, darkness_threshold):
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    valid_frame_count += 1

            if valid_frame_count >= min_valid_frames:
                valid_sequences.append(sequence_path)
            else:
                invalid_sequences.append(sequence_path)

    return valid_sequences, invalid_sequences

# Run this locally
dataset_path = "Gesture_Dataset_Dynamic"  # Change this!
valid, invalid = validate_sequences(dataset_path)

print("✅ Valid Sequences:", len(valid))
for path in valid:
    print("  -", path)

print("\n❌ Invalid Sequences:", len(invalid))
for path in invalid:
    print("  -", path)
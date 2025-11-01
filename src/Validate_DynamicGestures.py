import os
import cv2
import mediapipe as mp

SEQUENCE_ROOT = "Gesture_Dataset_Dynamic"
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

valid_sequences = []
invalid_sequences = []

THRESHOLD_VALID_FRAMES = 0.8  # Require at least 80% of frames to detect hands

print("🔍 Validating gesture sequences...\n")

for gesture_class in os.listdir(SEQUENCE_ROOT): #Loops through Gesture Classes, skips whatever is not a folder
    class_path = os.path.join(SEQUENCE_ROOT, gesture_class)
    if not os.path.isdir(class_path):
        continue

    for sequence_name in os.listdir(class_path): #Helps loops through the folders within the gesture classes
        sequence_path = os.path.join(class_path, sequence_name)
        if not os.path.isdir(sequence_path):
            continue

        frame_files = sorted(os.listdir(sequence_path)) # Lists all the frames inside a sequence folder
        total_frames = len(frame_files) # Counts how many frames there
        valid_frame_count = 0 # Variable to store how many valid frames

        for frame_name in frame_files: #Reads each frame using OpenCV
            frame_path = os.path.join(sequence_path, frame_name)
            image = cv2.imread(frame_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb) # Runs hand detection
            if result.multi_hand_landmarks: # If at least one hand is detected, then it's valid
                valid_frame_count += 1

        ratio = valid_frame_count / total_frames # Computes the ratio of the valid frames to the non-valid
        if ratio >= THRESHOLD_VALID_FRAMES: # If the ratio is above the threshold, then it's valid, or added to the list
            valid_sequences.append(sequence_path)
        else:
            invalid_sequences.append(sequence_path)

print(f"\n✅ VALID Sequences ({len(valid_sequences)}):")
for path in valid_sequences:
    print("  -", path)

print(f"\n❌ INVALID Sequences ({len(invalid_sequences)}):")
for path in invalid_sequences:
    print("  -", path)
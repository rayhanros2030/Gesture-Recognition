import cv2
import os
import numpy as np
import mediapipe as mp

# === SETTINGS ===
VIDEO_FOLDER = "Storage_Folder"            # Folder where your gesture .mp4 videos are stored
OUTPUT_FOLDER = "Storage_Folder"       # Folder where landmark sequences will be saved
FRAMES_PER_SEQUENCE = 20                   # Number of frames to extract per sequence

# === INIT MEDIAPIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# === CREATE OUTPUT FOLDER IF NEEDED ===
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# === LOOP THROUGH VIDEOS ===
for video_file in os.listdir(VIDEO_FOLDER):
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
        continue

    gesture_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    output_path = os.path.join(OUTPUT_FOLDER, f"{gesture_name}.npy")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // FRAMES_PER_SEQUENCE)

    sequence = []
    frame_idx = 0

    while cap.isOpened() and len(sequence) < FRAMES_PER_SEQUENCE:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                landmarks = np.array([[pt.x, pt.y, pt.z] for pt in lm.landmark]).flatten()
                sequence.append(landmarks)

        frame_idx += 1

    cap.release()

    # Save only if we have enough valid frames
    if len(sequence) == FRAMES_PER_SEQUENCE:
        np.save(output_path, np.array(sequence))
        print(f"✅ Saved: {output_path}")
    else:
        print(f"❌ Skipped (insufficient frames): {video_file}")

print("\n🎉 Done extracting landmark sequences.")
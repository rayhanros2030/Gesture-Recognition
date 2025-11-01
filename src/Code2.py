import cv2
import time
import numpy as np
import subprocess
import platform
import pyautogui
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cooldown time between gesture changes
COOLDOWN_TIME = 2  # Reduced for faster registration

# State variables
gesture_sequence = []
last_gesture_time = 0
last_detected_fingers = None

# Define function to count fingers
def count_fingers(landmarks):
    tips = [8, 12, 16, 20]
    count = 0
    if landmarks[4].x < landmarks[3].x:  # Thumb
        count += 1
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1
    return count

# Command executor
def execute_gesture_command(sequence):
    if sequence == [2, 0]:
        print("🎯 Two fingers then fist detected → Open YouTube in Chrome")
        open_chrome("https://www.youtube.com")
    elif sequence == [0]:
        print("🖐️ Fist alone detected → Open Chrome")
        open_chrome()

# OS-specific Chrome opener
def open_chrome(url=None):
    system = platform.system()
    if system == "Windows":
        command = "start chrome"
    elif system == "Darwin":
        command = "open -a 'Google Chrome'"
    else:
        command = "google-chrome"

    if url:
        command += f" {url}"

    subprocess.Popen(command, shell=True)

# Start webcam and recognition
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        now = time.time()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                fingers_up = count_fingers(landmarks)

                # Only register gesture if it changes from the last and cooldown is passed
                if (fingers_up != last_detected_fingers and now - last_gesture_time > COOLDOWN_TIME):
                    gesture_sequence.append(fingers_up)
                    print(f"✅ New gesture: {fingers_up} fingers")
                    last_gesture_time = now
                    last_detected_fingers = fingers_up

                    # Check if a valid sequence was completed
                    if len(gesture_sequence) >= 2 or (len(gesture_sequence) == 1 and fingers_up == 0):
                        execute_gesture_command(gesture_sequence)
                        gesture_sequence.clear()

        cv2.imshow("Compositional Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
"""
GESTURE RECOGNITION DEMO
========================
A simple demo that shows hand detection and tracking with gesture-based app control.

Features:
- Real-time hand landmark detection using MediaPipe
- Visual feedback showing detected hand landmarks
- Basic gesture count (detected fingers)
- Application control based on gestures
- No pre-trained models required - works out of the box!

Instructions:
1. Install dependencies: pip install opencv-python mediapipe numpy pyautogui
2. Run: python demo_gesture_detection.py
3. Show your hand to the camera
4. Use gestures to control apps
5. Press 'q' to quit

Gesture Actions:
- 1 finger  = Open Notepad/Text Editor
- 2 fingers = Open Calculator
- 3 fingers = Open YouTube
- 5 fingers = Open Google
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import subprocess
import platform

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def count_fingers(landmarks):
    """
    Simple finger counting based on landmark positions.
    Returns count of raised fingers (0-5).
    """
    # Finger tip IDs
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    
    # Finger pip IDs (joints that need to be lower for finger to be raised)
    pips = [3, 6, 10, 14, 18]
    
    count = 0
    
    # Check thumb separately (x-coordinate comparison)
    if landmarks.landmark[tips[0]].x > landmarks.landmark[pips[0]].x:
        count += 1
    
    # Check other fingers (y-coordinate comparison)
    for i in range(1, 5):
        if landmarks.landmark[tips[i]].y < landmarks.landmark[pips[i]].y:
            count += 1
    
    return count

def get_gesture_name(finger_count):
    """Map finger count to gesture names."""
    gestures = {
        0: "Fist",
        1: "One",
        2: "Two (Peace)",
        3: "Three",
        4: "Four",
        5: "Open Hand"
    }
    return gestures.get(finger_count, f"{finger_count} Fingers")

def trigger_app_action(finger_count, last_trigger_time, COOLDOWN=3):
    """Trigger application actions based on finger count gesture."""
    current_time = time.time()
    
    # Check cooldown
    if current_time - last_trigger_time < COOLDOWN:
        return last_trigger_time
    
    # Prevent multiple rapid triggers
    if hasattr(trigger_app_action, 'last_count'):
        if trigger_app_action.last_count == finger_count:
            return last_trigger_time
    trigger_app_action.last_count = finger_count
    
    system = platform.system().lower()
    
    # Different actions for different finger counts
    if finger_count == 1:  # One finger - Open Notepad
        print("Action: Opening Notepad...")
        if system == 'windows':
            subprocess.Popen(['notepad.exe'])
        elif system == 'darwin':  # macOS
            subprocess.Popen(['open', '-a', 'TextEdit'])
        elif system == 'linux':
            subprocess.Popen(['gedit'])
        return current_time
    
    elif finger_count == 2:  # Two fingers - Open Calculator
        print("Action: Opening Calculator...")
        if system == 'windows':
            subprocess.Popen(['calc.exe'])
        elif system == 'darwin':  # macOS
            subprocess.Popen(['open', '-a', 'Calculator'])
        elif system == 'linux':
            subprocess.Popen(['gnome-calculator'])
        return current_time
    
    elif finger_count == 3:  # Three fingers - Open YouTube
        print("Action: Opening YouTube...")
        if system == 'windows':
            subprocess.Popen(['start', 'https://www.youtube.com'], shell=True)
        elif system == 'darwin':  # macOS
            subprocess.Popen(['open', 'https://www.youtube.com'])
        elif system == 'linux':
            subprocess.Popen(['xdg-open', 'https://www.youtube.com'])
        return current_time
    
    elif finger_count == 5:  # Open Hand - Open Google
        print("Action: Opening Google...")
        if system == 'windows':
            subprocess.Popen(['start', 'https://www.google.com'], shell=True)
        elif system == 'darwin':  # macOS
            subprocess.Popen(['open', 'https://www.google.com'])
        elif system == 'linux':
            subprocess.Popen(['xdg-open', 'https://www.google.com'])
        return current_time
    
    return last_trigger_time

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Configure MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Detect up to 2 hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        print("Camera initialized. Show your hand to the camera!")
        print("Press 'q' to quit")
        print("Gesture Actions:")
        print("  1 finger  = Open Notepad")
        print("  2 fingers = Open Calculator")
        print("  3 fingers = Open YouTube")
        print("  5 fingers = Open Google")
        print("=" * 60)
        
        frame_count = 0
        last_trigger_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(frame_rgb)
            
            # Draw information on frame
            h, w, _ = frame.shape
            
            # Add title
            cv2.putText(frame, "Gesture Recognition Demo", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Show your hand! | Press 'q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            
            # Draw landmarks and connections for each detected hand
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get hand classification (Left or Right)
                    hand_label = results.multi_handedness[idx].classification[0].label
                    hand_score = results.multi_handedness[idx].classification[0].score
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Count fingers
                    finger_count = count_fingers(hand_landmarks)
                    gesture_name = get_gesture_name(finger_count)
                    
                    # Get bounding box
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    
                    x_min = int(min(x_coords) * w)
                    x_max = int(max(x_coords) * w)
                    y_min = int(min(y_coords) * h)
                    y_max = int(max(y_coords) * h)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x_min - 10, y_min - 40), (x_max + 10, y_max + 10),
                                 (0, 255, 0), 2)
                    
                    # Display information
                    info_y = y_min - 50 if y_min > 50 else y_max + 30
                    cv2.putText(frame, f"{hand_label} Hand", 
                               (x_min - 10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Gesture: {gesture_name}", 
                               (x_min - 10, info_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Also print to console every 30 frames
                    if frame_count % 30 == 0:
                        print(f"{hand_label} Hand - {gesture_name}")
                    
                    # Trigger app action based on gesture (only on first hand detected)
                    if idx == 0:
                        last_trigger_time = trigger_app_action(finger_count, last_trigger_time)
            else:
                # No hand detected
                if frame_count % 30 == 0:
                    print("No hand detected - Show your hand to the camera")
            
            # Display frame
            cv2.imshow("Gesture Recognition Demo", frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nDemo ended by user")
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Thanks for trying the demo!")

if __name__ == "__main__":
    main()


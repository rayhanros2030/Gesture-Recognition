import cv2
import os
import mediapipe as mp

# === Settings ===
gesture_name = "Handsup"            # Change this for each gesture
sequence_id = 17
# Change this for each new recording
output_dir = f"Storage_Folder/{gesture_name}/Seq_{sequence_id}"
os.makedirs(output_dir, exist_ok=True)

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# === Camera setup ===
cap = cv2.VideoCapture(0)
frame_count = 0
max_frames = 65  # Number of frames to save

print("📹 Recording gesture:", gesture_name)

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 🔄 Flip horizontally (mirror view)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        annotated = frame.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        save_path = os.path.join(output_dir, f"frame_{frame_count:03d}.jpg")
        cv2.imwrite(save_path, annotated)
        frame_count += 1
        cv2.imshow("Recording (Flipped)", annotated)
    else:
        cv2.imshow("Recording (Flipped)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done! Frames saved to:", output_dir)
import cv2
import torch
import joblib
import pyautogui
import time
import numpy as np
from PIL import Image
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp
import os

# === Static Gesture Setup ===
STATIC_MODEL_PATH = "static_landmark_model.pkl"
STATIC_ENCODER_PATH = "static_label_encoder.pkl"

static_model = joblib.load(STATIC_MODEL_PATH)
static_encoder: LabelEncoder = joblib.load(STATIC_ENCODER_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# === Dynamic Gesture Setup ===
SEQUENCE_LENGTH = 65
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# === CNN Encoder (Same as training) ===
class CNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.feature_extractor = base.features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

class TemporalConvNet(torch.nn.Module):
    def __init__(self, input_size=1280, num_classes=3):
        super().__init__()
        self.temporal = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.squeeze(-1)
        return self.classifier(x)

def load_dynamic_model(path="gesture_temporal_cnn_model.pth"):
    checkpoint = torch.load(path, map_location=DEVICE)
    cnn = CNNEncoder().to(DEVICE)
    temporal = TemporalConvNet(num_classes=len(checkpoint['label_map'])).to(DEVICE)
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    temporal.load_state_dict(checkpoint['temporal_state_dict'])
    cnn.eval()
    temporal.eval()
    return cnn, temporal, checkpoint['label_map']

# === Helper to extract landmarks ===
def extract_static_landmarks(image_rgb):
    result = hands.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None
    hand_landmarks = result.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)

# === Recognize Static Gesture ===
def recognize_static(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = extract_static_landmarks(img_rgb)
    if landmarks is None:
        return None
    pred = static_model.predict([landmarks])[0]
    return static_encoder.inverse_transform([pred])[0]

# === Recognize Dynamic Gesture ===
def recognize_dynamic(cap, cnn, temporal, label_map):
    inv_map = {v: k for k, v in label_map.items()}
    frames = []

    while len(frames) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        img_tensor = transform(pil_img)
        frames.append(img_tensor)

        cv2.putText(frame, f"Capturing dynamic gesture {len(frames)}/{SEQUENCE_LENGTH}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    seq_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)
    B, T, C, H, W = seq_tensor.shape
    seq_tensor = seq_tensor.view(B*T, C, H, W)

    with torch.no_grad():
        features = cnn(seq_tensor)
        features = features.view(B, T, -1)
        output = temporal(features)
        pred_idx = torch.argmax(output, dim=1).item()

    return inv_map[pred_idx]

# === Trigger Action ===
def trigger_action(gesture_seq):
    if gesture_seq == ["TwoFingers", "Handup"]:
        pyautogui.hotkey("command", "space")
        pyautogui.write("chrome")
        pyautogui.press("enter")
        time.sleep(2)
        pyautogui.write("https://www.youtube.com")
        pyautogui.press("enter")
        print("🎬 YouTube launched!")

# === Main Control ===
def run_gesture_control():
    cap = cv2.VideoCapture(0)
    mode = "static"
    cooldown_start = None
    sequence = []
    cnn, temporal, label_map = load_dynamic_model()

    print("🎯 Gesture control started. Thumbsup switches mode. Combo: TwoFingers + Handup")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        if cooldown_start and time.time() - cooldown_start < 8:
            remaining = int(8 - (time.time() - cooldown_start))
            cv2.putText(frame, f"Cooldown: {remaining}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        else:
            cooldown_start = None

        if mode == "static":
            gesture = recognize_static(frame)
            if gesture == "Thumbsup":
                print(f"🔀 Thumbsup detected! Switching to DYNAMIC mode...")
                mode = "dynamic"
                cooldown_start = time.time()
                continue
            elif gesture:
                print(f"🧊 Static Gesture: {gesture}")
                sequence.append(gesture)
                cooldown_start = time.time()

        elif mode == "dynamic":
            gesture = recognize_dynamic(cap, cnn, temporal, label_map)
            if gesture:
                print(f"🌀 Dynamic Gesture: {gesture}")
                sequence.append(gesture)
                cooldown_start = time.time()
            mode = "static"

        if len(sequence) == 2:
            print(f"✅ Gesture Sequence: {sequence}")
            trigger_action(sequence)
            sequence = []

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gesture_control()

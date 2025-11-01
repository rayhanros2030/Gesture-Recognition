import torch
import torch.nn as nn
import cv2
import time
from torchvision import transforms, models
from PIL import Image

# === Config ===
SEQUENCE_LENGTH = 32       # ✅ Reduced for more stability
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COOLDOWN_SECONDS = 3

# === CNN Encoder (same as training) ===
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

# === Temporal CNN with Dropout ===
class TemporalConvNet(nn.Module):
    def __init__(self, input_size=1280, num_classes=3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # ✅ Dropout to reduce overfitting
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):  # [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.temporal(x)  # [B, 128, 1]
        x = x.squeeze(-1)     # [B, 128]
        return self.classifier(x)  # [B, num_classes]

# === Load Model ===
def load_model(model_path='gesture_temporal_cnn_model.pth'):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    label_map = checkpoint['label_map']
    num_classes = len(label_map)

    cnn = CNNEncoder().to(DEVICE)
    temporal_cnn = TemporalConvNet(num_classes=num_classes).to(DEVICE)

    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    temporal_cnn.load_state_dict(checkpoint['temporal_state_dict'])

    cnn.eval()
    temporal_cnn.eval()
    return cnn, temporal_cnn, label_map

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# === Predict Live Gesture from Webcam ===
def predict_live_gesture():
    cnn, temporal_cnn, label_map = load_model()
    inv_map = {v: k for k, v in label_map.items()}

    cap = cv2.VideoCapture(0)
    print("🎥 Webcam started. Press 'q' to quit.\n")
    cooldown_start = None

    while True:
        # === Cooldown Timer ===
        if cooldown_start is not None:
            elapsed = time.time() - cooldown_start
            if elapsed < COOLDOWN_SECONDS:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"🕓 Cooldown: {COOLDOWN_SECONDS - int(elapsed)}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Temporal Gesture Recognizer", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            else:
                cooldown_start = None

        # === Collect Sequence ===
        frames = []
        print("🖐️ Show your gesture...")

        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_tensor = transform(pil_img)
            frames.append(img_tensor)
            cv2.putText(frame, f"Capturing frame {len(frames)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Temporal Gesture Recognizer", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # === Predict ===
        sequence_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)  # [1, T, 3, H, W]
        B, T, C, H, W = sequence_tensor.shape
        sequence_tensor = sequence_tensor.view(B*T, C, H, W)

        with torch.no_grad():
            features = cnn(sequence_tensor)  # [B*T, 1280]
            features = features.view(B, T, -1)  # [B, T, F]
            output = temporal_cnn(features)     # [B, num_classes]
            pred_idx = torch.argmax(output, dim=1).item()

            # ✅ Add softmax confidence
            probs = torch.softmax(output, dim=1)[0]
            conf = probs[pred_idx].item()

        gesture = inv_map[pred_idx]
        print(f"🧠 Predicted Gesture: {gesture} | Confidence: {conf:.2f}")
        cooldown_start = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_live_gesture()

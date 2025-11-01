import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import joblib

# Parameters
GESTURE_FOLDER = "Gesture_Dataset"
SEQUENCE_LENGTH = 25
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Step 1: Extract landmarks from videos
def extract_landmark_sequence(video_path, sequence_length=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    while len(sequence) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            coords = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
            sequence.append(np.array(coords).flatten())
    cap.release()
    return np.array(sequence) if len(sequence) == sequence_length else None

# Step 2: Load data
X, y = [], []
labels = os.listdir(GESTURE_FOLDER)
for label in labels:
    folder_path = os.path.join(GESTURE_FOLDER, label)
    for file in os.listdir(folder_path):
        if file.endswith(".mp4") or file.endswith(".mov"):
            video_path = os.path.join(folder_path, file)
            seq = extract_landmark_sequence(video_path)
            if seq is not None:
                X.append(seq)
                y.append(label)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X = np.array(X)
y_encoded = np.array(y_encoded)

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 4: Dataset and Dataloader
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = GestureDataset(X_train, y_train)
test_dataset = GestureDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Step 5: Define LSTM Model
class GestureLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=len(le.classes_)):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out

model = GestureLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Step 6: Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Save model and encoder
torch.save(model.state_dict(), "gesture_lstm_model_retrained.pth")
joblib.dump(le, "gesture_label_encoder.pkl")
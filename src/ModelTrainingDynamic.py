import os
import cv2
import numpy as np
import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import mediapipe as mp
from collections import Counter

# === Parameters ===
dataset_dir = "Gesture_Dataset_Dynamic"
sequence_length = 21  # number of frames per gesture sequence

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# === Load sequences ===
X = []
y = []

for gesture_label in os.listdir(dataset_dir):
    gesture_path = os.path.join(dataset_dir, gesture_label)
    if not os.path.isdir(gesture_path):
        continue

    for sequence_folder in os.listdir(gesture_path):
        seq_path = os.path.join(gesture_path, sequence_folder)
        if not os.path.isdir(seq_path):
            continue

        sequence_landmarks = []

        for frame_name in sorted(os.listdir(seq_path)):
            frame_path = os.path.join(seq_path, frame_name)
            image = cv2.imread(frame_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image)
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                coords = []
                for pt in lm.landmark:
                    coords.extend([pt.x, pt.y, pt.z])
                if len(coords) == 63:
                    sequence_landmarks.append(coords)

        if len(sequence_landmarks) == sequence_length:
            flat_sequence = np.array(sequence_landmarks).flatten()  # shape: (63*21,)
            X.append(flat_sequence)
            y.append(gesture_label)

# === Count before filtering ===
print("\n📊 Gesture sequence counts (before filtering):")
gesture_counts = Counter(y)
for label, count in gesture_counts.items():
    print(f"{label}: {count} sequences")

# === Filter out gesture classes with <2 sequences ===
filtered_X, filtered_y = [], []
for xi, yi in zip(X, y):
    if gesture_counts[yi] >= 2:
        filtered_X.append(xi)
        filtered_y.append(yi)

X = filtered_X
y = filtered_y

# === Count after filtering ===
print("\n✅ Gesture sequence counts (after filtering):")
final_counts = Counter(y)
for label, count in final_counts.items():
    print(f"{label}: {count} sequences")

# === Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === Train SVM ===
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("\n🧪 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Save ===
joblib.dump(clf, "landmark_sequence_svm_model.pkl")
joblib.dump(le, "landmark_sequence_label_encoder.pkl")

print("✅ Model and encoder saved!")
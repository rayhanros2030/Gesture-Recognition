import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

def extract_landmarks_from_image(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return None

# === Modify this to your actual dataset path ===
dataset_path = "Gesture_Dataset_Static"

X, y = [], []

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        img = cv2.imread(file_path)
        if img is None:
            continue
        landmarks = extract_landmarks_from_image(img)
        if landmarks is not None:
            X.append(landmarks)
            y.append(label)

X = np.array(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

n_estimators_list = [5, 10, 20, 50, 100, 200]
train_acc = []
test_acc = []

for n in n_estimators_list:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)

    train_acc.append(clf.score(X_train, y_train))
    test_acc.append(clf.score(X_test, y_test))

# === Plotting ===
plt.figure(figsize=(10,5))
plt.plot(n_estimators_list, train_acc, label='Train Accuracy')
plt.plot(n_estimators_list, test_acc, label='Test Accuracy')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees in Random Forest')
plt.legend()
plt.grid(True)
plt.show()

print("Classification Report:")
print(classification_report(y_test, clf.predict(X_test), target_names=label_encoder.classes_))

# Save model and label encoder
joblib.dump(clf, "static_landmark_model.pkl")
joblib.dump(label_encoder, "static_label_encoder.pkl")
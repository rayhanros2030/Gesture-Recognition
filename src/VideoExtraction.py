import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# === Config ===
SEQUENCE_LENGTH = 40
IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = "Gesture_Dataset_Dynamic"

# === CNN Encoder ===
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.feature_extractor = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x  # [B*T, 1280]

# === Temporal CNN ===
class TemporalConvNet(nn.Module):
    def __init__(self, input_size=1280, num_classes=3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_size, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):  # [B, T, F]
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.squeeze(-1)
        return self.classifier(x)

# === Dataset Loader ===
class GestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(root_dir)))}
        global NUM_CLASSES
        NUM_CLASSES = len(self.label_map)

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for seq in os.listdir(label_path):
                    seq_path = os.path.join(label_path, seq)
                    if os.path.isdir(seq_path):
                        frames = sorted(os.listdir(seq_path))[:SEQUENCE_LENGTH]
                        frame_paths = [os.path.join(seq_path, f) for f in frames]
                        if len(frame_paths) == SEQUENCE_LENGTH:
                            self.samples.append(frame_paths)
                            self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths = self.samples[idx]
        label = self.labels[idx]
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        frames_tensor = torch.stack(frames)
        return frames_tensor, label

# === Training Setup ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = GestureDataset(DATASET_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

cnn = CNNEncoder().to(DEVICE)
temporal_cnn = TemporalConvNet(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(cnn.parameters()) + list(temporal_cnn.parameters()), lr=1e-4)

# === Training Loop ===
train_losses = []
train_accuracies = []

print("🧠 Starting Temporal CNN training...")
for epoch in range(NUM_EPOCHS):
    cnn.train()
    temporal_cnn.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in dataloader:
        B, T, C, H, W = sequences.shape
        sequences = sequences.view(B*T, C, H, W).to(DEVICE)
        labels = labels.to(DEVICE)

        features = cnn(sequences)
        features = features.view(B, T, -1)
        outputs = temporal_cnn(features)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# === Save the Model ===
torch.save({
    'cnn_state_dict': cnn.state_dict(),
    'temporal_state_dict': temporal_cnn.state_dict(),
    'label_map': dataset.label_map
}, "gesture_temporal_cnn_model.pth")

# === Plot Training Curve ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, marker='o')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()
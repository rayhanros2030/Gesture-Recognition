import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ======== CONFIG ========
from config import SEQUENCE_LENGTH, IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, DATASET_PATHS
DATASET_DIR = DATASET_PATHS['dynamic']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== DATASET ========
class GestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = [] # Paths to image frames in each sequence
        self.labels = [] # Class index for each sequence
        self.transform = transform # Applies Resizing, tensor Conservation, etc.
        self.label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(root_dir)))} # Assigns a unique index to each gesture class
        global NUM_CLASSES
        NUM_CLASSES = len(self.label_map)

        for label in os.listdir(root_dir): # goes through each gesture class folder, then each sequence
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for seq in os.listdir(label_path):
                    seq_path = os.path.join(label_path, seq)
                    if os.path.isdir(seq_path):
                        frames = sorted(os.listdir(seq_path))[:SEQUENCE_LENGTH] # Sorts the frames alphabetically
                        frame_paths = [os.path.join(seq_path, f) for f in frames] # Takes the filenames, so that it has a path towards the image
                        if len(frame_paths) == SEQUENCE_LENGTH:
                            self.samples.append(frame_paths) # Stores list of frame paths into dataset
                            self.labels.append(self.label_map[label]) # Stores the class index into a list

    def __len__(self):
        return len(self.samples) # Returns the number of valid sequences

    def __getitem__(self, idx):
        frame_paths = self.samples[idx] # gets the list of full image paths
        label = self.labels[idx] # Gets numerical number for the images
        frames = [] # list

        for path in frame_paths:
            img = Image.open(path).convert('RGB') # Loads image from disk, important if it is in greyscale
            if self.transform:
                img = self.transform(img) #Ensures all images have the same format or same shape, by seeing if there was any reshaping or anything
            frames.append(img)

        frames_tensor = torch.stack(frames)  # [T, 3, H, W], Converts list of individual image tensors into one big tensor
        return frames_tensor, label

# ======== CNN Feature Extractor ========
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights='IMAGENET1K_V1') #Loads pre-trained MobileNet Architecture from Torchvision Models
        self.feature_extractor = base.features # Outputs high-level feature maps from input images, helps give leniency
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Compresses feature maps, ensures that the model outputs a consistent feature size

    def forward(self, x):  # [B*T, 3, H, W], defines how input images pass through the CNN to get features
        x = self.feature_extractor(x) # Extracts spatial patterns from images
        x = self.pool(x) # Turns the images into fixed size vectorsaC
        x = x.view(x.size(0), -1)
        return x  # [B*T, 1280]

# ======== Temporal CNN (1D Conv) ========
class TemporalConvNet(nn.Module): # handles temporal images
    def __init__(self, input_size=1280, num_classes=3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1), # Learns high-level temporal sequences
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes) # Layer performs the final gesture recognition, maps the 128-dim pooled vector to your gesture class

    def forward(self, x):  # [B, T, F]
        x = x.transpose(1, 2)  # → [B, F, T]
        x = self.temporal(x)  # → [B, 128, 1]
        x = x.squeeze(-1)     # → [B, 128], need a flat vector
        return self.classifier(x)  # → [B, num_classes]

# ======== TRAINING SETUP ========
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
]) # Expects all images to have the same size

dataset = GestureDataset(DATASET_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

cnn = CNNEncoder().to(DEVICE) # Extracts per-frame visuals
temporal_cnn = TemporalConvNet(num_classes=len(dataset.label_map)).to(DEVICE) # Forms complete dynamic gesture recognition, which learns patterns over time
criterion = nn.CrossEntropyLoss() # Defines loss function
optimizer = torch.optim.Adam(list(cnn.parameters()) + list(temporal_cnn.parameters()), lr=1e-4) # Adam Optimizer, adjusts which weights to change

# ======== TRAINING LOOP ========
print("🧠 Starting Temporal CNN training...")
for epoch in range(NUM_EPOCHS):
    cnn.train()
    temporal_cnn.train()
    total_loss = 0

    for sequences, labels in dataloader:
        B, T, C, H, W = sequences.shape # Makes the shape explicitly so you can manipulate it
        sequences = sequences.view(B*T, C, H, W).to(DEVICE) # reshapes the sequence so all frames become one batch
        labels = labels.to(DEVICE)

        features = cnn(sequences)  # [B*T, 1280]
        features = features.view(B, T, -1)  # [B, T, 1280]

        outputs = temporal_cnn(features)  # [B, num_classes]
        loss = criterion(outputs, labels)

        optimizer.zero_grad() # Clears out old gradients
        loss.backward()
        optimizer.step() # Takes steps, learns
        total_loss += loss.item()

    print(f"📍 Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss:.4f}")

# ======== SAVE MODEL ========
torch.save({
    'cnn_state_dict': cnn.state_dict(),
    'temporal_state_dict': temporal_cnn.state_dict(),
    'label_map': dataset.label_map,
}, 'gesture_temporal_cnn_model.pth')

print("✅ Model saved as gesture_temporal_cnn_model.pth")


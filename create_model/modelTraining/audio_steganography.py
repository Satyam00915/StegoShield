import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Updated Paths
dataset_path = "C:/old/college/sem 6/Special Project/Project/StegoShield/dataset/split_data/split_audio"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
save_model_path = os.path.join(os.getcwd(), "backend/models/audiol.pth")

# Function to convert waveform to spectrogram
def waveform_to_spectrogram(waveform, sr=22050, n_mels=128, target_shape=(128, 300)):
    n_fft = min(2048, len(waveform))
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
    return pad_spectrogram(spectrogram, target_shape)

# Function to pad spectrogram
def pad_spectrogram(spectrogram, target_shape=(128, 300)):
    _, height, width = spectrogram.shape
    if width < target_shape[1]:
        pad_width = target_shape[1] - width
        spectrogram = F.pad(spectrogram, (0, pad_width), mode='constant', value=0)
    elif width > target_shape[1]:
        spectrogram = spectrogram[:, :, :target_shape[1]]
    return spectrogram

# Audio Dataset Class
class AudioDataset(Dataset):
    def __init__(self, data_dir, sr=22050, n_mels=128):
        self.data_dir = data_dir
        self.sr = sr
        self.n_mels = n_mels
        self.file_paths = []
        self.labels = []
        
        for label, subdir in enumerate(["clean", "stego"]):
            full_path = os.path.join(data_dir, subdir)
            for file in os.listdir(full_path):
                self.file_paths.append(os.path.join(full_path, file))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        waveform, _ = librosa.load(file_path, sr=self.sr)
        spectrogram = waveform_to_spectrogram(waveform, sr=self.sr, n_mels=self.n_mels)
        return spectrogram, torch.tensor(label, dtype=torch.long)

# CNN Model
class AudioStegoCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioStegoCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        dummy_input = torch.randn(1, 1, 128, 300)
        out = self.cnn(dummy_input)
        flattened_size = out.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Train Function with Model Accuracy Calculation
def train(model, train_loader, val_loader, epochs=20, lr=0.0001, weight_decay=1e-4):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        scheduler.step(avg_train_loss)
    
        # Model Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy:.2f}%")
    
    torch.save(model.state_dict(), save_model_path)
    print(f"Training complete. Model saved at {save_model_path}")

# Load Dataset & Create Dataloaders
if __name__ == "__main__":
    train_dataset = AudioDataset(train_path)
    val_dataset = AudioDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    num_classes = 2
    model = AudioStegoCNN(num_classes)
    train(model, train_loader, val_loader)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
dataset_path = "dataset/audio"
clean_path = os.path.join(dataset_path, "clean")
stego_path = os.path.join(dataset_path, "stego")
labels_file = os.path.join(dataset_path, "labels.csv")
save_model_path = os.path.join(os.getcwd(), "audio_stego_model.pth")

# Data Augmentation
def add_noise(waveform, noise_level=0.005):
    noise = noise_level * np.random.randn(len(waveform))
    return waveform + noise

def time_shift(waveform, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(waveform))
    return np.roll(waveform, shift)

def pitch_shift(waveform, sr=22050, n_steps=2):
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)

def time_stretch(waveform, rate=0.8):
    return librosa.effects.time_stretch(waveform, rate)


def waveform_to_spectrogram(waveform, sr=22050, n_mels=128, target_shape=(128, 300)):
    waveform = add_noise(time_shift(pitch_shift(time_stretch(waveform))))
    n_fft = min(2048, len(waveform))
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
    return pad_spectrogram(spectrogram, target_shape)

def pad_spectrogram(spectrogram, target_shape=(128, 300)):
    _, height, width = spectrogram.shape
    if width < target_shape[1]:
        pad_width = target_shape[1] - width
        spectrogram = F.pad(spectrogram, (0, pad_width), mode='constant', value=0)
    elif width > target_shape[1]:
        spectrogram = spectrogram[:, :, :target_shape[1]]
    return spectrogram

# Audio Dataset
class AudioDataset(Dataset):
    def __init__(self, clean_dir, stego_dir, labels_file, sr=22050, n_mels=128):
        self.clean_dir = clean_dir
        self.stego_dir = stego_dir
        self.sr = sr
        self.n_mels = n_mels
        self.labels = self._load_labels(labels_file)
        self.file_names = list(self.labels.keys())
    
    def _load_labels(self, labels_file):
        df = pd.read_csv(labels_file)
        return {row["filename"]: 0 if row["label"] == "clean" else 1 for _, row in df.iterrows()}
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        filename = self.file_names[idx]
        label = self.labels[filename]
        audio_path = os.path.join(self.clean_dir if label == 0 else self.stego_dir, filename)
        waveform, _ = librosa.load(audio_path, sr=self.sr)
        spectrogram = waveform_to_spectrogram(waveform, sr=self.sr, n_mels=self.n_mels)
        return spectrogram, torch.tensor(label, dtype=torch.long)

# Enhanced CNN Model
class AudioStegoCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioStegoCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Dropout(0.5)
        )
        
        self.fc1 = nn.Linear(128 * 16 * 37, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train Function
def train(model, dataloader, epochs=20, lr=0.001, weight_decay=1e-4):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        scheduler.step()
    
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved at {save_model_path}")

# Load Dataset & Train Model
if __name__ == "__main__":
    audio_dataset = AudioDataset(clean_dir=clean_path, stego_dir=stego_path, labels_file=labels_file)
    audio_loader = DataLoader(audio_dataset, batch_size=16, shuffle=True)
    audio_model = AudioStegoCNN()
    train(audio_model, audio_loader)
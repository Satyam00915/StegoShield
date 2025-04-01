import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
test_data_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\dataset\split_data\split_audio\test"
model_path = "audio_model.pth"

# Function to convert waveform to spectrogram
def waveform_to_spectrogram(waveform, sr=22050, n_mels=128, target_shape=(128, 300)):
    n_fft = min(2048, len(waveform))
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Convert to tensor
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

    # Adjust shape: Padding or Cropping
    if spectrogram.shape[1] < target_shape[1]:  
        pad = target_shape[1] - spectrogram.shape[1]
        spectrogram = torch.nn.functional.pad(spectrogram, (0, pad))  # Pad on the right
    else:
        spectrogram = spectrogram[:, :target_shape[1]]  # Crop if too long
    
    return spectrogram.unsqueeze(0)  # Adding channel dimension

# Audio Dataset Class for Testing
class AudioTestDataset(Dataset):
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

# Load Model
class AudioStegoCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(AudioStegoCNN, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.5)
        )
        self.fc1 = torch.nn.Linear(64 * 16 * 37, 128)  # Adjusted based on spectrogram size
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load Model Weights
model = AudioStegoCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load Test Dataset
test_dataset = AudioTestDataset(test_data_path)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Test the Model
correct = 0
total = 0

with torch.no_grad():
    for spectrograms, labels in test_loader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        outputs = model(spectrograms)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")

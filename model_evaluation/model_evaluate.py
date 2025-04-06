import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import os
import cv2
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Paths
image_test_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\dataset\split_data\split_images\test"
audio_test_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\dataset\split_data\split_audio\test"
video_test_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\dataset\split_data\split_videos\test"

# Model Paths
image_model_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\saved_models\image_model.pth"
audio_model_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\saved_models\audio_model.pth"
video_model_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\saved_models\video_model.pth"

# Transformation for Image Data
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset Classes
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        for label, subdir in enumerate(["clean", "stego"]):
            path = os.path.join(root_dir, subdir)
            for file in os.listdir(path):
                self.data.append((os.path.join(path, file), label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        return image, torch.tensor(label, dtype=torch.long)

class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        for label, subdir in enumerate(["clean", "stego"]):
            path = os.path.join(root_dir, subdir)
            for file in os.listdir(path):
                self.data.append((os.path.join(path, file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        y, sr = librosa.load(audio_path, sr=16000)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0)  # (1, freq, time)
        return mel_spectrogram, torch.tensor(label, dtype=torch.long)

class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        for label, subdir in enumerate(["clean", "stego"]):
            path = os.path.join(root_dir, subdir)
            for file in os.listdir(path):
                self.data.append((os.path.join(path, file), label))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < 10:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)
        cap.release()
        if len(frames) < 10:
            frames.extend([frames[-1]] * (10 - len(frames)))
        frames = torch.stack(frames)
        return frames, torch.tensor(label, dtype=torch.long)

# Load Datasets & Dataloaders
image_test_dataset = ImageDataset(image_test_path)
audio_test_dataset = AudioDataset(audio_test_path)
video_test_dataset = VideoDataset(video_test_path)

image_test_loader = DataLoader(image_test_dataset, batch_size=8, shuffle=False)
audio_test_loader = DataLoader(audio_test_dataset, batch_size=8, shuffle=False)
video_test_loader = DataLoader(video_test_dataset, batch_size=4, shuffle=False)

# Model Classes
class ImageStegoModel(nn.Module):
    def __init__(self):
        super(ImageStegoModel, self).__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 2)

    def forward(self, x):
        return self.cnn(x)

class AudioStegoModel(nn.Module):
    def __init__(self):
        super(AudioStegoModel, self).__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 2)

    def forward(self, x):
        return self.cnn(x)

class VideoStegoModel(nn.Module):
    def __init__(self):
        super(VideoStegoModel, self).__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.shape
        cnn_features = [self.cnn(x[:, t, :, :, :]) for t in range(timesteps)]
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        return self.fc(lstm_out[:, -1, :])

# Function to Load Model
def load_model(model_class, model_path):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to Evaluate Model
def evaluate_model(model, test_loader):
    model.to(device)
    model.eval()
    
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Compute Metrics
    acc = accuracy_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions, average='weighted') * 100
    recall = recall_score(all_labels, all_predictions, average='weighted') * 100
    f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
    cm = confusion_matrix(all_labels, all_predictions)

    # Print results
    print(f"✅ Accuracy: {acc:.2f}%")
    print(f"🎯 Precision: {precision:.2f}%")
    print(f"🔄 Recall: {recall:.2f}%")
    print(f"🔥 F1 Score: {f1:.2f}%")

    # Plot Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Clean", "Stego"], yticklabels=["Clean", "Stego"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Load & Evaluate Models
image_model = load_model(ImageStegoModel, image_model_path)
audio_model = load_model(AudioStegoModel, audio_model_path)
video_model = load_model(VideoStegoModel, video_model_path)

print("\n📸 Evaluating Image Model...")
evaluate_model(image_model, image_test_loader)

print("\n🎵 Evaluating Audio Model...")
evaluate_model(audio_model, audio_test_loader)

print("\n📽️ Evaluating Video Model...")
evaluate_model(video_model, video_test_loader)

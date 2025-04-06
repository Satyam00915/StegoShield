import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import models
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

# Paths
dataset_path = "dataset_prep/dataset/split_data/split_audio"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
save_model_path = os.path.join(os.getcwd(), "backend/models/best_audio_model.pth")

# Function to convert waveform to spectrogram
def waveform_to_spectrogram(waveform, sr=22050, n_mels=128, target_shape=(128, 300)):
    waveform, _ = librosa.effects.trim(waveform)
    n_fft = min(2048, len(waveform))
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
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

# SpecAugment for spectrograms
def spec_augment(spec, time_mask_param=30, freq_mask_param=13):
    _, freq, time = spec.shape
    if time > time_mask_param:
        t = np.random.randint(0, time - time_mask_param)
        spec[:, :, t:t + time_mask_param] = 0
    if freq > freq_mask_param:
        f = np.random.randint(0, freq - freq_mask_param)
        spec[:, f:f + freq_mask_param, :] = 0
    return spec

# Augmentation Function
def augment_waveform(waveform, sr):
    if np.random.rand() < 0.3:
        waveform += 0.005 * np.random.randn(len(waveform))
    if np.random.rand() < 0.3:
        rate = np.random.uniform(0.8, 1.2)
        waveform = librosa.effects.time_stretch(y=waveform, rate=rate)
    if np.random.rand() < 0.3:
        steps = np.random.uniform(-1, 1)
        waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=steps)
    return waveform

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
        waveform = augment_waveform(waveform, self.sr)
        spectrogram = waveform_to_spectrogram(waveform, sr=self.sr, n_mels=self.n_mels)
        spectrogram = spec_augment(spectrogram)
        return spectrogram, torch.tensor(label, dtype=torch.long)

# ResNet34 Model for Audio Spectrograms
class ResNet34Audio(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34Audio, self).__init__()
        self.resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        for name, param in self.resnet34.named_parameters():
            if "conv1" in name or "bn1" in name:
                param.requires_grad = False

        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet34(x)

# Train Function with Accuracy + Val Loss
def train(model, train_loader, val_loader, epochs=75, lr=1e-4, weight_decay=5e-5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    writer = SummaryWriter()

    best_acc = 0
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        print(f"\n🔄 Epoch {epoch+1}/{epochs} Training...")
        for inputs, labels in tqdm(train_loader, desc="🛠️ Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        val_loss = 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="🔍 Validating", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        val_loss /= len(val_loader)
        scheduler.step(epoch)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)

        print(f"📘 Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), save_model_path)
            print(f"💾 Best model saved (Acc: {accuracy:.2f}%) ✅")
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("⏹️ Early stopping triggered! Best Val Accuracy:", best_acc)
            break

    writer.close()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clean", "Stego"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_audio.png")
    plt.close()

    print(f"\n🏁 Training complete. Best model saved at: {save_model_path}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = AudioDataset(train_path)
    val_dataset = AudioDataset(val_path)

    class_sample_count = np.array([train_dataset.labels.count(t) for t in np.unique(train_dataset.labels)])
    weights = 1. / class_sample_count
    samples_weights = np.array([weights[t] for t in train_dataset.labels])
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    num_classes = 2
    model = ResNet34Audio(num_classes)
    train(model, train_loader, val_loader)

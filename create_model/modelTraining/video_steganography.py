import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Video Dataset with Augmentation & Error Handling
class VideoDataset(Dataset):
    def __init__(self, root_dir, max_frames=10):
        self.root_dir = root_dir
        self.data = []
        self.max_frames = max_frames
        for label, subdir in enumerate(["clean", "stego"]):
            path = os.path.join(root_dir, subdir)
            if os.path.exists(path):
                for file in os.listdir(path):
                    self.data.append((os.path.join(path, file), label))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # Resize to match EfficientNet
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Handle videos with fewer frames
        if len(frames) < self.max_frames:
            frames.extend([frames[-1]] * (self.max_frames - len(frames)))
        
        frames = torch.stack(frames)
        return frames, torch.tensor(label, dtype=torch.long)

# Updated Model with EfficientNet + LSTM
class VideoStegoModel(nn.Module):
    def __init__(self):
        super(VideoStegoModel, self).__init__()
        self.cnn = models.efficientnet_v2_s(pretrained=True)
        self.cnn.classifier = nn.Linear(self.cnn.classifier[1].in_features, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.shape
        cnn_features = [self.cnn(x[:, t, :, :, :]) for t in range(timesteps)]
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        return self.fc(lstm_out[:, -1, :])

# Training Function with Balanced Loss
def train(model, train_loader, val_loader, epochs=15, lr=0.0005, save_path="backend/models/video.pth"):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save Best Model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved at {save_path}")

# Load Dataset & Train Model
if __name__ == "__main__":
    train_path = "C:/old/college/sem 6/Special Project/Project/StegoShield/dataset/split_data/split_videos/train"
    val_path = "C:/old/college/sem 6/Special Project/Project/StegoShield/dataset/split_data/split_videos/val"
    
    train_dataset = VideoDataset(train_path)
    val_dataset = VideoDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    video_model = VideoStegoModel()
    train(video_model, train_loader, val_loader)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Video Dataset
class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
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
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)

        cap.release()
        if len(frames) < 10:
            frames.extend([frames[-1]] * (10 - len(frames)))  # Padding with last frame if needed
        
        frames = torch.stack(frames)
        return frames, torch.tensor(label, dtype=torch.long)

# Video Steganography Detection Model
class VideoStegoModel(nn.Module):
    def __init__(self):
        super(VideoStegoModel, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.shape
        cnn_features = [self.cnn(x[:, t, :, :, :]) for t in range(timesteps)]
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        return self.fc(lstm_out[:, -1, :])

# Training Function
def train(model, dataloader, epochs=10, lr=0.001, save_path="video_model.pth"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for videos, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
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

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

# Load Dataset & Train Model
if __name__ == "__main__":
    video_dataset = VideoDataset("dataset/videos")
    video_loader = DataLoader(video_dataset, batch_size=4, shuffle=True)
    video_model = VideoStegoModel()
    train(video_model, video_loader)

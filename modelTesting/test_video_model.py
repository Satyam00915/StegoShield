import os
import torch
import torch.nn as nn
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
            if not os.path.exists(path):
                continue
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
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.shape
        cnn_features = [self.cnn(x[:, t, :, :, :]) for t in range(timesteps)]
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        return self.fc(lstm_out[:, -1, :])

# Load Test Dataset
test_data_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\dataset\split_data\split_videos\test"
test_dataset = VideoDataset(test_data_path)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load Trained Model
model_path = "video_model.pth"  # Make sure this is the correct path to the trained model
model = VideoStegoModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Evaluate Model
correct = 0
total = 0
with torch.no_grad():
    for videos, labels in tqdm(test_loader, desc="Testing Video Model"):
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")

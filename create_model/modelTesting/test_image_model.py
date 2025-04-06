import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import models
from tqdm import tqdm

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Dataset Class
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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
        img_path, label = self.data[idx]
        image = read_image(img_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Load Model
class ImageStegoCNN(nn.Module):
    def __init__(self):
        super(ImageStegoCNN, self).__init__()
        self.model = models.resnet18(pretrained=False)  # pretrained=False since we're loading custom weights
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# Define Transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Test Dataset
test_data_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\dataset\split_data\split_images\test"
test_dataset = ImageDataset(test_data_path, transform=image_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load Trained Model
model_path = "img_model.pth"  # Make sure this is the correct path to the trained model
model = ImageStegoCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Evaluate Model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")

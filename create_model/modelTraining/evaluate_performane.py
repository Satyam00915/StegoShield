import os
import torch
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.models import resnet18
from torch import nn
from tqdm import tqdm
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class (same as your training code)
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

# Load the model
class ImageStegoCNN(nn.Module):
    def __init__(self):
        super(ImageStegoCNN, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("✅ Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["Clean", "Stego"]))

    print("📊 Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

# Main
if __name__ == "__main__":
    val_path = "C:/old/college/sem 6/Special Project/Project/StegoShield/dataset/split_data/split_images/val"
    model_path = os.path.join(os.getcwd(), "backend/models/image_model.pth")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_dataset = ImageDataset(val_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = ImageStegoCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    evaluate(model, val_loader)

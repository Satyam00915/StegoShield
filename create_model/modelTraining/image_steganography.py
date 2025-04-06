import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from tqdm import tqdm

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Dataset
class ImageDataset(Dataset):
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

# Image Steganography Detection Model
class ImageStegoCNN(nn.Module):
    def __init__(self):
        super(ImageStegoCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# Training Function
def train(model, train_loader, val_loader, epochs=10, lr=0.001, save_path="backend/models/image_model.pth"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        correct, total = 0, 0
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
        
        # Evaluate model on validation set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {accuracy:.2f}%")

    # Save model after training
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved as {save_path}")

# Load Dataset & Train Model
if __name__ == "__main__":
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_path = "C:/old/college/sem 6/Special Project/Project/StegoShield/dataset/split_data/split_images/train"
    val_path = "C:/old/college/sem 6/Special Project/Project/StegoShield/dataset/split_data/split_images/val"
    save_model_path = os.path.join(os.getcwd(), "img_model.pth")

    train_dataset = ImageDataset(train_path, transform=image_transform)
    val_dataset = ImageDataset(val_path, transform=image_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    image_model = ImageStegoCNN()
    train(image_model, train_loader, val_loader, epochs=20, save_path=save_model_path)
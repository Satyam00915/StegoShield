import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import io

# 🟢 Image Model (ResNet18)
class ImageStegoCNN(nn.Module):
    def __init__(self):
        super(ImageStegoCNN, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # ✅ Fixed pretrained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# 🟢 Audio Model (CNN)
class AudioStegoCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioStegoCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        dummy_input = torch.randn(1, 1, 128, 300)
        out = self.cnn(dummy_input)
        flattened_size = out.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 🟢 Video Model (EfficientNet + LSTM)
class VideoStegoModel(nn.Module):
    def __init__(self):
        super(VideoStegoModel, self).__init__()
        self.cnn = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)  # ✅ Fixed weights
        self.cnn.classifier = nn.Sequential(  # ✅ Fixed classifier layer
            nn.Linear(self.cnn.classifier[1].in_features, 128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.shape
        cnn_features = [self.cnn(x[:, t, :, :, :]) for t in range(timesteps)]
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        return self.fc(lstm_out[:, -1, :])

# ✅ 🚀 Fixed Model Loading Function
def load_model():
    models_dict = {
        "image": ImageStegoCNN(),
        "audio": AudioStegoCNN(),
        "video": VideoStegoModel()
    }

    # 🔹 Load Weights
    models_dict["image"].load_state_dict(torch.load("backend/models/image_model.pth", map_location=torch.device('cpu')))
    models_dict["audio"].load_state_dict(torch.load("backend/models/audio_model.pth", map_location=torch.device('cpu')))
    models_dict["video"].load_state_dict(torch.load("backend/models/video_model.pth", map_location=torch.device('cpu')), strict=False)  # ✅ Fixed: `strict=False`

    # 🔹 Set to Evaluation Mode
    models_dict["image"].eval()
    models_dict["audio"].eval()
    models_dict["video"].eval()

    return models_dict

# Preprocess image for model prediction
def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Predict function to detect malicious payloads
def predict(file, models):
    filename = file.filename.lower()
    
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        model = models["image"]
        input_data = preprocess_image(file)
    
    elif filename.endswith(('.wav', '.mp3')):
        model = models["audio"]
        # TODO: Implement audio preprocessing
        input_data = torch.randn(1, 1, 128, 300)

    elif filename.endswith(('.mp4', '.avi', '.mkv')):
        model = models["video"]
        # TODO: Implement video preprocessing
        input_data = torch.randn(1, 10, 3, 224, 224)
    else:
        return "Unsupported file type", 0.0

    with torch.no_grad():
        output = model(input_data)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()

    result = "Malicious" if prediction == 1 else "Safe"
    return result, round(confidence, 2)

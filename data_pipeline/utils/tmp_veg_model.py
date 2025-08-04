import torch
from torch import nn
from torchvision import models
import numpy as np

model_path = '/home/dario/Downloads/best_ResNet18.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model with 10 classes
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 output classes to match the saved weights

# Load the complete state dict
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

# Load the image data
img_path = "/home/dario/Desktop/sample_veg_detection_data/VEGETATION/chile_tile_(3072, 7936).npy"
img = np.load(img_path)

# Preprocess the image
img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize to 0-1 range
img = img.astype(np.float32)
img = np.transpose(img, (2, 0, 1))  # Change shape to (C, H, W)
img = torch.tensor(img).unsqueeze(0)  # Add batch dimension
img = img.to(device)

# Perform inference
with torch.no_grad():
    output = model(img)
    output = torch.softmax(output, dim=1)  # Softmax for multi-class classification
    predicted_class = torch.argmax(output, dim=1)
    confidence = torch.max(output, dim=1)[0]
    
print("Prediction shape:", output.shape)
print("Class probabilities:", output.cpu().numpy())
print("Predicted class:", predicted_class.cpu().numpy())
print("Confidence:", confidence.cpu().numpy())
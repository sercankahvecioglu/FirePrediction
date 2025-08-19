import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # assuming 64x64 patches
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 16, 16]
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # return logits (no sigmoid)
    
def predict(model, image_path):
    # Detect file type
    if image_path.endswith('.npy'):
        array = np.load(image_path)  # shape: [H, W, 15]
        red = array[:, :, 3] / 10000.0
        green = array[:, :, 2] / 10000.0
        blue = array[:, :, 1] / 10000.0
        ndvi = array[:, :, 13]

        # Stack to [4, H, W]
        stacked = np.stack([red, green, blue, ndvi], axis=0)
    elif image_path.endswith('.png') or image_path.endswith('.jpg'):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((64, 64))  # Resize for consistency
        img = np.array(img).astype(np.float32) / 255.0

        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
        ndvi = np.zeros_like(red)

        # Stack to [4, 64, 64]
        stacked = np.stack([red, green, blue, ndvi], axis=0)

    elif image_path.endswith('.tif') or image_path.endswith('.tiff'):
        import rasterio
        with rasterio.open(image_path) as src:
        # Read bands using Sentinel-2 standard band numbers:
        # Band 4 = red (index 4)
        # Band 3 = green (index 3)
        # Band 2 = blue (index 2)
        # Band 8 = NIR (index 8)
         red = src.read(4).astype(np.float32) / 10000.0
         green = src.read(3).astype(np.float32) / 10000.0
         blue = src.read(2).astype(np.float32) / 10000.0
         nir = src.read(8).astype(np.float32) / 10000.0

        # Compute NDVI
         ndvi = (nir - red) / (nir + red + 1e-5)

        # Stack to [4, H, W]
         stacked = np.stack([red, green, blue, ndvi], axis=0)
    else:
        raise ValueError("Unsupported file format. Use .npy or .png/.jpg")

    # Resize if needed (for .npy)
    if stacked.shape[1:] != (64, 64):
        tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)  # [1, 4, H, W]
        tensor = F.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)
    else:
        tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)

    # Predict
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logit = model(tensor.to(device))
        prob = torch.sigmoid(logit).item()
        label = "VEGETATION" if prob > 0.4 else "NO_VEGETATION"
   
   # visualize = true 

    rgb_vis = np.stack([red, green, blue], axis=-1)
    rgb_vis = np.clip(rgb_vis, 0, 1)
    plt.figure(figsize=(4, 4))
    plt.imshow(rgb_vis)
    plt.title(f"Prediction: {label}", fontsize=14, color='green' if label == "VEGETATION" else 'red')
    plt.axis('off')
    plt.show()
    
def ndvi_veg_detector(input_folder, ndvi_threshold=0.3, min_veg_percentage=15.0):
    """
    Filter .npy files using NDVI threshold for vegetation detection
    
    Args:
        input_folder (str): Path to folder containing .npy files
        ndvi_threshold (float): NDVI threshold (typically 0.2-0.3)
        min_veg_percentage (float): Minimum percentage of vegetation pixels required (0-100)
    """
    npy_files = glob.glob(os.path.join(input_folder, "*.npy"))
    kept_files = []
    removed_files = []
    
    for npy_file in npy_files:
        try:
            # Load image data [height, width, channels]
            image_data = np.load(npy_file)
            
            # Assuming NIR is channel 7 and Red is channel 3 (adjust indices as needed)
            nir = image_data[..., 7].astype(np.float32)
            red = image_data[..., 3].astype(np.float32)
            
            # Calculate NDVI
            ndvi = np.divide(nir - red, nir + red, 
                           out=np.zeros_like(nir), where=(nir + red) != 0)
            
            # Calculate vegetation percentage
            total_pixels = ndvi.size
            veg_pixels = np.sum(ndvi > ndvi_threshold)
            veg_percentage = (veg_pixels / total_pixels) * 100
            
            # Keep if enough vegetation percentage
            if veg_percentage >= min_veg_percentage:
                kept_files.append(npy_file)
                label = "VEGETATION"
            else:
                removed_files.append(npy_file)
                label = "NO_VEGETATION"
                
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")
            # Remove input tile even on error

        red = image_data[:, :, 3] / 10000.0
        green = image_data[:, :, 2] / 10000.0
        blue = image_data[:, :, 1] / 10000.0
        ndvi = image_data[:, :, 13]
        rgb_vis = np.stack([red, green, blue], axis=-1)
        rgb_vis = np.clip(rgb_vis, 0, 1)
        plt.figure(figsize=(4, 4))
        plt.imshow(rgb_vis)
        plt.title(f"Prediction: {label}", fontsize=14, color='green' if label == "VEGETATION" else 'red')
        plt.axis('off')
        plt.show()
        
    return kept_files, removed_files

def veg_detector(input_folder):
    """
    Filter .npy files to keep only forest images (prediction = 1)
    
    Args:
        input_folder (str): Path to folder containing .npy files
        model_weights_path (str): Path to model weights (default: trained_models/vegcnn_weights.pth)
    """
    model_weights_path = "/Users/sercankahvecioglu/Desktop/sc25/FlameSentinels/code/final_demo/FirePrediction/server/utils/trained_models/WeightsProbe.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN()  # or whatever your model class is
    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for file in glob.glob(os.path.join(input_folder, "*.npy")):
        print(f"Processing file: {file}")
        # Predict using the model
        result = predict(model, file)
        print(result)

if __name__ == "__main__":
    # Example usage
    input_folder = "/Users/sercankahvecioglu/Desktop/sc25/FlameSentinels/code/final_demo/FirePrediction/server/utils/forest_detection/sample_veg_detection_data/NO_VEGETATION/"
    kept, removed = veg_detector(input_folder)
    print("Kept files:", kept)
    print("Removed files:", removed)
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import glob



# simple CNN arhcitecture for vegetation detection task
# produces 1 (after sigmoid & argmax) if forest, 0 otherwise
class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        # use adaptive pooling so FC receives fixed-size features regardless of input size
        self.adapt = nn.AdaptiveAvgPool2d((4,4))  # outputs [B, 64, 4, 4]
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)  # logits

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # raw logits (no sigmoid)
    


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
            else:
                # Remove input tile
                os.remove(npy_file)
                # Also remove corresponding label tile
                label_fname = os.path.join(os.path.dirname(input_folder), 'TILES_LABELS', os.path.basename(npy_file))
                if os.path.exists(label_fname):
                    os.remove(label_fname)
                removed_files.append(npy_file)
                
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")
            # Remove input tile even on error
            if os.path.exists(npy_file):
                os.remove(npy_file)
                # Also remove corresponding label tile
                label_fname = os.path.join(os.path.dirname(input_folder), 'TILES_LABELS', os.path.basename(npy_file))
                if os.path.exists(label_fname):
                    os.remove(label_fname)
            removed_files.append(npy_file)
    
    return kept_files, removed_files



def veg_detector(input_folder, model_weights_path=None):
    """
    Filter .npy files to keep only forest images (prediction = 1)
    
    Args:
        input_folder (str): Path to folder containing .npy files
        model_weights_path (str): Path to model weights (default: trained_models/vegcnn_weights.pth)
    """
    if model_weights_path is None:
        model_weights_path = "/home/dario/Desktop/FirePrediction/trained_models/vegcnn_weights.pth"
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(in_channels=14)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get all .npy files
    npy_files = glob.glob(os.path.join(input_folder, "*.npy"))
    
    kept_files = []
    removed_files = []
    
    with torch.no_grad():
        for npy_file in npy_files:
            try:
                # Load image data
                image_data = np.load(npy_file)
                
                # Ensure correct shape and data type
                if len(image_data.shape) == 3:
                    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
                
                # Convert to tensor and normalize if needed
                image_tensor = torch.FloatTensor(image_data).to(device)
                
                # Run inference
                outputs = model(image_tensor)
                # Apply sigmoid and threshold for binary classification
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).long()
                
                # Keep file if prediction is 1 (forest), otherwise remove
                if predicted.item() == 1:
                    kept_files.append(npy_file)
                else:
                    # Remove input tile
                    os.remove(npy_file)
                    # Also remove corresponding label tile
                    label_fname = os.path.join(os.path.dirname(input_folder), 'TILES_LABELS', os.path.basename(npy_file))
                    if os.path.exists(label_fname):
                        os.remove(label_fname)
                    removed_files.append(npy_file)
                    
            except Exception as e:
                print(f"Error processing {npy_file}: {e}")
                # Remove input tile even on error
                if os.path.exists(npy_file):
                    os.remove(npy_file)
                    # Also remove corresponding label tile
                    label_fname = os.path.join(os.path.dirname(input_folder), 'TILES_LABELS', os.path.basename(npy_file))
                    if os.path.exists(label_fname):
                        os.remove(label_fname)
                removed_files.append(npy_file)
    
    print(f"Processed {len(npy_files)} files")
    print(f"Kept {len(kept_files)} forest images")
    print(f"Removed {len(removed_files)} non-forest images")
    
    return kept_files, removed_files

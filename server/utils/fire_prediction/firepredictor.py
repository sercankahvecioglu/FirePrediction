import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .unet import UNet


# --------- CONSTANTS ------------ #

CURRENT_FOLDER = os.getcwd()
PARENT_FOLDER = os.path.dirname(CURRENT_FOLDER)
MODEL_FOLDER = os.path.join(PARENT_FOLDER, "server/utils/trained_models")
FIRE_MODEL = os.path.join(MODEL_FOLDER, "best_fireprediction_model.pth")

REQUIRED_BANDS = ['B08', 'B11', 'B12', 'NDVI', 'NDMI']
BANDS_INDICES = [7, 11, 12, 13, 14]

# --------------------------------- #

# -------- LOAD MODEL ------------- #

model = UNet()

print("Initializing Fire Prediction Model")

if torch.cuda.is_available():
    model.load_state_dict(torch.load(FIRE_MODEL, map_location=torch.device('cuda:0')))
    model = model.cuda()
    print("Using GPU...")
else:
    model.load_state_dict(torch.load(FIRE_MODEL, map_location=torch.device('cpu')))
    model = model.cpu()
    print("Using CPU...")



model.eval()  # Set model to evaluation mode

# ------------------------------- # 

def predict_fire(data: np.ndarray):
    """
    Predicts fire risk mask from input satellite data using the loaded UNet model.
    
    Args:
        data (np.ndarray): Array of shape (H, W, N) containing all bands.
                           Must include at least the indices specified in BANDS_INDICES.

    Returns:
        np.ndarray: Predicted mask of shape (H, W) with values 0 or 1.
    """

    # 1. Select required bands

    print("Selecting required bands...")
    input_data = data[:, :, BANDS_INDICES]  # Shape (H,W,5)
    
    # 2. Ensure it's 256x256 (resize if needed)
    print("Checking input dimensions...")
    if input_data.shape[0] != 256 or input_data.shape[1] != 256:
        raise ValueError(f"Input must be 256x256, got {input_data.shape[:2]}")

    # 3. Convert to tensor (1,5,256,256)
    print("Converting to tensor...")
    tensor_input = torch.tensor(input_data, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    if torch.cuda.is_available():
        tensor_input = tensor_input.cuda()

    # 3.5 Normalize input data
    for i in range(tensor_input.shape[1]):
       band = tensor_input[0, i]
       min_val, max_val = band.min(), band.max()
       tensor_input[0, i] = (band - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(band)

    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        output = model(tensor_input)  # Shape: (1,1,256,256)
    
    # 5. Threshold to get binary mask
    mask = (output > 0.5).int().squeeze().cpu().numpy()  # Shape (256,256)
    
    print("Inference completed. Mask shape:", mask.shape)

    return mask, output

if __name__ == "__main__":
    DATA_PATH = "/Users/diego/Documents/FirePrediction/server/data/TILES_IMAGES/14e611ab-d0e2-48c8-90ff-db0b4e59c4bc_tile_(0, 0).npy"

    data = np.load(DATA_PATH)

    # Add NDVI index

    nir = data[..., 7].astype(np.float32)
    red = data[..., 3].astype(np.float32)

    ndvi = np.divide(nir - red, nir + red,
                    out=np.zeros_like(nir), where=(nir + red) != 0)

    data = np.concatenate((data, ndvi[..., np.newaxis]), axis=-1)

    # Add NDMI index

    swir = data[..., 11].astype(np.float32)
    nir = data[..., 7].astype(np.float32)

    ndmi = np.divide(nir - swir, nir + swir,
                     out=np.zeros_like(nir), where=(nir + swir) != 0)

    data = np.concatenate((data, ndmi[..., np.newaxis]), axis=-1)

    mask, output = predict_fire(data)

    # Save the results
    np.save(os.path.join(os.getcwd(), "mask.npy"), mask)
    np.save(os.path.join(os.getcwd(), "output.npy"), output)
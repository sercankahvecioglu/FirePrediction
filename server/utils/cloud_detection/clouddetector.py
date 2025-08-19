import numpy as np
import os
import glob
import torch
import torch.nn.functional as F
from s2cloudless import S2PixelCloudDetector
from .unet import UNET

# Get the absolute path to the trained models directory relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
unet_model_path = os.path.join(current_dir, "..", "trained_models", "best_cloud_model.pth")

cloud_detector = S2PixelCloudDetector(threshold=0.5, average_over=4, dilation_size=2, all_bands=True)

# Initialize U-Net model (will be loaded when needed)
unet_model = None

def load_unet_model(model_weights_path: str = None):
    """
    Load the U-Net model with trained weights
    
    Params:
        model_weights_path (str): Path to the model weights file
    
    Returns:
        model: Loaded U-Net model
    """
    global unet_model
    
    if unet_model is None:
        # Initialize model with 13 input channels (Sentinel-2 bands) and 1 output channel (cloud mask)
        unet_model = UNET(in_channels=13, out_channels=1)
        
        # Load weights if path is provided
        if model_weights_path or unet_model_path:
            weights_path = model_weights_path or unet_model_path
            if os.path.exists(weights_path):
                unet_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                print(f"U-Net model weights loaded from {weights_path}")
            else:
                print(f"Warning: Model weights not found at {weights_path}")
        
        unet_model.eval()  # Set to evaluation mode
    
    return unet_model

def is_cloudy(tile_path:str, cloud_detector = cloud_detector, cloud_threshold:float = 0.4, job_id:str = None, delete = False):
    """
    Function to find cloudy tiles from s2cloudless's S2PixelCloudDetector and discard them from the tiles folder

    Params:
        tile_path (str): Path to the (input or label) tile to analyze
        cloud_threshold (float, optional): % of cloudy pixels above which an image is discarded (not downlinked). Defaults to 0.5.

    Returns:
        cloud_results (dict): dict of (clean_tiles_num, cloudy_tiles_num)
    """

    cloud_results = {'cloudy_tiles': 0, 'clean_tiles': 0}

    tile = np.load(tile_path)

    if tile.shape[-1] != 13:
        print("File has a n° of channels =\= 13 (probably it has already been processed). Cloud mask will not be calculated.")
        return cloud_results

    # Ensure tile has correct shape and data type
    if tile.ndim != 3:
        print(f"Warning: Tile {tile_path} has unexpected shape {tile.shape}")
        return cloud_results
        
    # Ensure data type is float
    tile = tile.astype(np.float32)

    cloud_mask = cloud_detector.get_cloud_masks(tile[np.newaxis, ...])[0]

    # Fix the percentage calculation
    total_pixels = cloud_mask.size
    cloudy_pixels = np.sum(cloud_mask)
    perc_cloudy = cloudy_pixels / total_pixels

    # if the cloudy pixels % exceeds the threshold, discard the image
    if perc_cloudy >= cloud_threshold:
        # remove image file from tiles folder
        cloud_results['cloudy_tiles'] += 1
        print(f"Cloudy tile ({os.path.basename(tile_path)}) found ({perc_cloudy*100:.1f}% of pixels are cloudy).")

        if delete:
            os.remove(tile_path)
            print("Removing image")
    else: 
        cloud_results['clean_tiles'] += 1
        print(f"Tile {os.path.basename(tile_path)} is not cloudy (only {perc_cloudy*100:.1f}% of pixels are cloudy).")

    return cloud_results, cloud_mask, perc_cloudy

def is_cloudy_unet(tile_path: str, model_weights_path: str = None, cloud_threshold: float = 0.4, job_id: str = None, delete: bool = False):
    """
    Function to find cloudy tiles using a custom U-Net model and discard them from the tiles folder

    Params:
        tile_path (str): Path to the (input or label) tile to analyze
        model_weights_path (str, optional): Path to the U-Net model weights
        cloud_threshold (float, optional): % of cloudy pixels above which an image is discarded. Defaults to 0.4.
        job_id (str, optional): Job identifier
        delete (bool, optional): Whether to delete cloudy tiles. Defaults to False.

    Returns:
        cloud_results (dict): dict of (clean_tiles_num, cloudy_tiles_num)
        cloud_mask (np.ndarray): Binary cloud mask
        perc_cloudy (float): Percentage of cloudy pixels
    """
    
    cloud_results = {'cloudy_tiles': 0, 'clean_tiles': 0}

    tile = np.load(tile_path)

    if tile.shape[-1] != 13:
        print("File has a n° of channels =\= 13 (probably it has already been processed). Cloud mask will not be calculated.")
        return cloud_results, None, 0.0

    # Ensure tile has correct shape and data type
    if tile.ndim != 3:
        print(f"Warning: Tile {tile_path} has unexpected shape {tile.shape}")
        return cloud_results, None, 0.0
        
    # Ensure data type is float
    tile = tile.astype(np.float32)
    
    # Load U-Net model
    model = load_unet_model(model_weights_path)
    
    # Prepare input tensor
    # Convert from HWC to CHW format and add batch dimension
    input_tensor = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).float()
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        
        # Apply sigmoid to get probabilities and threshold to get binary mask
        cloud_prob = torch.sigmoid(output)
        cloud_mask = (cloud_prob > 0.5).squeeze().cpu().numpy().astype(bool)
    
    # Calculate percentage of cloudy pixels
    total_pixels = cloud_mask.size
    cloudy_pixels = np.sum(cloud_mask)
    perc_cloudy = cloudy_pixels / total_pixels

    # if the cloudy pixels % exceeds the threshold, discard the image
    if perc_cloudy >= cloud_threshold:
        # remove image file from tiles folder
        cloud_results['cloudy_tiles'] += 1
        print(f"Cloudy tile ({os.path.basename(tile_path)}) found ({perc_cloudy*100:.1f}% of pixels are cloudy).")

        if delete:
            os.remove(tile_path)
            print("Removing image")
    else: 
        cloud_results['clean_tiles'] += 1
        print(f"Tile {os.path.basename(tile_path)} is not cloudy (only {perc_cloudy*100:.1f}% of pixels are cloudy).")

    return cloud_results, cloud_mask, perc_cloudy
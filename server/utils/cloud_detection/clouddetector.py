import numpy as np
import os
import glob
import torch
import torch.nn.functional as F
from s2cloudless import S2PixelCloudDetector
##from unet import UNET, FinalCloudUNet

cloud_detector = S2PixelCloudDetector(threshold=0.5, average_over=4, dilation_size=2, all_bands=True)

#model = UNET()
#test_model = FinalCloudUNet()

# Get the absolute path to the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize models (commented out - uncomment and set up paths when models are available)
model = None
test_model = None

# Uncomment these lines when models are available:
#model = UNET()
#test_model = FinalCloudUNet()
#model.load_state_dict(torch.load(os.path.join(current_dir, 'best_cloud_model_complete_v2.pth'), weights_only=False))
#checkpoint = torch.load(os.path.join(current_dir, 'final_cloud_model_rgb_256.pth'), weights_only=False)
#test_model.load_state_dict(checkpoint['model_state_dict'])
#test_model.load_state_dict(checkpoint['model_state_dict'])

def is_cloudy(tile_path: str, cloud_threshold: float = 0.4, job_id: str = None, delete: bool = False):
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

def is_cloudy_unet(tile_path: str, cloud_threshold: float = 0.4, job_id: str = None, delete: bool = False):
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
    
    # Check if model is available
    if model is None:
        print("Warning: U-Net model not loaded. Falling back to s2cloudless detector.")
        return is_cloudy(tile_path, cloud_threshold, job_id, delete)
    
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

    red = tile[:, :, 3]
    green = tile[:, :, 2]
    blue = tile[:, :, 1]

    image = np.stack([red, green, blue], axis=-1)
    
    # Prepare input tensor
    # Convert from HWC to CHW format and add batch dimension
    input_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

        # Apply softmax to get probabilities and threshold to get binary mask
        cloud_prob = torch.softmax(output, dim=1)
        cloud_mask = torch.argmax(cloud_prob, dim=1)  # shape (1, H, W), between 0 or 1
        cloud_mask = (cloud_mask > 0.5).squeeze().cpu().numpy().astype(bool)
    
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

    return cloud_results, cloud_mask, perc_cloudy, cloud_prob

def is_cloudy_test(tile_path: str, cloud_threshold: float = 0.4, job_id: str = None, delete: bool = False):
    """
    Function to find cloudy tiles using the FinalCloudUNet model and discard them from the tiles folder

    Params:
        tile_path (str): Path to the (input or label) tile to analyze
        cloud_threshold (float, optional): % of cloudy pixels above which an image is discarded. Defaults to 0.4.
        job_id (str, optional): Job identifier
        delete (bool, optional): Whether to delete cloudy tiles. Defaults to False.

    Returns:
        cloud_results (dict): dict of (clean_tiles_num, cloudy_tiles_num)
        cloud_mask (np.ndarray): Binary cloud mask
        perc_cloudy (float): Percentage of cloudy pixels
        cloud_prob (torch.Tensor): Cloud probability map
    """
    
    # Check if test model is available
    if test_model is None:
        print("Warning: FinalCloudUNet model not loaded. Falling back to s2cloudless detector.")
        result, cloud_mask, perc_cloudy = is_cloudy(tile_path, cloud_threshold, job_id, delete)
        return result, cloud_mask, perc_cloudy, None
    
    cloud_results = {'cloudy_tiles': 0, 'clean_tiles': 0}

    tile = np.load(tile_path)

    if tile.shape[-1] != 13:
        print("File has a n° of channels =\= 13 (probably it has already been processed). Cloud mask will not be calculated.")
        return cloud_results, None, 0.0, None

    # Ensure tile has correct shape and data type
    if tile.ndim != 3:
        print(f"Warning: Tile {tile_path} has unexpected shape {tile.shape}")
        return cloud_results, None, 0.0, None
        
    # Ensure data type is float
    tile = tile.astype(np.float32)

    red = tile[:, :, 3]
    green = tile[:, :, 2]
    blue = tile[:, :, 1]

    image = np.stack([red, green, blue], axis=-1)
    
    # Prepare input tensor
    # Convert from HWC to CHW format and add batch dimension
    input_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

    # Run inference with FinalCloudUNet
    with torch.no_grad():
        cloud_prob = test_model(input_tensor)  # Output is already sigmoid activated
        
        # Convert probabilities to binary mask
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

    return cloud_results, cloud_mask, perc_cloudy, cloud_prob
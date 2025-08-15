import numpy as np
import os
import glob
from s2cloudless import S2PixelCloudDetector

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True)

def process_single_tile(tile_path: str, cloud_detector=cloud_detector, cloud_threshold: float = 0.5, delete: bool = False):
    """
    Process a single tile for cloud detection.
    
    Params:
        tile_path (str): Path to the individual tile to analyze
        cloud_detector: S2PixelCloudDetector instance
        cloud_threshold (float): % of cloudy pixels above which an image is considered cloudy
        delete (bool): Whether to delete cloudy tiles
        
    Returns:
        tuple: (cloud_results_dict, cloud_mask, perc_cloudy)
    """
    cloud_results = {'cloudy_tiles': 0, 'clean_tiles': 0}
    
    if not os.path.exists(tile_path):
        print(f"Warning: Tile path {tile_path} does not exist")
        return cloud_results, None, 0.0
    
    try:
        tile = np.load(tile_path)
    except Exception as e:
        print(f"Error loading tile {tile_path}: {e}")
        return cloud_results, None, 0.0

    if tile.shape[-1] != 13:
        print(f"File {tile_path} has {tile.shape[-1]} channels instead of 13 (probably already processed). Cloud mask will not be calculated.")
        return cloud_results, None, 0.0

    # Ensure tile has correct shape and data type
    if tile.ndim != 3:
        print(f"Warning: Tile {tile_path} has unexpected shape {tile.shape}")
        return cloud_results, None, 0.0

    # Ensure data is in correct range (0-1 or 0-10000 depending on preprocessing)
    if tile.max() > 1.0:
        tile = tile / 10000.0  # Convert from reflectance values to 0-1 range
        
    # Ensure data type is float
    tile = tile.astype(np.float32)

    try:
        cloud_mask = cloud_detector.get_cloud_masks(tile[np.newaxis, ...])[0]
    except Exception as e:
        print(f"Error in cloud detection for {tile_path}: {e}")
        return cloud_results, None, 0.0

    # Calculate the percentage of cloudy pixels
    total_pixels = cloud_mask.size
    cloudy_pixels = np.sum(cloud_mask)
    perc_cloudy = cloudy_pixels / total_pixels if total_pixels > 0 else 0.0

    # if the cloudy pixels % exceeds the threshold, mark as cloudy
    if perc_cloudy >= cloud_threshold:
        cloud_results['cloudy_tiles'] = 1
        if delete:
            try:
                os.remove(tile_path)
                #print(f"Cloudy tile ({os.path.basename(tile_path)}) found ({perc_cloudy*100:.1f}% of pixels are cloudy). Removed.")
            except Exception as e:
                print(f"Error removing cloudy tile {tile_path}: {e}")
    else: 
        cloud_results['clean_tiles'] = 1
        #print(f"Tile {os.path.basename(tile_path)} is not cloudy (only {perc_cloudy*100:.1f}% of pixels are cloudy).")

    return cloud_results, cloud_mask, perc_cloudy

def is_cloudy(tile_path_or_dir: str, cloud_detector=cloud_detector, cloud_threshold: float = 0.5, job_id: str = None, delete: bool = False):
    """
    Function to find cloudy tiles from s2cloudless's S2PixelCloudDetector.
    Can process either a single tile file or all tiles in a directory.

    Params:
        tile_path_or_dir (str): Path to a single tile file (.npy) or directory containing tiles
        cloud_detector: S2PixelCloudDetector instance
        cloud_threshold (float, optional): % of cloudy pixels above which an image is discarded. Defaults to 0.5.
        job_id (str, optional): Job identifier (for compatibility)
        delete (bool, optional): Whether to delete cloudy tiles. Defaults to False.

    Returns:
        For single file: tuple (cloud_results_dict, cloud_mask, perc_cloudy)
        For directory: cloud_results_dict
    """
    
    # Check if input is a single file or directory
    if os.path.isfile(tile_path_or_dir) and tile_path_or_dir.endswith('.npy'):
        # Single tile processing
        return process_single_tile(tile_path_or_dir, cloud_detector, cloud_threshold, delete)
    
    elif os.path.isdir(tile_path_or_dir):
        # Directory processing (batch mode)
        return process_directory(tile_path_or_dir, cloud_detector, cloud_threshold, delete)
    
    else:
        print(f"Warning: {tile_path_or_dir} is neither a valid .npy file nor a directory")
        cloud_results = {'cloudy_tiles': 0, 'clean_tiles': 0}
        return cloud_results, None, 0.0

def process_directory(tile_dir: str, cloud_detector=cloud_detector, cloud_threshold: float = 0.5, delete: bool = False):
    """
    Process all tiles in a directory for cloud detection.
    
    Params:
        tile_dir (str): Path to directory containing tiles
        cloud_detector: S2PixelCloudDetector instance
        cloud_threshold (float): % of cloudy pixels above which an image is discarded
        delete (bool): Whether to delete cloudy tiles
        
    Returns:
        cloud_results (dict): dict of (clean_tiles_num, cloudy_tiles_num)
    """
    cloud_results = {'cloudy_tiles': 0, 'clean_tiles': 0}

    tile_files = glob.glob(os.path.join(tile_dir, '*.npy'))
    
    if not tile_files:
        print(f"No .npy files found in directory {tile_dir}")
        return cloud_results

    total_tiles = len(tile_files)
    print(f"Processing {total_tiles} tiles for cloud detection...")

    for i, tile_path in enumerate(tile_files, 1):
        result, _, _ = process_single_tile(tile_path, cloud_detector, cloud_threshold, delete)
        cloud_results['cloudy_tiles'] += result['cloudy_tiles']
        cloud_results['clean_tiles'] += result['clean_tiles']
        
        if i % 200 == 0:
            print(f"Processed {i}/{total_tiles} tiles. Clean: {cloud_results['clean_tiles']}, Cloudy: {cloud_results['cloudy_tiles']}")

    print(f"Completed processing {total_tiles} tiles. Final results - Clean: {cloud_results['clean_tiles']}, Cloudy: {cloud_results['cloudy_tiles']}")
    return cloud_results
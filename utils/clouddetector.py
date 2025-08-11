import numpy as np
import os
import glob
from s2cloudless import S2PixelCloudDetector, download_bands_and_valid_data_mask

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True)

def is_cloudy(tiles_path:str, cloud_threshold:float = 0.5):
    """
    Function to find cloudy tiles from s2cloudless's S2PixelCloudDetector and discard them from the tiles folder

    Params:
        tiles_path (str): Path to the (input or label) tiles parent folder
        cloud_threshold (float, optional): % of cloudy pixels above which an image is discarded (not downlinked). Defaults to 0.5.

    Returns:
        cloud_results (dict): dict of (clean_tiles_num, cloudy_tiles_num)
    """
    cloud_results = {'cloudy_tiles': 0, 'clean_tiles': 0}
    for fname in glob.glob(os.path.join(tiles_path, 'TILES_INPUT_DATA', '*.npy')):
        tile = np.load(fname)
        if tile.shape[-1] != 13:
            print("File has a n° of channels =\= 13 (probably it has already been processed). Cloud mask will not be calculated.")
            continue
        
        # Ensure tile has correct shape and data type
        if tile.ndim != 3:
            print(f"Warning: Tile {fname} has unexpected shape {tile.shape}")
            continue
            
        # Ensure data is in correct range (0-1 or 0-10000 depending on preprocessing)
        if tile.max() > 1.0:
            tile = tile / 10000.0  # Convert from reflectance values to 0-1 range
            
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
            os.remove(fname) 
            # remove also corresponding label
            label_fname = os.path.join(tiles_path, 'TILES_LABELS', os.path.basename(fname))
            if os.path.exists(label_fname):
                os.remove(label_fname)
            cloud_results['cloudy_tiles'] += 1
            print(f"Cloudy tile ({os.path.basename(fname)}) found ({perc_cloudy*100:.1f}% of pixels are cloudy). Removing it...")
        else: 
            cloud_results['clean_tiles'] += 1
            print(f"Tile {os.path.basename(fname)} is not cloudy (only {perc_cloudy*100:.1f}% of pixels are cloudy).")

    return cloud_results
import os
import numpy as np
from shutil import move

def patch_vegetation_detection(tiles_input_path: str, 
                             vegetation_threshold: float = 0.2,
                             ndvi_threshold: float = 0.2,
                             min_vegetation_pixels: int = 1000):
    """
    Vegetation detection model that calculates NDVI from raw bands and filters tiles.
    
    Args:
        tiles_input_path (str): Path to directory containing tile files
        vegetation_threshold (float): Minimum percentage of pixels that must have NDVI > ndvi_threshold
        ndvi_threshold (float): NDVI value threshold for considering a pixel as vegetation
        min_vegetation_pixels (int): Minimum absolute number of vegetation pixels required
    
    Returns:
        dict: Statistics about vegetation filtering results
    """
    
    # Create directory for low vegetation tiles
    low_vegetation_path = tiles_input_path.replace('TILES_INPUT_DATA', 'LOW_VEGETATION_tiles')
    os.makedirs(low_vegetation_path, exist_ok=True)
    
    # Get all tile files
    tile_files = [f for f in os.listdir(tiles_input_path) if f.endswith('.npy')]
    
    clean_tiles = 0
    low_vegetation_tiles = 0
    
    # Band indices for Sentinel-2
    # ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    red_band_idx = 3    # B04 (Red)
    nir_band_idx = 7    # B08 (NIR)
    
    for tile_file in tile_files:
        tile_path = os.path.join(tiles_input_path, tile_file)
        
        try:
            # Load tile data
            tile_data = np.load(tile_path)
            
            # Extract Red and NIR bands
            if tile_data.shape[2] >= 13:  # Ensure we have all 13 bands
                red = tile_data[:, :, red_band_idx].astype(np.float32)
                nir = tile_data[:, :, nir_band_idx].astype(np.float32)
                
                # Calculate NDVI: (NIR - Red) / (NIR + Red)
                # Add small epsilon to avoid division by zero
                epsilon = 1e-8
                ndvi = (nir - red) / (nir + red + epsilon)
                
                # Handle invalid values (set to -1 for non-vegetation)
                ndvi = np.where(np.isnan(ndvi) | np.isinf(ndvi), -1, ndvi)
                
                # Calculate vegetation statistics
                vegetation_pixels = np.sum(ndvi > ndvi_threshold)
                total_pixels = ndvi.shape[0] * ndvi.shape[1]
                vegetation_percentage = vegetation_pixels / total_pixels
                
                # Check if tile meets vegetation criteria
                if (vegetation_percentage >= vegetation_threshold and 
                    vegetation_pixels >= min_vegetation_pixels):
                    clean_tiles += 1
                else:
                    # Move tile to low vegetation folder
                    move(tile_path, os.path.join(low_vegetation_path, tile_file))
                    low_vegetation_tiles += 1
            else:
                print(f"Warning: Tile {tile_file} doesn't have enough bands, keeping it")
                clean_tiles += 1
                
        except Exception as e:
            print(f"Error processing tile {tile_file}: {e}")
            clean_tiles += 1  # Keep tile if there's an error
    
    return {
        'clean_tiles': clean_tiles,
        'low_vegetation_tiles': low_vegetation_tiles,
        'low_vegetation_path': low_vegetation_path
    }

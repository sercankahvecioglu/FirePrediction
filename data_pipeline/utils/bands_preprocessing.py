import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from rasterio.warp import reproject, Resampling
from scipy.ndimage import grey_closing
from ...FirePredictionModel.models.geodata_extraction import extract_geospatial_metadata

BAND_ORDER = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']


# extract the dNBR map (of values between 0 and 1) and the dNBR binary map
def extract_data_labels(pre_bands_fpath, post_bands_fpath, output_dir: str, masking: bool = True, thresh: list = [0.2 ,0.4]):
    """
    Create data labels from Sentinel-2 images by considering the dNBR value (with NDVI masking)
    
    Params:
        - pre_bands_fpath: Path to the pre-bands file
        - post_bands_fpath: Path to the post-bands file
        - output_dir: Location where to save the results
        - masking: bool param to determine whether to mask to 0 pixels that are not vegetation, using dNBR
            or not to do it (as these will be later discarded)
        - thresh: list param of threshold values to determine the division into classes of th efinal map

    Returns:
        numpy.ndarray: dNBR normalized image with shape (height, width, 1)
    """
    
    nbr_imgs = {}

    # B08 is at index 7, B12 is at index 12 (we have B8A after B08)
    pre_bands = np.load(pre_bands_fpath)[..., [7, 12]]
    post_bands = np.load(post_bands_fpath)[..., [7, 12]]
    
    # Process both datasets
    for i, bands_data in enumerate([pre_bands, post_bands]):
        
        b08_data = bands_data[:, :, 0] #B08
        b12_data = bands_data[:, :, 1] #B12

        # Calculate NBR
        denom_nbr = b08_data + b12_data
        denom_nbr[denom_nbr == 0] = 1e-6
        nbr_img = (b08_data - b12_data) / denom_nbr
        nbr_imgs[i] = nbr_img
        
    # get final dnbr image
    dnbr_img = nbr_imgs[0] - nbr_imgs[1]
    
    # apply masking: we don't care about pixel values with no significant change in NBR between pre- and post-
    if masking:
        # assign class 0 -> NO/LOW RISK, 1 -> MODERATE RISK, 2 -> HIGH RISK
        dnbr_map = np.where(dnbr_img < thresh[0], 0, np.where(dnbr_img < thresh[1], 1, 2))

    # apply closing operation to get smoother label areas (without low value pixels inside high risk areas)
    dnbr_map = grey_closing(dnbr_map, size=5)

    # add final channel dimension for later steps
    dnbr_map = dnbr_map[..., np.newaxis]

    # save this data as numpy file
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'dnbr_normalized.npy'), dnbr_map)

    print("Data saved successfully!")
    print(f"File saved in '{output_dir}' directory.")

    # free up ram 
    del pre_bands, post_bands

    # return path (for later tiling of labeled map, too)
    return os.path.join(output_dir, 'dnbr_normalized.npy')

#--------------------------------------------------------------------------------

# extract the single tiles of dim 256x256 (or whatever needed) from the large image
def extract_tiles_with_padding(image_path, name, tile_size, path):
    """
    Extract tiles using padding strategy to ensure complete coverage.
    
    Params:
        - image_path: Path of input image (having shape H, W, C)
        - name: The name of the location of the image (needed for data organizational purposes)
        - tile_size: Size of each tile (height, width, channels)
        - path: Location where to save tiles
    """
    os.makedirs(path, exist_ok=True)

    image = np.load(image_path)

    h, w, c = image.shape
    ph, pw, pc = tile_size
    
    # Calculate padding needed
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    
    # Pad with reflection to maintain natural patterns
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    # Extract non-overlapping tiles
    padded_h, padded_w, _ = padded_image.shape
    for i in range(0, padded_h, ph):
        for j in range(0, padded_w, pw):
            tile = padded_image[i:i+ph, j:j+pw, :]
            np.save(os.path.join(path, f'{name}_tile_{(i, j)}'), tile)

    print("All tiles extracted correctly!")

    del image # free up RAM space

#--------------------------------------------------------------------------------

# unified function to compute vegetation indices from single .npy tile file and add as new channels
def compute_veg_indices(tile_path: str, band_names: list = BAND_ORDER, indices: list = ['ndvi', 'ndmi']):
    """
    Compute vegetation indices (NDVI and/or NDMI) from a single .npy tile file, add them as new channels, and save
    
    Params:
        - tile_path (str): Path to the .npy tile file
        - band_names (list): List of band names in the order they appear in the tile
        - indices (list): List of indices to compute ('ndvi' and/or 'ndmi')
    
    Returns:
        - numpy.ndarray: Expanded tile data with vegetation indices as additional channels
        - list: Updated band names including the new indices
    """
    # load the tile data
    tile_data = np.load(tile_path)
    
    # check if vegetation indices are already present in the file 
    # (used when files from previous images are already prepared in the folder)
    expected_bands_with_indices = len(band_names) + len(indices)
    if tile_data.shape[2] >= expected_bands_with_indices:
        # return existing data with updated band names
        updated_band_names = band_names + [idx.upper() for idx in indices]
        return tile_data, updated_band_names
    
    # define band requirements for each index
    index_configs = {
        'ndvi': {'bands': ['B08', 'B04'], 'names': ['NIR', 'Red']},
        'ndmi': {'bands': ['B08', 'B11'], 'names': ['NIR', 'SWIR']}
    }
    
    # start with original tile data
    expanded_tile = tile_data.copy()
    updated_band_names = band_names.copy()
    
    # Ccmpute and add each requested index
    for index_type in indices:
        
        config = index_configs[index_type.lower()]
        required_bands = config['bands']
        band_descriptions = config['names']
        
        # get required band indices
        try:
            band1_idx = band_names.index(required_bands[0])
            band2_idx = band_names.index(required_bands[1])
        except ValueError:
            print(f"Warning: {required_bands[0]} ({band_descriptions[0]}) or {required_bands[1]} ({band_descriptions[1]}) bands not found. Skipping {index_type.upper()}...")
            continue
        
        # extract the specific bands
        band1_data = tile_data[:, :, band1_idx]
        band2_data = tile_data[:, :, band2_idx]
        
        # compute the index: (band1 - band2) / (band1 + band2)
        denom = band1_data + band2_data
        denom[denom == 0] = 1e-6  # avoid division by zero
        index_data = (band1_data - band2_data) / denom
        
        # add as new channel
        index_data = index_data[:, :, np.newaxis]
        expanded_tile = np.concatenate([expanded_tile, index_data], axis=2)
        updated_band_names.append(index_type.upper())

    del tile_data # free up ram space
    
    # save expanded tile (overwriting original)
    np.save(tile_path, expanded_tile)
    
    return expanded_tile, updated_band_names

#--------------------------------------------------------------------------------

# function to process full directories of tiles and add vegetation indices as additional channels
def process_tiles_directory_with_indices(tiles_dir: str, band_names: list, indices: list = ['ndvi', 'ndmi']):
    """
    Process all .npy tile files in a directory to add vegetation indices as additional channels
    
    Args:
        tiles_dir (str): Directory containing .npy tile files
        band_names (list): List of band names in order they appear in tiles
        indices (list): List of indices to compute ('ndvi' and/or 'ndmi')
    
    Returns:
        list: List of paths to saved expanded tile files
        list: Updated band names including the new indices
    """
    # Find all .npy files in the tiles directory
    tile_files = glob.glob(os.path.join(tiles_dir, '*.npy'))
    
    if not tile_files:
        print(f"No .npy files found in {tiles_dir}")
        return [], band_names
    
    print(f"Processing {len(tile_files)} tiles to add {', '.join([idx.upper() for idx in indices])} as additional channels...")
    
    saved_paths = []
    updated_band_names = None
    
    for tile_file in tile_files:
        try:
            _, current_band_names = compute_veg_indices(tile_file, band_names, indices)
            saved_paths.append(tile_file)
            if updated_band_names is None:
                updated_band_names = current_band_names
        except Exception as e:
            print(f"Error processing {tile_file}: {e}")
    
    print(f"Successfully processed {len(saved_paths)} tiles with expanded channels")
    if updated_band_names:
        print(f"Original channels: {len(band_names)}, Final channels: {len(updated_band_names)}")
        print(f"Added indices: {[name for name in updated_band_names if name not in band_names]}")
    
    return saved_paths, updated_band_names if updated_band_names else band_names

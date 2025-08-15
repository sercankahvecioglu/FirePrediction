import os
import numpy as np
import glob
import pandas as pd

BAND_ORDER = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']


# extract the dNBR map (of values between 0 and 1) and the dNBR binary map
def extract_data_labels(dataset_name, pre_bands_fpath, post_bands_fpath, output_dir: str, masking: bool = True, ndvi_thresh: float = 0.2):
    """
    Create data labels from Sentinel-2 images by considering the dNBR value (with NDVI masking)
    
    Params:
        - dataset_name: name of dataset, to use for filename saving
        - pre_bands_fpath: Path to the pre-bands file
        - post_bands_fpath: Path to the post-bands file
        - output_dir: Location where to save the results
        - masking: bool param to determine whether to mask to 0 pixels that are not vegetation, using dNBR
            or not to do it (as these will be later discarded)
        - ndvi_thresh: float param for NDVI threshold to mask non-vegetation areas (default: 0.2)

    Returns:
        numpy.ndarray: dNBR normalized image with shape (height, width, 1)
    """
    
    nbr_imgs = {}

    # load pre and post bands - we need B04 (Red), B08 (NIR), and B12 (SWIR) for NDVI and NBR
    # B04 is at index 3, B08 is at index 7, B12 is at index 12 (we have B8A after B08)
    pre_bands = np.load(pre_bands_fpath)[..., [3, 7, 12]]  # B04, B08, B12
    post_bands = np.load(post_bands_fpath)[..., [3, 7, 12]]  # B04, B08, B12
    
    # calculate NDVI from pre-fire image for vegetation masking
    pre_b04 = pre_bands[:, :, 0]  # Red band
    pre_b08 = pre_bands[:, :, 1]  # NIR band
    
    # calculate NDVI: (NIR - Red) / (NIR + Red)
    denom_ndvi = pre_b08 + pre_b04
    denom_ndvi[denom_ndvi == 0] = 1e-6
    ndvi_pre = (pre_b08 - pre_b04) / denom_ndvi
    
    # create vegetation mask: True for vegetation pixels (NDVI >= threshold)
    vegetation_mask = ndvi_pre >= ndvi_thresh
    
    # process both datasets for NBR calculation
    for i, bands_data in enumerate([pre_bands, post_bands]):
        
        b08_data = bands_data[:, :, 1]  # B08 (NIR)
        b12_data = bands_data[:, :, 2]  # B12 (SWIR)

        # Calculate NBR
        denom_nbr = b08_data + b12_data
        denom_nbr[denom_nbr == 0] = 1e-6
        nbr_img = (b08_data - b12_data) / denom_nbr
        nbr_imgs[i] = nbr_img
        
    # get final dnbr image
    dnbr_img = nbr_imgs[0] - nbr_imgs[1]

    # apply sigmoid function to get probabilitistic output 
    # (the sigmoid needs to be scaled to return the desired outputs for the ranges of dNBR)
    # the range of values are taken from UN-SPIDER guidelines (https://un-spider.org/advisory-support/recommended-practices/recommended-practice-burn-severity/in-detail/normalized-burn-ratio)
    # the approximate range of typical values is from -0.5 to 1, with medium risk at approx. 0.3/0.4
    shift = 0.35
    slope = 15
    #-----SIGMOID-----
    dnbr_heatmap = 1 / (1+np.exp(-slope*(dnbr_img - shift)))

    # apply masking: we don't care about pixel values with no significant change in NBR between pre- and post-
    if masking:
        # apply NDVI-based vegetation masking: set non-vegetation pixels to class 0
        dnbr_heatmap[~vegetation_mask] = 0

    # add final channel dimension for later steps
    dnbr_heatmap = dnbr_heatmap[..., np.newaxis]

    # save this data as numpy file
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'{dataset_name}_dnbr_heatmap.npy'), dnbr_heatmap)

    print("Data saved successfully!")
    print(f"File saved in '{output_dir}' directory.")

    # free up ram 
    del pre_bands, post_bands

    # return path (for later tiling of labeled map, too)
    return os.path.join(output_dir, f'{dataset_name}_dnbr_heatmap.npy')

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

    Returns: 
        - metadata: Pandas DataFrame containing metadata for each tile
    """
    os.makedirs(path, exist_ok=True)

    image = np.load(image_path)

    metadata = pd.DataFrame(columns=['tile_number', 'tile_name', 'tile_coordinates', 'cloud_percentage', 'cloud?', 'forest?'])

    h, w, c = image.shape
    ph, pw, pc = tile_size
    
    # Calculate padding needed
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    
    # Pad with reflection to maintain natural patterns
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    counter = 1
    
    # Extract non-overlapping tiles
    padded_h, padded_w, _ = padded_image.shape
    for i in range(0, padded_h, ph):
        for j in range(0, padded_w, pw):

            metadata = pd.concat([
                metadata,
                pd.DataFrame([{
                    'tile_number': counter,
                    'tile_name': f'{name}_tile_{(i, j)}',
                    'tile_coordinates': (i, j),
                    'cloud?': None,
                    'forest?': None
                }])
            ], ignore_index=True)

            counter += 1

            tile = padded_image[i:i+ph, j:j+pw, :]
            np.save(os.path.join(path, f'{name}_tile_{(i, j)}'), tile)

    print("All tiles extracted correctly!")

    del image # free up RAM space

    return metadata

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

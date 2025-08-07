import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from rasterio.warp import reproject, Resampling
from scipy.ndimage import grey_closing
from .geodata_extraction import extract_geospatial_metadata

# resample bands if needed
def resampling_bands(bands_data, reference_profile, target_resolution): 
    """
    Resample Data Bands (if needed)

    Args:
        bands_data: the data from image .jp2 file
        reference_profile: the profile of the reference band (we need it just for localization purposes)
        target_resolution: int representing target m resolution

    Returns:
        result object, being a dictionary with the data, profile, band names and ordering
    """
    # resample bands that don't match the target resolution
    resampled_bands_data = {}
    for band_name, band_info in bands_data.items():
        if band_info['profile']['width'] != reference_profile['width'] or \
            band_info['profile']['height'] != reference_profile['height']:
                
            print(f"Resampling band {band_name} from {band_info['native_resolution']}m to {target_resolution}m")
                
            # create output array with target dimensions
            resampled_data = np.zeros((reference_profile['height'], reference_profile['width']), dtype=np.float32)
                
            # reproject the band data
            reproject(
                source=band_info['data'],
                destination=resampled_data,
                src_transform=band_info['profile']['transform'],
                src_crs=band_info['profile']['crs'],
                dst_transform=reference_profile['transform'],
                dst_crs=reference_profile['crs'],
                resampling=Resampling.bilinear
            )
                
            # update profile for resampled band
            resampled_profile = reference_profile.copy()
            resampled_profile.update({
                'dtype': band_info['profile']['dtype'],
                'nodata': band_info['profile'].get('nodata')
            })
                
            resampled_bands_data[band_name] = {
                'data': resampled_data,
                'profile': resampled_profile,
                'file_path': band_info['file_path'],
                'original_resolution': f"{band_info['native_resolution']}m",
                'resampled_to': target_resolution
            }
        else:
            # band already at target resolution
            resampled_bands_data[band_name] = band_info.copy()
            resampled_bands_data[band_name]['original_resolution'] = f"{band_info['native_resolution']}m"
            resampled_bands_data[band_name]['resampled_to'] = target_resolution
        
    bands_data = resampled_bands_data
    print(f"All bands resampled to {target_resolution}m resolution")    
    
    # stack all bands into a single numpy array and return with unified profile
    # get the reference profile (all bands should have the same profile after resampling)
    reference_profile = next(iter(bands_data.values()))['profile']
    
    # stack all band data into a 3D array (height, width, bands)
    band_arrays = []
    band_names = []
    for band_name in sorted(bands_data.keys()):  
        band_arrays.append(bands_data[band_name]['data'])
        band_names.append(band_name)
    
    stacked_data = np.stack(band_arrays, axis=2)
    
    # create result structure
    result = {
        'data': stacked_data,
        'profile': reference_profile,
        'band_names': band_names,
        'band_order': {i: name for i, name in enumerate(band_names)}
    }

    return result

#--------------------------------------------------------------------------------

# read data from folders
def read_sent2_1c_bands(base_path: str, 
                        band_list:list=['B01', 'B02', 'B03', 'B4', 'B05', 'B06', 'B07', 
                                        'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']):
    """
    Read specific bands from a Sentinel-2 Level 1C dataset and resample to common resolution if needed
    
    Args:
        base_path (str): Base path name to the Sentinel-2 Level 1C product in the DATASETS folder (e.g., turkey, greece, italy...)
        band_list (list): List of bands to read (by default it reads all bands)
    
    Returns:
        dict: Dictionary with structure:
              {
                  'data': numpy array with shape (height, width, bands) containing stacked band data,
                  'profile': rasterio profile with geospatial metadata (unified for all bands),
                  'band_names': list of band names in the order they appear in the data array,
                  'band_order': dict mapping array index to band name,
                  'resampling_info': dict with resampling details (if resampling occurred)
              }
              All bands will be resampled to the highest resolution (lowest numerical value) and stacked together.
    """
    
    # Define native resolutions for Sentinel-2 Level 1C bands
    band_native_resolutions = {
        'B01': 60,  # Coastal aerosol
        'B02': 10,  # Blue
        'B03': 10,  # Green
        'B04': 10,  # Red
        'B05': 20,  # Vegetation Red Edge
        'B06': 20,  # Vegetation Red Edge
        'B07': 20,  # Vegetation Red Edge
        'B08': 10,  # NIR
        'B8A': 20,  # Vegetation Red Edge
        'B09': 60,  # Water vapour
        'B10': 60,  # Cirrus (available in L1C)
        'B11': 20,  # SWIR
        'B12': 20   # SWIR
    }

    # 'DATASETS' relative path where all images folders should be
    base_path = os.path.join('DATASETS', base_path)
    # path of actual image bands (handle folder with long name in between granule and img_data)
    img_data_pattern = os.path.join(base_path, 'GRANULE', '*', 'IMG_DATA')
    img_data_path = glob.glob(img_data_pattern)[0]
    
    if not os.path.exists(img_data_path):
        raise ValueError(f"IMG_DATA folder not found.")
    
    bands_data = {}
    
    # read each requested band
    for band in band_list:
        if band not in band_native_resolutions:
            print(f"Warning: Band {band} not recognized. Skipping...")
            continue
            
        # search for band file in the IMG_DATA folder
        # format: T{img_id}_{timestamp}_{band}.jp2
        band_file = glob.glob(os.path.join(img_data_path, f'*_{band}.jp2'))
        
        if band_file:
            print(f"Reading band {band} from: {band_file[0]}")
            
            with rasterio.open(band_file[0]) as src:
                bands_data[band] = {
                    'data': src.read(1).astype(np.float32),
                    'profile': src.profile,
                    'file_path': str(band_file[0]),
                    'native_resolution': band_native_resolutions[band]
                }
        else:
            print(f"Warning: Band {band} not found in {img_data_path}")
    
    if not bands_data:
        return {'data': None, 'profile': None, 'band_names': [], 'band_order': {}}
    
    # get the highest resolution (lowest numerical value) among the requested bands
    resolutions = [band_native_resolutions[band] for band in bands_data.keys()]
    target_resolution = min(resolutions)
    
    print(f"Target resolution: {target_resolution}m")
    
    # find a reference band at the target resolution for spatial reference
    reference_band = None
    reference_profile = None
    for band_name, band_info in bands_data.items():
        if band_info['native_resolution'] == target_resolution:
            reference_band = band_name
            reference_profile = band_info['profile']
            break
    
    if reference_band is None:
        raise ValueError(f"No reference band found at target resolution {target_resolution}m")
    
    # check if resampling is needed
    needs_resampling = any(band_info['native_resolution'] != target_resolution 
                          for band_info in bands_data.values())
    
    if needs_resampling:
        print(f"Multiple resolutions detected. Resampling all bands to {target_resolution}m")
        result = resampling_bands(bands_data, reference_profile, target_resolution)
    else:
        # no resampling needed - stack bands directly
        for band_name, band_info in bands_data.items():
            band_info['original_resolution'] = f"{band_info['native_resolution']}m"
            band_info['resampled_to'] = target_resolution
        
        # stack all band data into a 3D array (height, width, bands)
        band_arrays = []
        band_names = []
        for band_name in sorted(bands_data.keys()):  
            band_arrays.append(bands_data[band_name]['data'])
            band_names.append(band_name)
        
        stacked_data = np.stack(band_arrays, axis=2)
        
        # create result structure
        result = {
            'data': stacked_data,
            'profile': reference_profile,
            'band_names': band_names,
            'band_order': {i: name for i, name in enumerate(band_names)}
        }
    
    return result

#--------------------------------------------------------------------------------

# extract the dNBR map (of values between 0 and 1) and the dNBR binary map
def extract_data_labels_from_bands(pre_bands_data, post_bands_data, output_dir: str, thresh: float = 0.6, masking: bool = True):
    """
    Create data labels from Sentinel-2 images by considering the dNBR value (with NDVI masking)
    
    Params:
        - img_list: List of paths to Sentinel-2 Level 1C product directories [pre-fire, post-fire]
        - output_dir: Location where to save the results
        - thresh: threshold of dNBR values to use to create binary map
        - masking: bool param to determine whether to mask to 0 pixels that are not vegetation, using ndvi
            or not to do it (as these will be later discarded)

    Returns:
        numpy.ndarray: dNBR normalized image with shape (height, width, 1)
    """
    
    nbr_imgs = {}
    ndvi_imgs = {}
    
    # Process both datasets
    for i, bands_data in enumerate([pre_bands_data, post_bands_data]):
        band_order = bands_data['band_order']
        
        # Find B08 (NIR), B12 (SWIR)
        try:
            b08_idx = next(idx for idx, name in band_order.items() if name == 'B08')
            b12_idx = next(idx for idx, name in band_order.items() if name == 'B12')
            # if applying masking, find indices for ndvi
            if masking:
                b04_idx = next(idx for idx, name in band_order.items() if name == 'B04')
        except StopIteration:
            raise ValueError("Some bands not found in the provided bands data")
        
        b08_data = bands_data['data'][:, :, b08_idx]
        b12_data = bands_data['data'][:, :, b12_idx]

        # Calculate NBR
        denom_nbr = b08_data + b12_data
        denom_nbr[denom_nbr == 0] = 1e-6
        nbr_img = (b08_data - b12_data) / denom_nbr
        nbr_imgs[i] = nbr_img

        if masking:
            b08_data = bands_data['data'][:, :, b08_idx]
            b04_data = bands_data['data'][:, :, b04_idx]
            # Calculate NDVI
            denom_ndvi = b08_data + b04_data
            denom_ndvi[denom_ndvi == 0] = 1e-6
            ndvi_img = (b08_data - b04_data) / denom_ndvi
            ndvi_imgs[i] = ndvi_img

        
    # get final dnbr image
    dnbr_img = nbr_imgs[0] - nbr_imgs[1]
    
    # apply masking: we don't care about pixel values with no significant change in NBR between pre- and post-
    if masking:
        # Create a binary map in one step, without modifying dnbr_img
        dnbr_map = np.where(dnbr_img < 0.2, 0, np.where(dnbr_img < 0.4, 1, 2))

    # normalize between 0 and 1 to get heatmap of probabilities 
    #dnbr_img = (dnbr_img - np.min(dnbr_img)) / (np.max(dnbr_img) - np.min(dnbr_img))

    # apply closing operation to get smoother label areas (without low value pixels inside high risk areas)
    dnbr_map = grey_closing(dnbr_map, size=5)

    # save this data
    os.makedirs(output_dir, exist_ok=True)

    # Save as numpy file
    np.save(os.path.join(output_dir, 'dnbr_normalized.npy'), dnbr_map)

    print("Data saved successfully!")
    print(f"Files saved in '{output_dir}' directory:")
    print("- dnbr_normalized.npy (NumPy array)")
    print("- dnbr_binary_map.npy (NumPy array)")

    return dnbr_img

#--------------------------------------------------------------------------------

# extract the single tiles of dim 256x256 (or whatever needed) from the large image
def extract_tiles_with_padding(image, name, tile_size, path):
    """
    Extract tiles using padding strategy to ensure complete coverage.
    
    Params:
        - image: Input image (H, W, C)
        - name: The name of the location of the image (needed for data organizational purposes)
        - tile_size: Size of each tile (height, width, channels)
        - path: Location where to save tiles
    """
    os.makedirs(path, exist_ok=True)

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

    print("All tileed extracted correctly!")

#--------------------------------------------------------------------------------

# unified function to compute vegetation indices from single .npy tile file and add as new channels
def compute_veg_indices(tile_path: str, band_names: list, indices: list = ['ndvi', 'ndmi']):
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
    
    # Check if vegetation indices are already present
    expected_bands_with_indices = len(band_names) + len(indices)
    if tile_data.shape[2] >= expected_bands_with_indices:
        # Return existing data with updated band names
        updated_band_names = band_names + [idx.upper() for idx in indices]
        return tile_data, updated_band_names
    
    # define band requirements for each index
    index_configs = {
        'ndvi': {'bands': ['B08', 'B04'], 'names': ['NIR', 'Red']},
        'ndmi': {'bands': ['B08', 'B11'], 'names': ['NIR', 'SWIR']}
    }
    
    # Start with original tile data
    expanded_tile = tile_data.copy()
    updated_band_names = band_names.copy()
    
    # Compute and add each requested index
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
        
        # Add as new channel
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

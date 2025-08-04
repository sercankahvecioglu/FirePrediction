import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from utils.bands_preprocessing import read_sent2_1c_bands


#--------------------------------------------------------------------------------

# get ndvi = (nir - r)/(nir + r)
def get_ndvi_from_bands(bands_data):
    """
    Calculate NDVI from pre-extracted bands data
    
    Args:
        bands_data: Dictionary from read_sent2_1c_bands containing band data
    
    Returns:
        numpy.ndarray: NDVI data with shape (height, width, 1)
    """
    # Extract band data from stacked array
    band_order = bands_data['band_order']
    band_names = bands_data['band_names']
    
    # Find B08 (NIR) and B04 (Red) indices
    try:
        b08_idx = next(i for i, name in band_order.items() if name == 'B08')
        b04_idx = next(i for i, name in band_order.items() if name == 'B04')
    except StopIteration:
        raise ValueError("B08 (NIR) or B04 (Red) bands not found in the provided bands data")
    
    b08_data = bands_data['data'][:, :, b08_idx]
    b04_data = bands_data['data'][:, :, b04_idx]
    
    denom = b08_data + b04_data
    denom[denom==0] = 1e-6
    # avoid division by zero
    ndvi_img = (b08_data - b04_data)/denom
    
    # Add channel dimension to make it 3D (height, width, 1)
    ndvi_img = ndvi_img[:, :, np.newaxis]

    return ndvi_img

# Backward compatibility function
def get_ndvi(img_path):
    """Legacy function for backward compatibility - reads bands and calculates NDVI"""
    nir_r = read_sent2_1c_bands(img_path, ['B08', 'B04'], ['10m', '10m'])
    return {'data': get_ndvi_from_bands(nir_r), 'profile': nir_r['profile']}

# get ndmi = (b08 - b11)/(b08 + b11)
def get_ndmi_from_bands(bands_data):
    """
    Calculate NDMI from pre-extracted bands data
    
    Args:
        bands_data: Dictionary from read_sentinel2_bands containing band data
    
    Returns:
        numpy.ndarray: NDMI data with shape (height, width, 1)
    """
    # Extract band data from stacked array
    band_order = bands_data['band_order']
    band_names = bands_data['band_names']
    
    # Find B08 (NIR) and B11 (SWIR) indices
    try:
        b08_idx = next(i for i, name in band_order.items() if name == 'B08')
        b11_idx = next(i for i, name in band_order.items() if name == 'B11')
    except StopIteration:
        raise ValueError("B08 (NIR) or B11 (SWIR) bands not found in the provided bands data")

    b08_data = bands_data['data'][:, :, b08_idx]
    b11_data = bands_data['data'][:, :, b11_idx]

    denom = b08_data + b11_data
    denom[denom==0] = 1e-6
    # avoid division by zero
    ndmi_img = (b08_data - b11_data)/denom
    
    # Add channel dimension to make it 3D (height, width, 1)
    ndmi_img = ndmi_img[:, :, np.newaxis]

    return ndmi_img

# Backward compatibility function
def get_ndmi(img_path):
    """Legacy function for backward compatibility - reads bands and calculates NDMI"""
    nir_swir = read_sent2_1c_bands(img_path, ['B08', 'B11'])
    return {'data': get_ndmi_from_bands(nir_swir), 'profile': nir_swir['profile']}

# open a .tif image file with rasterio

import rasterio
import xarray as xr
import numpy as np
import os   
import matplotlib.pyplot as plt


img = rasterio.open('/home/dario/Desktop/FlameSentinels/tiles_post/tile_033_000.tif')

# extract all data about bands and metadata
data = img.read()


# first print the shape of the data and the metadata (bands and other info)
print("Data shape:", data.shape)
print("Metadata:", img.meta)


def display_full_resolution(rgb_array, save_filename=None):
    """Display and optionally save RGB image at full resolution"""
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_array)
    plt.axis('off')
    
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"Image saved as {save_filename}")
    
    plt.show()

def read_rgb_image(file_path):
    """Read RGB image from a file and return as numpy array"""
    with rasterio.open(file_path) as src:
        rgb_data = src.read([1, 2, 3])  # Assuming bands 1, 2, 3 are RGB
        rgb_data = np.moveaxis(rgb_data, 0, -1)  # Move bands to last dimension
    return rgb_data

# Read the RGB image
rgb_image = read_rgb_image('/home/dario/Desktop/FlameSentinels/tiles_post/tile_022_007.tif')

# Normalize the image for display (convert to 0-1 range)
def normalize_image(rgb_array):
    """Normalize RGB array for display"""
    # Convert to float and normalize each channel
    rgb_normalized = rgb_array.astype(np.float32)
    
    # Apply percentile stretch for better visualization
    for i in range(3):
        channel = rgb_normalized[:, :, i]
        # Get 2nd and 98th percentiles for contrast stretching
        p2, p98 = np.percentile(channel[channel > 0], [2, 98])
        # Normalize to 0-1 range
        channel_norm = np.clip((channel - p2) / (p98 - p2), 0, 1)
        rgb_normalized[:, :, i] = channel_norm
    
    return rgb_normalized

# Normalize and display the image
rgb_normalized = normalize_image(rgb_image)
print(f"Image shape: {rgb_image.shape}")
print(f"Original data range: {rgb_image.min()} - {rgb_image.max()}")
print(f"Normalized data range: {rgb_normalized.min()} - {rgb_normalized.max()}")

# Display the full resolution image
display_full_resolution(rgb_normalized, save_filename='tile_022_007_display.png')


#!/usr/bin/env python3
"""
Fire Data Visualization Script (Memory Efficient)

This script takes pre and post fire images and labels, processes them in chunks
to minimize RAM usage, and creates RGB visualizations and colored heatmaps.

Usage:
    python visualize_fire_data.py --region chile
    python visualize_fire_data.py --region california --tile_size 512
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

def process_image_chunk(image_path, start_row, start_col, chunk_size, target_channels=None):
    """
    Load and process a chunk of the image to minimize memory usage
    """
    # Load only the image header to get shape info
    image = np.load(image_path, mmap_mode='r')  # Memory-mapped reading
    h, w, c = image.shape
    
    # Calculate actual chunk bounds
    end_row = min(start_row + chunk_size, h)
    end_col = min(start_col + chunk_size, w)
    
    # Extract the chunk
    if target_channels is not None:
        chunk = np.array(image[start_row:end_row, start_col:end_col, :target_channels])
    else:
        chunk = np.array(image[start_row:end_row, start_col:end_col, :])
    
    return chunk

def normalize_band(band, percentile_clip=2):
    """Normalize a single band to 0-255 range with percentile clipping"""
    p_low, p_high = np.percentile(band, [percentile_clip, 100 - percentile_clip])
    band_clipped = np.clip(band, p_low, p_high)
    return ((band_clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

def create_rgb_image(tile_data, rgb_bands=[3, 2, 1]):  # B04(Red), B03(Green), B02(Blue)
    """
    Create RGB image from sentinel-2 bands
    Default uses B04, B03, B02 for natural color
    """
    # Extract RGB bands (adjust indices for 0-based indexing)
    red = tile_data[:, :, rgb_bands[0]]
    green = tile_data[:, :, rgb_bands[1]] 
    blue = tile_data[:, :, rgb_bands[2]]
    
    # Normalize each band
    red_norm = normalize_band(red)
    green_norm = normalize_band(green)
    blue_norm = normalize_band(blue)
    
    # Stack to create RGB image
    rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)
    return rgb_image

def create_heatmap(label_data, colormap='coolwarm'):
    """Create colored heatmap from label data"""
    # Squeeze to remove single dimension if present
    if label_data.ndim == 3 and label_data.shape[2] == 1:
        label_data = label_data.squeeze(axis=2)
    
    # Create colormap
    cmap = plt.get_cmap(colormap)
    
    # Normalize to 0-1 range
    label_norm = (label_data - label_data.min()) / (label_data.max() - label_data.min() + 1e-8)
    
    # Apply colormap
    colored_heatmap = cmap(label_norm)
    
    # Convert to 0-255 range and remove alpha channel
    heatmap_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
    
    return heatmap_image

def save_image(image_array, save_path):
    """Save image array as PNG file"""
    plt.figure(figsize=(8, 8))
    plt.imshow(image_array)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    plt.clf()  # Clear figure to free memory

def get_image_info(image_path):
    """Get image dimensions without loading into memory"""
    image = np.load(image_path, mmap_mode='r')
    return image.shape

def visualize_fire_data(region, tile_size=256, data_dir='data', labels_dir='full_labels'):
    """
    Main function to visualize fire data for a given region with minimal memory usage
    """
    # Paths
    base_dir = '/home/dario/Desktop/FirePrediction'
    pre_path = os.path.join(base_dir, data_dir, f'{region}_pre.npy')
    post_path = os.path.join(base_dir, data_dir, f'{region}_post.npy')
    label_path = os.path.join(base_dir, labels_dir, f'{region}_dnbr_heatmap.npy')
    
    # Output directories
    viz_pre_dir = os.path.join(base_dir, 'viz_pre')
    viz_post_dir = os.path.join(base_dir, 'viz_post')
    viz_labels_dir = os.path.join(base_dir, 'viz_labels')
    
    # Create output directories
    for dir_path in [viz_pre_dir, viz_post_dir, viz_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Check if input files exist
    for file_path, name in [(pre_path, 'pre'), (post_path, 'post'), (label_path, 'labels')]:
        if not os.path.exists(file_path):
            print(f"Error: {name} file not found at {file_path}")
            return
    
    print(f"Processing {region} fire data with memory-efficient approach...")
    
    # Get image dimensions without loading full images
    pre_shape = get_image_info(pre_path)
    post_shape = get_image_info(post_path)
    label_shape = get_image_info(label_path)
    
    print(f"Pre-fire shape: {pre_shape}")
    print(f"Post-fire shape: {post_shape}")
    print(f"Label shape: {label_shape}")
    
    # Calculate number of tiles
    h, w = pre_shape[0], pre_shape[1]
    n_tiles_h = (h + tile_size - 1) // tile_size  # Ceiling division
    n_tiles_w = (w + tile_size - 1) // tile_size
    total_tiles = n_tiles_h * n_tiles_w
    
    print(f"Will create {total_tiles} tiles ({n_tiles_h}x{n_tiles_w})")
    
    tile_count = 0
    
    # Process tiles row by row to minimize memory usage
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile_count += 1
            
            try:
                print(f"Processing tile {tile_count}/{total_tiles} at ({i}, {j})")
                
                # Load chunks from each image
                pre_chunk = process_image_chunk(pre_path, i, j, tile_size)
                post_chunk = process_image_chunk(post_path, i, j, tile_size)
                label_chunk = process_image_chunk(label_path, i, j, tile_size, target_channels=1)
                
                # Skip tiles that are too small or empty
                if pre_chunk.shape[0] < 32 or pre_chunk.shape[1] < 32:
                    print(f"  Skipping small tile at ({i}, {j})")
                    continue
                
                # Generate tile name
                tile_name = f'{region}_tile_{i}_{j}'
                
                # Create RGB images
                pre_rgb = create_rgb_image(pre_chunk)
                post_rgb = create_rgb_image(post_chunk)
                
                # Create heatmap
                label_heatmap = create_heatmap(label_chunk)
                
                # Save visualizations
                save_image(pre_rgb, os.path.join(viz_pre_dir, f'{tile_name}_pre_rgb.png'))
                save_image(post_rgb, os.path.join(viz_post_dir, f'{tile_name}_post_rgb.png'))
                save_image(label_heatmap, os.path.join(viz_labels_dir, f'{tile_name}_heatmap.png'))
                
                # Free memory
                del pre_chunk, post_chunk, label_chunk, pre_rgb, post_rgb, label_heatmap
                
                print(f"  Saved tile {tile_count}")
                
            except Exception as e:
                print(f"Error processing tile at ({i}, {j}): {e}")
                continue
    
    print(f"Visualization complete! Created {tile_count} tiles.")
    print(f"Check directories:")
    print(f"  - RGB pre-fire images: {viz_pre_dir}")
    print(f"  - RGB post-fire images: {viz_post_dir}")
    print(f"  - Fire severity heatmaps: {viz_labels_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize fire prediction data (memory efficient)')
    parser.add_argument('--region', type=str, required=True,
                       help='Region name (e.g., chile, california, greece)')
    parser.add_argument('--tile_size', type=int, default=256,
                       help='Size of tiles to extract (default: 256)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing input data (default: data)')
    parser.add_argument('--labels_dir', type=str, default='full_labels',
                       help='Directory containing labels (default: full_labels)')
    
    args = parser.parse_args()
    
    visualize_fire_data(args.region, args.tile_size, args.data_dir, args.labels_dir)

if __name__ == "__main__":
    main()

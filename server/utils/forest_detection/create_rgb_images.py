#!/usr/bin/env python3
"""
Standalone script to create RGB images from .npy satellite tiles
Usage: python create_rgb_images.py <input_folder> <output_folder>
"""

import os
import sys
import numpy as np
import glob
from PIL import Image


def create_rgb_images_from_tiles(input_folder, output_folder, rgb_channels=[3, 2, 1]):
    """
    Create RGB images from .npy tiles
    
    Args:
        input_folder (str): Path to folder containing .npy files
        output_folder (str): Path to save RGB images
        rgb_channels (list): Channels to use for RGB [R, G, B] - default [3, 2, 1]
    
    Returns:
        list: Paths of created images
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    npy_files = glob.glob(os.path.join(input_folder, "*.npy"))
    created_images = []
    
    print(f"Found {len(npy_files)} .npy files in {input_folder}")
    
    for i, npy_file in enumerate(npy_files):
        try:
            print(f"Processing {i+1}/{len(npy_files)}: {os.path.basename(npy_file)}")
            
            # Load image data
            image_data = np.load(npy_file)
            
            # Check if data is in [height, width, channels] format and transpose if needed
            if len(image_data.shape) == 3 and image_data.shape[2] < image_data.shape[0]:
                # Data is in [height, width, channels] format, transpose to [channels, height, width]
                image_data = np.transpose(image_data, (2, 0, 1))
            
            # Check if we have enough channels
            if image_data.shape[0] <= max(rgb_channels):
                print(f"  Warning: Not enough channels in {npy_file}. Has {image_data.shape[0]}, needs {max(rgb_channels)+1}")
                continue
            
            # Extract RGB channels
            red = image_data[rgb_channels[0]].astype(np.float32)
            green = image_data[rgb_channels[1]].astype(np.float32)
            blue = image_data[rgb_channels[2]].astype(np.float32)
            
            # Normalize to 0-1 range
            def normalize_channel(channel):
                # Handle cases where channel might have NaN or inf values
                channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
                
                min_val = np.percentile(channel, 2)  # Use 2nd percentile to avoid outliers
                max_val = np.percentile(channel, 98)  # Use 98th percentile to avoid outliers
                
                # Avoid division by zero
                if max_val == min_val:
                    return np.ones_like(channel) * 0.5  # Return middle gray if no variation
                
                normalized = (channel - min_val) / (max_val - min_val)
                return np.clip(normalized, 0, 1)
            
            red_norm = normalize_channel(red)
            green_norm = normalize_channel(green)
            blue_norm = normalize_channel(blue)
            
            # Stack channels to create RGB image [height, width, 3]
            rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)
            
            # Convert to 8-bit
            rgb_image_8bit = (rgb_image * 255).astype(np.uint8)
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(npy_file))[0]
            output_path = os.path.join(output_folder, f"{base_name}_rgb.png")
            
            # Save as PNG using PIL
            pil_image = Image.fromarray(rgb_image_8bit)
            pil_image.save(output_path)
            
            created_images.append(output_path)
            print(f"  ✓ Created: {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"  ✗ Error processing {npy_file}: {e}")
    
    return created_images


def main():
    if len(sys.argv) != 3:
        print("Usage: python create_rgb_images.py <input_folder> <output_folder>")
        print("Example: python create_rgb_images.py /path/to/npy/files /path/to/output/images")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        sys.exit(1)
    
    print(f"Creating RGB images from .npy tiles...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"RGB channels: [3, 2, 1] (Red, Green, Blue)")
    print("-" * 50)
    
    try:
        created_images = create_rgb_images_from_tiles(input_folder, output_folder)
        print("-" * 50)
        print(f"✅ Successfully created {len(created_images)} RGB images")
        print(f"📁 Images saved to: {output_folder}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

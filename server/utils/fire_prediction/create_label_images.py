#!/usr/bin/env python3
"""
Script to convert label files (.npy) to RGB images using matplotlib coolwarm colormap.
Processes all label files in TILES_LABELS and saves RGB images to LABELS_IMGS folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def create_label_images(labels_dir="TILES_LABELS", output_dir="LABELS_IMGS", n_classes=3):
    """
    Convert label files to RGB images using custom colormap.
    
    Args:
        labels_dir: Directory containing .npy label files
        output_dir: Directory to save RGB images
        n_classes: Number of classes in the labels (default: 3)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .npy files in the labels directory
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.npy')]
    
    if not label_files:
        print(f"No .npy files found in {labels_dir}")
        return
    
    print(f"Found {len(label_files)} label files to process")
    
    # Create a colormap for the classes
    # Custom color mapping: 0=navy blue, 1=yellow, 2=red
    colors = [(0, 0, 0.5, 1), (1, 1, 0, 1), (1, 0, 0, 1)]  # Navy Blue, Yellow, Red (RGBA)
    custom_cmap = ListedColormap(colors[:n_classes])
    
    # Process each label file
    for i, filename in enumerate(label_files):
        print(f"Processing {i+1}/{len(label_files)}: {filename}")
        try:
            # Load the label file
            label_path = os.path.join(labels_dir, filename)
            labels = np.load(label_path)
            
            # Remove channel dimension if present (e.g., shape (256, 256, 1) -> (256, 256))
            if len(labels.shape) == 3 and labels.shape[2] == 1:
                labels = labels.squeeze(2)
            
            # Ensure labels are in valid range [0, n_classes-1]
            labels = np.clip(labels, 0, n_classes - 1)
            
            # Create the image using the colormap
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(labels, cmap=custom_cmap, vmin=0, vmax=n_classes-1)
            ax.axis('off')  # Remove axes
            
            # Save the image
            output_filename = filename.replace('.npy', '_labels.png')
            output_path = os.path.join(output_dir, output_filename)
            
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()  # Close the figure to free memory
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"Successfully converted {len(label_files)} label files to images")
    print(f"Images saved in: {output_dir}")
    
    # Print some statistics
    sample_file = os.path.join(labels_dir, label_files[0])
    sample_data = np.load(sample_file)
    if len(sample_data.shape) == 3 and sample_data.shape[2] == 1:
        sample_data = sample_data.squeeze(2)
    
    unique_classes = np.unique(sample_data)
    print(f"Classes found in sample file: {unique_classes}")
    print(f"Color mapping:")
    for i, class_id in enumerate(range(n_classes)):
        color_rgb = [int(c * 255) for c in colors[i][:3]]  # Convert to RGB 0-255
        print(f"  Class {class_id}: RGB{tuple(color_rgb)}")


def create_input_images(input_dir="TILES_INPUT_DATA", output_dir="INPUT_IMGS", bands=[3, 2, 1]):
    """
    Convert input data files to RGB images using specified bands.
    
    Args:
        input_dir: Directory containing .npy input data files
        output_dir: Directory to save RGB images
        bands: List of band indices to use for RGB visualization (default: [3, 2, 1])
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .npy files in the input directory (excluding pkl files)
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.npy') and not f.endswith('.pkl')]
    
    if not input_files:
        print(f"No .npy files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} input files to process")
    
    # Process each input file
    for i, filename in enumerate(input_files):
        print(f"Processing {i+1}/{len(input_files)}: {filename}")
        try:
            # Load the input file
            input_path = os.path.join(input_dir, filename)
            input_data = np.load(input_path)
            
            # Check if we have enough bands
            if len(input_data.shape) != 3 or input_data.shape[2] < max(bands) + 1:
                print(f"Warning: {filename} doesn't have enough bands. Shape: {input_data.shape}")
                continue
            
            # Extract the specified bands (note: bands are 0-indexed)
            rgb_data = input_data[:, :, bands]  # Shape: (H, W, 3)
            
            # Normalize the data to [0, 1] range for visualization
            # Handle potential extreme values
            rgb_normalized = np.zeros_like(rgb_data)
            for channel in range(3):
                channel_data = rgb_data[:, :, channel]
                # Use percentile-based normalization to handle outliers
                p2, p98 = np.percentile(channel_data, [2, 98])
                channel_normalized = np.clip((channel_data - p2) / (p98 - p2), 0, 1)
                rgb_normalized[:, :, channel] = channel_normalized
            
            # Create the image
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(rgb_normalized)
            ax.axis('off')  # Remove axes
            
            # Save the image
            output_filename = filename.replace('.npy', '_input.png')
            output_path = os.path.join(output_dir, output_filename)
            
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()  # Close the figure to free memory
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"Successfully converted {len(input_files)} input files to images")
    print(f"Images saved in: {output_dir}")
    
    # Print some statistics from a sample file
    if input_files:
        sample_file = os.path.join(input_dir, input_files[0])
        sample_data = np.load(sample_file)
        print(f"Sample input file shape: {sample_data.shape}")
        print(f"Using bands {bands} for RGB visualization")
        if len(sample_data.shape) == 3:
            for i, band_idx in enumerate(bands):
                band_data = sample_data[:, :, band_idx]
                print(f"  Band {band_idx} -> RGB channel {['R', 'G', 'B'][i]}: min={band_data.min():.3f}, max={band_data.max():.3f}")


def create_combined_visualization(input_dir="TILES_INPUT_DATA", labels_dir="TILES_LABELS", 
                                output_dir="LABELS_IMGS", bands=[3, 2, 1], n_classes=3):
    """
    Create side-by-side visualization of input images and their corresponding labels.
    
    Args:
        input_dir: Directory containing .npy input data files
        labels_dir: Directory containing .npy label files
        output_dir: Directory to save combined RGB images
        bands: List of band indices to use for RGB visualization (default: [3, 2, 1])
        n_classes: Number of classes in the labels (default: 3)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .npy files in both directories
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.npy') and not f.endswith('.pkl')]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.npy')]
    
    # Find common files
    common_files = list(set(input_files) & set(label_files))
    
    if not common_files:
        print(f"No common .npy files found between {input_dir} and {labels_dir}")
        return
    
    print(f"Found {len(common_files)} common files to process")
    
    # Create a colormap for the classes
    colors = [(0, 0, 0.5, 1), (1, 1, 0, 1), (1, 0, 0, 1)]  # Navy Blue, Yellow, Red (RGBA)
    custom_cmap = ListedColormap(colors[:n_classes])
    
    # Process each common file
    for i, filename in enumerate(sorted(common_files)):
        print(f"Processing {i+1}/{len(common_files)}: {filename}")
        try:
            # Load the input file
            input_path = os.path.join(input_dir, filename)
            input_data = np.load(input_path)
            
            # Load the label file
            label_path = os.path.join(labels_dir, filename)
            labels = np.load(label_path)
            
            # Check input data has enough bands
            if len(input_data.shape) != 3 or input_data.shape[2] < max(bands) + 1:
                print(f"Warning: {filename} doesn't have enough bands. Shape: {input_data.shape}")
                continue
            
            # Extract and normalize RGB data
            rgb_data = input_data[:, :, bands]
            rgb_normalized = np.zeros_like(rgb_data)
            for channel in range(3):
                channel_data = rgb_data[:, :, channel]
                p2, p98 = np.percentile(channel_data, [2, 98])
                channel_normalized = np.clip((channel_data - p2) / (p98 - p2), 0, 1)
                rgb_normalized[:, :, channel] = channel_normalized
            
            # Process labels
            if len(labels.shape) == 3 and labels.shape[2] == 1:
                labels = labels.squeeze(2)
            labels = np.clip(labels, 0, n_classes - 1)
            
            # Create side-by-side visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot input image
            ax1.imshow(rgb_normalized)
            ax1.set_title(f'Input RGB (Bands {bands})', fontsize=14)
            ax1.axis('off')
            
            # Plot labels
            ax2.imshow(labels, cmap=custom_cmap, vmin=0, vmax=n_classes-1)
            ax2.set_title('Labels', fontsize=14)
            ax2.axis('off')
            
            # Add overall title
            fig.suptitle(filename.replace('.npy', ''), fontsize=16)
            
            # Save the combined image
            output_filename = filename.replace('.npy', '_combined.png')
            output_path = os.path.join(output_dir, output_filename)
            
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()  # Close the figure to free memory
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"Successfully created {len(common_files)} combined visualizations")
    print(f"Images saved in: {output_dir}")

def create_single_label_image(input_path, output_path, n_classes=3):
    """
    Convert a single label file to RGB image using custom colormap.
    
    Args:
        input_path: Path to the .npy label file
        output_path: Path where the RGB image will be saved
        n_classes: Number of classes in the labels (default: 3)
    """
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing single file: {input_path}")
    
    # Create a colormap for the classes
    # Custom color mapping: 0=navy blue, 1=yellow, 2=red
    colors = [(0, 0, 0.5, 1), (1, 1, 0, 1), (1, 0, 0, 1)]  # Navy Blue, Yellow, Red (RGBA)
    custom_cmap = ListedColormap(colors[:n_classes])
    
    try:
        # Load the label file
        labels = np.load(input_path)
        
        # Remove channel dimension if present (e.g., shape (256, 256, 1) -> (256, 256))
        if len(labels.shape) == 3 and labels.shape[2] == 1:
            labels = labels.squeeze(2)
        
        # Ensure labels are in valid range [0, n_classes-1]
        labels = np.clip(labels, 0, n_classes - 1)
        
        # Create the image using the colormap
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(labels, cmap=custom_cmap, vmin=0, vmax=n_classes-1)
        ax.axis('off')  # Remove axes
        
        # Save the image
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()  # Close the figure to free memory
        
        print(f"Successfully saved image to: {output_path}")
        
        # Print some statistics
        unique_classes = np.unique(labels)
        print(f"Classes found in file: {unique_classes}")
        print(f"Image shape: {labels.shape}")
        print(f"Color mapping:")
        for i, class_id in enumerate(range(n_classes)):
            color_rgb = [int(c * 255) for c in colors[i][:3]]  # Convert to RGB 0-255
            print(f"  Class {class_id}: RGB{tuple(color_rgb)}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        raise

if __name__ == "__main__":
    # Set the working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("="*60)
    print("FIRE PREDICTION DATA VISUALIZATION")
    print("="*60)
    
    # 1. Process TILES_LABELS - convert to categorical color images
    print("\n1. Processing TILES_LABELS directory...")
    create_label_images()
    
    # 2. Process TILES_INPUT_DATA - convert to RGB images using bands [3, 2, 1]
    print("\n2. Processing TILES_INPUT_DATA directory...")
    create_input_images()
    
    # 3. Create combined side-by-side visualizations
    print("\n3. Creating combined visualizations...")
    create_combined_visualization()
    
    # 4. Process the single file in full_labels (if exists)
    print("\n4. Processing full_labels directory...")
    full_labels_input = "full_labels/dnbr_normalized.npy"
    full_labels_output = "LABELS_IMGS/dnbr_normalized_full.png"
    
    if os.path.exists(full_labels_input):
        create_single_label_image(full_labels_input, full_labels_output)
    else:
        print(f"File not found: {full_labels_input}")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print("Generated image types:")
    print("- *_labels.png: Categorical visualization of label data")
    print("- *_input.png: RGB visualization of input data (bands 3,2,1)")
    print("- *_combined.png: Side-by-side input and label visualization")
    print(f"All images saved in: LABELS_IMGS/")
    print("="*60)

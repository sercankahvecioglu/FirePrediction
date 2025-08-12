import numpy as np
import scipy as sp
import torch
import os
import glob
from pathlib import Path


# NOTE: input data should be of type (h, w, c)

def rotate_90(data):
    """Rotate array 90 degrees counterclockwise"""
    return np.rot90(data, k=1, axes=(-3, -2))  # Rotate spatial dimensions (h, w)

def rotate_180(data):
    """Rotate array 180 degrees"""
    return np.rot90(data, k=2, axes=(-3, -2))  # Rotate spatial dimensions (h, w)

def flip_horizontal(data):
    """Flip array horizontally (left-right mirror)"""
    return np.flip(data, axis=-2)  # Flip along width (columns)

def flip_vertical(data):
    """Flip array vertically (up-down mirror)"""
    return np.flip(data, axis=-3)  # Flip along height (rows)

def augment_npy_files(folder_path):
    """
    Apply data augmentation to all .npy files in the given folder.
    
    Args:
        folder_path (str): Path to folder containing .npy files
    """
    folder_path = Path(folder_path)
    
    # Find all .npy files in the folder
    npy_files = list(folder_path.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {folder_path}")
        return
    
    print(f"Found {len(npy_files)} .npy files to augment")
    
    for npy_file in npy_files:
        print(f"Processing: {npy_file.name}")
        
        try:
            # Load the numpy array
            data = np.load(npy_file)
            
            # Apply augmentations and save with appropriate prefixes
            augmentations = {
                "rot90": rotate_90(data),
                "rot180": rotate_180(data),
                "flipped_h": flip_horizontal(data),
                "flipped_v": flip_vertical(data)
            }
            
            for prefix, augmented_data in augmentations.items():
                # Create new filename with prefix
                new_filename = f"{prefix}_{npy_file.name}"
                new_filepath = folder_path / new_filename
                
                # Save augmented data
                np.save(new_filepath, augmented_data)
                print(f"  Saved: {new_filename}")
                
        except Exception as e:
            print(f"Error processing {npy_file.name}: {str(e)}")
    
    print("Data augmentation completed!")

def main():
    """Main function to run data augmentation"""
    # Get the current working directory or specify folder path
    current_folder = "/home/dario/Desktop/sample_veg_detection_data/VEGETATION"
    
    # You can also specify a different folder by uncommenting and modifying the line below:
    # folder_to_process = input("Enter folder path (or press Enter for current directory): ").strip()
    # if folder_to_process:
    #     current_folder = folder_to_process
    
    print(f"Applying data augmentation to .npy files in: {current_folder}")
    augment_npy_files(current_folder)

if __name__ == "__main__":
    main()
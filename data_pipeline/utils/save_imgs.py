import os
import shutil
import glob
from pathlib import Path

def copy_npy_files_by_image_names(img_folder, npy_source_folder):
    """
    Copy .npy files that correspond to image names in the specified folder.
    
    Args:
        img_folder (str): Path to folder containing image files
        npy_source_folder (str): Path to folder containing .npy files
    """
    # Create output folder name
    img_folder_path = Path(img_folder)
    output_folder = str(img_folder_path.parent / f"{img_folder_path.name}_NPY")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Common image extensions
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.gif']
    
    # Get all image files in the input folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(img_folder, ext)))
        image_files.extend(glob.glob(os.path.join(img_folder, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {img_folder}")
        return
    
    copied_count = 0
    not_found_count = 0
    
    for img_path in image_files:
        # Get filename without extension
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Look for corresponding .npy file
        npy_filename = f"{img_filename}.npy"
        npy_source_path = os.path.join(npy_source_folder, npy_filename)
        
        if os.path.exists(npy_source_path):
            # Copy the .npy file to output folder
            npy_output_path = os.path.join(output_folder, npy_filename)
            try:
                shutil.copy2(npy_source_path, npy_output_path)
                print(f"Copied: {npy_filename}")
                copied_count += 1
            except Exception as e:
                print(f"Error copying {npy_filename}: {str(e)}")
        else:
            print(f"Not found: {npy_filename}")
            not_found_count += 1
    
    print(f"\nSummary:")
    print(f"- Total image files found: {len(image_files)}")
    print(f"- .npy files copied: {copied_count}")
    print(f"- .npy files not found: {not_found_count}")
    print(f"- Output folder: {output_folder}")

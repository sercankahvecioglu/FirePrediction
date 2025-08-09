import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils.bands_preprocessing import *
from utils.clouddetector import is_cloudy
from vegetation_filter import patch_vegetation_detection
import time
from shutil import copy2



# full processing pipeline to get all the tiles data needed (chosen bands, ndvi, ndmi, labels in [0, 1])
def full_sentinel2_data_pipeline(dataset_name: str, 
                                   base_path: str = '/home/dario/Desktop/FirePrediction',
                                   patch_size: tuple = (256, 256),
                                   cloud_threshold: float = 0.5):
    """
    Complete processing pipeline for satellite data including bands extraction, cloud filtering, 
    vegetation filtering, and NDVI/NDMI extraction.
    
    Args:
        dataset_name (str): Common name for the dataset folder (e.g., 'turkey', 'california', etc.)
        base_path (str): Base directory path where data folders are located
        patch_size (tuple): Size of tiles to extract (height, width)
        threshold (float): Threshold for binary dNBR classification
        cloud_threshold (float): Cloud probability threshold for S2PixelCloudDetector (0.0-1.0)
    """
    
    print(f"=== Starting processing pipeline for dataset: {dataset_name} ===")
    
    # Define paths
    full_labels_path = os.path.join(base_path, f'full_labels')
    
    # Output patch directories
    tiles_input_path = os.path.join(base_path, 'TILES_INPUT_DATA')
    tiles_labels_path = os.path.join(base_path, 'TILES_LABELS')

    os.makedirs(tiles_input_path, exist_ok=True)
    os.makedirs(tiles_labels_path, exist_ok=True)
    
    # Step 1: Extract data from folder and read all 13 bands
    all_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    
    #------------------------------------------------------------------------------------
    
    print("Obtaining Pre-Images geospatial information...")
    fname = os.path.join(base_path, "data_pkl", f"{dataset_name}_pre_tiles_data.pkl")
    copy2(fname, os.path.join(tiles_input_path, os.path.basename(fname)))
    print("Geospatial information saved successfully to input path!\n")

    #------------------------------------------------------------------------------------
    
    pre_path = os.path.join(base_path, 'data', f'{dataset_name}_pre.npy')
    post_path = os.path.join(base_path, 'data', f'{dataset_name}_post.npy')
    # Step 1: Create label map (dNBR)
    print("\n--- Step 1: Creating label map (dNBR) ---")
    os.makedirs(full_labels_path, exist_ok=True)
    dnbr_path = extract_data_labels(pre_path, post_path, full_labels_path)
    print("✓ dNBR label map created")

    #------------------------------------------------------------------------------------
    
    # Step 2: Divide into tiles of size (256, 256, 13)
    print("\n--- Step 2: Extracting tiles (256, 256, 13) ---")
    num_bands = len(all_bands)
    init_time = time.time()
    extract_tiles_with_padding(pre_path, dataset_name, (*patch_size, num_bands), tiles_input_path)
    dt = time.time() - init_time
    print(f"✓ input tiles extracted to {tiles_input_path} in {dt:.1f} seconds") 

    # Also extract label tiles
    dnbr_normmap = np.load(os.path.join(full_labels_path, 'dnbr_normalized.npy'))
    extract_tiles_with_padding(dnbr_path, dataset_name, (*patch_size, 1), tiles_labels_path)
    print(f"✓ label tiles extracted to {tiles_input_path}")

    #------------------------------------------------------------------------------------
    
    # Step 3: Apply cloud detection function
    print("\n--- Step 3: Applying cloud detection ---") # T: approx. 9 minutes per single image
    print(f"Using S2PixelCloudDetector with % of cloudy pixels threshold: {cloud_threshold}")
    init_time = time.time()
    cloud_results = is_cloudy(base_path, cloud_threshold)
    dt = time.time() - init_time
    print(f"✓ Cloud detection completed in {dt:.1f} seconds. Clean tiles: {cloud_results['clean_tiles']}, Cloudy tiles moved: {cloud_results['cloudy_tiles']}")
        
    #------------------------------------------------------------------------------------

    # Step 4: Apply vegetation detection model
    # TEMPORARY NDVI-BASED FILTERING
    print("\n--- Step 4: Applying vegetation detection ---")
    print(f"Using NDVI-based vegetation detection with {0.2*100}% vegetation coverage threshold")
    init_time = time.time()
    vegetation_results = patch_vegetation_detection(tiles_input_path, vegetation_threshold=0.2)
    dt = time.time() - init_time
    print(f"✓ Vegetation detection completed in {dt:.1f} seconds. Clean tiles: {vegetation_results['clean_tiles']}, Low vegetation tiles moved: {vegetation_results['low_vegetation_tiles']}")

    #------------------------------------------------------------------------------------

    # TODO: Implement filtering for black tiles (0 or smth like that for all rgb values)
    # - check if pixel is fully black 
    # - if so, discard it
    
    #------------------------------------------------------------------------------------

    # Step 5: Extract NDVI and NDMI for remaining tiles
    print("\n--- Step 5: Extracting NDVI and NDMI for remaining clean tiles ---")
    
    # Process remaining tiles in the tiles folder (after cloud & vegetation filtering)
    remaining_patch_files = [f for f in os.listdir(tiles_input_path) if f.endswith('.npy')]
    
    if remaining_patch_files:
        print(f"Processing {len(remaining_patch_files)} remaining clean tiles...")
        
        # Process NDVI & NDMI tiles
        process_tiles_directory_with_indices(tiles_input_path, all_bands)
        print(f"✓ NDVI & NDMI values added to the tiles in {tiles_input_path}")

    else:
        print("⚠ No clean tiles remaining after cloud detection!")
    
    # Print folder locations
    print(f"\n=== Processing pipeline completed for dataset: {dataset_name} ===")
    print("📁 OUTPUT FOLDER LOCATIONS:")
    print(f"  🔸 Clean bands tiles +  indices (15 bands): {tiles_input_path}")
    print(f"  🔸 Label tiles (dNBR): {tiles_labels_path}")
    print(f"  🔸 Full image label: {full_labels_path}")

    # Display dNBR heatmap
    ax = plt.imshow(dnbr_normmap[:, :, 0], cmap='coolwarm')
    plt.title('dNBR Heatmap', fontsize=12)
    plt.colorbar()
    plt.axis('off')
    plt.show()
    plt.savefig(os.path.join(full_labels_path, 'dnbr_heatmap.png'))


    return {
        'dataset_name': dataset_name,
        'bands_tiles': tiles_input_path,
        'label_tiles': tiles_labels_path,
        'full_img_results': full_labels_path,
        'cloud_results': cloud_results,
        'vegetation_results': vegetation_results
    }

if __name__ == '__main__':
    full_sentinel2_data_pipeline('france2')
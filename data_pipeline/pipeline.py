import os
import numpy as np
import matplotlib.pyplot as plt
from utils.bands_preprocessing import *
from FirePredictionModel.models.geodata_extraction import extract_geospatial_metadata
from utils.clouddetector import is_cloudy
import time



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
    full_img_path = os.path.join(base_path, f'{dataset_name}_full_img_results')
    
    # Output patch directories
    tiles_input_path = os.path.join(base_path, 'TILES_INPUT_DATA')
    tiles_labels_path = os.path.join(base_path, 'TILES_LABELS')

    os.makedirs(tiles_input_path, exist_ok=True)
    os.makedirs(tiles_labels_path, exist_ok=True)
    
    # Step 1: Extract data from folder and read all 13 bands
    all_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    """
    print("Saving geospatial information...")
    extract_geospatial_metadata(dataset_name, os.path.join(base_path, "DATASETS"), tiles_input_path)
    print("Geospatial information saved successfully!\n")

    # save band info (not 100% necessary, but better for clarity of bands info)
    with open(os.path.join(tiles_input_path, f'{dataset_name}_band_info.pkl'), 'wb') as f:
        band_info = {
            'band_names': pre_bands['band_names'],
            'band_order': pre_bands['band_order']
        }
        if 'resampling_info' in pre_bands:
            band_info['resampling_info'] = pre_bands['resampling_info']
        pickle.dump(band_info, f)
"""
    #------------------------------------------------------------------------------------
    
    pre_path = os.path.join(base_path, 'data', f'{dataset_name}_pre.npy')
    post_path = os.path.join(base_path, 'data', f'{dataset_name}_post.npy')
    # Step 1: Create label map (dNBR)
    print("\n--- Step 1: Creating label map (dNBR) ---")
    os.makedirs(full_img_path, exist_ok=True)
    dnbr_path = extract_data_labels(pre_path, post_path, full_img_path)
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
    dnbr_normmap = np.load(os.path.join(full_img_path, 'dnbr_normalized.npy'))
    extract_tiles_with_padding(dnbr_path, dataset_name, (*patch_size, 1), tiles_labels_path)
    print(f"✓ label tiles extracted to {tiles_input_path}")

    #------------------------------------------------------------------------------------
    
    # Step 3: Apply cloud detection function
    print("\n--- Step 3: Applying cloud detection ---")
    print(f"Using S2PixelCloudDetector with % of cloudy pixels threshold: {cloud_threshold}")
    init_time = time.time()
    cloud_results = is_cloudy(base_path, cloud_threshold)
    dt = time.time() - init_time
    print(f"✓ Cloud detection completed in {dt:.1f} seconds. Clean tiles: {cloud_results['clean_tiles']}, Cloudy tiles moved: {cloud_results['cloudy_tiles']}")
        
    #------------------------------------------------------------------------------------

    # Step 4: Placeholder for vegetation detection model
    # TODO: Implement vegetation detection model that removes tiles without sufficient vegetation
    # This model should:
    # - Analyze vegetation indices (NDVI/NDMI) in each remaining patch
    # - Apply a vegetation threshold (e.g., minimum 30% vegetation coverage)
    # - Move tiles with insufficient vegetation to a separate folder (e.g., "LOW_VEGETATION_tiles")
    # - Return statistics similar to cloud detection results
    # Example function call: vegetation_results = patch_vegetation_detection(tiles_input_path, vegetation_threshold=0.3)

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
    print(f"  🔸 Full image label: {full_img_path}")

    # Display dNBR heatmap
    ax = plt.imshow(dnbr_normmap[:, :, 0], cmap='coolwarm')
    plt.title('dNBR Heatmap', fontsize=12)
    plt.colorbar()
    plt.axis('off')
    plt.show()
    plt.savefig(os.path.join(full_img_path, 'dnbr_heatmap.png'))


    return {
        'dataset_name': dataset_name,
        'bands_tiles': tiles_input_path,
        'label_tiles': tiles_labels_path,
        'full_img_results': full_img_path,
        'cloud_results': cloud_results
    }

if __name__ == '__main__':
    full_sentinel2_data_pipeline('france2') 
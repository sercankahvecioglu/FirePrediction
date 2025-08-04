import os
import numpy as np
import matplotlib.pyplot as plt
from utils.bands_preprocessing import *
from utils.geodata_extraction import extract_geospatial_metadata



# full processing pipeline to get all the tiles data needed (chosen bands, ndvi, ndmi, labels in [0, 1])
def full_sentinel2_data_pipeline(dataset_name: str, 
                                   base_path: str = '/home/dario/Desktop/FlameSentinels',
                                   patch_size: tuple = (256, 256),
                                   threshold: float = 0.6,
                                   cloud_threshold: float = 0.6):
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
    
    # Step 1: Extract data from folder and read all 13 bands
    print("\n--- Step 1: Reading all 13 bands data ---")
    all_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    # extracting pre bands
    print("Reading pre-fire bands...")
    pre_bands = read_sent2_1c_bands(f'{dataset_name}_pre', all_bands)
    # saving geospatial profile
    extract_geospatial_metadata(dataset_name, tiles_input_path)

    # save band info (not 100% necessary, but better for clarity of bands info)
    with open(os.path.join(tiles_input_path, f'{dataset_name}_band_info.pkl'), 'wb') as f:
        band_info = {
            'band_names': bands_data['band_names'],
            'band_order': bands_data['band_order']
        }
        if 'resampling_info' in bands_data:
            band_info['resampling_info'] = bands_data['resampling_info']
        pickle.dump(band_info, f)

    # extracting post bands
    print("Reading post-fire bands...")
    post_bands = read_sent2_1c_bands(f'{dataset_name}_post', ['B04', 'B08', 'B8A', 'B12'])
    
    # Print information about the result
    print("Band names:", pre_bands['band_names'])
    print("Data shape:", pre_bands['data'].shape)
    print("Band order:", pre_bands['band_order'])
    if 'resampling_info' in pre_bands:
        print("Resampling info:", pre_bands['resampling_info'])

    #------------------------------------------------------------------------------------
    
    # Step 2: Create label map (dNBR) with NDVI masking
    print("\n--- Step 2: Creating label map (dNBR with NDVI masking) ---")
    os.makedirs(full_img_path, exist_ok=True)
    extract_data_labels_from_bands(pre_bands, post_bands, full_img_path, threshold)
    print("✓ dNBR label map with NDVI masking created")

    #------------------------------------------------------------------------------------
    
    # Step 3: Divide into tiles of size (256, 256, 13)
    print("\n--- Step 3: Extracting tiles (256, 256, 13) ---")
    bands_data = pre_bands['data']  # Use pre-fire data for tiles
    num_bands = len(all_bands)
    extract_tiles_with_padding(bands_data, dataset_name, (*patch_size, num_bands), tiles_input_path)
    
    # Also extract label tiles
    dnbr_normmap = np.load(os.path.join(full_img_path, 'dnbr_normalized.npy'))
    extract_tiles_with_padding(dnbr_normmap, dataset_name, (*patch_size, 1), tiles_labels_path)
    print(f"✓ tiles extracted to {tiles_input_path}")

    #------------------------------------------------------------------------------------

    # TODO: put band info and profile info into the tiles folder
    
    """
    # Step 4: Apply cloud detection function
    print("\n--- Step 4: Applying cloud detection ---")
    print(f"Using S2PixelCloudDetector with threshold: {cloud_threshold}")
    cloud_results = patch_cloud_detection(tiles_input_path, cloud_threshold)
    if cloud_results is not None:
        visualize_patch_results(cloud_results)
        print(f"✓ Cloud detection completed. Clean tiles: {cloud_results['clean_tiles']}, Cloudy tiles moved: {cloud_results['cloudy_tiles']}")
    else:
        print("❌ Cloud detection failed or no tiles found")
    """
        
    #------------------------------------------------------------------------------------

    # Step 5: Placeholder for vegetation detection model
    # TODO: Implement vegetation detection model that removes tiles without sufficient vegetation
    # This model should:
    # - Analyze vegetation indices (NDVI/NDMI) in each remaining patch
    # - Apply a vegetation threshold (e.g., minimum 30% vegetation coverage)
    # - Move tiles with insufficient vegetation to a separate folder (e.g., "LOW_VEGETATION_tiles")
    # - Return statistics similar to cloud detection results
    # Example function call: vegetation_results = patch_vegetation_detection(tiles_input_path, vegetation_threshold=0.3)

    #------------------------------------------------------------------------------------

    # Step 6: Placeholder for black pixel filtering
    # TODO: Implement filtering for black tiles (0 or smth like that for all rgb values)
    # - check if pixel is fully black (part of image padding)
    # - if so, discard it
    
    #------------------------------------------------------------------------------------

    # Step 7: Extract NDVI and NDMI for remaining tiles
    print("\n--- Step 6: Extracting NDVI and NDMI for remaining clean tiles ---")
    
    # Process remaining tiles in the tiles_BANDS folder (after cloud filtering)
    remaining_patch_files = [f for f in os.listdir(tiles_input_path) if f.endswith('.npy')]
    
    if remaining_patch_files:
        print(f"Processing {len(remaining_patch_files)} remaining clean tiles...")
        
        # Get band names for vegetation index computation
        band_names = pre_bands['band_names']
        
        # Process NDVI & NDMI tiles
        process_tiles_directory_with_indices(tiles_input_path, band_names)
        print(f"✓ NDVI & NDMI values added to the tiles in {tiles_input_path}")

    else:
        print("⚠ No clean tiles remaining after cloud detection!")
    
    # Step 8: Print folder locations
    print(f"\n=== Processing pipeline completed for dataset: {dataset_name} ===")
    print("📁 OUTPUT FOLDER LOCATIONS:")
    print(f"  🔸 Clean bands tiles +  indices (15 bands): {tiles_input_path}")
    print(f"  🔸 Label tiles (dNBR): {tiles_labels_path}")

    #if cloud_results:
    #    print(f"  🔸 Cloudy tiles (moved): {cloud_results['cloudy_dir']}")
    print(f"  🔸 Full image results: {full_img_path}")
    

    # Visualize the heatmap and binary map side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display dNBR heatmap
    im1 = ax1.imshow(dnbr_normmap[:, :, 0], cmap='coolwarm', vmin=0, vmax=1)
    ax1.set_title('dNBR Heatmap (NDVI Masked)', fontsize=12)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    dnbr_map = dnbr_normmap > 0.6
    
    # Display binary map
    im2 = ax2.imshow(dnbr_map[:, :, 0], cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'dNBR Binary Map (NDVI Masked, thresh={0.6})', fontsize=12)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()


    return {
        'dataset_name': dataset_name,
        'bands_tiles': tiles_input_path,
        'label_tiles': tiles_labels_path,
        'full_img_results': full_img_path,
        #'cloud_results': cloud_results
    }

if __name__ == '__main__':
    full_sentinel2_data_pipeline('chile') 
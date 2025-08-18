import os
import sys
import argparse

# Add server directory to the path to access forest_detection and cloud_detection modules
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


from bands_preprocessing import *
from utils.forest_detection.veg_model_eval import ndvi_veg_detector
from utils.cloud_detection import is_cloudy
from data_augmentation import data_augmentation

import time
from shutil import copy2
from s2cloudless import S2PixelCloudDetector

# Basic parent class for the pre-processing steps 
class BaseProcessor():
    """
        Base processing pipeline for satellite data including bands extraction, cloud filtering, 
        vegetation filtering, and NDVI/NDMI extraction.
        
        Args:
            dataset_name (str): Common name for the dataset folder (e.g., 'turkey', 'california', etc.)
            base_path (str): Base directory path where data folders are located
            patch_size (tuple): Size of tiles to extract (height, width)
            cloud_threshold (float): Cloudy pixels % threshold for tiles discarding (0.0-1.0)
    """
    def __init__(self, dataset_name: str, 
                        base_path: str = '/home/dario/Desktop/FirePrediction',
                        patch_size: tuple = (256, 256),
                        cloud_threshold: float = 0.5):
        
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.cloud_threshold = cloud_threshold
        self.patch_size = patch_size
        self.all_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    
    def _setup_directories(self):
        """Setup common directory paths"""
        self.tiles_input_path = os.path.join(self.base_path, 'TILES_INPUT_DATA')
        os.makedirs(self.tiles_input_path, exist_ok=True)
    
    def _copy_geospatial_info(self):
        """Copy pre-image geospatial information"""
        print("Obtaining Pre-Images geospatial information...")
        fname = os.path.join(self.base_path, "data_pkl", f"{self.dataset_name}_pre_tiles_data.pkl")
        copy2(fname, os.path.join(self.tiles_input_path, os.path.basename(fname)))
        print("Geospatial information saved successfully to input path!\n")
    
    def _extract_tiles(self):
        """Extract tiles from pre-fire image"""
        pre_path = os.path.join(self.base_path, 'data', f'{self.dataset_name}_pre.npy')
        print("\n--- Step 2: Extracting tiles (256, 256, 13) ---")
        num_bands = len(self.all_bands)
        init_time = time.time()
        extract_tiles_with_padding(pre_path, self.dataset_name, (*self.patch_size, num_bands), self.tiles_input_path)
        dt = time.time() - init_time
        print(f"✓ input tiles extracted to {self.tiles_input_path} in {dt:.1f} seconds")
    
    def _apply_cloud_detection(self):
        """Apply cloud detection filtering"""
        print("\n--- Step 3: Applying cloud detection ---")
        print(f"Using S2PixelCloudDetector with % of cloudy pixels threshold: {self.cloud_threshold}")
        init_time = time.time()
        cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True)
        cloud_results = is_cloudy(self.tiles_input_path, cloud_detector=cloud_detector, cloud_threshold=self.cloud_threshold)
        dt = time.time() - init_time
        print(f"✓ Cloud detection completed in {dt:.1f} seconds. Clean tiles: {cloud_results['clean_tiles']}, Cloudy tiles moved: {cloud_results['cloudy_tiles']}")
        return cloud_results
    
    def _apply_vegetation_detection(self):
        """Apply vegetation detection filtering"""
        print("\n--- Step 4: Applying vegetation detection ---")
        print(f"Using NDVI-based vegetation detection with {0.2*100}% vegetation coverage threshold")
        init_time = time.time()
        veg_tiles, no_veg_tiles = ndvi_veg_detector(self.tiles_input_path)
        #----------------------------------------------------------------
        dt = time.time() - init_time
        print(f"✓ Vegetation detection completed in {dt:.1f} seconds. Number of clean tiles: {len(veg_tiles)}, Number of low vegetation tiles moved: {len(no_veg_tiles)}")
        return veg_tiles, no_veg_tiles
    
    def _extract_indices(self):
        """Extract NDVI and NDMI for remaining clean tiles"""
        print("\n--- Step 5: Extracting NDVI and NDMI for remaining clean tiles ---")
        
        remaining_patch_files = [f for f in os.listdir(self.tiles_input_path) if f.endswith('.npy')]
        
        if remaining_patch_files:
            print(f"Processing {len(remaining_patch_files)} remaining clean tiles...")
            process_tiles_directory_with_indices(self.tiles_input_path, self.all_bands)
            print(f"✓ NDVI & NDMI values added to the tiles in {self.tiles_input_path}")
        else:
            print("⚠ No clean tiles remaining after cloud detection!")


# Class for the full pre-processing steps (including label extraction)
class TrainDataProcessor(BaseProcessor):
    """
        Complete processing pipeline for training data including labels generation.
        
        Args:
            dataset_name (str): Common name for the dataset folder (e.g., 'turkey', 'california', etc.)
            base_path (str): Base directory path where data folders are located
            patch_size (tuple): Size of tiles to extract (height, width)
            cloud_threshold (float): Cloudy pixels % threshold for tiles discarding (0.0-1.0)
            apply_augmentation (bool): Whether to apply data augmentation before NDVI/NDMI extraction
            augmentation_config (dict): Configuration for data augmentation parameters
    """
    
    def __init__(self, dataset_name: str, 
                        base_path: str = '/home/dario/Desktop/FirePrediction',
                        patch_size: tuple = (256, 256),
                        cloud_threshold: float = 0.5,
                        apply_augmentation: bool = True,
                        augmentation_config: dict = None):
        super().__init__(dataset_name, base_path, patch_size, cloud_threshold)
        self.apply_augmentation = apply_augmentation
        self.augmentation_config = augmentation_config or {
            'apply_rotations': True,
            'apply_flips': True,
            'apply_noise': True,
            'noise_factor': 0.05,
            'augmentation_probability': 0.5,
            'seed': 42
        }
    
    def _setup_directories(self):
        """Setup directories including labels path for training"""
        super()._setup_directories()
        self.full_labels_path = os.path.join(self.base_path, f'full_labels')
        self.tiles_labels_path = os.path.join(self.base_path, 'TILES_LABELS')
        os.makedirs(self.tiles_labels_path, exist_ok=True)
    
    def _create_labels(self):
        """Create dNBR label map"""
        pre_path = os.path.join(self.base_path, 'data', f'{self.dataset_name}_pre.npy')
        post_path = os.path.join(self.base_path, 'data', f'{self.dataset_name}_post.npy')
        print("\n--- Step 1: Creating label map (dNBR with NDVI masking) ---")
        os.makedirs(self.full_labels_path, exist_ok=True)
        dnbr_path = extract_data_labels(self.dataset_name, pre_path, post_path, self.full_labels_path)
        print("✓ dNBR label map with NDVI masking created")
        return dnbr_path
    
    def _extract_label_tiles(self):
        """Extract tiles from the dNBR label map"""
        dnbr_path = os.path.join(self.full_labels_path, f'{self.dataset_name}_dnbr_heatmap.npy')
        print("\n--- Step 2.5: Extracting label tiles from dNBR map ---")
        init_time = time.time()
        # Labels have 1 channel (height, width, 1)
        extract_tiles_with_padding(dnbr_path, self.dataset_name, (*self.patch_size, 1), self.tiles_labels_path)
        dt = time.time() - init_time
        print(f"✓ label tiles extracted to {self.tiles_labels_path} in {dt:.1f} seconds")
    
    def _apply_data_augmentation(self):
        """Apply data augmentation to clean tiles before NDVI/NDMI extraction"""
        if not self.apply_augmentation:
            print("⏭️  Data augmentation skipped (disabled)")
            return {'status': 'skipped'}
            
        print("\n--- Step 4.5: Applying data augmentation to clean tiles ---")
        init_time = time.time()
        
        # Count existing files before augmentation
        data_files = [f for f in os.listdir(self.tiles_input_path) if f.endswith('.npy')]
        label_files = [f for f in os.listdir(self.tiles_labels_path) if f.endswith('.npy')]
        
        if not data_files:
            print("⚠️  No data files to augment")
            return {'status': 'no_files'}
        
        try:
            # Apply data augmentation
            result = data_augmentation(
                data_folder=self.tiles_input_path,
                labels_folder=self.tiles_labels_path,
                augmentation_config=self.augmentation_config
            )
            
            dt = time.time() - init_time
            print(f"✓ Data augmentation completed in {dt:.1f} seconds")
            print(f"  Original files: {result.get('original_files', 0)}")
            print(f"  Augmented samples generated: {result.get('augmented_count', 0)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during data augmentation: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def run(self):
        pipeline_start = time.time()
        print(f"=== Starting processing pipeline for dataset: {self.dataset_name} ===")
        
        self._setup_directories()
        self._copy_geospatial_info()
        self._create_labels()
        self._extract_tiles()
        self._extract_label_tiles()
        cloud_results = self._apply_cloud_detection()
        vegetation_results = self._apply_vegetation_detection()
        augmentation_results = self._apply_data_augmentation()
        self._extract_indices()
        
        pipeline_duration = time.time() - pipeline_start
        print(f"\n=== Processing pipeline completed for dataset: {self.dataset_name} ===")
        print(f"⏱️  Total pipeline execution time: {pipeline_duration:.1f} seconds")
        print("📁 OUTPUT FOLDER LOCATIONS:")
        print(f"  🔸 Clean bands tiles +  indices (15 bands): {self.tiles_input_path}")
        print(f"  🔸 Label tiles (dNBR): {self.tiles_labels_path}")
        print(f"  🔸 Full image label: {self.full_labels_path}")

        return {
            'dataset_name': self.dataset_name,
            'bands_tiles': self.tiles_input_path,
            'label_tiles': self.tiles_labels_path,
            'full_img_results': self.full_labels_path,
            'cloud_results': cloud_results,
            'vegetation_results': vegetation_results,
            'augmentation_results': augmentation_results
        }


# Class for the on-satellite pre-processing steps
class SatelliteProcessor(BaseProcessor):
    """
        On-satellite processing pipeline for data before downlinking, with cloud filtering, 
        vegetation filtering, and NDVI/NDMI extraction.
        
        Args:
            dataset_name (str): Common name for the dataset folder (e.g., 'turkey', 'california', etc.)
            base_path (str): Base directory path where data folders are located
            patch_size (tuple): Size of tiles to extract (height, width)
            cloud_threshold (float): Cloudy pixels % threshold for tiles discarding (0.0-1.0)
    """
    
    def __init__(self, dataset_name: str, 
                        base_path: str = '/home/dario/Desktop/FirePrediction',
                        patch_size: tuple = (256, 256),
                        cloud_threshold: float = 0.5):
        super().__init__(dataset_name, base_path, patch_size, cloud_threshold)
    
    def run(self):
        pipeline_start = time.time()
        print(f"=== Starting processing pipeline for dataset: {self.dataset_name} ===")
        
        self._setup_directories()
        self._copy_geospatial_info()
        self._extract_tiles()
        cloud_results = self._apply_cloud_detection()
        vegetation_results = self._apply_vegetation_detection()
        self._extract_indices()
        
        pipeline_duration = time.time() - pipeline_start
        print(f"\n=== Processing pipeline completed for dataset: {self.dataset_name} ===")
        print(f"⏱️  Total pipeline execution time: {pipeline_duration:.1f} seconds")
        print("📁 OUTPUT FOLDER LOCATIONS:")
        print(f"  🔸 Clean bands tiles +  indices (15 bands): {self.tiles_input_path}")

        return {
            'dataset_name': self.dataset_name,
            'bands_tiles': self.tiles_input_path,
            'cloud_results': cloud_results,
            'vegetation_results': vegetation_results
        }


def main():
    """Main function to handle command line arguments and run the processing pipeline"""
    parser = argparse.ArgumentParser(description='Process satellite data for fire prediction')
    parser.add_argument('dataset_name', 
                       help='Name of the dataset to process (e.g., usa2, turkey, california, etc.)')
    parser.add_argument('--processor', '-p', 
                       choices=['train', 'satellite'], 
                       default='train',
                       help='Type of processor to use: train (full pipeline with labels) or satellite (no labels)')
    parser.add_argument('--cloud-threshold', '-c',
                       type=float,
                       default=0.5,
                       help='Cloudy pixels percentage threshold for tile discarding (0.0-1.0)')
    parser.add_argument('--patch-size',
                       type=int,
                       nargs=2,
                       default=[256, 256],
                       metavar=('HEIGHT', 'WIDTH'),
                       help='Size of tiles to extract (height width)')
    parser.add_argument('--no-augmentation',
                       action='store_true',
                       help='Disable data augmentation (only for train processor)')
    parser.add_argument('--base-path',
                       default='/home/dario/Desktop/FirePrediction',
                       help='Base directory path where data folders are located')
    
    args = parser.parse_args()
    
    # Convert patch size to tuple
    patch_size = tuple(args.patch_size)
    
    # Create processor based on type
    if args.processor == 'train':
        processor = TrainDataProcessor(
            dataset_name=args.dataset_name,
            base_path=args.base_path,
            patch_size=patch_size,
            cloud_threshold=args.cloud_threshold,
            apply_augmentation=not args.no_augmentation
        )
    else:  # satellite
        processor = SatelliteProcessor(
            dataset_name=args.dataset_name,
            base_path=args.base_path,
            patch_size=patch_size,
            cloud_threshold=args.cloud_threshold
        )
    
    # Run the processing pipeline
    try:
        results = processor.run()
        print(f"\n✅ Processing completed successfully for dataset: {args.dataset_name}")
        return results
    except Exception as e:
        print(f"\n❌ Error processing dataset {args.dataset_name}: {str(e)}")
        sys.exit(1)


# brief running check to see if the code works correctly
if __name__ == '__main__':
    main()
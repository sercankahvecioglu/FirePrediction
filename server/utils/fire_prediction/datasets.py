import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pickle
import json
from geodata_extraction import get_real_world_coords


def calculate_dataset_statistics(input_data_dir, band_indices=[7, 11, 12, 13, 14], 
                                save_path=None, force_recalculate=False):
    """
    Calculate global mean and std for the entire dataset and save to file.
    
    Args:
        input_data_dir (str): Directory containing .npy input files
        band_indices (list): Indices of bands to use
        save_path (str): Path to save statistics JSON file
        force_recalculate (bool): Whether to recalculate even if stats file exists
    
    Returns:
        tuple: (means, stds) as torch tensors
    """
    if save_path is None:
        save_path = os.path.join(input_data_dir, 'dataset_statistics.json')
    
    # Check if statistics already exist
    if os.path.exists(save_path) and not force_recalculate:
        print(f"Loading existing dataset statistics from {save_path}")
        with open(save_path, 'r') as f:
            stats = json.load(f)
            means = torch.tensor(stats['means'])
            stds = torch.tensor(stats['stds'])
        return means, stds
    
    print("Calculating dataset statistics...")
    input_files = glob.glob(os.path.join(input_data_dir, "*.npy"))
    
    if not input_files:
        raise ValueError(f"No .npy files found in {input_data_dir}")
    
    # Collect all band data for statistics calculation
    all_bands_data = [[] for _ in band_indices]
    
    for i, file_path in enumerate(input_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(input_files)}")
        
        # Load data and extract selected bands
        data = np.load(file_path)[..., band_indices]  # Shape: (H, W, C)
        
        # Collect data for each band
        for band_idx, band_data in enumerate(data.transpose(2, 0, 1)):  # (C, H, W)
            all_bands_data[band_idx].append(band_data.flatten())
    
    # Calculate statistics for each band
    means = []
    stds = []
    
    for band_idx, band_data_list in enumerate(all_bands_data):
        # Concatenate all data for this band
        all_band_data = np.concatenate(band_data_list)
        
        # Calculate statistics
        mean_val = float(np.mean(all_band_data))
        std_val = float(np.std(all_band_data))
        
        means.append(mean_val)
        stds.append(std_val)
        
        print(f"Band {band_indices[band_idx]}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    # Convert to tensors
    means = torch.tensor(means)
    stds = torch.tensor(stds)
    
    # Save statistics to file
    stats_dict = {
        'means': means.tolist(),
        'stds': stds.tolist(),
        'band_indices': band_indices,
        'num_files': len(input_files)
    }
    
    with open(save_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"Dataset statistics saved to {save_path}")
    return means, stds


def load_dataset_statistics(stats_path):
    """Load dataset statistics from JSON file."""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return torch.tensor(stats['means']), torch.tensor(stats['stds'])


class BaseSent2Dataset(Dataset):
    """Base class for Sentinel-2 datasets with common functionality"""
    
    def __init__(self, input_data_dir, transform=None, use_global_normalization=True, 
                 stats_path=None):
        self.input_data_dir = input_data_dir
        self.input_list = glob.glob(os.path.join(input_data_dir, "*.npy"))
        self.band_indices = [7, 11, 12, 13, 14]  # NIR, SWIR1, SWIR2, NDVI, NDMI
        self.use_global_normalization = use_global_normalization
        
        # Setup normalization
        if use_global_normalization:
            if stats_path is None:
                stats_path = os.path.join(input_data_dir, 'dataset_statistics.json')
            
            # Load or calculate statistics
            if os.path.exists(stats_path):
                self.global_means, self.global_stds = load_dataset_statistics(stats_path)
                print(f"Loaded global statistics from {stats_path}")
            else:
                print("Global statistics not found. Calculating...")
                self.global_means, self.global_stds = calculate_dataset_statistics(
                    input_data_dir, self.band_indices, stats_path
                )
            
            # Create normalization transform
            self.normalize_transform = transforms.Normalize(
                mean=self.global_means.tolist(),
                std=self.global_stds.tolist()
            )
        else:
            self.normalize_transform = None
        
        # Additional transforms provided by user
        self.transform = transform
    
    def __len__(self):
        return len(self.input_list)
    
    def _load_and_process_input(self, input_path):
        """Load and process input tensor with normalization"""
        input_tensor = np.load(input_path)[..., self.band_indices]
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1)
        
        if self.use_global_normalization:
            # Apply global normalization using torchvision.transforms.Normalize
            input_tensor = self.normalize_transform(input_tensor)
        else:
            # Fallback to per-sample min-max normalization
            for i in range(input_tensor.shape[0]):
                band = input_tensor[i]
                min_val, max_val = band.min(), band.max()
                input_tensor[i] = (band - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(band)
        
        # Apply additional transforms if provided
        return self.transform(input_tensor) if self.transform else input_tensor
    
    def _setup_geoinfo(self):
        """Setup geospatial information for datasets that need it"""
        self.geoinfo_paths = glob.glob(os.path.join(self.input_data_dir, "*_tiles_data.pkl"))
        self.country_ids = [os.path.basename(path).split('_pre_tiles_data.pkl')[0] for path in self.geoinfo_paths]
        
        self.geoinfo = []
        for path in self.geoinfo_paths:
            with open(path, 'rb') as f:
                self.geoinfo.append(pickle.load(f))
        
        self.geodict = dict(zip(self.country_ids, self.geoinfo))
    
    def _get_coordinates(self, input_path):
        """Extract real-world coordinates from input path"""
        country_id, realcoords = os.path.basename(input_path).split('_tile_')
        coordy, coordx = map(int, realcoords.replace('.npy', '').strip("()").split(", "))
        return get_real_world_coords(coordy, coordx, metadata=self.geodict[country_id])


class TrainingDataset(BaseSent2Dataset):
    """Optimized dataset for training - no geospatial information processing"""
    
    def __init__(self, input_data_dir, labels_dir, transform=None, target_transform=None,
                 use_global_normalization=True, stats_path=None):
        super().__init__(input_data_dir, transform, use_global_normalization, stats_path)
        self.labels_list = glob.glob(os.path.join(labels_dir, "*.npy"))
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_tensor = self._load_and_process_input(self.input_list[index])
        
        label_tensor = torch.tensor(np.load(self.labels_list[index]), dtype=torch.float32).permute(2, 0, 1)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)
        
        return input_tensor, label_tensor


class TestingDataset(BaseSent2Dataset):
    """Dataset for testing - includes geospatial information for evaluation"""
    
    def __init__(self, input_data_dir, labels_dir, transform=None, target_transform=None,
                 use_global_normalization=True, stats_path=None):
        super().__init__(input_data_dir, transform, use_global_normalization, stats_path)
        self.labels_list = glob.glob(os.path.join(labels_dir, "*.npy"))
        self.target_transform = target_transform
        self._setup_geoinfo()

    def __getitem__(self, index):
        input_path = self.input_list[index]
        input_tensor = self._load_and_process_input(input_path)
        coords = self._get_coordinates(input_path)
        
        label_tensor = torch.tensor(np.load(self.labels_list[index]), dtype=torch.float32).permute(2, 0, 1)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)
        
        return input_tensor, label_tensor, coords


class InferenceDataset(BaseSent2Dataset):
    """Dataset for inference - no labels required, includes geospatial information"""
    
    def __init__(self, input_data_dir, transform=None, use_global_normalization=True, 
                 stats_path=None):
        super().__init__(input_data_dir, transform, use_global_normalization, stats_path)
        self._setup_geoinfo()

    def __getitem__(self, index):
        input_path = self.input_list[index]
        input_tensor = self._load_and_process_input(input_path)
        coords = self._get_coordinates(input_path)
        
        return input_tensor, coords
    
    

if __name__ == '__main__':
    # Test all dataset classes with global normalization
    print("Testing datasets with global normalization...")
    
    # IMPORTANT: Testing and inference MUST use training statistics!
    training_stats_path = "/home/dario/Desktop/FirePrediction/inputs/dataset_statistics.json"
    
    datasets = {
        'Training': TrainingDataset(
            "/home/dario/Desktop/FirePrediction/inputs", 
            "/home/dario/Desktop/FirePrediction/labels",
            use_global_normalization=True
            # Will automatically create/use stats from training data
        ),
        'Testing': TestingDataset(
            "/home/dario/Desktop/FirePrediction/test_TEST_INPUT_DATA", 
            "/home/dario/Desktop/FirePrediction/TEST_LABELS",
            use_global_normalization=True,
            stats_path=training_stats_path  # CRITICAL: Use training statistics
        ),
        'Inference': InferenceDataset(
            "/home/dario/Desktop/FirePrediction/test_TEST_INPUT_DATA",
            use_global_normalization=True,
            stats_path=training_stats_path  # CRITICAL: Use training statistics
        )
    }
    
    for name, ds in datasets.items():
        print(f"{name} dataset length: {len(ds)}")
        if hasattr(ds, 'global_means'):
            print(f"  Global means: {ds.global_means}")
            print(f"  Global stds: {ds.global_stds}")
    
    # Verify that test/inference use the same statistics as training
    if hasattr(datasets['Training'], 'global_means'):
        train_means = datasets['Training'].global_means
        train_stds = datasets['Training'].global_stds
        
        for name in ['Testing', 'Inference']:
            if hasattr(datasets[name], 'global_means'):
                test_means = datasets[name].global_means
                test_stds = datasets[name].global_stds
                
                if torch.allclose(train_means, test_means) and torch.allclose(train_stds, test_stds):
                    print(f"✅ {name} uses same statistics as Training")
                else:
                    print(f"⚠️  WARNING: {name} uses different statistics than Training!")
                    print(f"   Training means: {train_means}")
                    print(f"   {name} means: {test_means}")
                    print(f"   This could cause poor model performance!")
    
    # Test loading a sample
    if len(datasets['Training']) > 0:
        sample = datasets['Training'][0]
        print(f"\nSample input shape: {sample[0].shape}")
        print(f"Sample input stats - Mean: {sample[0].mean():.6f}, Std: {sample[0].std():.6f}")
        print(f"Sample label shape: {sample[1].shape}")
        print("✅ Dataset testing completed!")
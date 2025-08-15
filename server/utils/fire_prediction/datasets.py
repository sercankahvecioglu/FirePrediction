import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np
import pickle
from geodata_extraction import get_real_world_coords


class BaseSent2Dataset(Dataset):
    """Base class for Sentinel-2 datasets with common functionality"""
    
    def __init__(self, input_data_dir, transform=None):
        self.input_data_dir = input_data_dir
        self.input_list = glob.glob(os.path.join(input_data_dir, "*.npy"))
        self.transform = transform
        self.band_indices = [7, 11, 12, 13, 14]  # NIR, SWIR1, SWIR2, NDVI, NDMI
    
    def __len__(self):
        return len(self.input_list)
    
    def _load_and_process_input(self, input_path):
        """Load and process input tensor with normalization"""
        input_tensor = np.load(input_path)[..., self.band_indices]
        input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)
        
        # Apply minmax normalization to all bands
        for i in range(input_tensor.shape[0]):
            band = input_tensor[i]
            min_val, max_val = band.min(), band.max()
            input_tensor[i] = (band - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(band)
        
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
    
    def __init__(self, input_data_dir, labels_dir, transform=None, target_transform=None):
        super().__init__(input_data_dir, transform)
        self.labels_list = glob.glob(os.path.join(labels_dir, "*.npy"))
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_tensor = self._load_and_process_input(self.input_list[index])
        
        label_tensor = torch.tensor(np.load(self.labels_list[index])).permute(2, 0, 1)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)
        
        return input_tensor, label_tensor


class TestingDataset(BaseSent2Dataset):
    """Dataset for testing - includes geospatial information for evaluation"""
    
    def __init__(self, input_data_dir, labels_dir, transform=None, target_transform=None):
        super().__init__(input_data_dir, transform)
        self.labels_list = glob.glob(os.path.join(labels_dir, "*.npy"))
        self.target_transform = target_transform
        self._setup_geoinfo()

    def __getitem__(self, index):
        input_path = self.input_list[index]
        input_tensor = self._load_and_process_input(input_path)
        coords = self._get_coordinates(input_path)
        
        label_tensor = torch.tensor(np.load(self.labels_list[index])).permute(2, 0, 1)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)
        
        return input_tensor, label_tensor, coords


class InferenceDataset(BaseSent2Dataset):
    """Dataset for inference - no labels required, includes geospatial information"""
    
    def __init__(self, input_data_dir, transform=None):
        super().__init__(input_data_dir, transform)
        self._setup_geoinfo()

    def __getitem__(self, index):
        input_path = self.input_list[index]
        input_tensor = self._load_and_process_input(input_path)
        coords = self._get_coordinates(input_path)
        
        return input_tensor, coords
    
    

if __name__ == '__main__':
    # Test all dataset classes
    datasets = {
        'Training': TrainingDataset("/home/dario/Desktop/FirePrediction/TILES_INPUT_DATA", "/home/dario/Desktop/FirePrediction/TILES_LABELS"),
        'Testing': TestingDataset("/home/dario/Desktop/FirePrediction/TILES_INPUT_DATA", "/home/dario/Desktop/FirePrediction/TILES_LABELS"),
        'Inference': InferenceDataset("/home/dario/Desktop/FirePrediction/TILES_INPUT_DATA")
    }
    
    for name, ds in datasets.items():
        print(f"{name} dataset length: {len(ds)}")
import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np
import pickle
from data_pipeline.utils.geodata_extraction import get_real_world_coords


class Sent2Dataset(Dataset):
    """
    Class for the creation of a dataset from Sentinel2 images 

    Initialization Params:
        - input_data_dir (str): full path for the directory containing input data tensors (in .npy format)
        - labels_dir (str): full path for the directory containing labels tensors (in .npy format)
        - transform (OPTIONAL): transformation to apply to each input image
        - target_transform (OPTIONAL): transformation to apply to each label
    """
    def __init__(self, input_data_dir, labels_dir, transform=None, target_transform=None):
        self.input_data_dir = input_data_dir
        self.input_list = glob.glob(os.path.join(input_data_dir, "*.npy"))
        self.labels_dir = labels_dir
        self.labels_list = glob.glob(os.path.join(labels_dir, "*.npy"))
        self.transform = transform
        self.target_transform = target_transform
        # geospatial info part
        self.geoinfo_paths = glob.glob(os.path.join(input_data_dir, "*_geoinfo.pkl"))
        self.bands_info_paths = glob.glob(os.path.join(input_data_dir, "*_band_info.pkl"))
        self.country_ids = [os.path.basename(path).split('_geoinfo.pkl')[0] for path in self.geoinfo_paths]
        # get pickle data extracted
        self.geoinfo = []
        for path in self.geoinfo_paths:
            with open(path, 'rb') as f:
                self.geoinfo.append(pickle.load(f))
        with open(self.bands_info_paths[0], "rb") as f:
            bands_info = pickle.load(f)
        self.bands_info = bands_info

        # retrieve indices for required bands (NIR, SWIR1, SWIR2, NDVI, NDMI)
        required_bands = ['B08', 'B11', 'B12', 'NDVI', 'NDMI']
        # bands_info is a dict idx:band, so we need to find indices where band matches required_bands
        self.band_indices = [idx for idx, band in bands_info['band_order'].items() if band in required_bands]
        
        # Separate indices for reflectance bands vs spectral indices
        self.reflectance_indices = [i for i, idx in enumerate(self.band_indices) if bands_info['band_order'][idx] in ['B08', 'B11', 'B12']]
        self.spectral_indices = [i for i, idx in enumerate(self.band_indices) if bands_info['band_order'][idx] in ['NDVI', 'NDMI']]

        # create dict with pkl info for each country
        self.geodict = dict(zip(self.country_ids, self.geoinfo))

    def __len__(self):
        return len(self.input_list)


    def __getitem__(self, index):
        """Returns: input tensor, output (labeled) tensor, coordinates string"""

        input_path = self.input_list[index]
        country_id, realcoords = os.path.basename(input_path).split('_tile_')
        # extract values of relative coordinates
        coordy, coordx = map(int, realcoords.replace('.npy', '').strip("()").split(", "))
        # get real-world coords from geoinfo dict
        geoinfo = self.geodict[country_id]
        coords = get_real_world_coords(coordy, coordx, geoinfo)
        label_path = self.labels_list[index]
        # Load input and select only required bands
        input_tensor = np.load(input_path)
        input_tensor = input_tensor[..., self.band_indices]

        label_tensor = np.load(label_path)

        input_tensor, label_tensor = torch.tensor(input_tensor), torch.tensor(label_tensor)

        # convert from [height, width, channels] to [channels, height, width] (neeed for the cnns in the unet)
        input_tensor = input_tensor.permute(2, 0, 1)
        label_tensor = label_tensor.permute(2, 0, 1)
        
        # Normalize different band types appropriately
        # Scale reflectance bands (0-10000) to (0-1)
        for i in self.reflectance_indices:
            input_tensor[i] = input_tensor[i] / 10000.0
        
        # Clip and scale spectral indices (-1 to 1) to (0-1)
        for i in self.spectral_indices:
            input_tensor[i] = torch.clamp(input_tensor[i], -1, 1)
            input_tensor[i] = (input_tensor[i] + 1) / 2.0
        
        # Apply transforms if provided
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        return input_tensor, label_tensor, coords





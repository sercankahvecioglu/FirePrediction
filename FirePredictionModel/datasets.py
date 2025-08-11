import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np
import pickle
from .geodata_extraction import get_real_world_coords


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
        self.geoinfo_paths = glob.glob(os.path.join(input_data_dir, "*_tiles_data.pkl"))
        self.country_ids = [os.path.basename(path).split('_pre_tiles_data.pkl')[0] for path in self.geoinfo_paths]
        # get pickle data extracted
        self.geoinfo = []
        for path in self.geoinfo_paths:
            with open(path, 'rb') as f:
                self.geoinfo.append(pickle.load(f))

        # retrieve indices for required bands (NIR, SWIR1, SWIR2, NDVI, NDMI)
        required_bands = ['B08', 'B11', 'B12', 'NDVI', 'NDMI']
        # select required indices for above defined bands
        self.band_indices = [7, 11, 12, 13, 14]
        
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
        coords = get_real_world_coords(coordy, coordx, metadata=geoinfo)
        label_path = self.labels_list[index]
        # Load input and select only required bands
        input_tensor = np.load(input_path)
        input_tensor = input_tensor[..., self.band_indices]

        label_tensor = np.load(label_path)

        input_tensor, label_tensor = torch.tensor(input_tensor), torch.tensor(label_tensor)

        # convert from [height, width, channels] to [channels, height, width] (neeed for the cnns in the unet)
        input_tensor = input_tensor.permute(2, 0, 1)
        label_tensor = label_tensor.permute(2, 0, 1)
        
        
        # apply minmax normalization to all bands (separately)
        for i in range(input_tensor.shape[0]):  # iterate over each band
            band = input_tensor[i]
            min_val = band.min()
            max_val = band.max()
            if max_val > min_val:  # avoid division by zero
                input_tensor[i] = (band - min_val) / (max_val - min_val)
            else:
                input_tensor[i] = torch.zeros_like(band)  # handle constant bands

        # apply transforms if provided
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        return input_tensor, label_tensor, coords




if __name__ == '__main__':
    ds = Sent2Dataset("/home/dario/Desktop/FirePrediction/TILES_INPUT_DATA", "/home/dario/Desktop/FirePrediction/TILES_LABELS")
    ds[3]
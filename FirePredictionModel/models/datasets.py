import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import pickle
from data_pipeline.utils.geodata_extraction import get_real_world_coords


class Sent2Dataset(Dataset):
    """
    Class for the creation of a dataset from Sentinel2 images 

    Initialization Params:
        input_data_dir (str): full path for the directory containing input data tensors (in .npy format)
        labels_dir (str): full path for the directory containing labels tensors (in .npy format)
        transform (OPTIONAL): transformation to apply to each input image
        target_transform (OPTIONAL): transformation to apply to each label
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
        self.country_ids = [os.path.basename(path).split('_geoinfo.pkl')[0] for path in self.geoinfo_paths]
        # get pickle data extracted
        self.geoinfo = []
        for path in self.geoinfo_paths:
            with open(path, 'rb') as f:
                self.geoinfo.append(pickle.load(f))
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
        # get real-world coords 
        pkl_path = f"/home/dario/Desktop/imgs_metadata/{country_id}_pre.pkl"
        coords = get_real_world_coords(coordy, coordx, pkl_path)
        label_path = self.labels_list[index]

        return np.load(input_path), np.load(label_path), coords
    

    



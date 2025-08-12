import os
import numpy as np
import torch
import torch
from torch.utils.data import DataLoader, random_split  
from FirePredictionModel.unet import UNet
import matplotlib.pyplot as plt
import pickle
from utils.bands_preprocessing import *
from utils.clouddetector import is_cloudy
from utils.data_preprocessing import SatellitePreprocessor
import time
from shutil import copy2

#----------------------------------

#...Preprocessing steps


# MODEL EXECUTION



n_channels = 5 # CHANGE WITH ACCEPTED NUMBRE OF BANDS
unet_model = UNet(n_channels, out_channels=3).to('cuda')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

unet_model.load_state_dict(torch.load('/home/dario/Desktop/FirePrediction/trained_models/best_fireprediction_model.pth', weights_only=True))

# set model to evaluation mode for production inference
unet_model.eval()

# TODO
# 1. Get geospatial information pkl file
# 2. Load dataset
# 3. Use geospatial info extraction function to associate geoinfo with each label
# 4. Pass dataset through model to obtain predictions
# 5. Obtain pngs of predictions (color mapping should be 0: navy blue, 1: yellow, 2: red)
# 6. Compute % of pixels with 1 or 2 class values in each tile
# 7. Obtain geoinfo of tiles having % of risky pixels above a certain threshold
# 8. Pass images & geoinfo of risky areas to backend for message sending
# (e.g., "Area at risk of potential wildfire detected around {coordinate_info}")

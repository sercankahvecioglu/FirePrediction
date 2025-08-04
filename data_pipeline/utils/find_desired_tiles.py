import os
import glob
import numpy as np
import matplotlib.pyplot as plt


SOURCE_DIR = '/home/dario/Desktop/FlameSentinels/sample_data/TILES_LABELS'

tiles = glob.glob(os.path.join(SOURCE_DIR, '*.npy'))

top5_vals = np.random.uniform(low=1e-18, high=1e-17, size=5)

top5_paths = {}

for tilepath in tiles:

    tile = np.load(tilepath)
    print(tile.shape)
    topval = np.mean(tile)
    print(topval)

    minval = np.min(top5_vals)
    minidx = np.where(top5_vals == minval)[0][0]
    if topval > min(top5_vals):
        top5_vals[minidx] = topval
        top5_paths[minidx] = tilepath

    break


print(top5_paths.values())
print(top5_paths.keys())
print(top5_vals)


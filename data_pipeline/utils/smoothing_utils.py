from scipy.ndimage import grey_erosion, grey_dilation, grey_opening, grey_closing
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    heatmap = np.load('/home/dario/Desktop/FlameSentinels/TILES_LABELS/greece_tile_(10752, 8960).npy')

    # Assuming heatmap is your [0,1] fire risk map
    kernel_size = 5  # or larger for more aggressive smoothing

    # Fill gaps in high-risk areas (most useful for your case)
    smoothed_heatmap = grey_closing(heatmap, size=kernel_size)

    # Or remove isolated high-risk pixels
    smoothed_heatmap2 = grey_opening(heatmap, size=kernel_size)

    # Or custom sequence
    temp = grey_dilation(heatmap, size=kernel_size)  # Fill gaps first
    smoothed_heatmap3 = grey_erosion(temp, size=kernel_size)  # Then smooth


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(heatmap, cmap='coolwarm', vmin=0, vmax=1)
    ax1.set_title('Original Heatmap')
    fig.colorbar(im1, orientation='vertical')
    im2 = ax2.imshow(smoothed_heatmap, cmap='coolwarm', vmin=0, vmax=1)
    ax2.set_title('Smoothed Heatmap')
    fig.colorbar(im2, orientation='vertical')

    plt.show()
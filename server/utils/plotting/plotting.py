import numpy as np
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


def create_rgb_visualization(image_data, output_folder, rgb_channels=[3, 2, 1], job_id=None):
    """
    Create RGB images from .npy tiles
    
    Args:
        image_path (str): Path to the .npy file
        output_folder (str): Path to save RGB images
        rgb_channels (list): Channels to use for RGB [R, G, B] - default [3, 2, 1]
    
    Returns:
        list: Paths of created images
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    try:
        print(f"Processing RGB image: {job_id}")

        # Check if data is in [height, width, channels] format and transpose if needed
        if len(image_data.shape) == 3 and image_data.shape[2] < image_data.shape[0]:
            # Data is in [height, width, channels] format, transpose to [channels, height, width]
            image_data = np.transpose(image_data, (2, 0, 1))
        
        # Check if we have enough channels
        if image_data.shape[0] <= max(rgb_channels):
            print(f"  Warning: Not enough channels in {job_id}. Has {image_data.shape[0]}, needs {max(rgb_channels)+1}")
            raise ValueError("Not enough channels in the image data.")
        
        # Extract RGB channels
        red = image_data[rgb_channels[0]].astype(np.float32)
        green = image_data[rgb_channels[1]].astype(np.float32)
        blue = image_data[rgb_channels[2]].astype(np.float32)
        
        # Normalize to 0-1 range
        def normalize_channel(channel):
            # Handle cases where channel might have NaN or inf values
            channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
            
            min_val = np.percentile(channel, 2)  # Use 2nd percentile to avoid outliers
            max_val = np.percentile(channel, 98)  # Use 98th percentile to avoid outliers
            
            # Avoid division by zero
            if max_val == min_val:
                return np.ones_like(channel) * 0.5  # Return middle gray if no variation
            
            normalized = (channel - min_val) / (max_val - min_val)
            return np.clip(normalized, 0, 1)
        
        red_norm = normalize_channel(red)
        green_norm = normalize_channel(green)
        blue_norm = normalize_channel(blue)
        
        # Stack channels to create RGB image [height, width, 3]
        rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)
        
        # Convert to 8-bit
        rgb_image_8bit = (rgb_image * 255).astype(np.uint8)
        
        # Create output filename
        output_path = os.path.join(output_folder, f"{job_id}_rgb.png")
        
        # Save as PNG using PIL
        pil_image = Image.fromarray(rgb_image_8bit)
        pil_image.save(output_path)
            
    except Exception as e:
        print(f"  ✗ Error processing {job_id}: {e}")
        return False

    return True

def create_cloud_mask_visualization(output_folder, metadata_path, job_id=None):
    """
    Combina todos los tiles en una sola imagen grande y muestra:
    - Imagen RGB completa
    - Imagen RGB con máscara de nubes encima
    """
    try:
        print(f"Processing cloud image for: {job_id}")

        metadata = pd.read_excel(metadata_path)
        CLOUD_MASK_FOLDER = os.path.join(output_folder, "CLOUD_IMAGES")
        TILES_IMAGES_PATH = os.path.join(output_folder, "TILES_IMAGES")
        DISPLAY_PATH = os.path.join(output_folder, "DISPLAY")

        # --- 1. Determinar tamaño de la grilla ---
        # Suponiendo que las coordenadas son tipo "(row,col)"
        coords = [eval(c) for c in metadata['tile_coordinates']]
        rows = sorted(set(r for r, _ in coords))
        cols = sorted(set(c for _, c in coords))

        tile_dict_rgb = {}
        tile_dict_mask = {}
        tile_dict_rgb_display = {}

        # --- 2. Cargar todos los tiles ---
        for i in range(metadata.shape[0]):
            coord = eval(metadata['tile_coordinates'][i])  # (row, col)
            tile_path = os.path.join(TILES_IMAGES_PATH, f"{job_id}_tile_{metadata['tile_coordinates'][i]}.npy")
            cloud_mask_path = os.path.join(CLOUD_MASK_FOLDER, f"{job_id}_{metadata['tile_coordinates'][i]}_cloud_mask.npy")

            is_cloudy = metadata['cloud?'][i]
            perc_cloud = metadata['cloud_percentage'][i]

            tile = np.load(tile_path)

            if perc_cloud > 0.4:
                # RED tile
                rgb_tile = np.zeros_like(tile[:,:,:3])  # Create a red tile
                #rgb_tile[..., 0] = 255  # Set red channel to 255
            else:
                rgb_tile = tile[:,:,[3,2,1]]  # Extract RGB bands

                # Do gamma correction
                gamma = 1.2  # >1 aclara sombras y realza color
                rgb_tile = np.power(rgb_tile * 255.0, gamma) / 255.0

            rgb_display = tile[:,:,[3,2,1]]  # Extract RGB bands for display
            gamma = 1.2  # >1 aclara sombras y realza color
            rgb_display = np.power(rgb_display * 255.0, gamma) / 255.0
            cloud_mask = np.load(cloud_mask_path)
            
            tile_dict_rgb[coord] = rgb_tile
            tile_dict_mask[coord] = cloud_mask
            tile_dict_rgb_display[coord] = rgb_display

        # --- 3. Reconstruir la imagen grande ---
        full_rgb_rows = []
        full_mask_rows = []
        full_rgb_display_rows = []
        for r in rows:
            rgb_row = [tile_dict_rgb[(r,c)] for c in cols]
            mask_row = [tile_dict_mask[(r,c)] for c in cols]
            rgb_display_row = [tile_dict_rgb_display[(r,c)] for c in cols]
            full_rgb_rows.append(np.hstack(rgb_row))
            full_mask_rows.append(np.hstack(mask_row))
            full_rgb_display_rows.append(np.hstack(rgb_display_row))

        full_rgb = np.vstack(full_rgb_rows)
        full_mask = np.vstack(full_mask_rows)
        full_rgb_display = np.vstack(full_rgb_display_rows)

        # --- 4. Visualizar ---
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(full_rgb)
        ax[0].set_title("Original RGB Image")
        ax[0].axis("off")

        ax[1].imshow(full_rgb_display)
        ax[1].imshow(full_mask, alpha=0.2)
        ax[1].set_title("Cloud Mask Overlay")
        ax[1].axis("off")
        plt.savefig(os.path.join(DISPLAY_PATH, f"{job_id}_cloud.png"))

    except Exception as e:
        print(f"  ✗ Error processing cloud mask for {job_id}: {e}")
        return False

    return True


def create_forest_picture(output_folder, metadata_path, job_id=None, cloud_job_id=None):
    """
    Create a forest picture from the image data.

    Args:
        image_path (str): Path to the input image
        output_folder (str): Path to the output folder
        metadata_path (str): Path to the metadata file
        job_id (str, optional): Job ID for tracking purposes

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing forest picture for: {job_id}")

        print("Loading metadata...")

        metadata = pd.read_excel(metadata_path)
        TILES_IMAGES_PATH = os.path.join(output_folder, "TILES_IMAGES")
        DISPLAY_PATH = os.path.join(output_folder, "DISPLAY")

        coords = [eval(c) for c in metadata['tile_coordinates']]
        rows = sorted(set(r for r, _ in coords))
        cols = sorted(set(c for _, c in coords))

        tile_dict_rgb = {}
        tile_dict_ndvi = {}

        for i in range(metadata.shape[0]):

            is_cloudy = metadata['cloud?'][i]
            is_forest = metadata['forest?'][i]

            coord = eval(metadata['tile_coordinates'][i])  # (row, col)

            if not is_cloudy and is_forest:

                tile_path = os.path.join(TILES_IMAGES_PATH, f"{cloud_job_id}_tile_{metadata['tile_coordinates'][i]}.npy")

                tile = np.load(tile_path)

                # Calculate ndvi for display purposes

                nir = tile[..., 7].astype(np.float32)
                red = tile[..., 3].astype(np.float32)

                ndvi = np.divide(nir - red, nir + red,
                                out=np.zeros_like(nir), where=(nir + red) != 0)

                rgb = tile[:,:,[3,2,1]]  # Extract RGB bands for display
                gamma = 1.2  # >1 aclara sombras y realza color
                rgb = np.power(rgb * 255.0, gamma) / 255.0

            else:
                if is_cloudy:
                    # Black
                    rgb = np.zeros((256, 256, 3), dtype=np.float32)
                else:
                    # Red
                    rgb = np.zeros((256, 256, 3), dtype=np.float32)
                    rgb[..., 0] = 255  # Set red channel to 255
                ndvi = np.zeros((256, 256), dtype=np.float32)

            tile_dict_rgb[coord] = rgb
            tile_dict_ndvi[coord] = ndvi

        # Display everything
        full_rgb_rows = []
        full_ndvi_rows = []

        for r in rows:
            rgb_row = [tile_dict_rgb[(r,c)] for c in cols]
            ndvi_row = [tile_dict_ndvi[(r,c)] for c in cols]
            full_rgb_rows.append(np.hstack(rgb_row))
            full_ndvi_rows.append(np.hstack(ndvi_row))

        full_rgb = np.vstack(full_rgb_rows)
        full_ndvi = np.vstack(full_ndvi_rows)

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(full_rgb)
        ax[0].set_title("Original RGB Image")
        ax[0].axis("off")

        ax[1].imshow(full_ndvi, cmap="RdYlGn")
        ax[1].set_title("NDVI Image")
        ax[1].axis("off")
        
        plt.savefig(os.path.join(DISPLAY_PATH, f"{job_id}_forest.png"))
        

    except Exception as e:
        print(f"  ✗ Error processing forest picture for {job_id}: {e}")
        return False

    return True

def create_heatmap(image_data, output_folder, metadata_path, job_id=None):
    """
    Create a heatmap from the image data.

    Args:
        image_path (str): Path to the input image
        output_folder (str): Path to the output folder
        metadata_path (str): Path to the metadata file
        job_id (str, optional): Job ID for tracking purposes

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing heatmap for: {job_id}")

        # Check if image data is valid
        if image_data is None:
            print(f"  ✗ Invalid image data for {job_id}")
            return False

        # Create output filename
        output_path = os.path.join(output_folder, f"{job_id}_heatmap.png")

        # Save as PNG using PIL
        pil_image = Image.fromarray(image_data.astype(np.uint8))
        pil_image.save(output_path)

    except Exception as e:
        print(f"  ✗ Error processing heatmap for {job_id}: {e}")
        return False

    return True
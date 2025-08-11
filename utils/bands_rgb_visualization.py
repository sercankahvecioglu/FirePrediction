import os
import numpy as np
from PIL import Image
import glob
import matplotlib.cm as cm

def extract_rgb_channels(input_folder, output_folder):
    """
    Extract RGB channels from multi-band images and save as PNG files.
    
    Args:
        input_folder (str): Path to folder containing input images
        output_folder (str): Path to folder where PNG images will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all files in the input folder (assuming numpy arrays or similar format)
    file_patterns = ['*.npy', '*.npz', '*.tif', '*.tiff']
    input_files = []
    
    for pattern in file_patterns:
        input_files.extend(glob.glob(os.path.join(input_folder, pattern)))
    
    if not input_files:
        print(f"No compatible files found in {input_folder}")
        return
    
    for file_path in input_files:
        try:
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            # Load the image data
            if file_path.endswith('.npy'):
                image_data = np.load(file_path)
            elif file_path.endswith('.npz'):
                # Assuming the array is stored with a key, try common keys
                npz_data = np.load(file_path)
                if len(npz_data.files) == 1:
                    image_data = npz_data[npz_data.files[0]]
                else:
                    # Try common keys
                    for key in ['arr_0', 'data', 'image']:
                        if key in npz_data.files:
                            image_data = npz_data[key]
                            break
                    else:
                        print(f"Could not determine array key in {file_path}")
                        continue
            else:  # TIFF files
                from PIL import Image as PILImage
                import tifffile
                image_data = tifffile.imread(file_path)
            
            # Check if image has correct shape
            if len(image_data.shape) != 3 or image_data.shape[2] != 15:
                print(f"Skipping {filename}: Expected shape (height, width, 15), got {image_data.shape}")
                continue
            
            # Extract RGB channels (indices 3, 2, 1 for R, G, B respectively)
            # Note: PIL expects RGB order, so we arrange as [R, G, B]
            red_channel = image_data[:, :, 3]    # Index 3 for Red
            green_channel = image_data[:, :, 2]  # Index 2 for Green
            blue_channel = image_data[:, :, 1]   # Index 1 for Blue
            
            # Stack channels to create RGB image
            rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=2)
            
            # Normalize to 0-255 range if needed
            if rgb_image.dtype != np.uint8:
                # Check if values are in 0-1 range (float) or larger range
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                else:
                    # Normalize to 0-255 range
                    rgb_min = rgb_image.min()
                    rgb_max = rgb_image.max()
                    if rgb_max > rgb_min:
                        rgb_image = ((rgb_image - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
                    else:
                        rgb_image = np.zeros_like(rgb_image, dtype=np.uint8)
            
            # Convert to PIL Image and save as PNG
            pil_image = Image.fromarray(rgb_image, mode='RGB')
            output_path = os.path.join(output_folder, f"{filename}.png")
            pil_image.save(output_path)
            
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def main():
    # Example usage
    input_folder = '/home/dario/Desktop/FlameSentinels/TILES_INPUT_DATA'
    output_folder = '/home/dario/Desktop/FlameSentinels/TILES_IMGS'
    
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        return
    
    extract_rgb_channels(input_folder, output_folder)
    print("RGB extraction completed!")

if __name__ == "__main__":
    main()
    """
    input_folder = '/home/dario/Desktop/FlameSentinels/TEST_LABELS'
    output_folder = '/home/dario/Desktop/FlameSentinels/TEST_LABELS_IMGS'
    
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        exit()
    
    input_files = glob.glob(os.path.join(input_folder, '*.npy'))

    for filepath in input_files:

        filename = os.path.splitext(os.path.basename(filepath))[0]

        img = np.load(filepath)
        
        # Squeeze to remove any single dimensions and ensure 2D array
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(axis=2)  # Remove the last dimension if it's 1
        elif len(img.shape) > 2:
            img = img.squeeze()  # Remove all single dimensions
        
        # Ensure values are in 0-1 range for colormap
        if img.max() > 1.0:
            img = img / img.max()  # Normalize to 0-1 range
        
        colormap = cm.get_cmap('coolwarm')  # Blue to red colormap
        colored_pred = colormap(img)  # This returns (height, width, 4) - RGBA
        
        # Convert to uint8 (0-255) and remove alpha channel
        colored_pred_uint8 = (colored_pred[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(colored_pred_uint8, mode='RGB')
        
        output_path = os.path.join(output_folder, f"{filename}.png")
        img.save(output_path)
            
        print(f"Saved: {output_path}")

    print("RGB extraction completed!")
    """
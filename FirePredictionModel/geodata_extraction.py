import xml.etree.ElementTree as ET
import pickle

def get_real_world_coords(coordy, coordx, metadata=None, pkl_path=None, tile_size=(256, 256), image_size=(12500, 12500)):
    """
    Get real-world coordinates of tile center from top-left pixel coordinates.
    
    Params:
        - coordy (int): y pixel coordinate of the top left, relative to the full image
        - coordx (int): x pixel coordinate of the top left, relative to the full image
        - metadata (dict, optional): geoinfo dictionary (preferred)
        - pkl_path (str, optional): path of the metadata pickle file (if dict not provided)
        - tile_size (tuple): size of the tile (default (256, 256))
        - image_size (tuple): size of the full image (default (12500, 12500))

    Returns:
        string of the coordinates
    """
    if metadata is None:
        if pkl_path is None:
            raise ValueError("Either metadata dict or pkl_path must be provided.")
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)

    # Extract spatial extent from metadata
    spatial_extent = metadata.get('spatial_extent', {})
    west = spatial_extent.get('west_bound')
    east = spatial_extent.get('east_bound')
    north = spatial_extent.get('north_bound')
    south = spatial_extent.get('south_bound')
    
    if None in [west, east, north, south]:
        raise ValueError("Incomplete spatial extent information in metadata")

    # Calculate center pixel coordinates of the tile
    center_y = coordy + tile_size[0] // 2
    center_x = coordx + tile_size[1] // 2

    # Convert pixel coordinates to geographic coordinates
    # Calculate the geographic span per pixel
    lon_per_pixel = (east - west) / image_size[1]
    lat_per_pixel = (north - south) / image_size[0]
    
    # Calculate real-world coordinates
    lon = west + (center_x * lon_per_pixel)
    lat = north - (center_y * lat_per_pixel)  # Subtract because image y increases downward

    lat_dir = 'N' if lat >= 0 else 'S'
    lon_dir = 'E' if lon >= 0 else 'W'

    return f"{abs(lat):.6f}°{lat_dir}, {abs(lon):.6f}°{lon_dir}"

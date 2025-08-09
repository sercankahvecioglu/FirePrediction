import xml.etree.ElementTree as ET
import pickle
import re
import os
import glob
from pyproj import Transformer

def extract_geospatial_metadata(country_id, base_path, pkl_path, when='pre', resolution=10):
    """
    Extract geospatial metadata from XML and save to pickle file.
    
    Params:
        - country_id (str): the name/id of the country of which to retrieve the metadata
        - base_path (str): path of DATASETS folder
        - pkl_path (str): output folder path
        - when (str): string representing the timing (pre or post) of the image w.r.t. the fire
        - resolution (int): optional argument for what resolution to consider.
        By default we resample all bands to 10m res

    Returns:
        metadata (dict): dictionary with necessary metadata for coordinate conversion
    """
    try:
        xml_path = glob.glob(os.path.join(base_path, f"{country_id}_{when}/GRANULE/*/MTD_TL.xml"))[0]
    except IndexError:
        raise FileNotFoundError(f"MTD_TL.xml not found for {country_id}_{when}")

    print(f"Processing: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find geoposition for specified resolution
    try:
        geoposition = next(geo for geo in root.findall(".//Geoposition") 
                          if geo.get('resolution') == str(resolution))
    except StopIteration:
        raise ValueError(f"No geoposition found for resolution {resolution}m")
    
    # Extract essential coordinate system information
    crs_element = root.find(".//HORIZONTAL_CS_CODE")
    crs_code = crs_element.text if crs_element is not None else None
    
    # Extract EPSG code from CRS
    epsg_code = None
    if crs_code:
        epsg_match = re.search(r'EPSG:(\d+)', crs_code)
        if epsg_match:
            epsg_code = int(epsg_match.group(1))
    
    if not epsg_code:
        raise ValueError(f"Could not extract EPSG code from CRS: {crs_code}")
    
    # Extract essential transformation parameters
    metadata = {
        'ULX': float(geoposition.find('ULX').text),
        'ULY': float(geoposition.find('ULY').text),
        'XDIM': float(geoposition.find('XDIM').text),
        'YDIM': float(geoposition.find('YDIM').text),
        'EPSG_CODE': epsg_code
    }

    pkl_path = os.path.join(pkl_path, f"{country_id}_geoinfo.pkl")
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata

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

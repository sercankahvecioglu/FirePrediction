import xml.etree.ElementTree as ET
import pickle
import re
import os
import glob
from pyproj import Transformer

def extract_geospatial_metadata(country_id, pkl_path, when='pre', resolution=10):
    """
    Extract geospatial metadata from XML and save to pickle file.
    
    Params:
        country_id (str): the name/id of the country of which to retrieve the metadata
        pkl_path (str): output folder path
        when (str): string representing the timing (pre or post) of the image w.r.t. the fire
        resolution (int): optional argument for what resolution to consider.
        By default we resample all bands to 10m res

    Returns:
        metadata (dict): dictionary with necessary metadata for coordinate conversion
    """
    try:
        xml_path = glob.glob(f"/home/dario/Desktop/flame_sentinel_data/DATASETS/{country_id}_{when}/GRANULE/*/MTD_TL.xml")[0]
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

    pkl_path = os.path.join(pkl_path, f"{country_id}_{when}.pkl")
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata

def get_real_world_coords(coordy, coordx, pkl_path, tile_size=(256, 256)):
    """
    Get real-world coordinates of tile center from top-left pixel coordinates.
    
    Params:
        coordy (int): y pixel coordinate of the top left, relative to the full image
        coordx (int): x pixel coordinate of the top left, relative to the full image
        pkl_path (str): path of the metadata pickle file path

    Returns:
        string of the coordinates
    """
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Calculate center pixel coordinates
    center_y = coordy + tile_size[0] // 2
    center_x = coordx + tile_size[1] // 2
    
    # Transform to UTM then to WGS84
    utm_x = metadata['ULX'] + (center_x * metadata['XDIM'])
    utm_y = metadata['ULY'] + (center_y * metadata['YDIM'])
    
    transformer = Transformer.from_crs(
        f"EPSG:{metadata['EPSG_CODE']}", "EPSG:4326", always_xy=True
    )
    lon, lat = transformer.transform(utm_x, utm_y)
    
    lat_dir = 'N' if lat >= 0 else 'S'
    lon_dir = 'E' if lon >= 0 else 'W'
    
    return f"{abs(lat):.6f}°{lat_dir}, {abs(lon):.6f}°{lon_dir}"


if __name__ == "__main__":
    # Extract metadata for all datasets
    print("Extracting metadata...")
    for path in glob.glob("/home/dario/Desktop/flame_sentinel_data/DATASETS/*"):
        try:
            country_id, when = os.path.basename(path).split("_")
            extract_geospatial_metadata(country_id, "/home/dario/Desktop/imgs_metadata", when=when)
            print(f"✓ Processed {country_id}_{when}")
        except Exception as e:
            print(f"✗ Failed to process {path}: {e}")
    
    # Example coordinate conversion
    fpath = "/home/dario/Desktop/flame_sentinel_data/TEST_INPUT_DATA/chile_tile_(7168, 6400).npy"
    
    country_id, realcoords = os.path.basename(fpath).split('_tile_')

    # Remove .npy extension first, then strip parentheses
    coords_str = realcoords.replace('.npy', '').strip('()')
    print(coords_str)
    # extract values of relative coordinates
    coordx, coordy = map(int, coords_str.split(', '))
    # get real-world coords 
    pkl_path = f"/home/dario/Desktop/imgs_metadata/{country_id}_pre.pkl"

    coords = get_real_world_coords(coordx, coordy, pkl_path)
    print(coords)
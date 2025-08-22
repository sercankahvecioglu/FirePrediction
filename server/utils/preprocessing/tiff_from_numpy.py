import pickle
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from datetime import datetime

def load_metadata_from_pickle(pickle_path):
    """Load metadata from pickle file"""
    with open(pickle_path, 'rb') as f:
        metadata = pickle.load(f)
    return metadata

def create_geotiff_from_pickle_metadata(npy_data, pickle_metadata, output_path, nodata=None):
    """
    Create GeoTIFF from NumPy array and pickle metadata
    
    Parameters:
    - npy_data: NumPy array (2D or 3D) with shape (bands, height, width) or (height, width)
    - pickle_metadata: Dictionary loaded from pickle file
    - output_path: Output GeoTIFF path
    - nodata: NoData value (optional)
    """
    
    # Ensure data has band dimension
    if npy_data.ndim == 2:
        npy_data = npy_data[np.newaxis, :, :]
    
    bands, height, width = npy_data.shape
    
    # Extract spatial extent
    spatial = pickle_metadata['spatial_extent']
    bounds = (
        spatial['west_bound'],   # west
        spatial['south_bound'],  # south
        spatial['east_bound'],   # east
        spatial['north_bound']   # north
    )
    
    # Calculate transform from bounds
    transform = from_bounds(*bounds, width, height)
    
    # Parse CRS - handle the URL format
    crs_code = pickle_metadata['technical_specs']['crs_code']
    if 'EPSG' in crs_code:
        # Extract EPSG code from URL format
        epsg_code = crs_code.split('/')[-1]
        crs = CRS.from_epsg(int(epsg_code))
    else:
        # Fallback to WGS84 if CRS parsing fails
        print(f"Warning: Could not parse CRS {crs_code}, using EPSG:4326")
        crs = CRS.from_epsg(4326)
    
    # Create rasterio profile
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': bands,
        'dtype': npy_data.dtype,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256
    }
    
    # Add nodata if specified
    if nodata is not None:
        profile['nodata'] = nodata
    
    # Create comprehensive tags from metadata
    tags = {
        'AREA_OR_POINT': 'Area',
        'TIFFTAG_SOFTWARE': 'Python/Rasterio',
        'TIFFTAG_DATETIME': datetime.now().strftime('%Y:%m:%d %H:%M:%S'),
        
        # Custom tags from pickle metadata
        'COUNTRY_ID': pickle_metadata.get('country_id', ''),
        'TIME_PERIOD': pickle_metadata.get('time_period', ''),
        'PRODUCT_TITLE': pickle_metadata.get('product_info', {}).get('title', ''),
        'CREATION_DATE': pickle_metadata.get('product_info', {}).get('creation_date', ''),
        'START_TIME': pickle_metadata.get('temporal_extent', {}).get('start_time', ''),
        'END_TIME': pickle_metadata.get('temporal_extent', {}).get('end_time', ''),
        'SPATIAL_RESOLUTION': str(pickle_metadata.get('technical_specs', {}).get('spatial_resolution', '')),
        'CENTER_LAT': str(spatial.get('center_lat', '')),
        'CENTER_LON': str(spatial.get('center_lon', '')),
        'ORGANIZATION': pickle_metadata.get('contact_info', {}).get('organization', ''),
        'CONTACT_EMAIL': pickle_metadata.get('contact_info', {}).get('email', ''),
        'KEYWORDS': ','.join(pickle_metadata.get('technical_specs', {}).get('keywords', []))
    }
    
    # Write the GeoTIFF
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(npy_data)
        dst.update_tags(**tags)
        
        # Add band descriptions if multiple bands
        if bands > 1:
            for i in range(bands):
                dst.set_band_description(i + 1, f'Band_{i + 1}')
    
    print(f"Successfully created GeoTIFF: {output_path}")
    print(f"Shape: {npy_data.shape}")
    print(f"CRS: {crs}")
    print(f"Bounds: {bounds}")
    print(f"Spatial Resolution: {pickle_metadata.get('technical_specs', {}).get('spatial_resolution', 'Unknown')}m")

def complete_workflow_example():
    """
    Complete example of loading pickle metadata and numpy data to create GeoTIFF
    """
    
    # Load your data
    npy_data = np.load('your_bands_data.npy')  # Replace with your .npy file
    
    # Load pickle metadata
    with open('your_metadata.pkl', 'rb') as f:  # Replace with your .pkl file
        pickle_metadata = pickle.load(f)
    
    # Create GeoTIFF
    create_geotiff_from_pickle_metadata(
        npy_data=npy_data,
        pickle_metadata=pickle_metadata,
        output_path='output_sentinel2.tif',
        nodata=-9999  # Set appropriate nodata value
    )

def verify_created_geotiff(geotiff_path):
    """
    Verify the created GeoTIFF by reading it back
    """
    with rasterio.open(geotiff_path) as src:
        print(f"\nVerification of {geotiff_path}:")
        print(f"Shape: {src.shape}")
        print(f"Bands: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Transform: {src.transform}")
        print(f"Tags: {src.tags()}")
        
        # Read a sample of the data
        sample_data = src.read(1, window=((0, min(10, src.height)), (0, min(10, src.width))))
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Data type: {sample_data.dtype}")
        print(f"Sample values: {sample_data.flatten()[:5]}")

# Example usage functions
def create_sample_data():
    """
    Create sample data matching your metadata for testing
    """
    # Based on your metadata, create sample Sentinel-2 data
    # Typical Sentinel-2 has multiple spectral bands
    bands = 12  # Sentinel-2 has 13 bands, using 12 here
    height, width = 1000, 1000  # Sample dimensions
    
    # Create realistic-looking remote sensing data
    sample_data = np.random.uniform(0, 10000, (bands, height, width)).astype(np.uint16)
    
    # Save sample data
    np.save('sample_sentinel2_data.npy', sample_data)
    
    # Create sample metadata (using your format)
    sample_metadata = {
        'country_id': 'izmir',
        'time_period': 'pre',
        'product_info': {
            'title': 'S2B_MSIL1C_20250625T085559_N0511_R007_T35SMC_20250625T112531.SAFE',
            'creation_date': '2014-01-01',
            'file_path': ''
        },
        'spatial_extent': {
            'west_bound': 25.847248649247085,
            'east_bound': 27.11247403813807,
            'south_bound': 37.85395064837389,
            'north_bound': 38.84894433319432,
            'center_lat': 38.3514474907841,
            'center_lon': 26.479861343692576
        },
        'temporal_extent': {
            'start_time': '2025-06-25T09:00:49',
            'end_time': '2025-06-25T09:13:34'
        },
        'technical_specs': {
            'spatial_resolution': 20,
            'crs_code': 'http://www.opengis.net/def/crs/EPSG/0/4936',
            'keywords': ['Orthoimagery', 'Land cover', 'Geographical names', 'data set series', 'processing']
        },
        'contact_info': {
            'organization': 'org_name',
            'email': 'org_name@org.ext'
        }
    }
    
    # Save sample metadata
    with open('sample_metadata.pkl', 'wb') as f:
        pickle.dump(sample_metadata, f)
    
    print("Sample data and metadata created!")
    return sample_data, sample_metadata

if __name__ == "__main__":
    # Example workflow
    
    # 1. Create sample data (remove this if you have real data)
    create_sample_data()
    
    # 2. Load and process your actual data
    npy_data = np.load('sample_sentinel2_data.npy')  # Replace with your file
    
    with open('sample_metadata.pkl', 'rb') as f:  # Replace with your file
        metadata = pickle.load(f)
    
    # 3. Create GeoTIFF
    create_geotiff_from_pickle_metadata(
        npy_data=npy_data,
        pickle_metadata=metadata,
        output_path='sentinel2_reconstructed.tif',
        nodata=0  # Adjust based on your data
    )
    
    # 4. Verify the result
    verify_created_geotiff('sentinel2_reconstructed.tif')
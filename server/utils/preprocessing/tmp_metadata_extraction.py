import xml.etree.ElementTree as ET
import pickle
from pathlib import Path
import os

class MetadataExtractor:
    """
    Class to extract essential metadata from Sentinel-2 XML files for use with Sentinel Hub API
    """
    
    def __init__(self, xml_folder_path: str):
        """
        Initialize the MetadataExtractor
        
        Args:
            xml_folder_path (str): Path to the folder containing XML files
        """
        self.xml_folder_path = Path(xml_folder_path)
        self.metadata_cache = {}
    

    def extract_geospatial_metadata(self, country_id, when='pre', resolution=10):
        """
        Extract geospatial metadata from XML files.
        
        Args:
            country_id (str): Country identifier
            pkl_path (str): Path to the XML file or directory
            when (str): Time period identifier ('pre', 'post')
            resolution (int): Spatial resolution in meters
        
        Returns:
            dict: Extracted metadata dictionary
        """
        
        xml_file = self.xml_folder_path / f"{country_id}_{when}_inspire.xml"
        
        # Check if the XML file exists
        if not xml_file.exists():
            raise FileNotFoundError(f"XML file not found: {xml_file}")
        
        print(f"Extracting metadata from: {xml_file}")
        
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Namespaces
        ns = {
            'gmd': 'http://www.isotc211.org/2005/gmd',
            'gco': 'http://www.isotc211.org/2005/gco',
            'gml': 'http://www.opengis.net/gml'
        }
        
        try:
            # Extract product identification
            title_elem = root.find('.//gmd:title/gco:CharacterString', ns)
            title = title_elem.text if title_elem is not None else "Unknown"
            
            # Extract geographic coordinates
            west_elem = root.find('.//gmd:westBoundLongitude/gco:Decimal', ns)
            east_elem = root.find('.//gmd:eastBoundLongitude/gco:Decimal', ns)
            south_elem = root.find('.//gmd:southBoundLatitude/gco:Decimal', ns)
            north_elem = root.find('.//gmd:northBoundLatitude/gco:Decimal', ns)
            
            west = float(west_elem.text) if west_elem is not None else None
            east = float(east_elem.text) if east_elem is not None else None
            south = float(south_elem.text) if south_elem is not None else None
            north = float(north_elem.text) if north_elem is not None else None
            
            # Extract temporal information
            begin_elem = root.find('.//gml:beginPosition', ns)
            end_elem = root.find('.//gml:endPosition', ns)
            
            begin_time = begin_elem.text if begin_elem is not None else None
            end_time = end_elem.text if end_elem is not None else None
            
            # Extract spatial resolution
            resolution_elem = root.find('.//gmd:denominator/gco:Integer', ns)
            spatial_resolution = int(resolution_elem.text) if resolution_elem is not None else resolution
            
            # Extract coordinate reference system
            crs_elem = root.find('.//gmd:code/gco:CharacterString', ns)
            crs_code = crs_elem.text if crs_elem is not None else "Unknown"
            
            # Extract keywords
            keywords = []
            for keyword in root.findall('.//gmd:keyword/gco:CharacterString', ns):
                if keyword.text:
                    keywords.append(keyword.text)
            
            # Extract organization info
            org_elem = root.find('.//gmd:organisationName/gco:CharacterString', ns)
            email_elem = root.find('.//gmd:electronicMailAddress/gco:CharacterString', ns)
            
            organization = org_elem.text if org_elem is not None else "Unknown"
            email = email_elem.text if email_elem is not None else "Unknown"
            
            # Extract creation date
            date_elem = root.find('.//gmd:date/gmd:CI_Date/gmd:date/gco:Date', ns)
            creation_date = date_elem.text if date_elem is not None else None
            
        except Exception as e:
            print(f"Warning: Error extracting some metadata: {e}")
            # Provide default values if extraction fails
            title = "Unknown"
            west = east = south = north = None
            begin_time = end_time = None
            spatial_resolution = resolution
            crs_code = "Unknown"
            keywords = []
            organization = email = "Unknown"
            creation_date = None
        
        # Build metadata dictionary
        metadata = {
            'country_id': country_id,
            'time_period': when,
            'product_info': {
                'title': title,
                'creation_date': creation_date,
                'file_path': xml_file
            },
            'spatial_extent': {
                'west_bound': west,
                'east_bound': east,
                'south_bound': south,
                'north_bound': north,
                'center_lat': (north + south) / 2 if north and south else None,
                'center_lon': (east + west) / 2 if east and west else None
            },
            'temporal_extent': {
                'start_time': begin_time,
                'end_time': end_time
            },
            'technical_specs': {
                'spatial_resolution': spatial_resolution,
                'crs_code': crs_code,
                'keywords': keywords
            },
            'contact_info': {
                'organization': organization,
                'email': email
            }
        }
        
        return metadata

# Use relative path from the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
XML_FOLDER_PATH = os.path.join(current_dir, "..", "..", "..", "copied_xml_files")

extractor = MetadataExtractor(XML_FOLDER_PATH)

countries = ['california', 'chile', 'france', 'france2', 'usa2', 'greece', 'sardinia']
when = ['pre', 'post']

for country in countries:
    for tim in when:
        tiles_data = extractor.extract_all_metadata(country, tim, "30m")
        output_dir = os.path.join(current_dir, "..", "..", "..")
        with open(os.path.join(output_dir, f"{country}_{tim}_tiles_data.pkl"), "wb") as f:
            pickle.dump(tiles_data, f)

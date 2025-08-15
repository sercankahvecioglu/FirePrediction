# This code serves as the main entry point for processing satellite images.
# It coordinates the various steps involved in image processing, including loading,
# processing, and saving the results.
#
# For running this code, ensure that you have the necessary dependencies installed
# and run the server before.

import requests
import numpy as np
import argparse
import time
import os
import sys
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from PIL import Image


class SatelliteImageProcessor:
    """
    CLI client for processing satellite images through the FastAPI server.
    
    This class handles the complete workflow:
    1. Cloud detection
    2. Forest detection  
    3. Fire prediction
    4. Image visualization and display
    """
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        """
        Initialize the processor with API base URL.
        
        Args:
            base_url (str): Base URL of the FastAPI server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def check_server_health(self) -> bool:
        """
        Check if the server is running and accessible.
        
        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def submit_cloud_detection(self, npy_file_path: str, tile_size: int = 256) -> Optional[str]:
        """
        Submit image for cloud detection processing.
        
        Args:
            npy_file_path (str): Path to the .npy file
            tile_size (int): Size of tiles for processing
            
        Returns:
            Optional[str]: Job ID if successful, None otherwise
        """
        try:
            with open(npy_file_path, 'rb') as f:
                files = {'file': (os.path.basename(npy_file_path), f, 'application/octet-stream')}
                data = {'tile_size': tile_size}
                
                response = self.session.post(
                    f"{self.base_url}/submit-image/cloud-detection",
                    files=files,
                    data=data
                )
                
            if response.status_code == 200:
                result = response.json()
                print(f"Cloud detection job submitted successfully")
                print(f"   Job ID: {result['job_id']}")
                return result['job_id']
            else:
                print(f"Failed to submit cloud detection job: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error submitting cloud detection job: {e}")
            return None
    
    def submit_forest_detection(self, cloud_job_id: str) -> Optional[str]:
        """
        Submit image for forest detection processing.
        
        Args:
            cloud_job_id (str): Job ID from cloud detection
            
        Returns:
            Optional[str]: Job ID if successful, None otherwise
        """
        try:
            data = {'cloud_job_id': cloud_job_id}
            response = self.session.post(
                f"{self.base_url}/submit-image/forest-detection",
                params=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Forest detection job submitted successfully")
                print(f"   Job ID: {result['job_id']}")
                return result['job_id']
            else:
                print(f"Failed to submit forest detection job: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error submitting forest detection job: {e}")
            return None
    
    def submit_fire_prediction(self, cloud_job_id: str) -> Optional[str]:
        """
        Submit image for fire prediction processing.
        
        Args:
            cloud_job_id (str): Job ID from cloud detection
            
        Returns:
            Optional[str]: Job ID if successful, None otherwise
        """
        try:
            data = {'cloud_job_id': cloud_job_id}
            response = self.session.post(
                f"{self.base_url}/submit-image/fire-prediction",
                params=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Fire prediction job submitted successfully")
                print(f"   Job ID: {result['job_id']}")
                return result['job_id']
            else:
                print(f"Failed to submit fire prediction job: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error submitting fire prediction job: {e}")
            return None
    
    def wait_for_job_completion(self, job_id: str, job_type: str, max_wait_time: int = 600) -> bool:
        """
        Wait for a job to complete with progress updates.
        
        Args:
            job_id (str): Job ID to monitor
            job_type (str): Type of job for display purposes
            max_wait_time (int): Maximum time to wait in seconds
            
        Returns:
            bool: True if job completed successfully, False otherwise
        """
        print(f"Waiting for {job_type} to complete...")
        start_time = time.time()
        last_progress = -1
        not_completed = True
        
        while not_completed:
            try:
                response = self.session.get(f"{self.base_url}/job-status/{job_id}")
                
                if response.status_code == 200:
                    status = response.json()
                    current_progress = status.get('progress', 0)
                    
                    # Show progress update if it changed
                    if current_progress != last_progress:
                        print(f"   Progress: {current_progress}% - {status.get('message', 'Processing...')}")
                        last_progress = current_progress
                    
                    if status['status'] == 'completed':
                        print(f"{job_type} completed successfully!")
                        not_completed = False
                        return True
                    elif status['status'] == 'failed':
                        print(f"{job_type} failed: {status.get('message', 'Unknown error')}")
                        return False
                        
                time.sleep(2)  # Poll every 2 seconds
                
            except Exception as e:
                print(f"Error checking job status: {e}")
                return False
        
        print(f"{job_type} timed out after {max_wait_time} seconds")
        return False
    
    def download_image(self, job_id: str, job_type: str, output_dir: str = "./results") -> Optional[str]:
        """
        Download processed image from completed job.
        
        Args:
            job_id (str): Job ID to download image for
            job_type (str): Type of job for filename
            output_dir (str): Directory to save the image
            
        Returns:
            Optional[str]: Path to downloaded image if successful, None otherwise
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            response = self.session.get(f"{self.base_url}/download-image/{job_id}")
            
            if response.status_code == 200:
                filename = f"{job_type}_{job_id}_result.png"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"{job_type} image saved to: {filepath}")
                return filepath
            else:
                print(f"Failed to download {job_type} image: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error downloading {job_type} image: {e}")
            return None
    
    def download_picture_by_name(self, picture_name: str, output_dir: str = "./results") -> Optional[str]:
        """
        Download a specific picture by name.
        
        Args:
            picture_name (str): Name of the picture to download
            output_dir (str): Directory to save the image
            
        Returns:
            Optional[str]: Path to downloaded image if successful, None otherwise
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            response = self.session.get(f"{self.base_url}/download-picture/{picture_name}")
            
            if response.status_code == 200:
                filepath = os.path.join(output_dir, picture_name)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"Picture saved to: {filepath}")
                return filepath
            else:
                print(f"Failed to download picture {picture_name}: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error downloading picture {picture_name}: {e}")
            return None
    
    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processing results for a completed job.
        
        Args:
            job_id (str): Job ID to get results for
            
        Returns:
            Optional[Dict[str, Any]]: Job results if successful, None otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/get-result/{job_id}")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get job results: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error getting job results: {e}")
            return None
    
    def display_images(self, image_paths: list, titles: list = None):
        """
        Display multiple images in a grid layout.
        
        Args:
            image_paths (list): List of image file paths
            titles (list): Optional list of titles for each image
        """
        # Filter out None paths
        valid_paths = [path for path in image_paths if path and os.path.exists(path)]
        
        if not valid_paths:
            print("No valid images to display")
            return
        
        num_images = len(valid_paths)
        if titles is None:
            titles = [f"Image {i+1}" for i in range(num_images)]
        
        # Calculate grid layout
        cols = min(2, num_images)
        rows = (num_images + cols - 1) // cols
        
        plt.figure(figsize=(15, 5 * rows))
        
        for i, (path, title) in enumerate(zip(valid_paths, titles[:num_images])):
            try:
                img = Image.open(path)
                plt.subplot(rows, cols, i + 1)
                plt.imshow(img)
                plt.title(title, fontsize=14)
                plt.axis('off')
            except Exception as e:
                print(f"Error displaying image {path}: {e}")
        
        plt.tight_layout()
        plt.show()
        print(f"Displayed {len(valid_paths)} images")
    
    def process_complete_workflow(self, npy_file_path: str, tile_size: int = 256, output_dir: str = "./results") -> bool:
        """
        Execute the complete satellite image processing workflow.
        
        Args:
            npy_file_path (str): Path to the input .npy file
            tile_size (int): Size of tiles for processing
            output_dir (str): Directory to save results
            
        Returns:
            bool: True if workflow completed successfully, False otherwise
        """
        print("Starting complete satellite image processing workflow...")
        print(f"   Input file: {npy_file_path}")
        print(f"   Tile size: {tile_size}")
        print(f"   Output directory: {output_dir}")
        print()
        
        # Check if file exists
        if not os.path.exists(npy_file_path):
            print(f"Input file not found: {npy_file_path}")
            return False
        
        # Check server health
        if not self.check_server_health():
            print("Server is not accessible. Please ensure the FastAPI server is running.")
            return False
        
        print("Server is healthy and accessible")
        print()
        
        # Step 1: Cloud Detection
        print("Step 1: Cloud Detection")
        cloud_job_id = self.submit_cloud_detection(npy_file_path, tile_size)
        if not cloud_job_id:
            return False
        
        if not self.wait_for_job_completion(cloud_job_id, "cloud detection"):
            return False
        print()
        
        # Step 2: Forest Detection
        print("Step 2: Forest Detection")
        forest_job_id = self.submit_forest_detection(cloud_job_id)
        if not forest_job_id:
            return False
        
        if not self.wait_for_job_completion(forest_job_id, "forest detection"):
            return False
        print()
        
        # Step 3: Fire Prediction
        print("Step 3: Fire Prediction")
        fire_job_id = self.submit_fire_prediction(cloud_job_id)
        if not fire_job_id:
            return False
        
        if not self.wait_for_job_completion(fire_job_id, "fire prediction"):
            return False
        print()
        
        # Step 4: Download all images
        print("Step 4: Downloading Results")
        
        # Get results to know image names
        fire_results = self.get_job_results(fire_job_id)
        if not fire_results:
            print("Failed to get fire prediction results")
            return False
        
        # Download all images using picture names from results
        image_paths = []
        image_titles = []
        
        # RGB image
        if fire_results.get('rgb_image_url'):
            rgb_path = self.download_picture_by_name(fire_results['rgb_image_url'], output_dir)
            if rgb_path:
                image_paths.append(rgb_path)
                image_titles.append("RGB Visualization")
        
        # Cloud detection image
        if fire_results.get('cloud_image_url'):
            cloud_path = self.download_picture_by_name(fire_results['cloud_image_url'], output_dir)
            if cloud_path:
                image_paths.append(cloud_path)
                image_titles.append("Cloud Detection")
        
        # Forest detection image
        if fire_results.get('forest_image_url'):
            forest_path = self.download_picture_by_name(fire_results['forest_image_url'], output_dir)
            if forest_path:
                image_paths.append(forest_path)
                image_titles.append("Forest Detection")
        
        # Fire prediction heatmap
        if fire_results.get('heatmap_image_url'):
            fire_path = self.download_picture_by_name(fire_results['heatmap_image_url'], output_dir)
            if fire_path:
                image_paths.append(fire_path)
                image_titles.append("Fire Risk Heatmap")
        
        print()
        
        # Step 5: Display images
        print("Step 5: Displaying Results")
        if image_paths:
            self.display_images(image_paths, image_titles)
        else:
            print("No images available for display")
            return False
        
        # Print summary
        print()
        print("Workflow completed successfully!")
        print(f"   Cloud detection job ID: {cloud_job_id}")
        print(f"   Forest detection job ID: {forest_job_id}")
        print(f"   Fire prediction job ID: {fire_job_id}")
        print(f"   Results saved to: {output_dir}")
        print(f"   Processing time: {fire_results.get('processing_time', 'Unknown')} seconds")
        
        return True


def main():
    """
    Main CLI function with argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Process satellite images through cloud detection, forest detection, and fire prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process.py --input image.npy
  python process.py --input image.npy --tile-size 512 --output ./my_results
  python process.py --input image.npy --server-url http://localhost:5001
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input .npy satellite image file'
    )
    
    parser.add_argument(
        '--tile-size', '-t',
        type=int,
        default=256,
        help='Size of tiles for processing (default: 256)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    parser.add_argument(
        '--server-url', '-s',
        type=str,
        default='http://localhost:5001',
        help='Base URL of the FastAPI server (default: http://localhost:5001)'
    )
    
    args = parser.parse_args()
    
    # Create processor instance
    processor = SatelliteImageProcessor(base_url=args.server_url)
    
    # Run the complete workflow
    success = processor.process_complete_workflow(
        npy_file_path=args.input,
        tile_size=args.tile_size,
        output_dir=args.output
    )
    
    if success:
        print("\nAll processing completed successfully!")
        sys.exit(0)
    else:
        print("\nProcessing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()


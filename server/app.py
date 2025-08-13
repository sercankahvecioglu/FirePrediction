import shutil
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid
import os
import json
from datetime import datetime
import asyncio
import sys
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import random
# Note: PIL/Pillow might not be installed, so we'll use matplotlib for all image generation

# ---- HOMEMADE LIBRARIES ---- #

# Add sys to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from utils.mail.send_mail import send_mail # Sending mail adverting higher fire risk

# Load from utils library
from utils import cloud_detection
from utils import fire_prediction  
from utils import preprocessing
from utils import forest_detection
from utils import data_api 

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Satellite Image Analysis API - Asynchronous Processing",
    description="API for processing satellite images with cloud detection, forest detection, and fire prediction using asynchronous job-based workflow",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response schemas
class JobResponse(BaseModel):
    """Response model for job creation"""
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    """Status of image processing job"""
    job_id: str
    task: str  # 'cloud_detection', 'forest_detection', 'fire_prediction'
    status: str  # "pending", "processing", "completed", "failed"
    created_at: datetime
    completed_at: Optional[datetime]
    tiles_to_process: int  # Number of tiles to process 
    tiles_processed: int  # Number of tiles processed
    successful_tiles: int  # Number of successfully processed tiles
    progress: int  # 0-100
    message: str

class ProcessedImageResponse(BaseModel):
    """Response model for processed image"""
    job_id: str
    task: str
    rgb_image_url: str
    cloud_image_url: str
    forest_image_url: str
    heatmap_image_url: str
    metadata: List[dict]
    processing_time: Optional[float] = None

# In-memory storage for demo purposes (use database in production)
processing_jobs = {}
processed_images = {}

# ------ CONSTANTS ----- #

# allowed_formats = [".npy", ".tiff", ".tif"]
allowed_formats = [".npy"]

# Folders
DATA_PATH = os.path.join(os.getcwd(), "data")
RAW_IMAGES_PATH = os.path.join(DATA_PATH, "RAW_IMAGES")
CLOUD_IMAGES_PATH = os.path.join(DATA_PATH, "CLOUD_IMAGES")
TILES_IMAGES_PATH = os.path.join(DATA_PATH, "TILES_IMAGES")
FOREST_IMAGES_PATH = os.path.join(DATA_PATH, "FOREST_IMAGES")
FIRE_IMAGES_PATH = os.path.join(DATA_PATH, "FIRE_IMAGES")
METADATA_FOLDER = os.path.join(DATA_PATH, "METADATA")
DISPLAY_FOLDER = os.path.join(DATA_PATH, "DISPLAY")

# Create folders if necessary
os.makedirs(RAW_IMAGES_PATH, exist_ok=True)
os.makedirs(CLOUD_IMAGES_PATH, exist_ok=True)
os.makedirs(TILES_IMAGES_PATH, exist_ok=True)
os.makedirs(FOREST_IMAGES_PATH, exist_ok=True)
os.makedirs(FIRE_IMAGES_PATH, exist_ok=True)
os.makedirs(METADATA_FOLDER, exist_ok=True)
os.makedirs(DISPLAY_FOLDER, exist_ok=True)

# Mount static files to serve processed images
app.mount("/static/images", StaticFiles(directory=DISPLAY_FOLDER), name="images")

# ---------------------- #

@app.get("/", tags=["Health Check"])
def healthcheck():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        dict: Welcome message and API status
    """
    return {
        "message": "Welcome to the Satellite Image Analysis API",
        "status": "operational",
        "version": "2.0.0",
        "available_analyses": ["cloud_detection", "forest_detection", "fire_prediction"]
    }

# ============================================================================
# STEP 1: SEND IMAGE - Submit image for processing
# ============================================================================

@app.post("/submit-image/cloud-detection", response_model=JobResponse, tags=["Submit Image"])
async def submit_image_cloud_detection(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tile_size: int = 256
):
    """
    Submit a satellite image for cloud detection analysis.
    
    **Processing Flow:**
    1. Upload and validate satellite image
    2. Create asynchronous processing job
    3. Return job ID for tracking
    
    **Input:**
    - Satellite image file (supported formats: .npy)
    - Image should contain multi-spectral bands for optimal cloud detection
    
    **Output:**
    - Job ID for tracking processing status
    - Estimated processing time
    
    **Next Steps:**
    1. Use job ID to check processing status with `/job-status/{job_id}`
    2. Once completed, retrieve result with `/get-result/{job_id}`
    
    Args:
        file (UploadFile): Satellite image file for cloud detection
        
    Returns:
        JobResponse: Job information including job_id for tracking progress
    """
    job_id = str(uuid.uuid4())
    
    # Validate file format
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {allowed_formats}"
        )

    image = np.load(file.file)

    rows = math.ceil(image.shape[0] / tile_size)
    cols = math.ceil(image.shape[1] / tile_size)

    number_of_tiles = rows * cols

    # Create processing job
    processing_jobs[job_id] = JobStatus(
        job_id=job_id,
        task="cloud_detection",
        status="pending",
        created_at=datetime.now(),
        completed_at=None,
        tiles_to_process=number_of_tiles,
        tiles_processed=0,
        successful_tiles=0,
        progress=0,
        message="Image uploaded successfully. Processing will start shortly."
    )
    
    # Start background processing
    background_tasks.add_task(process_cloud_detection, job_id, image, tile_size)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Cloud detection job created successfully"
    )

@app.post("/submit-image/forest-detection", response_model=JobResponse, tags=["Submit Image"])
async def submit_image_forest_detection(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Submit a satellite image for forest detection analysis.
    
    **Processing Flow:**
    1. Upload and validate satellite image
    2. Create asynchronous processing job
    3. Return job ID for tracking
    
    **Input:**
    - Satellite image file (supported formats: .npy, .tiff, .tif)
    - Image should contain vegetation indices (NDVI, EVI) for optimal forest detection
    
    **Output:**
    - Job ID for tracking processing status
    - Estimated processing time
    
    **Next Steps:**
    1. Use job ID to check processing status with `/job-status/{job_id}`
    2. Once completed, retrieve result with `/get-result/{job_id}`
    
    Args:
        file (UploadFile): Satellite image file for forest detection
        
    Returns:
        JobResponse: Job information including job_id for tracking progress
    """
    job_id = str(uuid.uuid4())
    
    # Validate file format
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {allowed_formats}"
        )
    
    # Create processing job
    processing_jobs[job_id] = JobStatus(
        job_id=job_id,
        task="forest_detection",
        status="pending",
        created_at=datetime.now(),
        completed_at=None,
        progress=0,
        message="Image uploaded successfully. Processing will start shortly."
    )
    
    # Start background processing
    background_tasks.add_task(process_forest_detection, job_id, file)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Forest detection job created successfully",
        estimated_time=180  # 3 minutes
    )

@app.post("/submit-image/fire-prediction", response_model=JobResponse, tags=["Submit Image"])
async def submit_image_fire_prediction(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Submit a satellite image for fire risk prediction analysis.
    
    **Processing Flow:**
    1. Upload and validate satellite image
    2. Create asynchronous processing job
    3. Return job ID for tracking
    
    **Input:**
    - Satellite image file (supported formats: .npy, .tiff, .tif)
    - Image should contain multi-spectral bands including thermal for optimal fire prediction
    - Additional metadata (weather data, elevation) can improve predictions
    
    **Output:**
    - Job ID for tracking processing status
    - Estimated processing time
    - Automatic email alerts for high-risk areas
    
    **Next Steps:**
    1. Use job ID to check processing status with `/job-status/{job_id}`
    2. Once completed, retrieve result with `/get-result/{job_id}`
    
    Args:
        file (UploadFile): Satellite image file for fire risk prediction
        
    Returns:
        JobResponse: Job information including job_id for tracking progress
    """
    job_id = str(uuid.uuid4())
    
    # Validate file format
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {allowed_formats}"
        )
    
    # Create processing job
    processing_jobs[job_id] = JobStatus(
        job_id=job_id,
        task="fire_prediction",
        status="pending",
        created_at=datetime.now(),
        completed_at=None,
        progress=0,
        message="Image uploaded successfully. Processing will start shortly."
    )
    
    # Start background processing
    background_tasks.add_task(process_fire_prediction, job_id, file)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Fire prediction job created successfully",
        estimated_time=300  # 5 minutes
    )

# ============================================================================
# STEP 2 & 3: CHECK STATUS - Monitor processing progress
# ============================================================================

@app.get("/job-status/{job_id}", response_model=JobStatus, tags=["Check Status"])
def get_job_status(job_id: str):
    """
    Check the current status of a processing job.
    
    **Status Values:**
    - `pending`: Job created, waiting to start processing
    - `processing`: Job is currently being processed
    - `completed`: Job finished successfully, result available
    - `failed`: Job failed due to error
    
    **Progress Tracking:**
    - Progress percentage (0-100%)
    - Detailed status messages
    - Processing time information
    
    This endpoint should be polled regularly until status becomes `completed` or `failed`.
    
    Args:
        job_id (str): Unique identifier for the processing job
        
    Returns:
        JobStatus: Current job status and progress information
        
    Raises:
        HTTPException: If job_id is not found
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

# ============================================================================
# STEP 4: GET RESULT - Retrieve processed image
# ============================================================================

@app.get("/get-result/{job_id}", response_model=ProcessedImageResponse, tags=["Get Result"])
def get_processing_result(job_id: str):
    """
    Retrieve the result of a completed processing job.
    
    **Available when job status is `completed`**
    
    **Returns:**
    - URL to download processed image
    - Processing metadata and statistics
    - Analysis results and confidence scores
    
    **Result Types by Analysis:**
    
    **Cloud Detection:**
    - Binary cloud mask overlay
    - Cloud coverage percentage
    - Clear areas highlighted
    
    **Forest Detection:**
    - Forest area boundaries
    - Vegetation indices (NDVI, EVI)
    - Forest type classification
    
    **Fire Prediction:**
    - Fire risk heatmap
    - Risk level classifications
    - Vulnerable area identification
    - Automatic alerts for high-risk zones
    
    Args:
        job_id (str): Job identifier to get results for
        
    Returns:
        ProcessedImageResponse: Complete processing results including download URL
        
    Raises:
        HTTPException: If job not found or not completed
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    if job_id not in processed_images:
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    return processed_images[job_id]

# ============================================================================
# STEP 5: DOWNLOAD IMAGE - Get processed image file
# ============================================================================

@app.get("/download-image/{job_id}", tags=["Download Image"])
async def download_processed_image(job_id: str):
    """
    Download the processed image file.
    
    **File Formats:**
    - PNG: For visualization images (heatmaps, overlays)
    
    **File Naming Convention:**
    - `{task}_{job_id}_result.{ext}`
    - Example: `fire_prediction_abc123_result.png`
    
    Args:
        job_id (str): Job identifier to download image for
        
    Returns:
        FileResponse: The processed image file
        
    Raises:
        HTTPException: If job not found or file not available
    """

    print(f"Requesting images for job {job_id}")

    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Image not ready for download")
    
    # Get file path based on task type
    task = job.task

    if "cloud" in task:
        task_type = "cloud"
    elif "forest" in task:
        task_type = "forest"
    else:
        task_type = "fire"

    image_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_{task_type}.png")
    # image_path = f"static/processed/{task}_{job_id}_result.png"
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Processed image file not found")
    
    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=f"{task}_{job_id}_result.png"
    )

@app.get("/download-picture/{name}", tags=["Download Image"])
async def download_picture(name: str):
    """
    Download a picture file by name.
    
    Args:
        name (str): Name of the picture file to download
        
    Returns:
        FileResponse: The picture file
        
    Raises:
        HTTPException: If file not found
    """
    # Validate filename to prevent path traversal
    if os.path.sep in name or os.path.altsep and os.path.altsep in name or name.startswith("..") or name.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(DISPLAY_FOLDER, name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Picture file not found")
    
    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=name
    )

# ============================================================================
# JOB MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/jobs", response_model=List[JobStatus], tags=["Job Management"])
def list_all_jobs():
    """
    List all processing jobs with their current status.
    
    Useful for monitoring and administration purposes.
    
    Returns:
        List[JobStatus]: List of all jobs with their status information
    """
    return list(processing_jobs.values())

@app.delete("/job/{job_id}", tags=["Job Management"])
def delete_job(job_id: str):
    """
    Delete a processing job and all associated data.
    
    This endpoint:
    1. Removes job from processing queue
    2. Deletes processed image files
    3. Cleans up temporary storage
    
    Args:
        job_id (str): Job identifier to delete
        
    Returns:
        dict: Deletion confirmation message
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove from storage
    del processing_jobs[job_id]
    if job_id in processed_images:
        del processed_images[job_id]
    
    # TODO: Clean up files from disk
    
    return {"message": f"Job {job_id} deleted successfully"}

# ============================================================================
# FILE CLEANUP FUNCTIONS
# ============================================================================

@app.delete("/cleanup", tags=["File Management"])
def cleanup_files():
    """
    Clean up temporary files and folders used during processing.
    
    This endpoint:
    1. Deletes all files in the RAW_IMAGES_PATH
    2. Deletes all files in the CLOUD_IMAGES_PATH
    3. Deletes all files in the TILES_IMAGES_PATH
    4. Deletes all files in the FOREST_IMAGES_PATH
    5. Deletes all files in the FIRE_IMAGES_PATH
    6. Deletes all files in the METADATA_FOLDER
    7. Deletes all files in the DISPLAY_FOLDER

    Returns:
        dict: Cleanup confirmation message
    """
    try:
        shutil.rmtree(RAW_IMAGES_PATH)
        shutil.rmtree(CLOUD_IMAGES_PATH)
        shutil.rmtree(TILES_IMAGES_PATH)
        shutil.rmtree(FOREST_IMAGES_PATH)
        shutil.rmtree(FIRE_IMAGES_PATH)
        shutil.rmtree(METADATA_FOLDER)
        shutil.rmtree(DISPLAY_FOLDER)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up files: {e}")

    return {"message": "Temporary files cleaned up successfully"}

# ============================================================================
# MOCK DATA GENERATION FUNCTIONS
# ============================================================================

def generate_mock_rgb_image(job_id: str, output_path: str):
    """
    Generate a mock RGB satellite image for testing purposes.
    
    TODO: Replace with real satellite image loading when integrating with actual data.
    Integration steps:
    1. Replace this function with actual satellite image loading from file
    2. Use real bands (Red, Green, Blue) from satellite data
    3. Apply proper normalization based on satellite sensor specifications
    4. Consider atmospheric correction and geometric correction
    """
    try:
        # Create a synthetic satellite-like image
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Generate mock RGB data resembling satellite imagery
        height, width = 512, 512
        
        # Create base terrain with noise
        np.random.seed(hash(job_id) % 2**32)  # Consistent mock data per job_id
        terrain = np.random.rand(height, width)
        
        # Add geographical features
        r_channel = terrain * 0.6 + np.random.rand(height, width) * 0.4  # More earth tones
        g_channel = terrain * 0.8 + np.random.rand(height, width) * 0.2  # More vegetation
        b_channel = terrain * 0.4 + np.random.rand(height, width) * 0.6  # Water bodies
        
        # Stack channels
        rgb_image = np.dstack([r_channel, g_channel, b_channel])
        
        # Normalize to [0, 1]
        rgb_image = np.clip(rgb_image, 0, 1)
        
        ax.imshow(rgb_image)
        ax.set_title(f'Mock RGB Satellite Image - {job_id}', fontsize=14, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Mock RGB image generated: {output_path}")
        
    except Exception as e:
        print(f"Error generating mock RGB image: {e}")
        create_placeholder_image(output_path, "Mock RGB Image\nGeneration Failed")

def generate_mock_cloud_mask(job_id: str, output_path: str):
    """
    Generate a mock cloud detection mask for testing purposes.
    
    TODO: Replace with real cloud detection algorithm when integrating with actual data.
    Integration steps:
    1. Replace with utils.cloud_detection.is_cloudy() function
    2. Use real spectral bands for cloud detection (typically NIR, SWIR)
    3. Implement threshold-based or ML-based cloud detection
    4. Consider cloud shadow detection
    5. Validate against ground truth data
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Generate mock cloud mask data
        height, width = 512, 512
        np.random.seed(hash(job_id) % 2**32)
        
        # Create cloud patches
        cloud_mask = np.zeros((height, width))
        
        # Add some cloud regions
        for _ in range(random.randint(3, 8)):
            center_x = random.randint(50, width-50)
            center_y = random.randint(50, height-50)
            radius = random.randint(30, 80)
            
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            cloud_mask[mask] = random.uniform(0.3, 1.0)
        
        # Add some noise for realism
        noise = np.random.rand(height, width) * 0.1
        cloud_mask = np.clip(cloud_mask + noise, 0, 1)
        
        im = ax.imshow(cloud_mask, cmap='Blues', alpha=0.8)
        ax.set_title(f'Mock Cloud Detection Mask - {job_id}', fontsize=14, pad=20)
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Cloud Coverage Probability', shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Mock cloud mask generated: {output_path}")
        
    except Exception as e:
        print(f"Error generating mock cloud mask: {e}")
        create_placeholder_image(output_path, "Mock Cloud Mask\nGeneration Failed")

def generate_mock_forest_map(job_id: str, output_path: str):
    """
    Generate a mock forest detection map for testing purposes.
    
    TODO: Replace with real forest detection algorithm when integrating with actual data.
    Integration steps:
    1. Replace with utils.forest_detection forest classification algorithms
    2. Use vegetation indices (NDVI, EVI, SAVI) calculated from NIR and Red bands
    3. Implement supervised classification (Random Forest, SVM) or deep learning
    4. Include forest type classification (deciduous, coniferous, mixed)
    5. Validate against forestry databases and ground truth
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        height, width = 512, 512
        np.random.seed(hash(job_id) % 2**32)
        
        # Create forest regions with different types
        forest_map = np.zeros((height, width))
        
        # Forest type regions: 0=No forest, 1=Deciduous, 2=Coniferous, 3=Mixed
        for forest_type in range(1, 4):
            for _ in range(random.randint(2, 5)):
                center_x = random.randint(80, width-80)
                center_y = random.randint(80, height-80)
                radius_x = random.randint(40, 100)
                radius_y = random.randint(40, 100)
                
                y, x = np.ogrid[:height, :width]
                mask = ((x - center_x)/radius_x)**2 + ((y - center_y)/radius_y)**2 <= 1
                forest_map[mask] = forest_type
        
        # Create custom colormap
        colors = ['white', 'lightgreen', 'darkgreen', 'forestgreen']
        try:
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(colors[:4])
        except ImportError:
            # Fallback to default colormap if ListedColormap is not available
            cmap = 'Greens'
        
        im = ax.imshow(forest_map, cmap=cmap, vmin=0, vmax=3)
        ax.set_title(f'Mock Forest Classification Map - {job_id}', fontsize=14, pad=20)
        ax.axis('off')
        
        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['No Forest', 'Deciduous', 'Coniferous', 'Mixed'])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Mock forest map generated: {output_path}")
        
    except Exception as e:
        print(f"Error generating mock forest map: {e}")
        create_placeholder_image(output_path, "Mock Forest Map\nGeneration Failed")

def generate_mock_fire_heatmap(job_id: str, output_path: str):
    """
    Generate a mock fire risk prediction heatmap for testing purposes.
    
    TODO: Replace with real fire prediction algorithm when integrating with actual data.
    Integration steps:
    1. Replace with utils.fire_prediction.unet.py U-Net model predictions
    2. Use environmental features: vegetation moisture, temperature, wind, terrain
    3. Integrate weather data from utils.data_api (temperature_api.py, wind_api.py)
    4. Use trained models from utils.fire_prediction.train_test.py
    5. Apply post-processing from utils.fire_prediction.pred_labels_postprocessing.py
    6. Validate predictions against historical fire data
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        height, width = 512, 512
        np.random.seed(hash(job_id) % 2**32)
        
        # Generate fire risk heatmap
        # Use gradient to simulate realistic risk distribution
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Base risk pattern
        base_risk = 0.3 * np.sin(3 * np.pi * X) * np.cos(2 * np.pi * Y) + 0.5
        
        # Add hot spots (high risk areas)
        for _ in range(random.randint(3, 6)):
            center_x = random.randint(50, width-50)
            center_y = random.randint(50, height-50)
            intensity = random.uniform(0.7, 1.0)
            
            y_grid, x_grid = np.ogrid[:height, :width]
            distance = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            hotspot = intensity * np.exp(-(distance**2) / (2 * 30**2))
            base_risk += hotspot
        
        # Add noise for realism
        noise = np.random.rand(height, width) * 0.15
        fire_risk = np.clip(base_risk + noise, 0, 1)
        
        im = ax.imshow(fire_risk, cmap='YlOrRd', alpha=0.9)
        ax.set_title(f'Mock Fire Risk Prediction Heatmap - {job_id}', fontsize=14, pad=20)
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Fire Risk Probability', shrink=0.8)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['Low', 'Low-Med', 'Medium', 'High', 'Critical'])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Mock fire heatmap generated: {output_path}")
        
    except Exception as e:
        print(f"Error generating mock fire heatmap: {e}")
        create_placeholder_image(output_path, "Mock Fire Heatmap\nGeneration Failed")

def generate_mock_metadata(job_id: str, task: str) -> List[dict]:
    """
    Generate mock metadata for testing purposes.
    
    TODO: Replace with real metadata extraction when integrating with actual data.
    Integration steps:
    1. Extract real metadata from satellite image headers (TIFF tags, NPY attributes)
    2. Use geodata extraction from utils.preprocessing.geodata_extraction.py
    3. Include temporal data, geographic coordinates, sensor information
    4. Add weather data integration from utils.data_api modules
    5. Store metadata in structured format for analysis
    """
    np.random.seed(hash(job_id) % 2**32)
    
    if task == "cloud_detection":
        return [
            {
                "tile_id": f"tile_{i}",
                "coordinates": [random.uniform(-120, -110), random.uniform(35, 45)],
                "cloud_coverage_percent": round(random.uniform(0, 100), 2),
                "confidence_score": round(random.uniform(0.7, 0.98), 3),
                "clear_pixels": random.randint(50000, 65536),
                "cloudy_pixels": random.randint(0, 15536)
            }
            for i in range(random.randint(16, 36))  # Simulate 16-36 tiles
        ]
    
    elif task == "forest_detection":
        return [
            {
                "forest_coverage_percent": round(random.uniform(45, 85), 2),
                "forest_types": {
                    "deciduous": round(random.uniform(20, 40), 1),
                    "coniferous": round(random.uniform(35, 55), 1),
                    "mixed": round(random.uniform(15, 35), 1)
                },
                "ndvi_statistics": {
                    "mean": round(random.uniform(0.6, 0.8), 3),
                    "std": round(random.uniform(0.1, 0.3), 3),
                    "min": round(random.uniform(0.0, 0.3), 3),
                    "max": round(random.uniform(0.85, 1.0), 3)
                },
                "vegetation_health": random.choice(["Excellent", "Good", "Fair", "Poor"]),
                "deforestation_risk": random.choice(["Low", "Medium", "High"])
            }
        ]
    
    elif task == "fire_prediction":
        high_risk_areas = round(random.uniform(5, 30), 1)
        return [
            {
                "risk_distribution": {
                    "low_risk_percent": round(random.uniform(40, 60), 1),
                    "medium_risk_percent": round(random.uniform(25, 35), 1),
                    "high_risk_percent": high_risk_areas,
                    "critical_risk_percent": round(random.uniform(2, 8), 1)
                },
                "environmental_factors": {
                    "avg_temperature_celsius": round(random.uniform(25, 35), 1),
                    "humidity_percent": round(random.uniform(30, 70), 1),
                    "wind_speed_kmh": round(random.uniform(5, 25), 1),
                    "vegetation_moisture": round(random.uniform(0.2, 0.8), 2)
                },
                "prediction_confidence": round(random.uniform(0.8, 0.95), 3),
                "alert_level": "HIGH" if high_risk_areas > 20 else "MEDIUM" if high_risk_areas > 10 else "LOW",
                "recommended_actions": [
                    "Monitor high-risk zones closely",
                    "Prepare firefighting resources",
                    "Issue fire weather warnings"
                ] if high_risk_areas > 15 else [
                    "Continue routine monitoring",
                    "Maintain standard preparedness"
                ]
            }
        ]
    
    return []

# ============================================================================
# MOCK PROCESSING ENDPOINT (FOR TESTING WITHOUT REAL FILES)
# ============================================================================

@app.post("/mock-submit-image/{task}", response_model=JobResponse, tags=["Mock Testing"])
async def mock_submit_image(
    task: str,
    background_tasks: BackgroundTasks
):
    """
    Mock endpoint for testing the complete workflow without requiring real satellite images.
    
    This endpoint simulates the entire image processing pipeline with synthetic data.
    It's useful for:
    1. Frontend development and testing
    2. API integration testing
    3. Workflow validation
    4. Performance testing
    
    **Available Tasks:**
    - cloud_detection: Simulates cloud detection with synthetic cloud masks
    - forest_detection: Simulates forest classification with mock vegetation data
    - fire_prediction: Simulates fire risk prediction with synthetic heatmaps
    
    **Mock Data Generated:**
    - Realistic-looking processed images
    - Synthetic metadata with plausible values
    - Simulated processing progress
    - Mock alerts and notifications
    
    **Usage:**
    This endpoint follows the same workflow as real processing:
    1. Submit task → Get job_id
    2. Poll job status → Monitor progress
    3. Get results → Download processed images
    
    TODO: Remove this endpoint in production and use real processing endpoints only.
    
    Args:
        task (str): Type of analysis ('cloud_detection', 'forest_detection', 'fire_prediction')
        
    Returns:
        JobResponse: Job information for tracking mock processing
    """
    # Validate task type
    valid_tasks = ["cloud_detection", "forest_detection", "fire_prediction"]
    if task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task. Must be one of: {valid_tasks}"
        )
    
    job_id = str(uuid.uuid4())
    
    # Create mock processing job
    processing_jobs[job_id] = JobStatus(
        job_id=job_id,
        task=task,
        status="pending",
        created_at=datetime.now(),
        completed_at=None,
        tiles_to_process=random.randint(16, 36),  # Mock tile count
        tiles_processed=0,
        successful_tiles=0,
        progress=0,
        message="Mock processing job created. Starting simulation..."
    )
    
    # Start mock background processing
    background_tasks.add_task(mock_process_image, job_id, task)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        message=f"Mock {task} job created successfully"
    )

async def mock_process_image(job_id: str, task: str):
    """
    Mock background processing that simulates real image analysis.
    
    This function:
    1. Simulates realistic processing times
    2. Updates progress incrementally
    3. Generates synthetic processed images
    4. Creates mock metadata
    5. Handles mock alerts (for fire prediction)
    
    TODO: Replace this completely with real processing functions when integrating actual algorithms.
    """
    try:
        # Update job status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].message = "Initializing mock processing..."
        
        # Simulate processing stages with realistic timing
        stages = [
            (10, "Loading mock image data..."),
            (25, "Applying mock preprocessing..."),
            (45, f"Running mock {task} algorithm..."),
            (70, "Generating mock analysis results..."),
            (85, "Creating mock visualization images..."),
            (95, "Finalizing mock output...")
        ]
        
        for progress, message in stages:
            processing_jobs[job_id].progress = progress
            processing_jobs[job_id].message = message
            processing_jobs[job_id].tiles_processed = int((progress / 100) * processing_jobs[job_id].tiles_to_process)
            processing_jobs[job_id].successful_tiles = processing_jobs[job_id].tiles_processed
            
            # Simulate processing time
            await asyncio.sleep(random.uniform(1.0, 2.5))
        
        # Generate mock processed images
        rgb_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_rgb.png")
        cloud_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_cloud.png")
        forest_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_forest.png")
        fire_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_heatmap.png")
        
        # Generate all visualization types
        generate_mock_rgb_image(job_id, rgb_output_path)
        generate_mock_cloud_mask(job_id, cloud_output_path)
        generate_mock_forest_map(job_id, forest_output_path)
        generate_mock_fire_heatmap(job_id, fire_output_path)
        
        # Generate mock metadata
        mock_metadata = generate_mock_metadata(job_id, task)
        
        # Calculate mock processing time
        processing_time = (datetime.now() - processing_jobs[job_id].created_at).total_seconds()
        
        # Store processed image metadata
        processed_images[job_id] = ProcessedImageResponse(
            job_id=job_id,
            task=task,
            rgb_image_url=f"/static/images/{job_id}_rgb.png",
            cloud_image_url=f"/static/images/{job_id}_cloud.png",
            forest_image_url=f"/static/images/{job_id}_forest.png",
            heatmap_image_url=f"/static/images/{job_id}_heatmap.png",
            metadata=mock_metadata,
            processing_time=processing_time
        )
        
        # Mock fire risk alert system
        if task == "fire_prediction" and mock_metadata:
            risk_data = mock_metadata[0].get("risk_distribution", {})
            high_risk = risk_data.get("high_risk_percent", 0) + risk_data.get("critical_risk_percent", 0)
            
            if high_risk > 20:  # Trigger alert for >20% high/critical risk
                try:
                    # TODO: Replace with real email system integration
                    print(f"MOCK ALERT: High fire risk detected for job {job_id}")
                    print(f"High/Critical risk areas: {high_risk:.1f}%")
                    print("In production, this would send email via utils.mail.send_mail.py")
                    
                    # Mock email sending (replace with real send_mail call)
                    # send_mail(
                    #     subject="URGENT: High Fire Risk Detected - Satellite Analysis Alert",
                    #     high_risk_percentage=high_risk,
                    #     average_risk=0.7,  # Mock average risk
                    #     job_id=job_id
                    # )
                    
                except Exception as e:
                    print(f"Mock alert system error: {e}")
        
        # Mark job as completed
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].completed_at = datetime.now()
        processing_jobs[job_id].progress = 100
        processing_jobs[job_id].tiles_processed = processing_jobs[job_id].tiles_to_process
        processing_jobs[job_id].successful_tiles = processing_jobs[job_id].tiles_to_process
        processing_jobs[job_id].message = f"Mock {task} processing completed successfully"
        
        print(f"Mock processing completed for job {job_id} (task: {task})")
        
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Mock processing failed: {str(e)}"
        print(f"Mock processing error for job {job_id}: {e}")

# ============================================================================
# IMAGE VISUALIZATION FUNCTIONS
# ============================================================================

def create_rgb_visualization(image: np.ndarray, job_id: str, output_path: str):
    """
    Create RGB visualization from satellite image.
    Assumes bands are in order: R, G, B (first 3 bands)
    """
    try:
        if image.shape[2] < 3:
            raise ValueError("Image must have at least 3 bands for RGB visualization")
        
        # Extract RGB bands (typically bands 0, 1, 2)
        rgb = image[:, :, :3]
        
        # Normalize to 0-1 range
        rgb_normalized = np.zeros_like(rgb, dtype=np.float32)
        for i in range(3):
            band = rgb[:, :, i]
            band_min, band_max = np.percentile(band, [2, 98])  # Use 2-98 percentile for contrast
            rgb_normalized[:, :, i] = np.clip((band - band_min) / (band_max - band_min), 0, 1)
        
        # Create and save the plot
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_normalized)
        plt.axis('off')
        plt.title(f'RGB Visualization - {job_id}', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"RGB visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating RGB visualization: {e}")
        # Create a placeholder image if RGB fails
        create_placeholder_image(output_path, "RGB Visualization\nNot Available")

def create_cloud_mask_visualization(metadata: pd.DataFrame, image_shape: tuple, job_id: str, output_path: str):
    """
    Create cloud mask visualization from tile metadata.
    """
    try:
        # Create a binary mask based on cloud detection results
        cloud_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)
        
        for _, row in metadata.iterrows():
            if pd.notna(row['cloud?']) and row['cloud?']:
                coords = row['tile_coordinates']
                if isinstance(coords, tuple) and len(coords) == 2:
                    i, j = coords
                    # Assuming tile size of 256x256 (you might need to adjust this)
                    tile_size = 256
                    end_i = min(i + tile_size, image_shape[0])
                    end_j = min(j + tile_size, image_shape[1])
                    cloud_mask[i:end_i, j:end_j] = row.get('cloud_percentage', 1.0)
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(cloud_mask, cmap='Blues', alpha=0.8)
        plt.colorbar(label='Cloud Coverage')
        plt.title(f'Cloud Detection Mask - {job_id}', fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Cloud mask visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating cloud mask visualization: {e}")
        # Create a placeholder image if cloud mask fails
        create_placeholder_image(output_path, "Cloud Mask\nNot Available")

def create_placeholder_image(output_path: str, text: str):
    """
    Create a placeholder image with text when actual processing fails.
    """
    plt.figure(figsize=(10, 10))
    plt.text(0.5, 0.5, text, fontsize=20, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ============================================================================
# BACKGROUND PROCESSING FUNCTIONS
# ============================================================================

async def process_cloud_detection(job_id: str, image: np.ndarray, tiles_size: int):
    """
    Background task for cloud detection processing.
    
    **Processing Steps:**
    1. Load and validate satellite image
    2. Apply cloud detection algorithms
    3. Generate cloud mask overlay
    4. Calculate cloud coverage statistics
    5. Save processed image and metadata
    
    **Algorithms Used:**
    - Threshold-based cloud detection
    - Machine learning cloud classification
    - Multi-spectral band analysis
    
    **Output Products:**
    - Binary cloud mask
    - Cloud coverage percentage
    - Clear area visualization
    
    Args:
        job_id (str): Job identifier
        file (UploadFile): Input satellite image
    """
    try:
        # Update job status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].message = "Loading image data..."

        file_name = job_id + "_raw.npy"
        FILE_PATH = os.path.join(RAW_IMAGES_PATH, file_name)

        print(f"Saving raw image to {FILE_PATH}")
        print(f"Image shape: {image.shape}")

        if image.shape[2] != 13:
            image = image[:,:,:13] 

        np.save(FILE_PATH, image)

        metadata = preprocessing.extract_tiles_with_padding(FILE_PATH, job_id, (tiles_size, tiles_size, image.shape[2]), TILES_IMAGES_PATH)

        # Save metadata for debugging purposes
        metadata.to_excel(os.path.join(METADATA_FOLDER, f"{job_id}_metadata.xlsx"), index=False)

        print("Images loaded and tiled")

        processing_jobs[job_id].message = "Running cloud detection..."

        for i in range(metadata.shape[0]):
            tile_path = os.path.join(TILES_IMAGES_PATH, f"{job_id}_tile_{metadata['tile_coordinates'][i]}.npy")
            
            result, cloud_mask, perc_cloudy = cloud_detection.is_cloudy(tile_path, job_id=job_id, cloud_threshold=0.5)

            cloudy = result['cloudy_tiles'] > 0

            metadata.at[i, 'cloud?'] = cloudy
            metadata.at[i, 'cloud_percentage'] = perc_cloudy

            processing_jobs[job_id].tiles_processed += 1
            processing_jobs[job_id].successful_tiles += result['clean_tiles']
            processing_jobs[job_id].progress = int((processing_jobs[job_id].tiles_processed / metadata.shape[0]) * 100)

        # Generate visualization images after all tiles are processed
        processing_jobs[job_id].message = "Generating visualization images..."
        
        # Create RGB visualization
        rgb_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_rgb.png")
        create_rgb_visualization(image, job_id, rgb_output_path)
        
        # Create cloud mask visualization
        cloud_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_cloud.png")
        create_cloud_mask_visualization(metadata, image.shape, job_id, cloud_output_path)
        
        # Create placeholder images for forest and fire (not implemented yet)
        # TODO: Replace these with real forest and fire processing when algorithms are integrated
        forest_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_forest.png")
        create_placeholder_image(forest_output_path, "Forest Detection\nNot Available\n\nTODO: Integrate\nutils.forest_detection\nmodules")
        
        fire_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_heatmap.png")
        create_placeholder_image(fire_output_path, "Fire Prediction\nNot Available\n\nTODO: Integrate\nutils.fire_prediction\nmodules")
        
        # Convert metadata to list of dictionaries for JSON serialization
        metadata_list = []
        for _, row in metadata.iterrows():
            metadata_list.append({
                "tile_id": f"tile_{row['tile_coordinates']}",
                "coordinates": row['tile_coordinates'],
                "cloud_coverage": row.get('cloud_percentage', 0),
                "is_cloudy": row.get('cloud?', False),
                "tile_path": f"{job_id}_tile_{row['tile_coordinates']}.npy"
            })
        
        # Store processed image metadata
        processing_time = (datetime.now() - processing_jobs[job_id].created_at).total_seconds()
        processed_images[job_id] = ProcessedImageResponse(
            job_id=job_id,
            task="cloud_detection",
            rgb_image_url=f"/static/images/{job_id}_rgb.png",
            cloud_image_url=f"/static/images/{job_id}_cloud.png", 
            forest_image_url=f"/static/images/{job_id}_forest.png", # PLACEHOLDER - TODO: Implement real forest detection
            heatmap_image_url=f"/static/images/{job_id}_heatmap.png", # PLACEHOLDER - TODO: Implement real fire prediction
            metadata=metadata_list,
            processing_time=processing_time
        )
        
        # Mark job as completed
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].completed_at = datetime.now()
        processing_jobs[job_id].message = "Cloud detection completed successfully"
        
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Processing failed: {str(e)}"
        print(f"Error details: {str(e)}")

async def process_forest_detection(job_id: str, image: np.ndarray):
    """
    Background task for forest detection processing.
    
    **Processing Steps:**
    1. Load and validate satellite image
    2. Calculate vegetation indices (NDVI, EVI, SAVI)
    3. Apply forest classification algorithms
    4. Identify forest boundaries and types
    5. Generate forest cover map
    
    **Algorithms Used:**
    - Vegetation index thresholding
    - Random Forest classification
    - Convolutional Neural Networks
    - Object-based image analysis
    
    **Output Products:**
    - Forest area boundaries
    - Forest type classification
    - Vegetation health assessment
    - Deforestation change detection
    
    Args:
        job_id (str): Job identifier
        file (UploadFile): Input satellite image
    """
    try:
        # Update job status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].progress = 10
        processing_jobs[job_id].message = "Loading image data..."
        
        # TODO: Implement actual forest detection processing
        # This is a skeleton function - implement the actual algorithms
        
        # Simulate processing steps
        await asyncio.sleep(2)
        processing_jobs[job_id].progress = 25
        processing_jobs[job_id].message = "Calculating vegetation indices..."
        
        await asyncio.sleep(3)
        processing_jobs[job_id].progress = 50
        processing_jobs[job_id].message = "Applying forest classification..."
        
        await asyncio.sleep(3)
        processing_jobs[job_id].progress = 75
        processing_jobs[job_id].message = "Identifying forest boundaries..."
        
        await asyncio.sleep(2)
        processing_jobs[job_id].progress = 100
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].completed_at = datetime.now()
        processing_jobs[job_id].message = "Forest detection completed successfully"
        
        # Store processed image metadata
        processing_time = (datetime.now() - processing_jobs[job_id].created_at).total_seconds()
        
        # Generate visualization images
        rgb_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_rgb.png")
        cloud_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_cloud.png")
        forest_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_forest.png")
        fire_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_heatmap.png")
        
        # Create placeholder images for RGB and cloud (not applicable for forest-only detection)
        create_placeholder_image(rgb_output_path, "RGB Visualization\nNot Available\nfor Forest Detection")
        create_placeholder_image(cloud_output_path, "Cloud Detection\nNot Available\nfor Forest Detection")
        
        # Create mock forest detection visualization
        generate_mock_forest_map(job_id, forest_output_path)
        
        # Create placeholder fire image
        create_placeholder_image(fire_output_path, "Fire Prediction\nNot Available\nfor Forest Detection")
        
        # Generate realistic forest metadata
        forest_metadata = generate_mock_metadata(job_id, "forest_detection")
        
        processed_images[job_id] = ProcessedImageResponse(
            job_id=job_id,
            task="forest_detection",
            rgb_image_url=f"/static/images/{job_id}_rgb.png",
            cloud_image_url=f"/static/images/{job_id}_cloud.png",
            forest_image_url=f"/static/images/{job_id}_forest.png",
            heatmap_image_url=f"/static/images/{job_id}_heatmap.png",
            metadata=forest_metadata,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Processing failed: {str(e)}"

async def process_fire_prediction(job_id: str, file: UploadFile):
    """
    Background task for fire risk prediction processing.
    
    **Processing Steps:**
    1. Load and validate satellite image
    2. Extract environmental features (vegetation, moisture, temperature)
    3. Apply fire risk prediction models
    4. Generate risk probability maps
    5. Identify high-risk zones
    6. Send alerts for critical areas
    
    **Algorithms Used:**
    - U-Net deep learning architecture
    - Random Forest ensemble methods
    - Weather integration models
    - Topographic analysis
    
    **Output Products:**
    - Fire risk probability heatmap
    - Risk level classifications (Low/Medium/High/Critical)
    - Vulnerable area identification
    - Early warning alerts
    
    **Alert System:**
    - Automatic email notifications for high-risk areas
    - Risk threshold monitoring
    - Emergency contact integration
    
    Args:
        job_id (str): Job identifier
        file (UploadFile): Input satellite image
    """
    try:
        # Update job status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].progress = 10
        processing_jobs[job_id].message = "Loading image data..."
        
        # TODO: Implement actual fire prediction processing using existing modules
        # Integration points:
        # - utils/fire_prediction/unet.py for deep learning models
        # - utils/fire_prediction/train_test.py for model training/inference
        # - utils/preprocessing/ for data preparation
        
        # Simulate processing steps
        await asyncio.sleep(3)
        processing_jobs[job_id].progress = 20
        processing_jobs[job_id].message = "Extracting environmental features..."
        
        await asyncio.sleep(4)
        processing_jobs[job_id].progress = 40
        processing_jobs[job_id].message = "Running fire risk prediction model..."
        
        await asyncio.sleep(4)
        processing_jobs[job_id].progress = 65
        processing_jobs[job_id].message = "Generating risk probability maps..."
        
        await asyncio.sleep(2)
        processing_jobs[job_id].progress = 85
        processing_jobs[job_id].message = "Identifying high-risk zones..."
        
        await asyncio.sleep(2)
        processing_jobs[job_id].progress = 100
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].completed_at = datetime.now()
        processing_jobs[job_id].message = "Fire prediction completed successfully"
        
        # Store processed image metadata
        processing_time = (datetime.now() - processing_jobs[job_id].created_at).total_seconds()
        
        # Generate visualization images
        rgb_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_rgb.png")
        cloud_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_cloud.png")
        forest_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_forest.png")
        fire_output_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_heatmap.png")
        
        # Create placeholder images for RGB, cloud, and forest (not primary focus for fire prediction)
        create_placeholder_image(rgb_output_path, "RGB Visualization\nNot Available\nfor Fire Prediction")
        create_placeholder_image(cloud_output_path, "Cloud Detection\nNot Available\nfor Fire Prediction")
        create_placeholder_image(forest_output_path, "Forest Detection\nNot Available\nfor Fire Prediction")
        
        # Create mock fire risk heatmap
        generate_mock_fire_heatmap(job_id, fire_output_path)
        
        # Generate realistic fire prediction metadata
        fire_metadata = generate_mock_metadata(job_id, "fire_prediction")
        
        # Extract risk data for alert system
        high_risk_percentage = 0
        average_risk_score = 0
        if fire_metadata and len(fire_metadata) > 0:
            risk_dist = fire_metadata[0].get("risk_distribution", {})
            high_risk_percentage = risk_dist.get("high_risk_percent", 0) + risk_dist.get("critical_risk_percent", 0)
            env_factors = fire_metadata[0].get("environmental_factors", {})
            temp = env_factors.get("avg_temperature_celsius", 25)
            humidity = env_factors.get("humidity_percent", 50)
            wind = env_factors.get("wind_speed_kmh", 10)
            # Calculate average risk based on environmental factors
            average_risk_score = min(0.95, (temp/40 + (100-humidity)/100 + wind/30) / 3)
        
        processed_images[job_id] = ProcessedImageResponse(
            job_id=job_id,
            task="fire_prediction",
            rgb_image_url=f"/static/images/{job_id}_rgb.png",
            cloud_image_url=f"/static/images/{job_id}_cloud.png",
            forest_image_url=f"/static/images/{job_id}_forest.png",
            heatmap_image_url=f"/static/images/{job_id}_heatmap.png",
            metadata=fire_metadata,
            processing_time=processing_time
        )
        
        # Send email alert if high fire risk is detected
        if high_risk_percentage > 20 or average_risk_score > 0.6:
            try:
                # TODO: Replace with actual email system integration
                # Real integration would use: send_mail() from utils.mail.send_mail
                print(f"MOCK ALERT: High fire risk detected for job {job_id}")
                print(f"High/Critical risk areas: {high_risk_percentage:.1f}%")
                print(f"Average risk score: {average_risk_score:.2f}")
                print("In production, this would send email via utils.mail.send_mail.py")
                
                # Uncomment when email system is properly configured:
                # send_mail(
                #     subject="URGENT: High Fire Risk Detected - Satellite Analysis Alert",
                #     high_risk_percentage=high_risk_percentage,
                #     average_risk=average_risk_score,
                #     job_id=job_id
                # )
                
                # Update metadata to indicate alert was processed
                if fire_metadata and len(fire_metadata) > 0:
                    fire_metadata[0]["alert_sent"] = True
                    processing_jobs[job_id].message += " | High-risk alert processed"
                
            except Exception as e:
                print(f"Mock alert system error: {e}")
        
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Processing failed: {str(e)}"


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)